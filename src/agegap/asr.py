"""Whisper adapter.

fp16 on MPS is used because it is 2.6x faster than fp32 on CPU and was verified
to give an identical WER on a reference set (0.0274 for large-v3 on the
LibriSpeech dummy split under fp32/cpu, fp32/mps and fp16/mps alike). That check
is in ``tests/test_precision.py`` and is not decorative: a quantised Whisper KV
cache is capable of collapsing this model from 1.91% WER to 100%, so reduced
precision here is measured, never assumed.
"""

from __future__ import annotations

import numpy as np
import torch
from transformers import (WhisperForConditionalGeneration, WhisperProcessor,
                          logging as hf_logging)
from transformers.models.whisper.english_normalizer import EnglishTextNormalizer

hf_logging.set_verbosity_error()

_NORMALIZER = EnglishTextNormalizer({})


def normalize(text: str) -> str:
    return _NORMALIZER(text)


def pick_device() -> tuple[str, torch.dtype]:
    if torch.cuda.is_available():
        return "cuda", torch.float16
    if torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32


class Whisper:
    def __init__(self, name: str = "openai/whisper-large-v3",
                 device: str | None = None, dtype: torch.dtype | None = None):
        auto_dev, auto_dt = pick_device()
        self.device = device or auto_dev
        self.dtype = dtype or (auto_dt if device is None else torch.float32)
        self.name = name
        self.processor = WhisperProcessor.from_pretrained(name)
        self.model = WhisperForConditionalGeneration.from_pretrained(
            name, dtype=self.dtype).to(self.device).eval()

    def _generate(self, batch: list[np.ndarray], max_new_tokens: int,
                  device: str, dtype: torch.dtype) -> list[str]:
        feats = self.processor(batch, sampling_rate=16_000,
                               return_tensors="pt").input_features
        feats = feats.to(device, dtype)
        with torch.no_grad():
            ids = self.model.generate(feats, max_new_tokens=max_new_tokens,
                                      language="en", task="transcribe")
        return self.processor.batch_decode(ids, skip_special_tokens=True)

    def transcribe(self, clips: list[np.ndarray], *, batch_size: int = 16,
                   max_new_tokens: int = 200, retries: int = 3) -> list[str]:
        """Transcribe, surviving a Metal command-buffer failure.

        Sustained MPS load can drop a command buffer with "operations encoded
        on it may not have completed", which kills the process mid-run. Worse,
        the wording admits the possibility of a *silently* incomplete result,
        so a failed batch is never accepted: it is retried after clearing the
        cache, and if MPS keeps failing that batch is redone on CPU, which is
        slower but cannot come back quietly wrong.
        """
        out: list[str] = []
        for i in range(0, len(clips), batch_size):
            batch = clips[i:i + batch_size]
            for attempt in range(retries):
                try:
                    out.extend(self._generate(batch, max_new_tokens,
                                              self.device, self.dtype))
                    break
                except RuntimeError as exc:
                    if self.device == "mps":
                        torch.mps.empty_cache()
                    if attempt == retries - 1:
                        # Fall back rather than accept a possibly-partial result.
                        print(f"      MPS failed {retries}x, redoing batch on CPU: "
                              f"{str(exc)[:70]}", flush=True)
                        self.model.to("cpu", torch.float32)
                        out.extend(self._generate(batch, max_new_tokens,
                                                  "cpu", torch.float32))
                        self.model.to(self.device, self.dtype)
        return out

    def release(self) -> None:
        del self.model
        if self.device == "mps":
            torch.mps.empty_cache()
        elif self.device == "cuda":
            torch.cuda.empty_cache()


class Wav2Vec2:
    """A CTC recognizer, as an architectural control on the Whisper result.

    Whisper is an encoder-decoder model whose decoder is a language model, so
    it can repair a mangled acoustic frame from context. wav2vec2 is pure CTC:
    frame-wise, greedy, no decoder and no implicit LM. If the age effect
    appears in both, it cannot be an artifact of Whisper's decoder, which is
    the first thing anyone should suspect about a result showing older speakers
    are *easier* to transcribe.

    Absolute WER is much higher here (LibriSpeech-only training, no LM). Only
    the between-bracket comparison is meaningful.
    """

    def __init__(self, name: str = "facebook/wav2vec2-large-960h-lv60-self",
                 device: str | None = None):
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

        auto_dev, _ = pick_device()
        self.device = device or auto_dev
        self.dtype = torch.float32          # CTC in fp16 on MPS is not worth the risk
        self.name = name
        self.processor = Wav2Vec2Processor.from_pretrained(name)
        self.model = Wav2Vec2ForCTC.from_pretrained(name).to(self.device).eval()

    def transcribe(self, clips: list[np.ndarray], *, batch_size: int = 8,
                   **_) -> list[str]:
        out: list[str] = []
        for i in range(0, len(clips), batch_size):
            batch = clips[i:i + batch_size]
            inp = self.processor(batch, sampling_rate=16_000, return_tensors="pt",
                                 padding=True)
            vals = inp.input_values.to(self.device, self.dtype)
            mask = getattr(inp, "attention_mask", None)
            with torch.no_grad():
                logits = self.model(
                    vals, attention_mask=mask.to(self.device) if mask is not None else None
                ).logits
            ids = torch.argmax(logits, dim=-1)
            out.extend(self.processor.batch_decode(ids))
        return out

    def release(self) -> None:
        del self.model
        if self.device == "mps":
            torch.mps.empty_cache()
