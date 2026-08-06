"""Reduced precision is verified here, not assumed.

The benchmark runs fp16 on MPS because it is ~2.6x faster than fp32 on CPU.
That shortcut is only safe if it changes no answers, and the reason to doubt it
is concrete: a quantised Whisper cross-attention KV cache can take large-v3
from 1.91% WER to 100%, with the decoder emitting a language token until it
hits the generation cap. Different mechanism from ordinary fp16 inference, but
close enough that publishing an age effect on unverified fp16 would be
indefensible if it turned out to be a precision artifact.

Slow: downloads large-v3 and decodes 73 clips three ways.

    python3 -m pytest tests/test_precision.py -q -m slow
"""

from __future__ import annotations

import io
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

pytestmark = pytest.mark.slow

REFERENCE_WER = 0.0274        # large-v3, librispeech dummy, fp32/cpu
TOLERANCE = 0.002


def _librispeech():
    import soundfile as sf
    from datasets import Audio, load_dataset

    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean",
                      split="validation").cast_column("audio", Audio(decode=False))
    clips, refs = [], []
    for r in ds:
        x, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
        assert sr == 16_000
        clips.append(x if x.ndim == 1 else x.mean(1))
        refs.append(r["text"])
    return clips, refs


def _wer_under(device, dtype):
    import torch

    from agegap import asr, stats

    clips, refs = _librispeech()
    model = asr.Whisper("openai/whisper-large-v3", device=device, dtype=dtype)
    hyps = model.transcribe(clips)
    model.release()
    return stats.pooled_wer([asr.normalize(t) for t in refs],
                            [asr.normalize(t) for t in hyps])[0]


def test_fp16_on_mps_matches_fp32_on_cpu():
    import torch

    if not torch.backends.mps.is_available():
        pytest.skip("no MPS device")
    fast = _wer_under("mps", torch.float16)
    assert fast == pytest.approx(REFERENCE_WER, abs=TOLERANCE), (
        f"fp16/mps WER {fast:.4f} differs from the fp32/cpu reference "
        f"{REFERENCE_WER:.4f}; every age number in the README is suspect until "
        f"this is explained")


def test_fp32_on_cpu_still_matches_the_recorded_reference():
    """Guards against a transformers upgrade silently moving the baseline."""
    import torch

    slow = _wer_under("cpu", torch.float32)
    assert slow == pytest.approx(REFERENCE_WER, abs=TOLERANCE)
