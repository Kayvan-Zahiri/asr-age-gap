"""Acoustic measures: recording quality, speaking rate, and endpoint safety.

The last one is the point. A voice agent decides the caller has finished by
waiting for a fixed stretch of silence, typically 500-800 ms. Any pause *inside*
an utterance that exceeds that threshold is heard as the end of the turn, and
the agent starts talking over someone who was mid-sentence. That failure does
not appear in WER at all: the words the model did receive can be transcribed
perfectly while the caller is cut off every time.

So ``endpoint_cuts`` asks a question WER cannot: at a given silence threshold,
what fraction of utterances contain a pause long enough to be mistaken for the
end of the turn?
"""

from __future__ import annotations

import numpy as np

FRAME_MS = 25


def _frame_db(x: np.ndarray, sr: int, frame_ms: int = FRAME_MS) -> np.ndarray:
    n = max(1, int(sr * frame_ms / 1000))
    usable = len(x) // n * n
    if usable == 0:
        return np.array([])
    frames = x[:usable].reshape(-1, n)
    rms = np.sqrt((frames ** 2).mean(1) + 1e-12)
    return 20 * np.log10(rms + 1e-12)


def recording_quality(x: np.ndarray, sr: int) -> dict[str, float]:
    """Percentile SNR plus equipment tells.

    The absolute SNR is inflated by near-digital silence in mp3-encoded pauses,
    so it is only meaningful as a comparison between brackets measured the same
    way, which is how it is used.
    """
    db = _frame_db(x, sr)
    if db.size == 0:
        return {}
    noise, speech = np.percentile(db, 10), np.percentile(db, 90)
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    cum = np.cumsum(spec) / (spec.sum() + 1e-20)
    return {"snr_db": float(speech - noise), "noise_db": float(noise),
            "speech_db": float(speech),
            "rolloff_hz": float(np.searchsorted(cum, 0.99) * sr / len(x) / 2),
            "clipping": float((np.abs(x) > 0.99).mean())}


def speech_mask(x: np.ndarray, sr: int, *, rel_db: float = 25.0) -> np.ndarray:
    """Per-frame speech/silence, thresholded relative to this clip's own peak.

    A relative threshold is required: an absolute one would label quiet
    speakers as silent, and quiet speakers are exactly the population under
    study, so an absolute floor would manufacture the result.
    """
    db = _frame_db(x, sr)
    if db.size == 0:
        return np.array([], dtype=bool)
    return db > (np.percentile(db, 95) - rel_db)


def _internal_gaps_ms(mask: np.ndarray) -> list[float]:
    """Lengths of silence runs strictly between the first and last speech frame.

    Leading and trailing silence is excluded: it is recording margin, not a
    pause taken by the speaker.
    """
    idx = np.flatnonzero(mask)
    if idx.size < 2:
        return []
    gaps, run = [], 0
    for v in mask[idx[0]:idx[-1] + 1]:
        if v:
            if run:
                gaps.append(run * FRAME_MS)
            run = 0
        else:
            run += 1
    return gaps


def timing(x: np.ndarray, sr: int, n_words: int) -> dict[str, float]:
    """Speaking rate and pause profile, measured with both detectors.

    Articulation rate divides words by *voiced* time, so it inherits whatever
    the detector gets wrong. The energy threshold was already caught calling
    breathy trailing-off speech silent, which would shorten voiced time for
    exactly the older speakers under study and inflate their apparent rate.
    So the WebRTC figures are primary, the energy ones are kept alongside, and
    ``rate_wps_gross`` needs no detector at all: words over wall-clock
    duration, which cannot be gamed by a threshold.
    """
    dur_s = len(x) / sr
    out: dict[str, float] = {
        "dur_s": dur_s,
        "rate_wps_gross": n_words / dur_s if dur_s > 0.2 else float("nan"),
    }
    webrtc, energy = _webrtc_flags(x, sr), speech_mask(x, sr)
    for label, mask, frame in (("", webrtc, 30),
                               ("_energy", energy, FRAME_MS)):
        if mask.size == 0:
            continue
        gaps = _gaps_from_flags(mask, frame)
        voiced_s = float(mask.sum() * frame / 1000)
        out[f"voiced_s{label}"] = voiced_s
        out[f"rate_wps{label}"] = (n_words / voiced_s if voiced_s > 0.2
                                   else float("nan"))
        out[f"n_gaps{label}"] = len(gaps)
        out[f"max_gap_ms{label}"] = max(gaps) if gaps else 0.0
        out[f"total_pause_ms{label}"] = float(sum(gaps))
    return out


def _webrtc_flags(x: np.ndarray, sr: int = 16_000, *, aggressiveness: int = 2,
                  frame_ms: int = 30) -> np.ndarray:
    import webrtcvad

    vad = webrtcvad.Vad(aggressiveness)
    pcm = (np.clip(x, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
    n = int(sr * frame_ms / 1000) * 2
    return np.array([vad.is_speech(pcm[i:i + n], sr)
                     for i in range(0, len(pcm) - n, n)], dtype=bool)


def webrtc_gaps_ms(x: np.ndarray, sr: int = 16_000, *, aggressiveness: int = 2,
                   frame_ms: int = 30) -> list[float]:
    """Internal pause lengths according to WebRTC VAD.

    This is the default detector because the energy threshold above was caught
    manufacturing the result. On the eighties bracket the two disagreed on 36%
    of clips: relative-energy called the median longest pause 662 ms where
    WebRTC called it 345 ms, because breathy trailing-off speech drops under an
    energy floor while a trained detector still hears voicing. Since older
    speech is exactly what is quiet and breathy, the cheap detector's error is
    correlated with the variable under study, which is the worst possible
    property for a measurement instrument to have.
    """
    import webrtcvad

    vad = webrtcvad.Vad(aggressiveness)
    pcm = (np.clip(x, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
    n = int(sr * frame_ms / 1000) * 2
    flags = [vad.is_speech(pcm[i:i + n], sr) for i in range(0, len(pcm) - n, n)]
    return _gaps_from_flags(np.array(flags, dtype=bool))


def _gaps_from_flags(mask: np.ndarray, frame_ms: int = 30) -> list[float]:
    idx = np.flatnonzero(mask)
    if idx.size < 2:
        return []
    gaps, run = [], 0
    for v in mask[idx[0]:idx[-1] + 1]:
        if v:
            if run:
                gaps.append(run * frame_ms)
            run = 0
        else:
            run += 1
    return gaps


def endpoint_cuts(x: np.ndarray, sr: int, thresholds_ms=(400, 500, 700, 1000),
                  *, detector: str = "webrtc") -> dict[str, bool]:
    """Would a fixed-silence endpointer cut this utterance short?

    True at threshold T means the clip contains an internal pause of at least
    T ms, which an agent waiting T ms would read as the end of the turn and
    start talking over a caller who is still speaking.
    """
    if detector == "webrtc":
        gaps = webrtc_gaps_ms(x, sr)
    elif detector == "energy":
        gaps = _internal_gaps_ms(speech_mask(x, sr))
    else:
        raise ValueError(f"unknown detector {detector!r}")
    longest = max(gaps) if gaps else 0.0
    return {str(int(t)): bool(longest >= t) for t in thresholds_ms}
