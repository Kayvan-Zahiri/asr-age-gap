"""A simulated PSTN/VoIP channel, because that is what the target runs on.

Common Voice is 48 kHz into a laptop mic. careCycle and Kaigo are 8 kHz phone
calls. Benchmarking clean studio audio answers a question nobody in voice-health
is asking, so every clip is scored twice: wideband, and through this.

Stages match G.711 telephony: band-limit to 300-3400 Hz, decimate to 8 kHz,
mu-law compand to 8 bits, then upsample back to 16 kHz for the model.
"""
from __future__ import annotations
import numpy as np
from scipy.signal import butter, sosfilt, resample_poly

MU = 255.0

def mulaw_roundtrip(x: np.ndarray) -> np.ndarray:
    """G.711 mu-law: compand, quantise to 8 bits, expand. Lossy on purpose."""
    peak = np.max(np.abs(x)) or 1.0
    s = np.clip(x / peak, -1.0, 1.0)
    comp = np.sign(s) * np.log1p(MU * np.abs(s)) / np.log1p(MU)
    q = np.round((comp + 1.0) * 127.5) / 127.5 - 1.0          # 8-bit
    exp = np.sign(q) * ((1.0 + MU) ** np.abs(q) - 1.0) / MU
    return (exp * peak).astype(np.float32)

def drop_packets(x: np.ndarray, sr: int, rate: float, ms: int = 20,
                 rng: np.random.Generator | None = None) -> np.ndarray:
    """Zero whole 20 ms frames, the way a jitter buffer loses them."""
    if rate <= 0: return x
    rng = rng or np.random.default_rng(0)
    n = int(sr * ms / 1000)
    y = x.copy()
    for i in range(0, len(y) - n, n):
        if rng.random() < rate: y[i:i+n] = 0.0
    return y

def telephone(x: np.ndarray, sr: int, *, loss: float = 0.0,
              rng: np.random.Generator | None = None) -> np.ndarray:
    """48/16 kHz wideband in, 16 kHz narrowband-degraded out."""
    if x.ndim > 1: x = x.mean(1)
    x = x.astype(np.float32)
    sos = butter(4, [300, 3400], btype="band", fs=sr, output="sos")
    x = sosfilt(sos, x).astype(np.float32)
    g = np.gcd(int(sr), 8000)
    x8 = resample_poly(x, 8000 // g, int(sr) // g).astype(np.float32)
    x8 = mulaw_roundtrip(x8)
    if loss: x8 = drop_packets(x8, 8000, loss, rng=rng)
    return resample_poly(x8, 2, 1).astype(np.float32)          # 8k -> 16k
