"""Pull selected clips out of the Common Voice shard tars.

A tar is read once, start to finish, and every wanted member is taken on that
pass. Seeking per file would re-walk the archive for each clip and turn a
one-minute read into an hour.
"""

from __future__ import annotations

import io
import os
import tarfile

import numpy as np
import soundfile as sf
from huggingface_hub import hf_hub_download
from scipy.signal import resample_poly

REPO = "fsicoli/common_voice_17_0"
TARGET_SR = 16_000


def shard_path(shard: int) -> str:
    """Download the shard if absent, return its local path."""
    return hf_hub_download(REPO, f"audio/en/train/en_train_{shard}.tar",
                           repo_type="dataset")


def to_16k_mono(x: np.ndarray, sr: int) -> np.ndarray:
    if x.ndim > 1:
        x = x.mean(1)
    x = x.astype(np.float32)
    if sr != TARGET_SR:
        g = np.gcd(int(sr), TARGET_SR)
        x = resample_poly(x, TARGET_SR // g, int(sr) // g).astype(np.float32)
    return x


def read_shard(shard: int, wanted: set[str]) -> dict[str, np.ndarray]:
    """Decode every wanted member of one shard in a single sequential pass."""
    out: dict[str, np.ndarray] = {}
    if not wanted:
        return out
    with tarfile.open(shard_path(shard)) as tf:
        for member in tf:
            if not member.isfile():
                continue
            name = os.path.basename(member.name)
            if name not in wanted:
                continue
            try:
                raw = tf.extractfile(member).read()
                x, sr = sf.read(io.BytesIO(raw), dtype="float32")
            except Exception:
                continue          # a handful of clips fail to decode; skip, do not fake
            out[name] = to_16k_mono(x, sr)
    return out
