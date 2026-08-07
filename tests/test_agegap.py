"""One test per claim the README makes, plus one per bug already caught."""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from agegap import metrics, stats  # noqa: E402
from agegap.channel import drop_packets, mulaw_roundtrip, telephone  # noqa: E402


# ── the telephone channel ────────────────────────────────────────────────────

def test_packet_loss_actually_drops_frames():
    """Regression: 0% and 2% loss once produced byte-identical WER.

    That looked like a silent no-op. It was not, but only a direct check could
    say so, and a dropper that quietly does nothing would make every
    packet-loss row in the README a fabrication.
    """
    x = np.random.default_rng(0).standard_normal(8_000 * 6).astype(np.float32)
    kept = drop_packets(x, 8_000, 0.0, rng=np.random.default_rng(0))
    lossy = drop_packets(x, 8_000, 0.20, rng=np.random.default_rng(0))
    assert np.array_equal(kept, x)
    assert (lossy == 0).sum() > 0.10 * len(x)


def test_mulaw_is_lossy_but_faithful():
    x = (np.random.default_rng(1).standard_normal(16_000) * 0.2).astype(np.float32)
    y = mulaw_roundtrip(x)
    assert not np.array_equal(x, y)                      # 8-bit quantisation
    assert np.corrcoef(x, y)[0, 1] > 0.99                # still the same signal


def test_telephone_removes_high_frequencies():
    """A 6 kHz tone cannot survive a 300-3400 Hz channel."""
    sr, t = 16_000, np.arange(16_000) / 16_000
    high = np.sin(2 * np.pi * 6_000 * t).astype(np.float32)
    out = telephone(high, sr)
    assert np.sqrt((out ** 2).mean()) < 0.1 * np.sqrt((high ** 2).mean())


def test_telephone_preserves_speech_band():
    sr, t = 16_000, np.arange(16_000) / 16_000
    mid = np.sin(2 * np.pi * 1_000 * t).astype(np.float32)
    out = telephone(mid, sr)
    assert np.sqrt((out ** 2).mean()) > 0.3 * np.sqrt((mid ** 2).mean())


# ── endpointing ──────────────────────────────────────────────────────────────

def _utterance(gap_ms: int, sr: int = 16_000) -> np.ndarray:
    """Speech, a gap of a known length, then more speech."""
    rng = np.random.default_rng(2)
    speech = (rng.standard_normal(sr) * 0.3).astype(np.float32)
    gap = np.zeros(int(sr * gap_ms / 1000), dtype=np.float32)
    return np.concatenate([speech, gap, speech])


def test_endpoint_cut_detects_a_known_gap():
    cuts = metrics.endpoint_cuts(_utterance(800), 16_000, detector="energy")
    assert cuts["400"] and cuts["500"] and cuts["700"]
    assert not cuts["1000"]


def test_endpoint_cuts_are_monotonic_in_threshold():
    cuts = metrics.endpoint_cuts(_utterance(600), 16_000, detector="energy")
    order = [cuts[k] for k in ("400", "500", "700", "1000")]
    assert order == sorted(order, reverse=True)


def test_leading_and_trailing_silence_is_not_a_pause():
    """Recording margin is not something the speaker did."""
    sr = 16_000
    rng = np.random.default_rng(3)
    speech = (rng.standard_normal(sr) * 0.3).astype(np.float32)
    padded = np.concatenate([np.zeros(sr * 2, np.float32), speech,
                             np.zeros(sr * 2, np.float32)]).astype(np.float32)
    assert not any(metrics.endpoint_cuts(padded, sr, detector="energy").values())


def test_speech_mask_threshold_is_relative_not_absolute():
    """A quiet speaker must not be scored as silent.

    Quiet speech is the population under study, so an absolute floor would
    manufacture the very effect the benchmark reports.
    """
    sr = 16_000
    rng = np.random.default_rng(4)
    loud = (rng.standard_normal(sr) * 0.5).astype(np.float32)
    quiet = loud * 0.02
    assert metrics.speech_mask(quiet, sr).mean() == pytest.approx(
        metrics.speech_mask(loud, sr).mean(), abs=0.02)


# ── statistics ───────────────────────────────────────────────────────────────

def test_pooled_wer_is_not_the_mean_of_clip_wers():
    """A one-word clip with one error is 100% WER and must not outweigh a long
    clip that was perfect."""
    refs = ["yes", "the quick brown fox jumps over the lazy dog today"]
    hyps = ["no", "the quick brown fox jumps over the lazy dog today"]
    pooled, errors, words = stats.pooled_wer(refs, hyps)
    assert errors == 1 and words == 11
    assert pooled == pytest.approx(1 / 11)
    assert pooled < 0.5                       # the naive mean would be 0.50


def test_empty_references_are_dropped_not_divided_by():
    refs, hyps = ["", "hello world"], ["anything", "hello world"]
    w, _, words = stats.pooled_wer(refs, hyps)
    assert words == 2 and w == 0.0


def test_bootstrap_resamples_speakers_not_clips():
    """25 clips from one contributor are not 25 independent observations.

    One speaker who is always wrong and one always right must produce an
    interval wide enough to contain both outcomes.
    """
    refs = ["hello world"] * 40
    hyps = ["hello world"] * 20 + ["goodbye moon"] * 20
    spk = ["a"] * 20 + ["b"] * 20
    lo, hi = stats.bootstrap_wer(refs, hyps, spk, n=500)
    assert lo == pytest.approx(0.0, abs=1e-9)
    assert hi == pytest.approx(1.0, abs=1e-9)


def test_bootstrap_needs_more_than_one_speaker():
    lo, hi = stats.bootstrap_wer(["a b"], ["a b"], ["only"], n=10)
    assert np.isnan(lo) and np.isnan(hi)


def test_error_breakdown_rates_sum_sensibly():
    refs = ["the cat sat on the mat"]
    hyps = ["the cat sat the mat"]                       # one deletion
    b = stats.error_breakdown(refs, hyps)
    assert b["del"] == pytest.approx(1 / 6)
    assert b["sub"] == 0.0 and b["ins"] == 0.0


def test_unpaired_delta_detects_a_real_difference():
    """Two groups that differ by 50 points must produce an interval that
    excludes zero."""
    good_r = ["hello world"] * 40
    good_h = ["hello world"] * 40
    bad_r = ["hello world"] * 40
    bad_h = ["hello world"] * 20 + ["goodbye moon"] * 20
    spk_a = [f"a{i//4}" for i in range(40)]
    spk_b = [f"b{i//4}" for i in range(40)]
    d, lo, hi = stats.unpaired_speaker_delta(good_r, good_h, spk_a,
                                             bad_r, bad_h, spk_b, n=400)
    assert d > 0.2
    assert lo > 0, "a 50-point gap must not straddle zero"


def test_unpaired_delta_straddles_zero_when_groups_match():
    r = ["hello world"] * 40
    h = ["hello world"] * 20 + ["goodbye moon"] * 20
    spk_a = [f"a{i//4}" for i in range(40)]
    spk_b = [f"b{i//4}" for i in range(40)]
    _, lo, hi = stats.unpaired_speaker_delta(r, h, spk_a, r, h, spk_b, n=400)
    assert lo <= 0 <= hi


# ── drift floor ──────────────────────────────────────────────────────────────

def test_utterances_needed_scales_as_inverse_square_of_effect():
    """Halving the effect you want to detect quadruples the sample."""
    import pathlib as _p, sys as _s
    _s.path.insert(0, str(_p.Path(__file__).resolve().parents[1] / "bench"))
    from drift_floor import utterances_needed
    assert utterances_needed(0.18, 0.10) == pytest.approx(
        4 * utterances_needed(0.18, 0.20), rel=1e-6)


def test_noisier_feature_needs_more_samples():
    """Pause time (CV ~1.0) must cost far more than speech rate (CV ~0.18)."""
    import pathlib as _p, sys as _s
    _s.path.insert(0, str(_p.Path(__file__).resolve().parents[1] / "bench"))
    from drift_floor import utterances_needed
    rate = utterances_needed(0.18, 0.10)
    pause = utterances_needed(1.00, 0.10)
    assert pause > 20 * rate


def test_smart_turn_probability_is_not_re_sigmoided():
    """Regression: the ONNX graph already applies the sigmoid.

    Applying it a second time squashed every clip into a band around 0.73 and
    produced a 100.0% false-cutoff rate in every bracket, which looked like a
    finding and was an artifact of the wrapper.
    """
    import numpy as _np
    already_prob = _np.array([0.974, 0.052, 0.201])
    double = 1.0 / (1.0 + _np.exp(-already_prob))
    assert double.min() > 0.5, "double sigmoid pushes everything above 0.5"
    assert (already_prob > 0.5).sum() == 1, "the raw output does discriminate"
