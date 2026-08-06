"""Speaker-level resampling, and WER that is pooled rather than averaged.

Two easy mistakes this module exists to avoid.

Clips from one person are not independent observations. Bootstrapping over
clips treats 25 utterances from one contributor as 25 samples and reports an
interval far tighter than the data supports, which matters enormously here
because a single speaker can hold thousands of clips. Resampling is over
speakers.

And WER is a ratio of totals, not a mean of per-clip ratios. Averaging clip
WERs lets one short utterance with a single error outweigh a long one that was
transcribed perfectly, and lets a clip whose reference normalises to the empty
string divide by zero.
"""

from __future__ import annotations

import collections

import jiwer
import numpy as np


def pooled_wer(refs: list[str], hyps: list[str]) -> tuple[float, int, int]:
    """Corpus WER: total edits over total reference words.

    Returns (wer, n_errors, n_words). Empty references are dropped, since they
    carry no words to be wrong about.
    """
    pairs = [(r, h) for r, h in zip(refs, hyps) if r.strip()]
    if not pairs:
        return float("nan"), 0, 0
    m = jiwer.process_words([r for r, _ in pairs], [h for _, h in pairs])
    errors = m.substitutions + m.deletions + m.insertions
    words = m.substitutions + m.deletions + m.hits
    return errors / words, errors, words


def error_breakdown(refs: list[str], hyps: list[str]) -> dict[str, float]:
    """Substitution / deletion / insertion rates, each over reference words."""
    pairs = [(r, h) for r, h in zip(refs, hyps) if r.strip()]
    if not pairs:
        return {}
    m = jiwer.process_words([r for r, _ in pairs], [h for _, h in pairs])
    words = m.substitutions + m.deletions + m.hits
    return {"sub": m.substitutions / words, "del": m.deletions / words,
            "ins": m.insertions / words, "words": words}


def bootstrap_wer(refs: list[str], hyps: list[str], speakers: list[str],
                  *, n: int = 2000, seed: int = 0,
                  alpha: float = 0.05) -> tuple[float, float]:
    """Percentile CI for pooled WER, resampling speakers with replacement."""
    by: dict[str, list[tuple[str, str]]] = collections.defaultdict(list)
    for r, h, s in zip(refs, hyps, speakers):
        if r.strip():
            by[s].append((r, h))
    keys = list(by)
    if len(keys) < 2:
        return float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n):
        pick = rng.integers(0, len(keys), len(keys))
        rs, hs = [], []
        for i in pick:
            for r, h in by[keys[i]]:
                rs.append(r)
                hs.append(h)
        draws.append(pooled_wer(rs, hs)[0])
    lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def unpaired_speaker_delta(a_refs, a_hyps, a_spk, b_refs, b_hyps, b_spk,
                           *, n: int = 2000, seed: int = 0
                           ) -> tuple[float, float, float]:
    """CI for WER(b) - WER(a) across two independent groups of speakers.

    Needed because comparing two brackets by whether their separate intervals
    overlap is the wrong test: non-overlap implies a difference, but overlap
    does not imply its absence, so the naive reading silently under-reports.
    Each group is resampled over its own speakers and the difference is taken
    inside the loop.
    """
    def group(refs, hyps, spk):
        by: dict[str, list[tuple[str, str]]] = collections.defaultdict(list)
        for r, h, s in zip(refs, hyps, spk):
            if r.strip():
                by[s].append((r, h))
        return by, list(by)

    a_by, a_keys = group(a_refs, a_hyps, a_spk)
    b_by, b_keys = group(b_refs, b_hyps, b_spk)
    if len(a_keys) < 2 or len(b_keys) < 2:
        return float("nan"), float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n):
        ar, ah, br, bh = [], [], [], []
        for i in rng.integers(0, len(a_keys), len(a_keys)):
            for r, h in a_by[a_keys[i]]:
                ar.append(r); ah.append(h)
        for i in rng.integers(0, len(b_keys), len(b_keys)):
            for r, h in b_by[b_keys[i]]:
                br.append(r); bh.append(h)
        draws.append(pooled_wer(br, bh)[0] - pooled_wer(ar, ah)[0])
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(np.median(draws)), float(lo), float(hi)


def paired_speaker_delta(a_refs, a_hyps, b_refs, b_hyps, speakers,
                         *, n: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """CI for the WER change between two conditions on the SAME speakers.

    Used for the wideband-versus-telephone contrast, where each speaker appears
    in both arms and the paired structure removes between-speaker variance.
    """
    by: dict[str, list[tuple[str, str, str, str]]] = collections.defaultdict(list)
    for ar, ah, br, bh, s in zip(a_refs, a_hyps, b_refs, b_hyps, speakers):
        if ar.strip():
            by[s].append((ar, ah, br, bh))
    keys = list(by)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n):
        pick = rng.integers(0, len(keys), len(keys))
        ar_, ah_, br_, bh_ = [], [], [], []
        for i in pick:
            for ar, ah, br, bh in by[keys[i]]:
                ar_.append(ar); ah_.append(ah); br_.append(br); bh_.append(bh)
        draws.append(pooled_wer(br_, bh_)[0] - pooled_wer(ar_, ah_)[0])
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(np.median(draws)), float(lo), float(hi)
