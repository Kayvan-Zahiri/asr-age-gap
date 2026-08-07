"""How much must a person's speech change before you could possibly detect it?

    python3 bench/drift_floor.py

Several companies now sell daily phone check-ins for older adults that claim to
flag cognitive decline from "voice biomarkers" — speech rate, pause structure,
response latency. One cites 70-90% sensitivity. None publish a validation.

Validating the clinical claim needs DementiaBank or ADReSS, which are gated.
But there is a prior question that needs no clinical labels at all, and it
bounds every one of those claims from below:

**How much does a healthy person's speech vary from utterance to utterance?**

A drift detector can only see a change that exceeds the speaker's own noise. If
speech rate wobbles 20% between two clips recorded minutes apart, then a
genuine 10% slowing is invisible until you have averaged enough calls to shrink
the standard error below it. That required number of calls is computable, and
it decides whether a daily-call product can work at all.

This measures the within-speaker spread of the exact features those products
use, on Common Voice speakers with enough clips to characterise themselves, and
converts it into the number of sessions needed to detect a given change.

**Why the answer here is optimistic, i.e. a floor.** Common Voice clips from
one contributor are often recorded in a single sitting: same room, same
microphone, same time of day, same mood. Real day-to-day variation can only be
larger. So a session count that already looks impractical here is worse in the
field.
"""

from __future__ import annotations

import collections
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from agegap import audio, metrics, sample  # noqa: E402

RES = pathlib.Path(__file__).resolve().parents[1] / "results"
BRACKETS = ("twenties", "sixties", "seventies")
MIN_CLIPS = 40           # enough to characterise one speaker
MAX_CLIPS = 60           # cap the work per speaker
MAX_SPEAKERS = 12        # per bracket

FEATURES = {
    "rate_wps": "speech rate (words / voiced second)",
    "total_pause_ms": "total pause time per utterance",
    "n_gaps": "number of internal pauses",
    "dur_s": "utterance duration",
}


UTTERANCES_PER_CALL = 40      # a 5-minute call, mostly the caller talking


def utterances_needed(cv: float, effect: float, power_z: float = 2.8) -> float:
    """Utterances to resolve a relative change of `effect`, given a
    within-speaker coefficient of variation `cv`.

    The unit is one utterance, not one call. Standard error of a mean over n
    utterances is cv/sqrt(n); a change must clear roughly `power_z` standard
    errors (2.8 gives ~80% power, two-sided, alpha 0.05).

    **Converting to calls is optimistic.** Dividing by UTTERANCES_PER_CALL
    assumes utterances within a call are independent, and they are not: they
    share mood, time of day, fatigue, background noise and whatever the person
    is talking about. The effective sample size per call is therefore smaller
    than the raw count, so the call figures below are a lower bound on what a
    real product would need.
    """
    if effect <= 0:
        return float("inf")
    return (power_z * cv / effect) ** 2


def main() -> int:
    clips = sample.load_train(BRACKETS)
    by_spk: dict[tuple[str, str], list] = collections.defaultdict(list)
    for c in clips:
        if c.shard <= 11:
            by_spk[(c.speaker, c.age)].append(c)

    chosen = []
    for b in BRACKETS:
        cands = sorted(((k, v) for k, v in by_spk.items() if k[1] == b),
                       key=lambda kv: -len(kv[1]))
        for (spk, age), cs in cands[:MAX_SPEAKERS]:
            if len(cs) >= MIN_CLIPS:
                chosen.append((spk, age, cs[:MAX_CLIPS]))
    print(f"  {len(chosen)} speakers, "
          f"{sum(len(c) for _, _, c in chosen)} clips\n")

    want_by_shard: dict[int, set[str]] = collections.defaultdict(set)
    meta = {}
    for spk, age, cs in chosen:
        for c in cs:
            want_by_shard[c.shard].add(c.path)
            meta[c.path] = (spk, age, c.n_words)

    rows = []
    for shard in sorted(want_by_shard):
        wave = audio.read_shard(shard, want_by_shard[shard])
        for path, x in wave.items():
            spk, age, nw = meta[path]
            t = metrics.timing(x, 16_000, nw)
            if not t or t.get("rate_wps") != t.get("rate_wps"):
                continue
            rows.append({"speaker": spk, "age": age, **{k: t[k] for k in FEATURES
                                                        if k in t}})
        print(f"    shard {shard}: {len(wave)} clips", flush=True)
        del wave
    json.dump(rows, open(RES / "drift_floor.json", "w"))

    print(f"\n  within-speaker coefficient of variation "
          f"(spread of one person's own clips)\n")
    print("  feature                              " +
          "".join(f"{b[:9]:>12}" for b in BRACKETS))
    cvs: dict[str, dict[str, float]] = {}
    for f, label in FEATURES.items():
        cvs[f] = {}
        line = f"  {label[:36]:<38}"
        for b in BRACKETS:
            per = collections.defaultdict(list)
            for r in rows:
                if r["age"] == b and f in r:
                    per[r["speaker"]].append(r[f])
            vals = [np.std(v, ddof=1) / np.mean(v)
                    for v in per.values() if len(v) > 5 and np.mean(v) > 0]
            cv = float(np.median(vals)) if vals else float("nan")
            cvs[f][b] = cv
            line += f"{cv:>11.1%} "
        print(line)

    print("\n  UTTERANCES needed to detect a change, 80% power\n")
    print("  feature                   change    " +
          "".join(f"{b[:9]:>11}" for b in BRACKETS))
    need = {}
    for f, label in FEATURES.items():
        for eff in (0.10, 0.20):
            line = f"  {label[:24]:<26} {eff:>5.0%}    "
            for b in BRACKETS:
                n = utterances_needed(cvs[f][b], eff)
                need[(f, eff, b)] = n
                line += f"{n:>10.0f} " if n < 1e4 else f"{'>10k':>10} "
            print(line)

    print(f"\n  DAILY CALLS needed, at {UTTERANCES_PER_CALL} utterances per call")
    print("  (optimistic: utterances within one call are correlated,")
    print("   so the true effective sample size is smaller than the count)\n")
    print("  feature                   change    " +
          "".join(f"{b[:9]:>11}" for b in BRACKETS))
    for f, label in FEATURES.items():
        for eff in (0.10, 0.20):
            line = f"  {label[:24]:<26} {eff:>5.0%}    "
            for b in BRACKETS:
                d = need[(f, eff, b)] / UTTERANCES_PER_CALL
                line += f"{d:>10.1f} "
            print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
