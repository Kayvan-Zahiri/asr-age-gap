"""Turn one or more result files into the tables the README publishes.

    python3 bench/analyze.py results/run_whisper-large-v3_native.json

Given two files it also prints the controlled-versus-confounded contrast, which
is the point of keeping the uncontrolled run around: the size of the accent
artifact is itself a result.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from agegap import stats  # noqa: E402

THRESHOLDS = (400, 500, 700, 1000)


def load(path: str) -> dict:
    return json.load(open(path))


def wer_table(recs: list[dict], brackets: list[str]) -> None:
    print("  bracket      n    spk   wideband WER          telephone WER        channel cost")
    for b in brackets:
        s = [r for r in recs if r["age"] == b]
        if not s:
            continue
        spk = [r["speaker"] for r in s]
        refs = [r["ref"] for r in s]
        cells = []
        for cond in ("wideband", "telephone"):
            hyps = [r[f"hyp_{cond}"] for r in s]
            w, _, _ = stats.pooled_wer(refs, hyps)
            lo, hi = stats.bootstrap_wer(refs, hyps, spk)
            cells.append(f"{w:6.2%} [{lo:5.2%},{hi:5.2%}]")
        d, dlo, dhi = stats.paired_speaker_delta(
            refs, [r["hyp_wideband"] for r in s],
            refs, [r["hyp_telephone"] for r in s], spk)
        print(f"  {b:<11} {len(s):>4} {len(set(spk)):>6}  {cells[0]}  {cells[1]}"
              f"  {d:+6.2%} [{dlo:+5.2%},{dhi:+5.2%}]")


def relative_to_baseline(recs: list[dict], brackets: list[str],
                         cond: str = "wideband") -> None:
    """Each bracket's WER against the youngest, with a speaker bootstrap on the
    difference. An overlapping interval means the data does not support a claim
    of difference, and that is worth printing rather than hiding."""
    base_b = brackets[0]
    base = [r for r in recs if r["age"] == base_b]
    b_refs = [r["ref"] for r in base]
    b_hyps = [r[f"hyp_{cond}"] for r in base]
    b_spk = [r["speaker"] for r in base]
    bw, _, _ = stats.pooled_wer(b_refs, b_hyps)
    print(f"\n  {cond} WER relative to the {base_b} ({bw:.2%})\n")
    print("  bracket       WER   ratio   difference vs baseline [95% CI]   verdict")
    for b in brackets:
        s = [r for r in recs if r["age"] == b]
        if not s or b == base_b:
            if s:
                print(f"  {b:<11} {bw:>6.2%} {1.0:>6.2f}x   "
                      f"{'(baseline)':>32}")
            continue
        refs = [r["ref"] for r in s]
        hyps = [r[f"hyp_{cond}"] for r in s]
        spk = [r["speaker"] for r in s]
        w, _, _ = stats.pooled_wer(refs, hyps)
        d, lo, hi = stats.unpaired_speaker_delta(b_refs, b_hyps, b_spk,
                                                 refs, hyps, spk)
        # The interval on the DIFFERENCE is the test. Comparing two separate
        # intervals for overlap under-reports real differences.
        verdict = "excludes zero" if (lo > 0 or hi < 0) else "includes zero"
        print(f"  {b:<11} {w:>6.2%} {w/bw:>6.2f}x   "
              f"{d:>+8.2%} [{lo:>+6.2%},{hi:>+6.2%}]        {verdict}")


def cutoff_table(recs: list[dict], brackets: list[str]) -> None:
    print("\n  premature-cutoff rate, WebRTC VAD (energy VAD in brackets)\n")
    print("  bracket      " + "".join(f"{t}ms".rjust(16) for t in THRESHOLDS))
    for b in brackets:
        s = [r for r in recs if r["age"] == b]
        if not s:
            continue
        cells = []
        for t in THRESHOLDS:
            w = np.mean([r["cuts"][str(t)] for r in s])
            e = np.mean([r["cuts_energy"][str(t)] for r in s]) \
                if "cuts_energy" in s[0] else float("nan")
            cells.append(f"{w:6.1%} ({e:5.1%})")
        print(f"  {b:<11} " + "".join(c.rjust(16) for c in cells))


def acoustics(recs: list[dict], brackets: list[str]) -> None:
    print("\n  acoustic profile (the confound checks)\n")
    print("  bracket      SNR dB   noise dB   words/sec   voiced s   pause ms")
    for b in brackets:
        s = [r for r in recs if r["age"] == b]
        if not s:
            continue
        med = lambda k: np.median([r[k] for r in s if r.get(k) == r.get(k)])
        print(f"  {b:<11} {med('snr_db'):>7.1f} {med('noise_db'):>10.1f}"
              f" {med('rate_wps'):>11.2f} {med('voiced_s'):>10.2f}"
              f" {med('total_pause_ms'):>10.0f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    args = ap.parse_args()

    loaded = []
    for f in args.files:
        d = load(f)
        loaded.append((f, d))
        brackets = d.get("brackets") or ["twenties", "sixties", "seventies"]
        label = (f"accents={d.get('accents','?')}"
                 f"{' matched' if d.get('matched') else ' UNMATCHED'}")
        print(f"\n{'='*84}\n  {pathlib.Path(f).name}   {d['model']}   {label}\n{'='*84}\n")
        wer_table(d["records"], brackets)
        relative_to_baseline(d["records"], brackets, "wideband")
        relative_to_baseline(d["records"], brackets, "telephone")
        cutoff_table(d["records"], brackets)
        acoustics(d["records"], brackets)

    if len(loaded) == 2:
        print(f"\n{'='*84}\n  what ignoring accent costs (wideband WER)\n{'='*84}\n")

        def label(d):
            a = d.get("accents", "?")
            return f"{a}, matched" if d.get("matched") else f"{a}, unmatched"

        print("  bracket      " + "".join(f"{label(d):>22}" for _, d in loaded))
        brackets = loaded[0][1].get("brackets") or ["twenties", "sixties", "seventies"]
        for b in brackets:
            cells = []
            for _, d in loaded:
                s = [r for r in d["records"] if r["age"] == b]
                if not s:
                    cells.append("n/a".rjust(22))
                    continue
                w, _, _ = stats.pooled_wer([r["ref"] for r in s],
                                           [r["hyp_wideband"] for r in s])
                cells.append(f"{w:.2%}  (n={len(s)})".rjust(22))
            print(f"  {b:<11} " + "".join(cells))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
