"""Measure WER and endpoint safety by speaker age, wideband and over a phone.

    python3 bench/run.py [--per-bracket N] [--model NAME] [--limit-shard N]

Writes results/<tag>.jsonl incrementally and prints the README's tables.

Two properties this runner has that the first version did not, both bought by
losing a three-hour run to a laptop restart:

**It checkpoints.** Every batch is appended to a JSONL as it completes, so an
interrupted run resumes from where it stopped instead of starting over. The
first version held all results in memory and serialised once at the end, which
means a crash at 95% produced exactly nothing.

**It reports its own throughput.** That run silently degraded from 0.4 s/clip
to 2.6 s/clip, and because nothing was logged between phases there was no way
to see it happening or to tell a slow run from a wedged one. Each batch now
prints its rate.

It also aborts rather than reporting on a partial sample. A benchmark that
quietly drops half its clips and still prints a number is worse than one that
crashes, because the number gets believed.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from agegap import asr, audio, metrics, sample, stats  # noqa: E402
from agegap.channel import telephone  # noqa: E402

DEFAULT_BRACKETS = ("twenties", "sixties", "seventies")
RESULTS = pathlib.Path(__file__).resolve().parents[1] / "results"
ACCENT_SETS = {"us": sample.US_ENGLISH, "native": sample.NATIVE_ANGLOPHONE,
               "any": None}


def load_checkpoint(path: pathlib.Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    done = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue          # a torn final line from a hard kill
            done[r["path"]] = r
    return done


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-bracket", type=int, default=1500)
    ap.add_argument("--cap", type=int, default=25)
    ap.add_argument("--limit-shard", type=int, default=11)
    ap.add_argument("--model", default="openai/whisper-large-v3")
    ap.add_argument("--loss", type=float, default=0.0)
    ap.add_argument("--min-decode-rate", type=float, default=0.95)
    ap.add_argument("--accents", choices=sorted(ACCENT_SETS), default="native")
    ap.add_argument("--no-match-accents", action="store_true",
                    help="skip exact accent matching (reproduces the confounded run)")
    ap.add_argument("--brackets", nargs="+", default=list(DEFAULT_BRACKETS))
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=120)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    brackets = tuple(args.brackets)

    tag = args.tag or (f"{args.model.split('/')[-1]}_{args.accents}"
                       f"{'-raw' if args.no_match_accents else ''}")
    RESULTS.mkdir(exist_ok=True)
    ckpt = RESULTS / f"{tag}.jsonl"

    print(f"  model {args.model}, shards 0..{args.limit_shard}, "
          f"cap {args.cap}/speaker, accents={args.accents}"
          f"{'' if args.no_match_accents else ' (matched)'}\n")

    clips = sample.load_train(brackets)
    drawn = sample.draw(clips, brackets=brackets, cap=args.cap,
                        max_shard=args.limit_shard, per_bracket=args.per_bracket,
                        accents=ACCENT_SETS[args.accents])
    if not args.no_match_accents:
        drawn = sample.match_accents(drawn, brackets)
    print(sample.describe(drawn, brackets))

    done = load_checkpoint(ckpt)
    todo = [c for c in drawn if c.path not in done]
    print(f"\n  {len(drawn)} clips total, {len(done)} already done, "
          f"{len(todo)} to go  ->  {ckpt.name}\n")

    if todo:
        model = asr.Whisper(args.model)
        print(f"  device {model.device} / {model.dtype}\n")
        by_shard: dict[int, list] = collections.defaultdict(list)
        for c in todo:
            by_shard[c.shard].append(c)

        failed = 0
        t_start = time.time()
        n_done = 0
        with open(ckpt, "a") as out:
            for shard in sorted(by_shard):
                want = by_shard[shard]
                wave = audio.read_shard(shard, {c.path for c in want})
                got = [c for c in want if c.path in wave]
                failed += len(want) - len(got)
                print(f"    shard {shard}: {len(got)}/{len(want)} decoded", flush=True)

                for i in range(0, len(got), args.batch_size):
                    batch = got[i:i + args.batch_size]
                    wide = [wave[c.path] for c in batch]
                    tel = [telephone(x, 16_000, loss=args.loss) for x in wide]
                    t0 = time.time()
                    hw = model.transcribe(wide, batch_size=len(batch),
                                          max_new_tokens=args.max_new_tokens)
                    ht = model.transcribe(tel, batch_size=len(batch),
                                          max_new_tokens=args.max_new_tokens)
                    dt = time.time() - t0
                    for c, a, b in zip(batch, hw, ht):
                        x = wave[c.path]
                        rec = {"path": c.path, "age": c.age, "gender": c.gender,
                               "speaker": c.speaker, "accent": c.accent,
                               "n_words": c.n_words, "shard": c.shard,
                               "ref": asr.normalize(c.sentence),
                               "hyp_wideband": asr.normalize(a),
                               "hyp_telephone": asr.normalize(b),
                               **metrics.recording_quality(x, 16_000),
                               **metrics.timing(x, 16_000, c.n_words),
                               "cuts": metrics.endpoint_cuts(x, 16_000),
                               "cuts_energy": metrics.endpoint_cuts(
                                   x, 16_000, detector="energy")}
                        out.write(json.dumps(rec) + "\n")
                    out.flush()
                    n_done += len(batch)
                    rate = dt / (2 * len(batch))
                    eta = (len(todo) - n_done) * rate * 2 / 60
                    print(f"      {n_done}/{len(todo)}  {rate:.2f}s/clip  "
                          f"eta {eta:.0f}m", flush=True)
                del wave

        model.release()
        seen = len(load_checkpoint(ckpt))
        if seen / len(drawn) < args.min_decode_rate:
            print(f"\n  ABORT: {seen}/{len(drawn)} clips "
                  f"({seen/len(drawn):.1%}) below --min-decode-rate")
            return 1
        print(f"\n  wall {(time.time()-t_start)/60:.1f} min, {failed} undecodable")

    recs = list(load_checkpoint(ckpt).values())
    json.dump({"model": args.model, "cap": args.cap, "loss": args.loss,
               "accents": args.accents, "matched": not args.no_match_accents,
               "brackets": list(brackets), "records": recs},
              open(RESULTS / f"{tag}.json", "w"))
    print(f"  wrote {RESULTS / (tag + '.json')}  ({len(recs)} records)")
    report(recs, brackets)
    return 0


def report(recs: list[dict], brackets=DEFAULT_BRACKETS) -> None:
    print("\n  WER by age bracket, speaker-level 95% CI\n")
    print("  bracket     n   spk    wideband              telephone")
    for b in brackets:
        s = [r for r in recs if r["age"] == b]
        if not s:
            continue
        cells = []
        for cond in ("wideband", "telephone"):
            refs = [r["ref"] for r in s]
            hyps = [r[f"hyp_{cond}"] for r in s]
            spk = [r["speaker"] for r in s]
            w, _, _ = stats.pooled_wer(refs, hyps)
            lo, hi = stats.bootstrap_wer(refs, hyps, spk)
            cells.append(f"{w:6.2%} [{lo:5.2%},{hi:5.2%}]")
        print(f"  {b:<10} {len(s):>4} {len({r['speaker'] for r in s}):>5}  "
              + "  ".join(cells))

    print("\n  premature-cutoff rate: clips with an internal pause >= threshold\n")
    print("  bracket      400ms   500ms   700ms  1000ms   words/sec")
    for b in brackets:
        s = [r for r in recs if r["age"] == b]
        if not s:
            continue
        cuts = [np.mean([r["cuts"][str(t)] for r in s])
                for t in (400, 500, 700, 1000)]
        rates = [r["rate_wps"] for r in s if r.get("rate_wps") == r.get("rate_wps")]
        print(f"  {b:<10} " + "".join(f"{c:>7.1%} " for c in cuts)
              + f"  {np.median(rates):>9.2f}")


if __name__ == "__main__":
    raise SystemExit(main())
