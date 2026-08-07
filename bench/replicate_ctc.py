"""Re-run the exact primary sample through a CTC model as an architecture control.

    python3 bench/replicate_ctc.py

Whisper's decoder is a language model and can repair mangled acoustics from
context. If older speakers only look easier because Whisper's LM likes their
word choices, a CTC model with no decoder should not show the same effect.
Same clips, same references, same speaker bootstrap.
"""
from __future__ import annotations
import collections, json, pathlib, sys, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from agegap import asr, audio, stats  # noqa: E402

RES = pathlib.Path(__file__).resolve().parents[1] / "results"
recs = json.load(open(RES / "primary.json"))["records"]
brackets = ("twenties", "sixties", "seventies")
print(f"  replicating on the exact primary sample: {len(recs)} clips\n")

by_shard = collections.defaultdict(list)
for r in recs:
    by_shard[r["shard"]].append(r)

model = asr.Wav2Vec2()
print(f"  {model.name} on {model.device}\n")
out, t0 = [], time.time()
for shard in sorted(by_shard):
    want = by_shard[shard]
    wave = audio.read_shard(shard, {r["path"] for r in want})
    got = [r for r in want if r["path"] in wave]
    hyps = model.transcribe([wave[r["path"]] for r in got])
    for r, h in zip(got, hyps):
        out.append({**{k: r[k] for k in ("path", "age", "speaker", "ref")},
                    "hyp_ctc": asr.normalize(h)})
    print(f"    shard {shard}: {len(got)} clips  ({time.time()-t0:.0f}s)", flush=True)
    del wave
model.release()

json.dump({"model": model.name, "records": out}, open(RES / "ctc_control.json", "w"))
print(f"\n  wrote ctc_control.json ({len(out)} records)\n")

print("  CTC word error rate by bracket, speaker-level 95% CI\n")
print("  bracket      n     WER                   vs twenties")
base = [r for r in out if r["age"] == "twenties"]
for b in brackets:
    s = [r for r in out if r["age"] == b]
    w, _, _ = stats.pooled_wer([r["ref"] for r in s], [r["hyp_ctc"] for r in s])
    lo, hi = stats.bootstrap_wer([r["ref"] for r in s], [r["hyp_ctc"] for r in s],
                                 [r["speaker"] for r in s])
    if b == "twenties":
        print(f"  {b:<11} {len(s):>4} {w:>6.2%} [{lo:5.2%},{hi:5.2%}]   baseline")
        continue
    d, dlo, dhi = stats.unpaired_speaker_delta(
        [r["ref"] for r in base], [r["hyp_ctc"] for r in base],
        [r["speaker"] for r in base],
        [r["ref"] for r in s], [r["hyp_ctc"] for r in s], [r["speaker"] for r in s])
    flag = "excludes zero" if (dlo > 0 or dhi < 0) else "includes zero"
    print(f"  {b:<11} {len(s):>4} {w:>6.2%} [{lo:5.2%},{hi:5.2%}]   "
          f"{d:+6.2%} [{dlo:+6.2%},{dhi:+6.2%}]  {flag}")
