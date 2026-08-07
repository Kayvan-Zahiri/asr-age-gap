"""Does Pipecat's semantic turn model close the age gap a fixed threshold opens?

    python3 bench/smart_turn_eval.py

Mark Backman (Daily/Pipecat) made a fair correction to an earlier version of
this work: modern stacks do not endpoint on VAD alone. Pipecat's default is
smart-turn, a semantic model that listens to the waveform and grants more time
when it judges the turn incomplete. So the fixed-threshold result does not
describe what Pipecat actually ships.

That relocates the question rather than answering it. A learned turn model
inherits whatever speech its training data contained, and smart-turn v3's
published benchmark stratifies by language across 23 languages but not by
speaker age. Several of its listed corpora (chirp3, orpheus, rime) are
synthetic TTS, which has idealised prosody: TTS does not produce the halting,
breath-limited pausing of an eighty-year-old.

So this measures smart-turn directly, on the same age-matched sample.

**The test.** For each clip, locate the internal pauses with WebRTC VAD. For
each pause, feed smart-turn only the audio *up to* that pause and ask whether
the turn is complete. The speaker demonstrably continues afterwards, so a
"complete" verdict is a false cutoff: the agent would start talking. The rate
of those, per age bracket, is the number nobody publishes.

A positive control runs the model on whole clips, where "complete" is correct.
"""

from __future__ import annotations

import collections
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from agegap import audio, metrics, stats  # noqa: E402

RES = pathlib.Path(__file__).resolve().parents[1] / "results"
SR = 16_000
FRAME_MS = 30
MIN_PREFIX_S = 1.0        # smart-turn needs something to judge


def load_model():
    import onnxruntime as ort
    from huggingface_hub import hf_hub_download
    from transformers import WhisperFeatureExtractor

    path = hf_hub_download("pipecat-ai/smart-turn-v3", "smart-turn-v3.0.onnx")
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(path, sess_options=so), \
        WhisperFeatureExtractor(chunk_length=8), path


def predict(session, fx, chunks: list[np.ndarray]) -> np.ndarray:
    """P(turn complete) per chunk, matching pipecat-ai/smart-turn's inference.py.

    Two details that are easy to get wrong and produce a plausible-looking
    constant if you do:

    The ONNX graph already applies the sigmoid, so its output is a probability.
    Applying sigmoid again squashes everything into a narrow band around 0.73
    and every clip reads as "complete" regardless of content.

    Padding goes at the *start*. The model judges the end of a turn, so the
    audio must sit flush against the right edge of the window.
    """
    prepped = []
    for x in chunks:
        n = 8 * SR
        x = x[-n:] if len(x) > n else np.pad(x, (n - len(x), 0))
        prepped.append(x.astype(np.float32))
    feats = fx(prepped, sampling_rate=SR, return_tensors="np",
               padding="max_length", max_length=8 * SR, truncation=True,
               do_normalize=True).input_features
    out = session.run(None, {"input_features": feats.astype(np.float32)})[0]
    return np.asarray(out).reshape(-1)          # already a probability


def pause_prefixes(x: np.ndarray) -> list[np.ndarray]:
    """Audio up to the start of each internal pause of at least 200 ms."""
    flags = metrics._webrtc_flags(x, SR)
    idx = np.flatnonzero(flags)
    if idx.size < 2:
        return []
    out, run_start = [], None
    for i in range(idx[0], idx[-1] + 1):
        if not flags[i]:
            if run_start is None:
                run_start = i
        else:
            if run_start is not None:
                if (i - run_start) * FRAME_MS >= 200:
                    cut = int(run_start * FRAME_MS / 1000 * SR)
                    if cut >= MIN_PREFIX_S * SR:
                        out.append(x[:cut])
                run_start = None
    return out


def main() -> int:
    recs = json.load(open(RES / "primary.json"))["records"]
    brackets = ("twenties", "sixties", "seventies")
    session, fx, path = load_model()
    print(f"  smart-turn-v3.0.onnx\n  sample: {len(recs)} clips\n")

    by_shard = collections.defaultdict(list)
    for r in recs:
        by_shard[r["shard"]].append(r)

    rows, control = [], []
    for shard in sorted(by_shard):
        want = by_shard[shard]
        wave = audio.read_shard(shard, {r["path"] for r in want})
        got = [r for r in want if r["path"] in wave]
        whole = [wave[r["path"]] for r in got]
        for r, p in zip(got, predict(session, fx, whole)):
            control.append({"age": r["age"], "speaker": r["speaker"], "p": float(p)})
        chunks, owners = [], []
        for r in got:
            for pre in pause_prefixes(wave[r["path"]]):
                chunks.append(pre)
                owners.append(r)
        for i in range(0, len(chunks), 32):
            ps = predict(session, fx, chunks[i:i + 32])
            for r, p in zip(owners[i:i + 32], ps):
                rows.append({"age": r["age"], "speaker": r["speaker"],
                             "path": r["path"], "p_complete": float(p)})
        print(f"    shard {shard}: {len(got)} clips, {len(chunks)} mid-pause prefixes",
              flush=True)
        del wave

    json.dump({"model": "pipecat-ai/smart-turn-v3.0",
               "false_cutoff_rows": rows, "control_rows": control},
              open(RES / "smart_turn.json", "w"))

    print("\n  POSITIVE CONTROL: whole clips, 'complete' is the correct answer\n")
    print("  bracket      n    P(complete) median   says complete @0.5")
    for b in brackets:
        s = [c for c in control if c["age"] == b]
        print(f"  {b:<11} {len(s):>4} {np.median([c['p'] for c in s]):>14.3f}"
              f" {np.mean([c['p'] > 0.5 for c in s]):>20.1%}")

    print("\n  FALSE CUTOFFS: prefix ends at a mid-utterance pause, speaker continues\n")
    print("  bracket      prefixes   says COMPLETE (false cutoff)   vs twenties")
    base = [r for r in rows if r["age"] == "twenties"]
    rng = np.random.default_rng(0)

    def boot(a, bset):
        by = collections.defaultdict(list)
        for r in bset:
            by[r["speaker"]].append(r["p_complete"] > 0.5)
        ks = list(by)
        d = []
        for _ in range(3000):
            pick = rng.integers(0, len(ks), len(ks))
            d.append(np.mean([v for i in pick for v in by[ks[i]]]))
        return np.percentile(d, [2.5, 97.5])

    for b in brackets:
        s = [r for r in rows if r["age"] == b]
        rate = np.mean([r["p_complete"] > 0.5 for r in s])
        lo, hi = boot(b, s)
        if b == "twenties":
            print(f"  {b:<11} {len(s):>8} {rate:>16.1%} [{lo:.1%},{hi:.1%}]   baseline")
            continue
        bl = np.mean([r["p_complete"] > 0.5 for r in base])
        print(f"  {b:<11} {len(s):>8} {rate:>16.1%} [{lo:.1%},{hi:.1%}]"
              f"   {(rate-bl)*100:+.1f}pp")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
