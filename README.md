# asr-age-gap

Voice agents are being pointed at elderly callers, and the assumed risk is that
speech recognition will not hear them. **That assumption is wrong, and it is
hiding the failure that is actually happening.**

Measured on 2,760 Common Voice clips, matched between age brackets on accent,
gender and speaker so the only thing varying is age, and checked against a
second 3,189-clip draw that controls for none of it:

```
                 word error rate          premature cutoff (700 ms)
twenties   n=920      6.53%                        8.0%
sixties    n=920      5.23%   -1.31pp *           19.7%   +11.6pp *
seventies  n=920      4.67%   -1.86pp *           16.6%    +8.5pp *

* speaker-bootstrapped 95% interval excludes zero
```

**Whisper transcribes older speakers more accurately, not less.** And at the
500-800 ms endpoint threshold production voice stacks ship with, those same
speakers get talked over two to two and a half times as often.

`python3 bench/run.py` reproduces both. No API key, no spend.

## 1. The recognition penalty is not there

Large-v3, wideband, speaker-level 95% intervals:

```
bracket      n    spk    WER                  vs twenties
twenties   920    606    6.53% [5.77, 7.40]   baseline
sixties    920    240    5.23% [4.65, 5.86]   -1.31pp [-2.33, -0.30]  excludes zero
seventies  920    116    4.67% [4.12, 5.31]   -1.86pp [-2.91, -0.88]  excludes zero
```

Every error type falls with age, so this is not one category masking another:

```
bracket        sub      del      ins
twenties     5.22%    0.57%    0.74%
sixties      4.19%    0.41%    0.64%
seventies    3.67%    0.41%    0.59%
```

Deletions in particular do *not* rise, which is the result you would expect if
quiet or breathy speech were being dropped. It is not being dropped.

## 2. The turn-taking penalty is large

A voice agent decides the caller has finished by waiting for a fixed stretch of
silence. A pause *inside* an utterance that exceeds that threshold is heard as
the end of the turn, and the agent starts talking over someone mid-sentence.

WER is blind to this. The words the model did receive can be transcribed
perfectly while the caller is cut off every time.

Share of utterances containing an internal pause at least this long:

```
bracket        400 ms   500 ms   700 ms   1000 ms
twenties        23.3%    17.3%     8.0%      2.0%
sixties         41.5%    33.7%    19.7%      8.7%
seventies       37.3%    28.6%    16.6%      6.2%
```

All six age-versus-baseline differences exclude zero. At 700 ms the gap is
+11.6pp [+7.7, +15.7] for the sixties and +8.5pp [+4.0, +13.6] for the
seventies.

The mechanism is in the timing. Older speakers take **twice as many internal
pauses** and spend twice as long in them:

```
bracket     words/voiced-s   pauses/clip   pause total
twenties             2.47           1.0        240 ms
sixties              2.20           2.0        480 ms
seventies            2.20           2.0        420 ms
```

**It is not a clean gradient.** The sixties are cut off slightly more than the
seventies and their intervals overlap. This reads as an effect that arrives by
60 and plateaus, not a straight line, and it is drawn that way rather than
smoothed.

The eighties, run separately because matching against them would have shrunk
every bracket eightfold, are the sharpest case. Only 14 speakers exist, so the
intervals are wide — and the effect clears them anyway:

```
threshold   twenties   eighties   difference [95% CI]
    400ms      27.9%      49.2%   +21.8% [+6.3%, +36.6%]
    500ms      15.6%      40.2%   +24.8% [+8.9%, +38.7%]
    700ms       4.1%      22.1%   +18.1% [+7.9%, +27.2%]
   1000ms       0.0%       6.6%    +6.5% [+1.7%, +14.5%]
```

At 700 ms that is a 5.4x gap. Their WER, meanwhile, is 6.24% against 6.01% —
a difference of +0.15pp whose interval comfortably includes zero. **The two
findings diverge further with age**: recognition stays flat while turn-taking
gets steadily worse.

**The result survives a re-draw.** Running the whole benchmark again without
accent matching — 3,189 clips, 1,434 speakers, 30 accents instead of 8 —
reproduces the cutoff rates almost exactly:

```
                twenties   sixties   seventies
matched             8.0%     19.7%       16.6%
unmatched           7.5%     19.3%       16.6%
```

This is a re-draw from one corpus, not an independent replication: the two
samples share 52% of their speakers, though only 18% of their clips. It shows
the numbers are not an artifact of one particular draw or of the accent
matching. It does not show they generalise beyond Common Voice.

## 3. The accent confound is real, and it does not drive the result

Common Voice is globally crowdsourced and its younger contributors skew
non-native. The twenties bracket is 11.9% India-and-South-Asia English and
46.9% native anglophone; the sixties are 64.4%. Whisper is worse on non-native
English, so age and accent are genuinely entangled in this corpus, and an
uncontrolled comparison has an obvious alternative explanation.

Brackets are therefore matched on the (accent, gender) pair, holding both
identical by construction: 920 clips per bracket, 8 accents, 392/528
male/female **in every bracket**.

**I expected that to change the answer. It does not.** Running the benchmark
both ways, on samples whose accent composition could hardly be more different:

```
                   matched            unmatched
                (8 accents,         (30/15/13 accents,
              identical mix)         differing mix)
twenties          6.53%                 6.60%
sixties           5.23%                 4.90%
seventies         4.67%                 5.11%
```

The largest disagreement is 0.44pp. Both arms put the sixties and seventies
below the twenties, and in both the difference excludes zero. Matching is still
the right thing to do — it removes a live alternative explanation and it makes
the trend monotonic — but the finding does not rest on it.

This section originally claimed the opposite, on the strength of a 40-clip
pilot in which the twenties scored 10.54%. At full sample that figure is 6.60%.
The pilot was noise and the story built on it was wrong.

## 4. What this can and cannot claim

**Common Voice's older speakers are volunteers.** They chose to sit down at a
computer and record themselves for Mozilla. They are tech-comfortable and
almost certainly healthier of voice than the median 75-year-old on a
post-discharge call. Dysarthria, post-stroke speech and cognitive decline are
absent from this corpus by construction.

So this measures **healthy aging, not clinical aging**. The right reading of
finding 1 is "age alone does not break recognition", not "recognition is fine
for elderly patients". Those are different claims and only the first is
supported here. A follow-up on a disordered-speech corpus is the honest next
step.

Two narrower limits. Read speech is not conversational speech, and someone
reading a prompt pauses differently than someone answering a question — though
that cuts against finding 2 being an artifact, since read speech should if
anything *understate* natural pausing. And the eighties bracket has 27 speakers
in the entire split, so it is reported separately rather than folded into the
main comparison, where matching against it would have shrunk every bracket
eightfold. Its intervals are correspondingly wide.

## 5. Confounds that were checked and came back clean

Reported because a reader will ask, not because they changed anything.

- **Recording quality.** Median SNR 53.2 / 55.7 / 55.9 dB across brackets. No
  equipment disadvantage, so the WER result is not a microphone result.
- **Sentence length.** Median 11 words for the twenties and sixties, 10 for the
  seventies. Short utterances are genuinely harder here — 7.72% WER at 1-7
  words against 4.24% at 14+ — so the seventies carry the *harder* sentences,
  and the effect survives holding length fixed:

  ```
  words        twenties   sixties   seventies
   1-7            8.75%     8.33%       6.04%
   8-10           5.82%     6.37%       4.93%
  11-13           6.75%     4.63%       4.50%
  14-40           5.66%     3.23%       3.89%
  ```
- **Clip validation.** Older clips carry a *lower* community downvote rate
  (10.1% against 14.2%).
- **Speaker prolificacy.** One contributor holds 9,792 clips in a single shard;
  half of all 60+ audio in the split comes from about seven people. Capped at
  25 clips per speaker, and all intervals resample speakers rather than clips.

## 6. Three bugs the harness caught in itself

**The endpoint finding was nearly an artifact.** A relative-energy VAD reported
the eighties being cut off at 42.9%. WebRTC VAD — what production stacks
actually run — disagreed on 36% of those clips and put the median longest pause
at 345 ms where the energy detector said 662 ms. Breathy trailing-off speech
falls under an energy floor, and older speech is exactly what is breathy, so
the cheap detector's error was correlated with the variable under study. WebRTC
is now primary and both are recorded; on the published sample they agree on
80-90% of clips and give the same conclusion.

**Accent matching silently undid the gender balance.** Taking an accent-wise
subset need not preserve the male/female split, and one run came out 54/46 in
the sixties against 50/50 in the seventies. Matching on the (accent, gender)
pair costs 25 clips per bracket and fixes it exactly.

**A fp16 shortcut was verified rather than assumed.** The run uses fp16 on MPS
because it is 2.6x faster than fp32 on CPU. A quantised Whisper KV cache is
capable of taking large-v3 from 1.91% WER to 100%, so the shortcut was checked
against a reference: identical 0.0274 WER under fp32/cpu, fp32/mps and fp16/mps.
`tests/test_precision.py`.

## Running it

```bash
pip install torch transformers huggingface_hub jiwer numpy scipy soundfile webrtcvad-wheels
python3 bench/run.py                      # primary, accent-matched
python3 bench/analyze.py results/primary.json
python3 -m pytest tests/ -q               # 23 tests, one per claim above
```

The corpus is the CC-0 Common Voice 17 English set via the ungated
`fsicoli/common_voice_17_0` mirror. Shard membership is computable from
`train.tsv` row order, so only the shards the sample needs are downloaded
(~19 GB of 45 GB). The run checkpoints every batch to JSONL and resumes where
it stopped.

`METHOD.md` has the full design, including why a same-sentence paired design is
impossible in this corpus.

## Layout

| path | what |
|---|---|
| `src/agegap/sample.py` | stratified draw: speaker cap, gender balance, accent matching |
| `src/agegap/channel.py` | G.711 telephony simulation, mu-law and packet loss |
| `src/agegap/metrics.py` | SNR, speaking rate, and the endpoint-cutoff measure |
| `src/agegap/stats.py` | pooled WER and speaker-level bootstrap intervals |
| `src/agegap/asr.py` | Whisper adapter, device and precision selection |
| `bench/run.py` | the run: chunked, checkpointed, fail-closed |
| `bench/analyze.py` | the tables above, plus the controlled/uncontrolled contrast |
| `tests/` | 23 tests, including one per bug in section 6 |
