# asr-age-gap

I thought speech recognition would be worse for old people. I measured it. It's better.

Then I looked for where the real problem is, and found it in when the computer
decides you've stopped talking.

## The two numbers

I took 2,760 clips from Common Voice. I matched the age groups on accent,
gender and speaker, so age is the only thing that differs between them.

```
             word errors        cut off mid-sentence
             (lower better)     (at a 700ms pause)
twenties         6.53%                 8.0%
sixties          5.23%                19.7%
seventies        4.67%                16.6%
```

**Left column:** Whisper makes fewer mistakes on people in their seventies than
on people in their twenties. That surprised me, so I checked it on a second
model built a completely different way, and got the same answer.

**Right column:** older people pause more in the middle of a sentence. A voice
assistant waits for silence to decide your turn is over. Longer pauses mean it
cuts you off. At a 700ms wait, it happens to 8% of sentences from
twenty-somethings and 20% from people in their sixties.

Word error rate can't see that second problem at all. The words that get
through are transcribed fine. The caller just gets interrupted.

## One important correction

A maintainer at Daily/Pipecat, Mark Backman, told me that real voice products
don't use a plain silence timer any more. He's right. Their default is a model
that listens to your voice and decides if you sound finished.

So I tested that model too. It cuts the gap roughly in half, to the point where
I can no longer tell it apart from zero. Details in section 3.

So: if your product waits a fixed amount of silence, the problem is real. If it
uses a smarter turn detector, most of it goes away.

## Run it

```bash
python3 bench/run.py
```

No API key. No paid services. Everything downloads from public data and runs on
your machine.

---

The rest of this file is the detail: how I controlled for things that could
have faked the result, three mistakes I made and caught, and what this does not
prove.

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

**It is not a Whisper artifact.** The obvious objection is that Whisper's
decoder is a language model, so it might be repairing older speakers' word
choices rather than hearing them better. So the same clips were re-run through
wav2vec2, which is pure CTC: frame-wise, greedy, no decoder and no implicit LM.

```
bracket      Whisper enc-dec        wav2vec2 CTC
twenties           6.53%               14.23%
sixties            5.23%  -1.31pp      10.30%  -3.94pp [-5.52,-2.40]
seventies          4.67%  -1.86pp      10.52%  -3.72pp [-5.49,-2.08]
```

Absolute WER is much higher for wav2vec2 (LibriSpeech-only training, no LM), so
only the between-bracket comparison transfers. The effect is *larger* there and
still excludes zero, which puts it in the acoustics rather than in a decoder.

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
seventies and their intervals overlap. So the effect looks like it arrives by 60 and then levels off. It is drawn
that way here instead of as a straight line.

The eighties, run separately because matching against them would have shrunk
every bracket eightfold, are the sharpest case. Only 14 speakers exist, so the
intervals are wide. The effect clears them anyway:

```
threshold   twenties   eighties   difference [95% CI]
    400ms      27.9%      49.2%   +21.8% [+6.3%, +36.6%]
    500ms      15.6%      40.2%   +24.8% [+8.9%, +38.7%]
    700ms       4.1%      22.1%   +18.1% [+7.9%, +27.2%]
   1000ms       0.0%       6.6%    +6.5% [+1.7%, +14.5%]
```

At 700 ms that is a 5.4x gap. Their WER, meanwhile, is 6.24% against 6.01%,
a difference of +0.15pp whose interval comfortably includes zero. **The two
findings diverge further with age**: recognition stays flat while turn-taking
gets steadily worse.

**It survives a re-draw.** Running the whole benchmark again without
accent matching, on 3,189 clips from 1,434 speakers with 30 accents,
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

## 3. A semantic turn model closes most of the gap

Mark Backman of Daily/Pipecat read an earlier version of this and pointed out
that it described the wrong thing: production stacks do not endpoint on a fixed
VAD threshold. Pipecat's default is
[smart-turn](https://github.com/pipecat-ai/smart-turn), a semantic model that
listens to the waveform and grants more time when the turn sounds unfinished.

He is right, so smart-turn v3 was measured on the identical sample. For each
clip, the audio *up to* an internal pause is fed to the model and it is asked
whether the turn is complete. The speaker demonstrably continues, so
"complete" is a false cutoff.

```
                fixed 700ms threshold      smart-turn v3
twenties               8.0%                     75.6%
sixties               19.7%  +11.6pp *          81.6%   +5.9pp [-0.9,+12.9]
seventies             16.6%   +8.5pp *          79.7%   +4.0pp [-3.6,+11.6]
```

The gap roughly halves and stops excluding zero. Positive control on whole
utterances is flat at 90-91% across brackets.

Two things this does not say. The absolute 76-82% rate is not an error
rate: many internal pauses are legitimate clause boundaries where a turn could
plausibly end, and without human labels on which prefixes sound complete, only
the between-bracket comparison is interpretable. And "includes zero" is not
"no effect". Both point estimates stay positive, and 86 seventies speakers
cannot resolve four points either way.

The practical reading: if you endpoint on a fixed threshold, the age gap is
real and large. If you use a semantic turn model, most of it goes away. The
published smart-turn benchmark stratifies 31,527 samples across 23 languages
but not by speaker age, and its training mix leans on synthetic TTS, which does
not pause the way an eighty-year-old does.

## 4. What a person's own speech noise costs a drift detector

Several products now offer daily phone check-ins for older adults that claim to
flag cognitive decline from voice biomarkers. Validating that needs gated
clinical corpora. But a prior question needs no clinical labels and bounds the
claim from below: **how much does one healthy person's speech vary between
utterances?** A drift detector can only see change that clears the speaker's
own noise.

Measured on 36 speakers with 40+ clips each:

```
feature                        within-speaker CV
speech rate                          ~18%
utterance duration                   ~23%
number of internal pauses            ~76%
total pause time                   ~96-111%
```

Converted to the sample needed to resolve a 10% change at 80% power:

```
feature                  utterances    calls @40/call
speech rate                      25          0.6
utterance duration               41          1.0
number of internal pauses       447         11
total pause time                758         19
```

Pause features, the most frequently cited voice biomarker, vary by about
100% within the same speaker, often within one sitting. Detecting a 10% shift
in total pause time takes roughly three weeks of daily calls *per reading*, so
a "six-week trend" is two or three noisy measurements. Speech rate and duration
are comfortably usable.

Both directions of error are stated: Common Voice clips from one contributor
are often a single sitting, so real day-to-day variance is larger; and
utterances within one call are correlated, so dividing by 40 overstates the
effective sample. Both push the true requirement up. **These are floors.**

## 5. The accent confound is real, and it does not drive the result

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
the right thing to do. It removes a live alternative explanation and makes
the trend monotonic. But the finding does not rest on it.

This section originally claimed the opposite, on the strength of a 40-clip
pilot in which the twenties scored 10.54%. At full sample that figure is 6.60%.
The pilot was noise and the story built on it was wrong.

## 6. What this can and cannot claim

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
reading a prompt pauses differently than someone answering a question. Though
that cuts against finding 2 being an artifact, since read speech should if
anything *understate* natural pausing. And the eighties bracket has 27 speakers
in the entire split, so it is reported separately rather than folded into the
main comparison, where matching against it would have shrunk every bracket
eightfold. Its intervals are correspondingly wide.

## 7. Confounds that were checked and came back clean

None of these changed the result. They are here because a reader will ask.

- **Recording quality.** Median SNR 53.2 / 55.7 / 55.9 dB across brackets. No
  equipment disadvantage, so the WER result is not a microphone result.
- **Sentence length.** Median 11 words for the twenties and sixties, 10 for the
  seventies. Short utterances are harder here: 7.72% WER at 1-7
  words against 4.24% at 14+. So the seventies carry the *harder* sentences,
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

## 8. Bugs the harness caught in itself

**The endpoint finding was nearly an artifact.** A relative-energy VAD reported
the eighties being cut off at 42.9%. WebRTC VAD, which is what production stacks
actually run, disagreed on 36% of those clips and put the median longest pause
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
| `bench/replicate_ctc.py` | the wav2vec2 architecture control |
| `bench/smart_turn_eval.py` | Pipecat smart-turn v3 on the same age-matched clips |
| `bench/drift_floor.py` | within-speaker noise, and what it costs a drift detector |
| `tests/` | 26 tests, including one per bug in section 8 |
