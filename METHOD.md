# Method

Every design decision below was forced by something found in the corpus, not
chosen in advance. The order is roughly the order the problems appeared.

## The corpus

Mozilla Common Voice 17, English, via the ungated mirror `fsicoli/common_voice_17_0`.
CC-0. Each clip carries self-reported `age`, `gender` and `accents`, which is
what makes the comparison possible at all.

Only `train` is usable. Common Voice ships audio for `train`, `dev`, `test` and
`other`; `validated.tsv` lists 1.8M clips but 344k of them have no audio in any
shard.

## Why there is no paired design

The strongest version of this benchmark would compare the *same sentence* read
by a young and an old speaker, removing text difficulty entirely. 13,886
sentences in `validated` are read by both a twenties speaker and a 60+ speaker.

Zero of them have audio on both sides.

This is structural rather than bad luck: multi-read sentences are exactly the
ones Common Voice holds out of `train` to prevent sentence leakage between
splits, so one representative lands in `test` and the remaining reads ship no
audio at all. Text difficulty therefore has to be handled by matching
distributions rather than by pairing. It is a mild problem here, because median
sentence length is 10-11 words in every bracket.

## Shard membership is computable

Shard N contains exactly rows `[N*40000, (N+1)*40000)` of `train.tsv` in file
order. Verified at 40,000 of 40,000 members on shard 0.

This matters practically: `train` audio is 45 GB across 28 tars, and the sample
can be drawn from metadata first so only the shards it needs are ever
downloaded.

It also matters statistically, because shard membership is *not* random with
respect to age:

```
shard   60+ clips   60+ speakers
    0         368             88
    1         501            111
   ...
   22       8,991              4
   23       6,924              1
   24       8,515              1
   25       9,792              1
```

The late shards are a handful of extraordinarily prolific contributors. Half of
all 60+ audio in `train` comes from roughly seven people. The early shards hold
the speaker diversity, so the benchmark reads shards 0-11 and stops: past that
point another 22 GB buys the sixties bracket 48 more speakers.

## Four controls

**Per-speaker cap.** One contributor holds 9,792 clips in a single shard.
Uncapped, a bracket's WER is a description of one person's voice. Capped at 25.

**Gender balance.** The twenties are 68% male, the sixties 45%. An age
comparison over an unbalanced sample partly measures a gender shift. Each
bracket is truncated to twice the smaller side.

**Accent matching.** This one nearly inverted the result. Common Voice is
globally crowdsourced and its younger contributors skew non-native: the
twenties are 11.9% India-and-South-Asia English and 46.9% native anglophone,
against 64.4% for the sixties. Whisper is measurably worse on non-native
English, so the *uncontrolled* comparison shows younger speakers as harder to
transcribe and would have been reported as a finding.

Restricting to native varieties is not sufficient either. Inside that stratum
the twenties are 64.5% United States English and the eighties 35.9%, with
Australian English at 5.0% against 21.9% — a 28.6 point swing. So
`match_accents` keeps, for each accent, the same number of clips in every
bracket, making the composition identical by construction.

**Speaker-level bootstrap.** Clips from one contributor are not independent
observations. Resampling clips would treat 25 utterances from one person as 25
samples. All intervals resample speakers.

## Controls that turned out to be unnecessary

Checked, found clean, reported anyway because a reader will ask.

- **Recording quality.** Older speakers do not have worse microphones. Median
  SNR is 48.9 dB for the twenties and 58.4 dB for the sixties, with the sixties
  showing a *lower* noise floor. Any age effect is therefore not an equipment
  effect, and in the sixties the estimate is conservative.
- **Sentence length.** Median 10-11 words in every bracket.
- **Sentence domain.** Blank for all English clips in this release.
- **Clip validation.** Older clips have a *lower* downvote rate (10.1% for the
  seventies against 14.2% for the twenties), so they are if anything the better
  validated half.

## The telephone channel

Common Voice is 48 kHz into a laptop microphone. The companies this benchmark
is aimed at run 8 kHz phone calls. Scoring only wideband audio answers a
question they are not asking, so every clip is scored twice: as recorded, and
through a simulated G.711 channel — band-limited to 300-3400 Hz, decimated to
8 kHz, mu-law companded to 8 bits, and resampled back to 16 kHz for the model.
Optional 20 ms packet dropping models a jitter buffer.

## Endpointing

A voice agent decides the caller has finished by waiting for a fixed stretch of
silence, typically 500-800 ms. A pause *inside* an utterance that exceeds that
threshold is heard as the end of the turn, and the agent talks over someone who
was mid-sentence.

WER cannot see this failure at all: the words the model did receive can be
transcribed perfectly while the caller is cut off every time. So the benchmark
also reports, per bracket and per threshold, the share of utterances containing
an internal pause long enough to be mistaken for the end of the turn. Leading
and trailing silence is excluded, since recording margin is not a pause the
speaker took.

## Precision

The run uses fp16 on MPS, which is 2.6x faster than fp32 on CPU. That shortcut
is verified rather than assumed: large-v3 scores an identical 0.0274 WER on the
LibriSpeech dummy split under fp32/cpu, fp32/mps and fp16/mps. The check is
`tests/test_precision.py`. A quantised Whisper KV cache is capable of taking
this model from 1.91% WER to 100%, so reduced precision here does not get
taken on trust.
