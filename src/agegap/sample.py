"""Draw a stratified, speaker-capped, gender-balanced sample from Common Voice.

Three properties of the corpus force this design, and all three were found by
reading the metadata rather than assumed:

1. Shard membership is deterministic. Tar N holds exactly rows
   [N*40000, (N+1)*40000) of ``train.tsv`` in file order, verified on 40,000
   of 40,000 members. So the sample can be drawn first and only the shards it
   needs get downloaded, instead of pulling all 45 GB.

2. A few speakers dominate. One shard holds 9,792 clips from a single
   contributor; shards 22-25 are ~34,000 clips from about seven people, half
   of all 60+ audio in the split. Uncapped, "the eighties" would be a
   description of four individuals. Hence ``cap``.

3. Gender skews the opposite way with age. The twenties are 68% male, the
   sixties 45%. An uncorrected age comparison partly measures a gender shift,
   so brackets are balanced male/female and truncated to the smaller side.

Sentence difficulty is *not* matched by pairing. 13,886 sentences are read by
both a twenties and a 60+ speaker, but zero have audio on both sides: multi-read
sentences are exactly the ones held out of train to stop sentence leakage, so
one copy lands in ``test`` and the rest ship no audio at all. Median sentence
length is 10-11 words in every bracket, so the residual difference is small and
is reported rather than engineered away.
"""

from __future__ import annotations

import collections
import csv
from dataclasses import dataclass

from huggingface_hub import hf_hub_download

REPO = "fsicoli/common_voice_17_0"          # ungated mirror of the CC-0 corpus
SHARD_SIZE = 40_000
GENDERS = {"male_masculine": "m", "female_feminine": "f"}

# Accent is the confound that nearly inverted this benchmark. Common Voice is
# globally crowdsourced and its younger contributors skew non-native: the
# twenties are 11.9% India-and-South-Asia English and only 46.9% native
# anglophone, against 64.4% for the sixties. Whisper is measurably worse on
# non-native English, so an uncontrolled age comparison reports younger
# speakers as harder to transcribe and calls it a finding. Holding accent fixed
# is what makes the remaining difference attributable to age.
US_ENGLISH = frozenset({"United States English"})
NATIVE_ANGLOPHONE = frozenset({
    "United States English", "England English", "Canadian English",
    "Australian English", "New Zealand English", "Scottish English",
    "Irish English", "Northern Irish", "Welsh English",
})


@dataclass(frozen=True)
class Clip:
    path: str
    age: str
    gender: str
    speaker: str
    sentence: str
    shard: int
    accent: str = ""

    @property
    def n_words(self) -> int:
        return len(self.sentence.split())


def _tsv(split: str) -> str:
    return hf_hub_download(REPO, f"transcript/en/{split}.tsv", repo_type="dataset")


def load_train(brackets: tuple[str, ...]) -> list[Clip]:
    """Every train clip carrying a usable age and a binary gender label.

    ``shard`` is computed from row order, which is what makes selective
    downloading possible.
    """
    out: list[Clip] = []
    with open(_tsv("train"), encoding="utf-8", errors="replace") as fh:
        for i, r in enumerate(csv.DictReader(fh, delimiter="\t", quoting=csv.QUOTE_NONE)):
            age = (r.get("age") or "").strip()
            gender = GENDERS.get((r.get("gender") or "").strip())
            if age in brackets and gender:
                # Labels are comma-joined and the first token is the variety,
                # e.g. "England English,South London and Essex".
                accent = (r.get("accents") or "").split(",")[0].strip()
                out.append(Clip(r["path"], age, gender, r["client_id"],
                                r["sentence"], i // SHARD_SIZE, accent))
    return out


def draw(clips: list[Clip], *, brackets: tuple[str, ...], cap: int = 25,
         max_shard: int = 8, per_bracket: int | None = None,
         accents: frozenset[str] | None = US_ENGLISH,
         seed: int = 0) -> list[Clip]:
    """Cap per speaker, balance gender, hold accent fixed, then truncate.

    ``accents=None`` disables the accent restriction and reproduces the
    confounded comparison, which is kept only so the benchmark can show what
    ignoring it costs.

    Selection inside a speaker is by a seeded shuffle, so the sample is
    reproducible without depending on corpus row order.
    """
    import random

    rng = random.Random(seed)
    chosen: list[Clip] = []
    for bracket in brackets:
        pool = [c for c in clips if c.age == bracket and c.shard <= max_shard
                and (accents is None or c.accent in accents)]
        by_speaker: dict[tuple[str, str], list[Clip]] = collections.defaultdict(list)
        for c in pool:
            by_speaker[(c.speaker, c.gender)].append(c)

        sides: dict[str, list[Clip]] = {"m": [], "f": []}
        for (_, gender), cs in by_speaker.items():
            rng.shuffle(cs)
            sides[gender].extend(cs[:cap])

        # Balance by interleaving speakers so a truncation cannot silently
        # become "the first few speakers in each gender".
        for side in sides.values():
            rng.shuffle(side)
        k = min(len(sides["m"]), len(sides["f"]))
        take = sides["m"][:k] + sides["f"][:k]
        rng.shuffle(take)
        if per_bracket:
            take = take[:per_bracket]
        chosen.extend(take)
    return chosen


def match_accents(chosen: list[Clip], brackets: tuple[str, ...]) -> list[Clip]:
    """Force every bracket to the same accent *and* gender composition.

    Restricting to native-anglophone varieties is not enough on its own. Inside
    that stratum the twenties are 64.5% United States English and the eighties
    35.9%, with Australian English running 5.0% against 21.9% — a 28.6 point
    swing that a bare native/non-native filter leaves untouched.

    Matching is on the (accent, gender) pair rather than accent alone. Matching
    on accent by itself silently undoes the gender balance ``draw`` established,
    because taking an accent-wise subset need not preserve the male/female split
    — one run came out 54/46 in the sixties against 50/50 in the seventies.
    Keying on the pair costs about 25 clips per bracket and makes both
    compositions identical by construction rather than approximately equal.

    For each stratum this keeps ``min`` clips across the brackets. The cost is
    sample size, and it is worth paying: an accent difference of that size is
    large enough to produce an "age effect" on its own.
    """
    import random

    rng = random.Random(0)
    per: dict[str, dict[tuple[str, str], list[Clip]]] = {
        b: collections.defaultdict(list) for b in brackets}
    for c in chosen:
        if c.age in per:
            per[c.age][(c.accent, c.gender)].append(c)

    strata = set.intersection(*(set(per[b]) for b in brackets)) if brackets else set()
    out: list[Clip] = []
    for stratum in sorted(strata):
        k = min(len(per[b][stratum]) for b in brackets)
        for b in brackets:
            pool = list(per[b][stratum])
            rng.shuffle(pool)
            out.extend(pool[:k])
    return out


def describe(sample: list[Clip], brackets: tuple[str, ...]) -> str:
    lines = ["  bracket      clips  speakers   m/f      med words  accents"]
    for b in brackets:
        s = [c for c in sample if c.age == b]
        if not s:
            continue
        g = collections.Counter(c.gender for c in s)
        words = sorted(c.n_words for c in s)
        lines.append(
            f"  {b:<11} {len(s):>6} {len({c.speaker for c in s}):>9}"
            f"  {g['m']:>4}/{g['f']:<4} {words[len(words)//2]:>9}"
            f" {len({c.accent for c in s}):>8}")
    return "\n".join(lines)


def shards_needed(sample: list[Clip]) -> list[int]:
    return sorted({c.shard for c in sample})
