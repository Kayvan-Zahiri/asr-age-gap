"""The sampler's three guarantees, tested on synthetic clips so they need no
corpus download."""
from __future__ import annotations
import pathlib, sys, collections
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from agegap.sample import (Clip, NATIVE_ANGLOPHONE, draw,  # noqa: E402
                          match_accents)

BR = ("twenties", "seventies")

ACC = ["United States English", "England English", "Australian English"]


def _clips():
    """Two brackets whose accent mix differs, which is the real corpus's
    problem in miniature: the twenties skew US, the seventies skew Australian."""
    out = []
    for bi, b in enumerate(BR):
        # one hyper-prolific speaker per bracket
        for i in range(500):
            out.append(Clip(f"{b}_whale_{i}.mp3", b, "m", f"{b}_whale",
                            "a b c d e", 0, ACC[0]))
        for s in range(20):
            for i in range(10):
                g = "m" if s % 2 else "f"
                # skew the accent mix in opposite directions per bracket
                a = ACC[(s + bi * 2) % len(ACC)]
                out.append(Clip(f"{b}_{s}_{i}.mp3", b, g, f"{b}_spk{s}",
                                "a b c d e", s % 9, a))
    return out

def test_no_speaker_exceeds_the_cap():
    """One Common Voice contributor holds 9,792 clips in a single shard.
    Uncapped, a bracket becomes a description of one person."""
    s = draw(_clips(), brackets=BR, cap=25, max_shard=8, accents=NATIVE_ANGLOPHONE)
    c = collections.Counter(x.speaker for x in s)
    assert c and max(c.values()) <= 25

def test_genders_are_balanced_within_each_bracket():
    """The twenties are 68% male and the sixties 45%; uncorrected, an age
    comparison partly measures a gender shift."""
    s = draw(_clips(), brackets=BR, cap=25, max_shard=8, accents=NATIVE_ANGLOPHONE)
    for b in BR:
        g = collections.Counter(x.gender for x in s if x.age == b)
        assert g["m"] == g["f"], (b, g)

def test_shard_limit_is_respected():
    s = draw(_clips(), brackets=BR, cap=25, max_shard=3, accents=NATIVE_ANGLOPHONE)
    assert s and max(x.shard for x in s) <= 3

def test_draw_is_deterministic_under_a_seed():
    a = [x.path for x in draw(_clips(), brackets=BR, cap=25, max_shard=8, accents=NATIVE_ANGLOPHONE, seed=7)]
    b = [x.path for x in draw(_clips(), brackets=BR, cap=25, max_shard=8, accents=NATIVE_ANGLOPHONE, seed=7)]
    assert a == b

def test_truncation_does_not_collapse_onto_few_speakers():
    """per_bracket must not silently become 'the first speaker alphabetically'."""
    s = draw(_clips(), brackets=BR, cap=25, max_shard=8, accents=NATIVE_ANGLOPHONE, per_bracket=40, seed=1)
    for b in BR:
        spk = {x.speaker for x in s if x.age == b}
        assert len(spk) >= 5, (b, spk)


def test_accent_filter_excludes_out_of_stratum_clips():
    c = _clips() + [Clip("x.mp3", BR[0], "m", "spk_x", "a b", 0,
                         "India and South Asia (India")]
    s = draw(c, brackets=BR, cap=25, max_shard=8, accents=NATIVE_ANGLOPHONE)
    assert all(x.accent in NATIVE_ANGLOPHONE for x in s)


def test_match_accents_equalises_composition_exactly():
    """A native-only filter is not enough: inside that stratum the real corpus
    still swings 28.6 points between brackets."""
    s = draw(_clips(), brackets=BR, cap=25, max_shard=8, accents=NATIVE_ANGLOPHONE)
    before = {b: collections.Counter(x.accent for x in s if x.age == b) for b in BR}
    assert before[BR[0]] != before[BR[1]], "fixture must start unbalanced"

    m = match_accents(s, BR)
    after = {b: collections.Counter(x.accent for x in m if x.age == b) for b in BR}
    assert after[BR[0]] == after[BR[1]], after
    assert m, "matching must not empty the sample"


def test_match_accents_preserves_gender_balance():
    """Regression: matching on accent alone silently undid draw()'s gender
    balance, because an accent-wise subset need not keep the male/female split.
    One real run came out 54/46 in the sixties against 50/50 in the seventies.

    The guarantee is that the split is *identical across brackets*, which is
    what stops gender confounding the age comparison. It is deliberately not
    "exactly 50/50 within a bracket": on the real corpus joint matching lands
    at 506/532, and asserting an even split here would pass on this fixture
    while encoding a promise the code does not make.
    """
    s = draw(_clips(), brackets=BR, cap=25, max_shard=8, accents=NATIVE_ANGLOPHONE)
    m = match_accents(s, BR)
    counts = [collections.Counter(x.gender for x in m if x.age == b) for b in BR]
    assert counts[0] == counts[1], counts
    assert sum(counts[0].values()) > 0
