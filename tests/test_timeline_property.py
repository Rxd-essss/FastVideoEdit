"""Property / fuzz tests for the interval math the whole cut pipeline stands on
(audit #86): vpipe.timeline (merge_intervals / Timeline / remap_words) and
vpipe.subtitles.build_cues.

The hand suite guards these functions with a handful of 2-interval cases; this
module throws hundreds of overlapping / touching / inverted / past-EOF /
float-jittered interval sets plus whisper-shaped words at them and asserts the
invariants the pipeline actually depends on.

hypothesis is DEV-ONLY (requirements-dev.txt) and is NOT installed on CI
(.github/workflows/ci.yml installs requirements.txt + pytest + httpx only), so
the importorskip below skips this entire module there -- CI stays green and the
no-ffmpeg / hermetic philosophy is untouched. Run locally after
``pip install -r requirements-dev.txt``.
"""
import pytest

hypothesis = pytest.importorskip("hypothesis")   # CI has no hypothesis -> whole module skipped
from hypothesis import given, settings, strategies as st

from vpipe.timeline import Timeline, merge_intervals, remap_words
from vpipe.subtitles import build_cues
from vpipe.config import SubsCfg, MaskingCfg, ProfanityLists
from vpipe.detect.profanity import ProfanityMatcher
from vpipe.models import Word

_D = 100.0
# Adversarial removed sets: overlapping, touching, inverted, negative, past-EOF,
# float-jittered -- exactly the shape a hand-edited cutlist.json or an
# unvalidated PUT /api/cutlist can carry into Timeline's clamp chokepoint.
_ivs = st.lists(st.tuples(st.floats(-10, 110), st.floats(-10, 110)), max_size=40)


@settings(deadline=None, max_examples=200)
@given(_ivs)
def test_kept_and_removed_partition_duration(raw):
    tl = Timeline(raw, _D)
    kept = tl.kept_segments()
    # kept + removed exactly partition [0, D]: their durations sum to D.
    total = sum(b - a for a, b in kept) + tl.total_removed
    assert abs(total - _D) < 1e-6
    for a, b in kept:
        assert 0.0 <= a <= b <= _D
    # kept segments are ordered and disjoint (non-overlapping).
    for (a1, b1), (a2, b2) in zip(kept, kept[1:]):
        assert b1 <= a2


@settings(deadline=None, max_examples=200)
@given(_ivs, st.floats(-10, 110))
def test_remap_clamped_in_range(raw, t):
    tl = Timeline(raw, _D)
    c = tl.remap_clamped(t)
    assert 0.0 <= c <= tl.new_duration() + 1e-9


@settings(deadline=None, max_examples=200)
@given(_ivs)
def test_remap_clamped_monotonic_non_decreasing(raw):
    # remap_clamped never moves backwards as original time advances.
    tl = Timeline(raw, _D)
    prev = -1.0
    for i in range(0, 101):
        c = tl.remap_clamped(i * (_D / 100.0))
        assert c + 1e-9 >= prev
        prev = c


@settings(deadline=None, max_examples=200)
@given(st.lists(st.tuples(st.floats(-5, 55), st.floats(-5, 55)), max_size=30))
def test_merge_intervals_disjoint_sorted(raw):
    m = merge_intervals(raw)
    for a, b in m:
        assert b > a                     # zero-width / inverted dropped
    for (a1, b1), (a2, b2) in zip(m, m[1:]):
        assert a1 <= a2 and b1 < a2      # sorted, strictly non-overlapping (touching merged)


@st.composite
def _words(draw):
    """Whisper-shaped words: strictly increasing starts, positive width.

    Real transcripts never stack many zero-width words at a single instant;
    holding width >= 0.05 mirrors that and keeps the fuzz aimed at the INTERVAL
    math (Timeline / remap) rather than at a degenerate cue-packer input.
    """
    n = draw(st.integers(1, 30))
    t = draw(st.floats(0.0, 5.0))
    out = []
    for _ in range(n):
        t += draw(st.floats(0.0, 4.0))       # inter-word gap
        w = draw(st.floats(0.05, 2.0))       # word width
        out.append(Word("слово", t, t + w))
        t += w
    return out


@settings(deadline=None, max_examples=200)
@given(_words(), _ivs)
def test_build_cues_invariants(words, raw):
    tl = Timeline(raw, _D)
    remapped = remap_words(words, tl)
    matcher = ProfanityMatcher(ProfanityLists(roots=[], allow=[]))
    total = tl.new_duration()
    cues = build_cues(remapped, matcher, SubsCfg(), MaskingCfg(), total)
    for c in cues:
        assert 0.0 <= c.start <= c.end <= total + 1e-6
    for c1, c2 in zip(cues, cues[1:]):
        assert c1.end <= c2.start + 1e-6     # monotonic, non-overlapping
