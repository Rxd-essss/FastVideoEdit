from vpipe.models import Word
from vpipe.timeline import Timeline, merge_intervals, remap_words


def test_merge():
    assert merge_intervals([(0, 1), (0.5, 2), (3, 4)]) == [(0, 2), (3, 4)]
    assert merge_intervals([(2, 3), (0, 1)]) == [(0, 1), (2, 3)]
    assert merge_intervals([(1, 1), (2, 1.5)]) == []   # zero / inverted dropped


def test_remap_basic():
    tl = Timeline([(1, 2), (5, 6)], duration=10)
    assert tl.total_removed == 2
    assert tl.new_duration() == 8
    assert tl.remap(0) == 0
    assert tl.remap(3) == 2        # one cut (1..2) before it
    assert tl.remap(7) == 5        # two cuts before it
    assert tl.inside(1.5) is True
    assert tl.remap(1.5) is None


def test_kept_segments():
    tl = Timeline([(1, 2), (5, 6)], duration=10)
    assert tl.kept_segments() == [(0, 1), (2, 5), (6, 10)]


def test_kept_full_when_no_cuts():
    tl = Timeline([], duration=10)
    assert tl.kept_segments() == [(0, 10)]
    assert tl.new_duration() == 10


def test_remap_words_drops_inside():
    words = [Word("a", 0.0, 0.5), Word("b", 1.2, 1.8), Word("c", 7.0, 7.5)]
    tl = Timeline([(1, 2), (5, 6)], duration=10)
    out = remap_words(words, tl)
    assert [w.word for w in out] == ["a", "c"]   # "b" was inside (1..2)
    assert out[1].start == 5.0                    # 7 - 2 removed
