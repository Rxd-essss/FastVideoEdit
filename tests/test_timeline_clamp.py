# -*- coding: utf-8 -*-
"""Wave 4 (#74): Timeline clamps removed intervals into [0, duration] so an
unvalidated PUT /api/cutlist or a hand-edited cutlist.json with a past-EOF or
negative interval can't emit a degenerate kept segment beyond the media."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vpipe.timeline import Timeline, merge_intervals                  # noqa: E402


def test_clamp_past_eof():
    # FAILS before the fix: removed stays [(50,62),(65,70)], kept_segments emits
    # a spurious (62,65) segment past EOF and new_duration() == 43.
    dur = 60.0
    tl = Timeline([(50, dur + 2), (dur + 5, dur + 10)], dur)
    assert tl.removed == merge_intervals([(50, 60)])   # == [(50, 60)]
    assert tl.kept_segments() == [(0, 50)]             # no segment past 60
    assert tl.new_duration() == 50
    assert all(a < dur for a in tl.starts)             # no start >= duration


def test_clamp_negative_start():
    tl = Timeline([(-5, 10)], 60.0)
    assert tl.removed == [(0, 10)]                      # -5 clamped to 0
    assert tl.kept_segments() == [(10, 60)]
    assert tl.new_duration() == 50


def test_valid_cutlist_unchanged():
    # In-bounds intervals: clamp is a byte-for-byte no-op.
    tl = Timeline([(1, 2), (5, 6)], 10.0)
    assert tl.removed == [(1, 2), (5, 6)]
    assert tl.kept_segments() == [(0, 1), (2, 5), (6, 10)]
    assert tl.new_duration() == 8
