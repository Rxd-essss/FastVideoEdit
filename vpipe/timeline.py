"""Timeline math: merge removed intervals and remap timestamps onto the
shortened (post-cut) timeline.

After we delete a set of [start, end] intervals from the original video, every
surviving timestamp shifts left by the total duration of all cuts that lie
before it. This module computes that mapping in O(log n) per lookup.
"""
from __future__ import annotations

from bisect import bisect_right
from typing import Iterable, Optional

from .models import Word


def merge_intervals(intervals: Iterable[tuple[float, float]],
                    gap: float = 0.0) -> list[tuple[float, float]]:
    """Sort and merge overlapping/touching intervals into disjoint ones."""
    iv = sorted([(float(a), float(b)) for a, b in intervals if b > a])
    merged: list[list[float]] = []
    for a, b in iv:
        if merged and a <= merged[-1][1] + gap:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return [(a, b) for a, b in merged]


class Timeline:
    """Maps original timestamps to post-cut timestamps given removed intervals."""

    def __init__(self, removed: Iterable[tuple[float, float]], duration: float):
        self.removed = merge_intervals(removed)
        self.duration = float(duration)
        self.starts = [a for a, _ in self.removed]
        self.cum: list[float] = []   # removed duration BEFORE interval i begins
        c = 0.0
        for a, b in self.removed:
            self.cum.append(c)
            c += (b - a)
        self.total_removed = c

    def _idx(self, t: float) -> int:
        return bisect_right(self.starts, t) - 1

    def removed_before(self, t: float) -> float:
        """Total removed duration strictly before original time ``t``.

        If ``t`` is inside a cut, counts the portion of that cut up to ``t``.
        """
        i = self._idx(t)
        if i < 0:
            return 0.0
        a, b = self.removed[i]
        if t >= b:
            return self.cum[i] + (b - a)
        return self.cum[i] + (t - a)   # t within [a, b)

    def inside(self, t: float) -> bool:
        i = self._idx(t)
        if i < 0:
            return False
        a, b = self.removed[i]
        return a <= t < b

    def remap(self, t: float) -> Optional[float]:
        """New timestamp, or None if ``t`` falls inside a removed interval."""
        if self.inside(t):
            return None
        return t - self.removed_before(t)

    def remap_clamped(self, t: float) -> float:
        """New timestamp, clamped to [0, new_duration].

        Timestamps inside a cut snap to the cut's seam; a timestamp past the
        original duration (whisper can emit word ends slightly beyond it) is
        clamped to the new end rather than landing past it.
        """
        return min(self.new_duration(), max(0.0, t - self.removed_before(t)))

    def new_duration(self) -> float:
        return max(0.0, self.duration - self.total_removed)

    def kept_segments(self) -> list[tuple[float, float]]:
        """The complement of the removed intervals within [0, duration]."""
        kept: list[tuple[float, float]] = []
        cursor = 0.0
        for a, b in self.removed:
            if a > cursor:
                kept.append((cursor, a))
            cursor = max(cursor, b)
        if cursor < self.duration:
            kept.append((cursor, self.duration))
        return kept

    def removed_overlap(self, a: float, b: float) -> float:
        """Total removed duration overlapping the original interval [a, b)."""
        if b <= a:
            return 0.0
        return max(0.0, self.removed_before(b) - self.removed_before(a))


def remap_words(words: list[Word], tl: Timeline) -> list[Word]:
    """Shift words onto the post-cut timeline, dropping words inside cuts.

    A word is dropped when more than half of its duration falls inside a cut
    (overlap fraction > 50%). The surviving word's start/end are clamped to the
    kept side of any seam so a partially-cut word doesn't bleed across the trim.
    """
    out: list[Word] = []
    for w in words:
        dur = max(0.0, w.end - w.start)
        if dur <= 0.0:
            # Zero/negative-width word: fall back to the midpoint test.
            if tl.inside(0.5 * (w.start + w.end)):
                continue
        else:
            # Drop when the majority of the word lies inside a removed interval.
            if tl.removed_overlap(w.start, w.end) > 0.5 * dur:
                continue
        # remap_clamped snaps any endpoint that landed inside a cut onto the
        # cut's seam, keeping the surviving word on the kept side.
        ns = tl.remap_clamped(w.start)
        ne = tl.remap_clamped(w.end)
        if ne <= ns:
            ne = ns + 0.02
        out.append(Word(w.word, ns, ne, w.prob))
    return out
