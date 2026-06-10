"""Pause / dead-air detection from word-level timestamps."""
from __future__ import annotations

from ..config import PausesCfg
from ..models import ACTION_REMOVE, TYPE_PAUSE, CutSegment, Word


def detect(words: list[Word], duration: float, cfg: PausesCfg) -> list[CutSegment]:
    """Detect dead-air spans from word gaps.

    A gap is flagged when it exceeds ``min_silence``. We pad the cut inward by
    ``pad_start`` on the lead-in (the side following speech) and ``pad_end`` on
    the side before the next speech, so a sliver of breath/room-tone is kept on
    each edge. The EFFECTIVE silence a span must reach before any audio is
    actually removed is therefore ``min_silence + pad_start + pad_end`` (the
    pads eat into the raw gap, and spans shorter than ``min_keep`` are dropped).
    Every span is clamped to [0, duration] and only added when start < end.
    """
    out: list[CutSegment] = []
    dur = max(0.0, float(duration))

    def add(a: float, b: float) -> None:
        # Clamp to the media bounds and only keep a valid, long-enough span.
        a = min(max(0.0, a), dur)
        b = min(max(0.0, b), dur)
        if b > a and (b - a) >= cfg.min_keep:
            out.append(CutSegment(
                id="", start=round(a, 3), end=round(b, 3),
                type=TYPE_PAUSE, action=ACTION_REMOVE, enabled=True,
                text=f"пауза {b - a:.1f}с"))

    if not words:
        return out

    # Leading silence (keep a little lead-in before the first word). Use
    # pad_start consistently for the lead-in side (the side adjacent to speech).
    first = words[0].start
    if first > cfg.min_silence:
        add(0.0, first - cfg.pad_start)

    # Gaps between consecutive words.
    for w0, w1 in zip(words, words[1:]):
        gap = w1.start - w0.end
        if gap > cfg.min_silence:
            add(w0.end + cfg.pad_start, w1.start - cfg.pad_end)

    # Trailing silence.
    last = words[-1].end
    if dur - last > cfg.min_silence:
        add(last + cfg.pad_start, dur)

    return out
