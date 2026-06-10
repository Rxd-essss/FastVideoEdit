"""Cut-list helpers: human-readable review file + resolution into the concrete
intervals the renderer needs."""
from __future__ import annotations

from pathlib import Path

from .models import (ACTION_CENSOR, ACTION_REMOVE, CutList, CutSegment)
from .timeline import Timeline, merge_intervals


def _fmt_t(t: float) -> str:
    m, s = divmod(t, 60)
    h, m = divmod(int(m), 60)
    return f"{h:d}:{m:02d}:{s:06.3f}" if h else f"{m:02d}:{s:06.3f}"


def save_txt(cl: CutList, path: str | Path) -> None:
    """Write a readable review file (timecodes + reasons per cut)."""
    lines = [
        f"# Cut list for: {cl.source}",
        f"# Duration: {cl.duration:.1f}s   Segments: {len(cl.segments)}",
        "# Legend: [x] = will be applied, [ ] = disabled (kept).",
        "#         Edit the JSON file (toggle \"enabled\", change \"action\") and re-run with --apply.",
        "",
    ]
    by_type: dict[str, int] = {}
    for s in cl.segments:
        by_type[s.type] = by_type.get(s.type, 0) + 1
    lines.append("# Counts: " + ", ".join(f"{k}={v}" for k, v in sorted(by_type.items())))
    lines.append("")
    for s in sorted(cl.segments, key=lambda x: x.start):
        box = "x" if s.enabled else " "
        info = s.reason or s.text
        lines.append(
            f"[{box}] {s.id:<7} {_fmt_t(s.start)}–{_fmt_t(s.end)} "
            f"({s.duration:.2f}s)  {s.type}/{s.action}  {info}")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def resolve(cl: CutList) -> tuple[list[tuple[float, float]], list[CutSegment]]:
    """Return (merged removed intervals, effective censor segments).

    Each enabled censor is CLIPPED to the kept (surviving) timeline rather than
    being kept/dropped all-or-nothing by its midpoint. A censor that straddles a
    cut is split into its surviving sub-interval(s) and each one is returned as
    its own censor CutSegment, so profanity is fully muted in whatever audio
    survives the removals. A censor fully inside a cut produces no segments
    (the audio there is gone anyway).

    Note: the returned censor segments are in *original* timeline coordinates —
    the renderer/censor builder works against the original media before trims —
    they are simply restricted to the spans that will still exist after cutting.
    """
    removed = merge_intervals(cl.enabled_removes())
    tl = Timeline(removed, cl.duration)
    kept = tl.kept_segments()

    censors: list[CutSegment] = []
    for c in cl.enabled_censors():
        cs, ce = c.start, c.end
        parts: list[tuple[float, float]] = []
        for ka, kb in kept:
            if kb <= cs:
                continue
            if ka >= ce:
                break
            a = max(cs, ka)
            b = min(ce, kb)
            if b - a > 1e-4:
                parts.append((a, b))
        if not parts:
            continue                       # entire censor fell inside a cut
        if len(parts) == 1 and parts[0] == (cs, ce):
            censors.append(c)              # untouched: keep the original object
            continue
        for k, (a, b) in enumerate(parts):
            sub_id = c.id if (len(parts) == 1) else (f"{c.id}.{k + 1}" if c.id else "")
            censors.append(CutSegment(
                id=sub_id, start=round(a, 3), end=round(b, 3),
                type=c.type, action=c.action, enabled=True,
                text=c.text, reason=c.reason, word=c.word))
    return removed, censors
