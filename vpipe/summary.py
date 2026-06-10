"""Stage 9 — final summary report."""
from __future__ import annotations

from .models import ACTION_CENSOR, ACTION_REMOVE, CutList


def _fmt(t: float) -> str:
    m, s = divmod(int(round(t)), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def summarize(cl: CutList, new_duration: float, render_res: dict,
              subs_res: dict, chapters_res: dict, log=print) -> None:
    by_type: dict[str, float] = {}
    n_by_type: dict[str, int] = {}
    censored = 0
    for s in cl.segments:
        if not s.enabled:
            continue
        if s.action == ACTION_REMOVE:
            by_type[s.type] = by_type.get(s.type, 0.0) + s.duration
            n_by_type[s.type] = n_by_type.get(s.type, 0) + 1
        elif s.action == ACTION_CENSOR:
            censored += 1

    saved = cl.duration - new_duration
    log("")
    log("=" * 56)
    log("  SUMMARY")
    log("=" * 56)
    log(f"  Duration:   {_fmt(cl.duration)}  ->  {_fmt(new_duration)}"
        f"   (saved {_fmt(saved)}, {100 * saved / cl.duration:.0f}%)" if cl.duration else "")
    log(f"  Removed segments:")
    for t in sorted(n_by_type):
        log(f"    {t:<10} {n_by_type[t]:>4}   ({_fmt(by_type[t])})")
    if not n_by_type:
        log("    (none)")
    log(f"  Censored words: {censored}")
    log(f"  Encoder:        {render_res.get('encoder', '?')}")
    log(f"  Subtitle cues:  {subs_res.get('cues', 0)}")
    log(f"  Chapters:       {chapters_res.get('chapters', 0)}")
    log("=" * 56)
    log(f"  Output: {render_res.get('out', '?')}")
    log("=" * 56)
