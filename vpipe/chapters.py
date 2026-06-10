"""Stage 8 — YouTube chapters from the final (cut) transcript.

Uses the local LLM to split the talking into themed chapters with short Russian
titles (the model returns segment INDICES, we map to exact times), then enforces
YouTube's rules: first marker exactly 00:00, >=3 chapters, each >=10 s, ascending.
A non-LLM fallback splits evenly with keyword-ish titles.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .config import ChaptersCfg, LlmCfg, MaskingCfg
from .detect.profanity import ProfanityMatcher
from .llm import OllamaClient, segment_windows
from .models import Segment, Transcript, Word
from .subtitles import mask_text
from .timeline import Timeline, remap_words

_SYSTEM = (
    "Ты помогаешь оформить описание YouTube-видео. Дана расшифровка речи по "
    "сегментам с номерами и временем. Раздели видео на смысловые ГЛАВЫ. Для "
    "каждой главы укажи номер сегмента, с которого она начинается, и короткий "
    "русский заголовок (2–5 слов, без точки в конце). Первая глава — сегмент 0. "
    "Главы строго по возрастанию, не слишком частые (каждая не короче 10 секунд), "
    "минимум 3 главы. Верни строго JSON."
)

_SCHEMA = {
    "type": "object",
    "properties": {
        "chapters": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "title": {"type": "string"},
                },
                "required": ["index", "title"],
            },
        }
    },
    "required": ["chapters"],
}


@dataclass
class Chapter:
    time: float
    title: str


def _fmt(t: float, with_hours: bool) -> str:
    t = max(0, int(round(t)))
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    if with_hours:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{(h * 60 + m):02d}:{s:02d}"


def _final_segments(transcript: Transcript, tl: Timeline) -> list[Segment]:
    out: list[Segment] = []
    for s in transcript.segments:
        mid = 0.5 * (s.start + s.end)
        if tl.inside(mid):
            continue
        out.append(Segment(tl.remap_clamped(s.start), tl.remap_clamped(s.end), s.text))
    out.sort(key=lambda x: x.start)
    return out


def _title_at(time: float, words: list[Word], maxlen: int = 42) -> str:
    picked: list[str] = []
    for w in words:
        if w.start < time - 0.01:
            continue
        picked.append(w.word.strip())
        joined = " ".join(picked)
        if len(joined) >= 20 or w.word.strip().endswith((".", "!", "?", "…")):
            break
        if len(picked) >= 6:
            break
    title = " ".join(picked).strip(" .,!?–-").strip()
    if not title:
        title = "Глава"
    title = title[0].upper() + title[1:]
    return title[:maxlen]


def enforce_rules(chs: list[Chapter], new_duration: float,
                  cfg: ChaptersCfg) -> list[Chapter]:
    chs = sorted(chs, key=lambda c: c.time)
    # First marker must be exactly 0.
    if not chs or chs[0].time > 0.5:
        chs.insert(0, Chapter(0.0, chs[0].title if chs else "Вступление"))
    chs[0].time = 0.0
    # Drop chapters closer than min_length to the previously kept one. The end
    # guard only drops chapters within min_length of the very end (a chapter at
    # e.g. 23:50 of a 24:00 video is too short to count), not everything past
    # duration-1 — that used to silently swallow the final third of long videos.
    kept: list[Chapter] = []
    for c in chs:
        if not kept:
            kept.append(c)
        elif c.time - kept[-1].time >= cfg.min_length \
                and c.time <= new_duration - cfg.min_length:
            kept.append(c)
    # Cap: keep the first marker, then sample the rest EVENLY across the whole
    # timeline so the cap doesn't just take the opening N chapters and drop the
    # entire back half of a long video.
    if len(kept) > cfg.max_chapters:
        n = cfg.max_chapters
        rest = kept[1:]
        if n <= 1:
            kept = [kept[0]]
        else:
            picked = [kept[0]]
            m = len(rest)
            take = n - 1
            for j in range(take):
                # evenly spaced indices across the remaining chapters
                idx = round(j * (m - 1) / (take - 1)) if take > 1 else m - 1
                picked.append(rest[idx])
            # dedupe (rounding can collide) while preserving order
            seen: set[float] = set()
            kept = []
            for c in picked:
                if c.time not in seen:
                    seen.add(c.time)
                    kept.append(c)

    # Force YouTube's >=3 rule: if we still have too few chapters but the video
    # is long enough to hold them, inject evenly-spaced markers. Without this a
    # long video that the LLM under-segmented (or a sparse fallback) silently
    # ships with 1–2 chapters and fails YouTube's chaptering. The cap always
    # wins, so a max_chapters < min_chapters config never force-adds.
    if len(kept) < cfg.min_chapters \
            and cfg.max_chapters >= cfg.min_chapters \
            and new_duration >= cfg.min_chapters * cfg.min_length:
        target = min(cfg.max_chapters,
                     max(cfg.min_chapters,
                         int(new_duration // max(cfg.min_length, 1.0))))
        existing = sorted(c.time for c in kept)
        titles = {round(c.time, 3): c.title for c in kept}
        slots = [i * new_duration / target for i in range(target)]
        merged: list[Chapter] = []
        last = -1e9
        for t in slots:
            # snap to a nearby existing marker so we keep good LLM titles
            near = next((e for e in existing if abs(e - t) < cfg.min_length / 2), None)
            tt = near if near is not None else t
            if tt - last < cfg.min_length and merged:
                continue
            title = titles.get(round(tt, 3), "Глава")
            merged.append(Chapter(tt, title))
            last = tt
        if merged:
            merged[0].time = 0.0
            kept = merged
    return kept


def _fallback(final_segs: list[Segment], words: list[Word],
              new_duration: float, cfg: ChaptersCfg, log) -> list[Chapter]:
    log("  chapters: using even-split fallback.")
    # Pick N so each chapter is >= min_length and we aim for ~1 per 90s.
    max_by_len = max(1, int(new_duration // cfg.min_length))
    n = min(max_by_len, max(cfg.min_chapters, int(new_duration // 90) + 1))
    n = max(1, min(n, cfg.max_chapters))
    chs: list[Chapter] = []
    for i in range(n):
        t = i * new_duration / n
        chs.append(Chapter(t, _title_at(t, words)))
    return enforce_rules(chs, new_duration, cfg)


def generate(transcript: Transcript, removed: list[tuple[float, float]],
             cfg: ChaptersCfg, out_path: str | Path,
             llm: Optional[OllamaClient] = None,
             matcher: Optional[ProfanityMatcher] = None,
             mask: Optional[MaskingCfg] = None, log=print,
             on_progress=None, on_stage=None) -> dict:
    tl = Timeline(removed, transcript.duration)
    new_duration = tl.new_duration()
    final_segs = _final_segments(transcript, tl)
    words = remap_words(transcript.all_words(), tl)

    chapters: Optional[list[Chapter]] = None
    if llm is not None and final_segs:
        with_hours = new_duration >= 3600
        # Long videos overflow a single prompt — split into windows and offset
        # per-window indices back to GLOBAL segment indices, deduping overlap.
        by_start: dict[float, str] = {}
        windows = segment_windows(len(final_segs), llm.cfg)
        n_win = len(windows)
        for wi, (win_start, win_end) in enumerate(windows):
            # Surface per-window progress so a long video doesn't look frozen on
            # a single «Главы…» stage for minutes.
            if on_stage and n_win > 1:
                on_stage(f"Главы… {wi + 1}/{n_win}")
            if on_progress:
                on_progress(wi / max(1, n_win))
            window = final_segs[win_start:win_end]
            lines = [f"{local_i} | {_fmt(s.start, with_hours)} | {s.text}"
                     for local_i, s in enumerate(window)]
            user = ("Сегменты (после монтажа):\n" + "\n".join(lines) +
                    "\n\nВерни JSON: {\"chapters\": [{\"index\": <номер>, \"title\": <заголовок>}]}")
            try:
                # Keep qwen3 warm BETWEEN windows (default keep_alive=0 would
                # unload+reload it once per window on a long video); unload only
                # after the last window so it frees VRAM for the next stage.
                ka = 0 if wi == n_win - 1 else 60
                data = llm.chat_json(_SYSTEM, user, _SCHEMA, keep_alive=ka)
            except Exception as e:  # noqa: BLE001
                log(f"  chapters: window [{win_start}:{win_end}] failed ({e}); skipped.")
                continue
            for c in data.get("chapters", []):
                local_idx = c.get("index")
                if isinstance(local_idx, bool) or not isinstance(local_idx, int):
                    continue
                if not (0 <= local_idx < len(window)):
                    continue
                t = final_segs[win_start + local_idx].start
                if t not in by_start:
                    by_start[t] = str(c.get("title", "")).strip()
        if by_start:
            raw = [Chapter(t, by_start[t]) for t in sorted(by_start)]
            chapters = enforce_rules(raw, new_duration, cfg)

    if not chapters or len(chapters) < cfg.min_chapters:
        if not final_segs:
            log("  chapters: no transcript — skipped.")
            return {"chapters": 0, "path": None}
        chapters = _fallback(final_segs, words, new_duration, cfg, log)

    with_hours = new_duration >= 3600 or chapters[-1].time >= 3600

    def _title(t: str) -> str:
        t = t.strip() or "Глава"
        return mask_text(t, matcher, mask) if (matcher and mask) else t

    lines = [f"{_fmt(c.time, with_hours)} {_title(c.title)}" for c in chapters]
    Path(out_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    note = "" if len(chapters) >= cfg.min_chapters else \
        f"  (note: only {len(chapters)} chapters — video too short for YouTube's 3-chapter rule)"
    log(f"  {len(chapters)} chapters -> {Path(out_path).name}{note}")
    return {"chapters": len(chapters), "path": str(out_path)}
