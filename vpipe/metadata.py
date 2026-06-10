"""Stage 9 — YouTube metadata (title / description / tags / hook) from the final
(cut) transcript via the local LLM.

Mirrors chapters.py: same OllamaClient.chat_json(system, user, schema) call shape
(temperature 0, think:false handled in llm.py), the same _final_segments() helper
to project the transcript onto the post-cut timeline, and the same graceful
degradation when ``llm`` is None.

The model gets the final transcript (and, if available, the already-generated
chapters list) and returns a clickable-but-not-clickbait Russian title, a 2–4
paragraph description, 10–15 tags and one short hook. normalize() then enforces
the hard limits (title length, tag count/dedup, hook length) so the UI never
sees an over-long or duplicate-laden payload regardless of what the LLM emits.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from .config import MaskingCfg, MetadataCfg
from .detect.profanity import ProfanityMatcher
from .llm import OllamaClient, segment_windows
from .models import Segment, Transcript
from .subtitles import mask_text
from .timeline import Timeline

_EMPTY: dict = {"title": "", "description": "", "tags": [], "hook": ""}

_SYSTEM = (
    "Ты помогаешь оформить YouTube-видео на русском языке. Дана расшифровка речи "
    "после монтажа (нумерованные сегменты со временем), иногда — список глав. "
    "Сгенерируй метаданные ролика. Требования:\n"
    "- title: кликабельный, но НЕ кликбейтный заголовок, до 100 символов, без "
    "кавычек по краям, на русском;\n"
    "- description: 2–4 абзаца живого описания по сути ролика; если переданы "
    "главы, добавь их списком в конце в формате «00:00 Название»;\n"
    "- tags: 10–15 коротких тегов (ключевых слов/фраз) на русском, без решёток "
    "и без дубликатов;\n"
    "- hook: одна короткая цепляющая фраза-завлекалка (до 200 символов).\n"
    "Опирайся ТОЛЬКО на содержание расшифровки, не выдумывай фактов. "
    "Верни строго JSON."
)

_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "description": {"type": "string"},
        "tags": {"type": "array", "items": {"type": "string"}},
        "hook": {"type": "string"},
    },
    "required": ["title", "description", "tags", "hook"],
}


def _final_segments(transcript: Transcript, tl: Timeline) -> list[Segment]:
    """Project the transcript onto the post-cut timeline (copy of chapters.py).

    Segments whose midpoint falls inside a removed interval are dropped; the rest
    are remapped to FINAL coordinates and sorted by start time.
    """
    out: list[Segment] = []
    for s in transcript.segments:
        mid = 0.5 * (s.start + s.end)
        if tl.inside(mid):
            continue
        out.append(Segment(tl.remap_clamped(s.start), tl.remap_clamped(s.end), s.text))
    out.sort(key=lambda x: x.start)
    return out


def _read_chapters_block(chapters_path: Optional[str | Path]) -> str:
    """Read a chapters.txt file into a prompt block, or '' if absent/unreadable."""
    if not chapters_path:
        return ""
    p = Path(chapters_path)
    if not p.exists():
        return ""
    try:
        text = p.read_text(encoding="utf-8").strip()
    except OSError:
        return ""
    if not text:
        return ""
    return "Главы:\n" + text + "\n\n"


def normalize(raw: dict, cfg: MetadataCfg) -> dict:
    """Coerce/trim a raw LLM dict to the contract {title, description, tags, hook}.

    - title trimmed to cfg.max_title_chars
    - tags coerced to non-empty strings, order-preserving case-insensitive dedup,
      then capped at cfg.n_tags
    - hook trimmed to cfg.max_hook_chars
    Never raises: missing/odd fields degrade to empty values.
    """
    raw = raw or {}

    def _s(v) -> str:
        return v.strip() if isinstance(v, str) else ""

    title = _s(raw.get("title"))[: cfg.max_title_chars].strip()
    description = _s(raw.get("description"))
    hook = _s(raw.get("hook"))[: cfg.max_hook_chars].strip()

    tags: list[str] = []
    seen: set[str] = set()
    raw_tags = raw.get("tags")
    if isinstance(raw_tags, list):
        for t in raw_tags:
            tag = _s(t).lstrip("#").strip()
            if not tag:
                continue
            key = tag.casefold()
            if key in seen:
                continue
            seen.add(key)
            tags.append(tag)
            if len(tags) >= cfg.n_tags:
                break

    return {"title": title, "description": description, "tags": tags, "hook": hook}


def generate(transcript: Transcript, removed: list[tuple[float, float]],
             cfg: MetadataCfg, llm: Optional[OllamaClient] = None,
             chapters_path: Optional[str | Path] = None,
             matcher: Optional[ProfanityMatcher] = None,
             mask: Optional[MaskingCfg] = None, log=print) -> dict:
    """Generate YouTube metadata from the final (post-cut) transcript.

    Returns {'title', 'description', 'tags':[...], 'hook'} — all empty when the
    LLM is unavailable, the transcript is empty, or the call fails (graceful: the
    caller never has to handle an exception to fall back to "no metadata").
    """
    if llm is None:
        log("  metadata: LLM off — skipped.")
        return dict(_EMPTY)

    tl = Timeline(removed, transcript.duration)
    new_duration = tl.new_duration()
    final_segs = _final_segments(transcript, tl)
    if not final_segs:
        log("  metadata: no transcript — skipped.")
        return dict(_EMPTY)

    with_hours = new_duration >= 3600

    def _fmt(t: float) -> str:
        t = max(0, int(round(t)))
        h, rem = divmod(t, 3600)
        m, s = divmod(rem, 60)
        if with_hours:
            return f"{h:d}:{m:02d}:{s:02d}"
        return f"{(h * 60 + m):02d}:{s:02d}"

    # A long transcript overflows the context. Like chapters.py we lean on the
    # configured window size, but metadata only needs ONE pass over a
    # representative slice, so we take the FIRST window [0, size). For typical
    # videos this is the whole thing.
    windows = segment_windows(len(final_segs), llm.cfg)
    win_start, win_end = windows[0]
    window = final_segs[win_start:win_end]

    def _clean(text: str) -> str:
        text = (text or "").strip()
        return mask_text(text, matcher, mask) if (matcher and mask) else text

    lines = [f"{i} | {_fmt(s.start)} | {_clean(s.text)}"
             for i, s in enumerate(window)]
    chapters_block = _read_chapters_block(chapters_path)
    user = (
        "Транскрипт (после монтажа):\n" + "\n".join(lines) + "\n\n" +
        chapters_block +
        "Верни JSON: {\"title\": <заголовок>, \"description\": <описание>, "
        "\"tags\": [<тег>, ...], \"hook\": <короткая фраза>}"
    )

    try:
        data = llm.chat_json(_SYSTEM, user, _SCHEMA)
    except Exception as e:  # noqa: BLE001 — graceful: caller wants empty metadata
        log(f"  metadata: LLM call failed ({e}); skipped.")
        return dict(_EMPTY)

    result = normalize(data, cfg)
    # Mask any profanity the model may have echoed from the transcript into the
    # user-visible fields (same policy as chapters titles).
    if matcher and mask:
        result["title"] = mask_text(result["title"], matcher, mask)
        result["description"] = mask_text(result["description"], matcher, mask)
        result["hook"] = mask_text(result["hook"], matcher, mask)
        result["tags"] = [mask_text(t, matcher, mask) for t in result["tags"]]
    log(f"  metadata: title «{result['title'][:48]}», {len(result['tags'])} tags.")
    return result
