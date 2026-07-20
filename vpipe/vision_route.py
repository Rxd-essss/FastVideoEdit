# -*- coding: utf-8 -*-
"""Vision-route stage for the montage pipeline («variant C»).

The TEXT planner (enrich_llm) proposes a visual for a moment from the transcript
alone — it cannot SEE the frame, so it slaps a diagram over a talking head or
over a screen that already shows the concept («слоп»). This stage closes that
gap: for each visual-bearing candidate it samples a few real frames around the
moment, shows them to a local vision LLM together with the transcript quote and
the proposed visual, and applies a strict-JSON verdict:

  * ``insert=false``  -> veto: disable the item (never rendered).
  * ``insert=true`` + ``reroute`` -> if the VLM's ``kind`` disagrees with the
    chosen candidate AND the planner already produced a matching candidate,
    switch ``payload.selected`` to it. NEVER invents an asset.

Runs in its own Ollama VRAM slot (the caller unloads the text model first and
this model after). Fully best-effort: a dead Ollama, a missing model, an ffmpeg
hiccup or a parse miss leaves the item untouched — the montage never fails here.

The module is deliberately duck-typed on the item/payload objects (only reads
``.type/.enabled/.quote/.t_start/.status/.status_note`` and, for images,
``.payload.candidates/.selected/.style_hint``) so it needs no enrich imports and
is trivial to unit-test with SimpleNamespace stand-ins.
"""
from __future__ import annotations

import base64
import subprocess
from pathlib import Path
from typing import Callable, Optional

LogFn = Callable[[str], None]


def _noop(*_a, **_k) -> None:
    pass


# Item types that overlay a visual and are worth judging. cta_* plates are
# fixed-format calls-to-action (not content visuals) — left alone.
VISUAL_TYPES = ("image", "list_card", "animation")

# VLM ``kind`` -> which candidate ``source`` values satisfy it (re-route target).
_KIND_TO_SOURCES = {
    "diagram": ("schematic",),
    "image": ("diffusion", "stock"),
    "list_card": ("list_card",),
}

VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "scene": {"type": "string",
                  "enum": ["talking_head", "screen_share", "slide",
                           "code_editor", "terminal", "broll", "other"]},
        "insert": {"type": "boolean"},
        "kind": {"type": "string",
                 "enum": ["diagram", "image", "list_card", "none"]},
        "confidence": {"type": "integer", "minimum": 0, "maximum": 100},
        "reason": {"type": "string"},
        # Заполняется ТОЛЬКО когда kind=list_card: заголовок + 2-4 пункта,
        # дословно по смыслу кадра/реплики (кросс-тип image→list_card без SD).
        "card": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "items": {"type": "array", "items": {"type": "string"},
                          "minItems": 2, "maxItems": 4},
            },
        },
    },
    "required": ["scene", "insert", "kind", "reason"],
}

SYSTEM = (
    "Ты — арт-директор монтажа YouTube-ролика. Тебе дают несколько подряд идущих "
    "кадров одного момента видео, реплику спикера в этот момент и ВИЗУАЛ, который "
    "авто-монтажёр хочет наложить поверх кадра. Твоя работа — не допустить «слоп». "
    "Правила:\n"
    "1. Если на кадре крупный план говорящего человека (talking_head) и визуал не "
    "иллюстрирует ничего конкретного — insert=false.\n"
    "2. Если экран УЖЕ показывает нужный контент (демонстрация, слайд, код, "
    "вайтборд, терминал) и наложение будет дублировать/загораживать его — "
    "insert=false.\n"
    "3. Если момент реально выиграет от визуала (называется конкретная сущность, "
    "которую он поясняет, а на экране её нет) — insert=true и выбери подходящий "
    "kind: diagram (схема/структура), image (фото/иллюстрация), list_card "
    "(список пунктов), либо none если ничего не подходит.\n"
    "4. Если выбрал kind=list_card — ОБЯЗАТЕЛЬНО заполни поле card: title (ёмкий "
    "заголовок) и items (2-4 коротких пункта) СТРОГО по смыслу кадра и реплики "
    "(реальные слова/факты с экрана, не вода).\n"
    "Отвечай СТРОГО одним JSON-объектом по схеме."
)


def _offsets(n: int) -> list[float]:
    """Seconds relative to the moment start to sample at. Symmetric-ish around
    the beginning of the reply so we catch the actual on-screen context."""
    n = max(1, min(5, int(n)))
    base = [0.5, -1.0, 2.0, 0.0, 3.5]
    return base[:n]


def _sample_frames(ffmpeg_bin: str, video: Path, t: float, n: int,
                   size: int, timeout: float) -> list[str]:
    """Extract up to ``n`` frames around ``t`` via ffmpeg → base64 PNGs (piped,
    no temp files). A failed grab is skipped, not fatal."""
    out: list[str] = []
    for off in _offsets(n):
        tt = max(0.0, float(t) + off)
        cmd = [ffmpeg_bin, "-nostdin", "-y", "-ss", f"{tt:.2f}", "-i", str(video),
               "-frames:v", "1", "-vf", f"scale={int(size)}:-2",
               "-f", "image2pipe", "-vcodec", "png", "pipe:1"]
        try:
            r = subprocess.run(cmd, capture_output=True, timeout=timeout)
        except (subprocess.SubprocessError, OSError):
            continue
        if r.returncode == 0 and r.stdout:
            out.append(base64.b64encode(r.stdout).decode("ascii"))
    return out


def _chosen_source(item) -> str:
    """``source`` of the currently selected visual (image items with candidates);
    "" otherwise."""
    pl = getattr(item, "payload", None)
    cands = getattr(pl, "candidates", None) if pl is not None else None
    sel = getattr(pl, "selected", 0) if pl is not None else 0
    if cands and 0 <= sel < len(cands):
        c = cands[sel]
        return str(c.get("source", "")) if isinstance(c, dict) else ""
    return ""


def _describe_visual(item) -> str:
    pl = getattr(item, "payload", None)
    t = getattr(item, "type", "?")
    if t == "image":
        src = _chosen_source(item) or getattr(pl, "asset_kind", "?")
        concept = (getattr(pl, "concept", "")
                   or getattr(pl, "image_query_en", ""))
        style = getattr(pl, "style_hint", "?")
        return f"картинка/схема (source={src}, style={style}): {concept}"
    if t == "list_card":
        title = getattr(pl, "title", "")
        items = getattr(pl, "items", []) or []
        texts = [getattr(i, "text", getattr(i, "get", lambda _k: "")("text")
                         if hasattr(i, "get") else "") for i in items[:4]]
        return f"карточка-список «{title}»: {texts}"
    if t == "animation":
        return f"анимированный значок (preset={getattr(pl, 'preset', '?')})"
    return str(t)


def _reroute_image(item, kind: str) -> Optional[str]:
    """Switch ``payload.selected`` to a candidate whose source matches the VLM's
    ``kind``, if one exists and it's not already selected. Returns the new source
    on a switch, else None. Never invents a candidate."""
    pl = getattr(item, "payload", None)
    cands = getattr(pl, "candidates", None) if pl is not None else None
    if not cands:
        return None
    want = _KIND_TO_SOURCES.get(kind, ())
    if not want:
        return None
    cur = getattr(pl, "selected", 0)
    if 0 <= cur < len(cands) and isinstance(cands[cur], dict) \
            and cands[cur].get("source") in want:
        return None                              # already the right kind
    for i, c in enumerate(cands):
        if isinstance(c, dict) and c.get("source") in want:
            pl.selected = i
            return str(c.get("source"))
    return None


def _valid_card(v: dict) -> Optional[tuple]:
    """Extract (title, bullets) from a verdict's ``card`` if it's usable for a
    cross-type image→list_card conversion (title + ≥2 non-empty bullets)."""
    card = v.get("card")
    if not isinstance(card, dict):
        return None
    title = str(card.get("title", "")).strip()
    bullets = [str(x).strip() for x in (card.get("items") or [])
               if isinstance(x, (str, int, float)) and str(x).strip()]
    if title and len(bullets) >= 2:
        return title, bullets[:4]
    return None


def _apply(item, v: dict, *, reroute: bool, min_conf: int,
           card_factory=None) -> str:
    """Apply one verdict to one item in place. Returns a short outcome tag:
    keep / veto / convert:list_card / reroute:<src> / low_conf / noop.

    ``card_factory(item, title, bullets) -> bool`` (optional) performs the
    enrich-specific image→list_card mutation; kept as an injected callback so
    this module stays decoupled from the enrich payload types."""
    conf = v.get("confidence")
    if isinstance(conf, int) and conf < int(min_conf):
        return "low_conf"                        # not confident enough to act
    insert = v.get("insert")
    kind = str(v.get("kind", ""))
    reason = str(v.get("reason", ""))[:200]
    scene = str(v.get("scene", ""))
    if insert is False or (insert is True and kind == "none"):
        item.enabled = False
        item.status_note = f"vision: {scene} — {reason}"
        return "veto"
    if insert is True:
        if reroute and getattr(item, "type", "") == "image":
            # Кросс-тип: схема/картинка → текстовая карточка-список (без SD).
            if kind == "list_card" and card_factory is not None:
                card = _valid_card(v)
                if card and card_factory(item, card[0], card[1]):
                    item.status_note = f"vision→list_card: {scene}"
                    return "convert:list_card"
            newsrc = _reroute_image(item, kind)  # switch among existing candidates
            if newsrc:
                item.status_note = f"vision→{kind}: {scene}"
                return f"reroute:{newsrc}"
        return "keep"
    return "noop"                                # malformed verdict — leave it


def route(items: list, video, ffmpeg_bin: str, client, cfg,
          log: LogFn = _noop,
          on_progress: Optional[Callable[[int, int], None]] = None,
          card_factory=None) -> dict:
    """Judge each enabled visual candidate against real frames and apply the
    verdict in place. ``client`` is a ready OllamaClient bound to the vision
    model. ``cfg`` is a VisionRouteCfg. ``card_factory(item, title, bullets)``
    (optional) does the enrich-specific image→list_card mutation. Returns stats
    ``{judged, veto, convert, reroute, keep, skipped, error}``.

    Best-effort throughout: any per-item failure (no frames, transport error,
    parse miss) increments ``error`` and leaves the item untouched.
    """
    stats = {"judged": 0, "veto": 0, "convert": 0, "reroute": 0, "keep": 0,
             "skipped": 0, "error": 0}
    video = Path(video)
    targets = [it for it in items
               if getattr(it, "type", "") in VISUAL_TYPES
               and getattr(it, "enabled", True)]
    total = len(targets)
    if total == 0:
        return stats
    for idx, it in enumerate(targets):
        if on_progress is not None:
            try:
                on_progress(idx, total)
            except Exception:  # noqa: BLE001 — progress must never break the run
                pass
        imgs = _sample_frames(ffmpeg_bin, video, getattr(it, "t_start", 0.0),
                              cfg.frames, cfg.frame_size, cfg.timeout_s)
        if not imgs:
            stats["skipped"] += 1
            log(f"  vision: нет кадров для «{getattr(it, 'quote', '')[:40]}» — пропуск")
            continue
        quote = (getattr(it, "quote", "") or "").strip()
        user = (f"Реплика спикера: «{quote}»\n\n"
                f"Авто-монтажёр предлагает наложить: {_describe_visual(it)}\n\n"
                f"Кадры момента приложены. Верни JSON-вердикт.")
        try:
            v = client.chat_json(SYSTEM, user, VERDICT_SCHEMA,
                                 keep_alive="5m", images=imgs)
        except Exception as e:  # noqa: BLE001 — transport/parse — best-effort
            stats["error"] += 1
            log(f"  vision: вердикт не получен ({e}) — пропуск")
            continue
        stats["judged"] += 1
        outcome = _apply(it, v, reroute=bool(cfg.reroute),
                         min_conf=int(cfg.min_confidence),
                         card_factory=card_factory)
        if outcome == "veto":
            stats["veto"] += 1
        elif outcome.startswith("convert"):
            stats["convert"] += 1
        elif outcome.startswith("reroute"):
            stats["reroute"] += 1
        elif outcome == "keep":
            stats["keep"] += 1
    if on_progress is not None:
        try:
            on_progress(total, total)
        except Exception:  # noqa: BLE001
            pass
    log(f"  vision-route: судил {stats['judged']}, вето {stats['veto']}, "
        f"→карточка {stats['convert']}, пере-роут {stats['reroute']}, "
        f"оставил {stats['keep']}, без кадров {stats['skipped']}, "
        f"ошибок {stats['error']}")
    return stats
