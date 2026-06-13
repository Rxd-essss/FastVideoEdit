"""Auto-enrichment plan (ENRICH_PLAN §1, §2.1, §7-P1): schema models + planner.

This module owns three things:

1. The on-disk schema of ``out/<stem>.enrich.json`` (§1.2) — dataclass models
   for the six suggestion types, sanitizing ``load_enrich``/``save_enrich``
   (atomic .tmp -> os.replace, version/hash/cutlist_rev), and
   ``compute_cutlist_rev`` (sha1 of the canonicalized enabled removes).
2. ``plan_render()`` — the deterministic planner that remaps the ORIGINAL-time
   plan onto the FINAL (post-cut) timeline via ``Timeline.remap_clamped`` and
   enforces every R5 anti-cringe limit IN CODE (qwen ignores numeric bans —
   proven twice in R4): the >50%-in-cut drop, window conflicts with the
   card > cta > image > animation priority, the >=2 s gap, the >=0.5 s seam
   offset, the clean head/tail zones and all density ceilings.
3. ``RenderEnrich`` — the render-ready product (§2.1) consumed by
   vpipe/render.py (P1.3) and vpipe/enrich_cards.py (the ASS builder): the
   planner additionally exposes ``cards``/``cta_texts`` so the serve layer can
   build ``enrich_{base}.ass`` in work_dir and fill ``cards_ass``.

The plan stays in ORIGINAL coordinates on disk (like cutlist/clips — the UI
lives there); only RenderEnrich carries FINAL (post-concat) seconds.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Optional, Union

from .models import CutList, Word
from .timeline import Timeline

# --- suggestion types / statuses (§1.2) --------------------------------------
ENR_IMAGE = "image"
ENR_ANIMATION = "animation"
ENR_LIST_CARD = "list_card"
ENR_CTA_SUBSCRIBE = "cta_subscribe"
ENR_CTA_LIKE = "cta_like"
ENR_CTA_COMMENT = "cta_comment"
ENR_TYPES = (ENR_IMAGE, ENR_ANIMATION, ENR_LIST_CARD,
             ENR_CTA_SUBSCRIBE, ENR_CTA_LIKE, ENR_CTA_COMMENT)

ST_OK = "ok"
ST_CONFLICT = "conflict"
ST_IN_CUT = "in_cut"
ST_OFF_LIMITS = "off_limits"
_STATUSES = (ST_OK, ST_CONFLICT, ST_IN_CUT, ST_OFF_LIMITS)

# --- planner limits — R5-дефолты, не выносить в промпт ------------------------
# (qwen3:8b игнорирует числовые запреты — доказано R4 дважды; промпт владеет
#  смыслом, КОД владеет всеми числами. Менять только через новое R5-ревью.)
OVERLAYS_PER_MIN = 2.5        # средняя плотность ВСЕХ enrich-окон (карточки тоже)
OVERLAYS_PER_MIN_HARD = 4.0   # абсолютный потолок (страховка для density v1.1)
IMAGES_PER_MIN = 2            # не больше 2 картинок в любом 60-с окне
IMAGES_WINDOW_S = 60.0
CTA_PER_10_MIN = 2            # не больше 2 CTA в любом 10-минутном окне
CTA_WINDOW_S = 600.0
CTA_MIN_T_FINAL = 60.0        # CTA не раньше 60-й секунды финального таймлайна
SCREEN_TIME_FRAC_MAX = 0.25   # суммарное экранное время оверлеев <=25% ролика
MIN_GAP_S = 2.0               # зазор между ЛЮБЫМИ окнами (один оверлей одновременно)
SEAM_GAP_S = 0.5              # края окна не ближе 0.5 c к шву выреза
CLEAN_HEAD_S = 30.0           # первые 30 c финального таймлайна — чистые
CLEAN_TAIL_S = 20.0           # последние 20 c — чистые
IN_CUT_DROP_FRAC = 0.5        # >50% окна в вырезах -> status in_cut (правило remap_words)
MIN_WINDOW_S = 1.0            # окно, схлопнувшееся у швов короче 1 c -> off_limits
MAX_STILLS = 6                # engine-лимит §2.1 п.5: PNG-входов на рендер
MAX_ANIMS = 3                 # engine-лимит §2.1 п.5: WebM-входов на рендер

# --- schema clamps — R5-дефолты, не выносить в промпт -------------------------
IMAGE_DUR_MIN, IMAGE_DUR_DEF, IMAGE_DUR_MAX = 2.5, 3.0, 4.0      # §3.3
ANIM_DUR_MIN, ANIM_DUR_DEF, ANIM_DUR_MAX = 1.5, 2.5, 4.0
CTA_SUB_DUR = (2.0, 4.0, 8.0)         # (min, default, max), §1.2 payload
CTA_LIKE_DUR = (2.0, 3.0, 8.0)
CTA_COMMENT_DUR = (3.0, 5.0, 10.0)
CARD_HOLD_MIN, CARD_HOLD_DEF, CARD_HOLD_MAX = 1.0, 1.2, 1.5
CARD_ITEMS_MAX = 6                    # жёсткий потолок пунктов (§1.2)
CARD_ITEM_TEXT_MAX = 60               # жёсткий лимит текста пункта (§1.2)
CARD_TITLE_MAX = 60
CTA_QUESTION_MAX = 120                # жёсткий лимит вопроса cta_comment (§1.2)
CARD_READ_CPS = 13.0                  # дочитывание: 13 симв/с (§2.2)
CARD_READ_FACTOR = 2.0                # x2 запас на дочитывание (§2.2)
CARD_TAIL_MAX_S = 6.0                 # дочитывание не растягивает карточку дольше
CARD_LEAD_S = 0.3                     # t0 карточки = remap(intro) - 0.3 c (§2.2)
IMG_WIDTH_MIN, IMG_WIDTH_DEF, IMG_WIDTH_MAX = 0.30, 0.32, 0.34   # PiP 30-34% (R5)
ANIM_WIDTH_MIN, ANIM_WIDTH_DEF, ANIM_WIDTH_MAX = 0.14, 0.18, 0.22
FADE_MS_MAX = 1000
CTA_WIDTH_FRAC = 220.0 / 1920.0       # ~220 px @1080p (§2.3)
PIP_PAD_PX = 48                       # отступ PiP от углов @1080p (R2 §1)
CTA_BOTTOM_PX = 160                   # CTA: overlay=48:H-h-160 @1080p (§2.3)
_FALLBACK_WORD_S = 0.3                # ширина "слова" при невалидном word_idx

# Приоритет конфликтов: card > cta > image > animation (R5/§9).
_PRIO = {ENR_LIST_CARD: 3, ENR_CTA_SUBSCRIBE: 2, ENR_CTA_LIKE: 2,
         ENR_CTA_COMMENT: 2, ENR_IMAGE: 1, ENR_ANIMATION: 0}

# --- asset locations (§2.3, §4) -----------------------------------------------
_DATA_DIR = Path(__file__).resolve().parent / "data" / "enrich"
CTA_ASSET_DIR = _DATA_DIR / "cta"
EMOJI_SVG_DIR = _DATA_DIR / "emoji" / "noto"
FONTS_DIR = _DATA_DIR / "fonts"
EMOJI_CACHE_DIR = Path("cache") / "enrich_emoji"
EMOJI_PNG_SIZE = 256
# Имя файла CTA-пака по варианту (§2.3); никакого play-логотипа и слова YouTube.
_CTA_FILES = {"sub_like": "subscribe_like.webm", "like": "like.webm",
              "comment": "comment.webm", "bell": "bell.webm"}

_POS_XY = {"top_right": ("W-w-{pad}", "{pad}"),
           "top_left": ("{pad}", "{pad}")}

LogFn = Callable[..., None]


def _noop(*_a, **_k) -> None:
    pass


# --- small numeric / text guards ----------------------------------------------
def _f(v, default: float = 0.0, lo: Optional[float] = None,
       hi: Optional[float] = None) -> float:
    """Float with NaN/Infinity/garbage guard and optional clamping."""
    try:
        x = float(v)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(x):
        return default
    if lo is not None:
        x = max(lo, x)
    if hi is not None:
        x = min(hi, x)
    return x


def _i(v, default: int = 0, lo: Optional[int] = None,
       hi: Optional[int] = None) -> int:
    if isinstance(v, bool):           # bool is int — отсекаем (паттерн _valid_int)
        return default
    try:
        x = int(v)
    except (TypeError, ValueError):
        return default
    if lo is not None:
        x = max(lo, x)
    if hi is not None:
        x = min(hi, x)
    return x


def _s(v, default: str = "") -> str:
    return v.strip() if isinstance(v, str) else default


def _trim_text(v, limit: int) -> str:
    """Collapse whitespace and hard-trim to ``limit`` chars at a word boundary."""
    s = " ".join(_s(v).split())
    if len(s) <= limit:
        return s
    cut = s[:limit + 1]
    sp = cut.rfind(" ")
    return (cut[:sp] if sp > 0 else s[:limit]).rstrip()


def _enum(v, allowed: tuple, default: str) -> str:
    return v if v in allowed else default


def _abs_path(v) -> str:
    """Только абсолютный путь к ассету (path-traversal guard, код-ревью P2):
    относительные строки вроде «../../config.yaml» из правленного руками
    enrich.json / тела /api/enrich/save не должны попадать во входы
    ffmpeg-графа. Пустая строка = «ассета нет» (рендер честно дропнет item
    со status_note «ассет не найден»)."""
    s = _s(v)
    return s if s and Path(s).is_absolute() else ""


# --- payload models (§1.2) -----------------------------------------------------
@dataclass
class ImagePayload:
    concept: str = ""
    image_query_en: str = ""
    style_hint: str = "photo"          # photo | diagram | icon
    asset_kind: str = "none"           # emoji | user | none
    asset_path: str = ""
    emoji: str = ""
    position: str = "top_right"        # top_right | top_left
    width_frac: float = IMG_WIDTH_DEF
    kenburns: bool = False
    fade_ms: int = 220

    def to_dict(self) -> dict:
        return {"concept": self.concept, "image_query_en": self.image_query_en,
                "style_hint": self.style_hint, "asset_kind": self.asset_kind,
                "asset_path": self.asset_path, "emoji": self.emoji,
                "position": self.position,
                "width_frac": round(self.width_frac, 3),
                "kenburns": self.kenburns, "fade_ms": self.fade_ms}

    @staticmethod
    def sanitize(d: dict) -> "ImagePayload":
        return ImagePayload(
            concept=_s(d.get("concept")),
            image_query_en=_s(d.get("image_query_en")),
            style_hint=_enum(d.get("style_hint"),
                             ("photo", "diagram", "icon"), "photo"),
            asset_kind=_enum(d.get("asset_kind"),
                             ("emoji", "user", "none"), "none"),
            asset_path=_abs_path(d.get("asset_path")),
            emoji=_s(d.get("emoji")),
            position=_enum(d.get("position"),
                           ("top_right", "top_left"), "top_right"),
            width_frac=_f(d.get("width_frac"), IMG_WIDTH_DEF,
                          IMG_WIDTH_MIN, IMG_WIDTH_MAX),
            kenburns=bool(d.get("kenburns", False)),
            fade_ms=_i(d.get("fade_ms"), 220, 0, FADE_MS_MAX))


@dataclass
class AnimationPayload:
    preset: str = "pop_in"             # pop_in | pulse (anim_presets.json)
    asset_kind: str = "none"
    asset_path: str = ""
    emoji: str = ""
    position: str = "top_right"
    width_frac: float = ANIM_WIDTH_DEF
    fade_ms: int = 180

    def to_dict(self) -> dict:
        return {"preset": self.preset, "asset_kind": self.asset_kind,
                "asset_path": self.asset_path, "emoji": self.emoji,
                "position": self.position,
                "width_frac": round(self.width_frac, 3),
                "fade_ms": self.fade_ms}

    @staticmethod
    def sanitize(d: dict) -> "AnimationPayload":
        return AnimationPayload(
            preset=_enum(d.get("preset"), ("pop_in", "pulse"), "pop_in"),
            asset_kind=_enum(d.get("asset_kind"),
                             ("emoji", "user", "none"), "none"),
            asset_path=_abs_path(d.get("asset_path")),
            emoji=_s(d.get("emoji")),
            position=_enum(d.get("position"),
                           ("top_right", "top_left"), "top_right"),
            width_frac=_f(d.get("width_frac"), ANIM_WIDTH_DEF,
                          ANIM_WIDTH_MIN, ANIM_WIDTH_MAX),
            fade_ms=_i(d.get("fade_ms"), 180, 0, FADE_MS_MAX))


@dataclass
class CardItem:
    text: str = ""
    word_idx: int = -1                 # ОРИГИНАЛЬНЫЙ индекс в all_words(); -1 = неизвестен
    t_word: float = 0.0                # ОРИГИНАЛЬНЫЕ секунды произнесения пункта

    def to_dict(self) -> dict:
        return {"text": self.text, "word_idx": self.word_idx,
                "t_word": round(self.t_word, 3)}


@dataclass
class ListCardPayload:
    title: str = ""
    mode: str = "scrim"                # scrim (дефолт A из R5) | panel (v1.1)
    items: list[CardItem] = field(default_factory=list)
    hold_s: float = CARD_HOLD_DEF

    def to_dict(self) -> dict:
        return {"title": self.title, "mode": self.mode,
                "items": [it.to_dict() for it in self.items],
                "hold_s": round(self.hold_s, 3)}

    @staticmethod
    def sanitize(d: dict) -> "ListCardPayload":
        raw = d.get("items") if isinstance(d.get("items"), list) else []
        items: list[CardItem] = []
        for r in raw:
            if not isinstance(r, dict):
                continue
            text = _trim_text(r.get("text"), CARD_ITEM_TEXT_MAX)
            t_word = r.get("t_word")
            if not text or not isinstance(t_word, (int, float)) \
                    or isinstance(t_word, bool) or not math.isfinite(float(t_word)):
                continue                       # пустой текст / битое время — дроп пункта
            items.append(CardItem(text=text,
                                  word_idx=_i(r.get("word_idx"), -1, -1),
                                  t_word=max(0.0, float(t_word))))
        items.sort(key=lambda it: it.t_word)
        return ListCardPayload(
            title=_trim_text(d.get("title"), CARD_TITLE_MAX),
            mode=_enum(d.get("mode"), ("scrim", "panel"), "scrim"),
            items=items[:CARD_ITEMS_MAX],      # items <=6 ЖЁСТКО (§1.2)
            hold_s=_f(d.get("hold_s"), CARD_HOLD_DEF,
                      CARD_HOLD_MIN, CARD_HOLD_MAX))


@dataclass
class CtaSubscribePayload:
    variant: str = "sub_like"          # один значок «подпишись + лайк» (R5)
    position: str = "bottom_left"
    duration_s: float = CTA_SUB_DUR[1]

    def to_dict(self) -> dict:
        return {"variant": self.variant, "position": self.position,
                "duration_s": round(self.duration_s, 3)}

    @staticmethod
    def sanitize(d: dict) -> "CtaSubscribePayload":
        lo, de, hi = CTA_SUB_DUR
        return CtaSubscribePayload(
            variant=_enum(d.get("variant"), ("sub_like",), "sub_like"),
            position="bottom_left",            # низ-лево, низ-право занят вотермаркой (R5)
            duration_s=_f(d.get("duration_s"), de, lo, hi))


@dataclass
class CtaLikePayload:
    position: str = "bottom_left"
    duration_s: float = CTA_LIKE_DUR[1]

    def to_dict(self) -> dict:
        return {"position": self.position, "duration_s": round(self.duration_s, 3)}

    @staticmethod
    def sanitize(d: dict) -> "CtaLikePayload":
        lo, de, hi = CTA_LIKE_DUR
        return CtaLikePayload(position="bottom_left",
                              duration_s=_f(d.get("duration_s"), de, lo, hi))


@dataclass
class CtaCommentPayload:
    question: str = ""                 # РЕДАКТИРУЕМОЕ поле (R4: качество — лотерея)
    position: str = "bottom_left"
    duration_s: float = CTA_COMMENT_DUR[1]

    def to_dict(self) -> dict:
        return {"question": self.question, "position": self.position,
                "duration_s": round(self.duration_s, 3)}

    @staticmethod
    def sanitize(d: dict) -> "CtaCommentPayload":
        lo, de, hi = CTA_COMMENT_DUR
        return CtaCommentPayload(
            question=_trim_text(d.get("question"), CTA_QUESTION_MAX),
            position="bottom_left",
            duration_s=_f(d.get("duration_s"), de, lo, hi))


Payload = Union[ImagePayload, AnimationPayload, ListCardPayload,
                CtaSubscribePayload, CtaLikePayload, CtaCommentPayload]

_PAYLOAD_SANITIZE = {
    ENR_IMAGE: ImagePayload.sanitize,
    ENR_ANIMATION: AnimationPayload.sanitize,
    ENR_LIST_CARD: ListCardPayload.sanitize,
    ENR_CTA_SUBSCRIBE: CtaSubscribePayload.sanitize,
    ENR_CTA_LIKE: CtaLikePayload.sanitize,
    ENR_CTA_COMMENT: CtaCommentPayload.sanitize,
}


# --- item / plan models (§1.2) ---------------------------------------------------
@dataclass
class EnrichItem:
    id: str
    type: str
    payload: Payload
    enabled: bool = True
    source: str = "llm"                # llm | user
    score: int = 50                    # 0..100, клампится кодом
    word_start: int = 0                # ОРИГИНАЛЬНЫЕ word-индексы
    word_end: int = 0
    t_start: float = 0.0               # ОРИГИНАЛЬНЫЕ секунды
    t_end: float = 0.0
    quote: str = ""
    reason: str = ""
    status: str = ST_OK                # ok | conflict | in_cut | off_limits
    status_note: str = ""
    edited: bool = False

    def to_dict(self) -> dict:
        return {"id": self.id, "type": self.type, "enabled": self.enabled,
                "source": self.source, "score": self.score,
                "word_start": self.word_start, "word_end": self.word_end,
                "t_start": round(self.t_start, 3), "t_end": round(self.t_end, 3),
                "quote": self.quote, "reason": self.reason,
                "status": self.status, "status_note": self.status_note,
                "edited": self.edited, "payload": self.payload.to_dict()}


def new_item_id() -> str:
    return f"enr_{uuid.uuid4().hex[:6]}"


def item_from_dict(d: dict, log: LogFn = _noop) -> Optional[EnrichItem]:
    """Sanitize one raw item dict. Unknown ``type`` -> None (skip, never raise)."""
    if not isinstance(d, dict):
        log("enrich: пропускаю не-словарь в items")
        return None
    t = d.get("type")
    if t not in ENR_TYPES:
        log(f"enrich: пропускаю незнакомый type={t!r} (id={d.get('id')!r})")
        return None
    payload = _PAYLOAD_SANITIZE[t](
        d.get("payload") if isinstance(d.get("payload"), dict) else {})
    word_start = _i(d.get("word_start"), 0, 0)
    word_end = max(word_start, _i(d.get("word_end"), word_start, 0))
    t_start = _f(d.get("t_start"), 0.0, 0.0)
    t_end = _f(d.get("t_end"), 0.0, 0.0)
    # Длительности окон — по типу (§1.2 / §3.3); ВСЁ числовое клампит код.
    if t == ENR_IMAGE:
        dur = t_end - t_start
        dur = IMAGE_DUR_DEF if dur <= 0 else min(IMAGE_DUR_MAX,
                                                 max(IMAGE_DUR_MIN, dur))
        t_end = t_start + dur
    elif t == ENR_ANIMATION:
        dur = t_end - t_start
        dur = ANIM_DUR_DEF if dur <= 0 else min(ANIM_DUR_MAX,
                                                max(ANIM_DUR_MIN, dur))
        t_end = t_start + dur
    elif t in (ENR_CTA_SUBSCRIBE, ENR_CTA_LIKE, ENR_CTA_COMMENT):
        t_end = t_start + payload.duration_s   # для cta_* окно = duration_s
    else:                                      # list_card: окно выводит планировщик
        t_end = max(t_end, t_start)
    return EnrichItem(
        id=_s(d.get("id")) or new_item_id(),
        type=t, payload=payload,
        enabled=bool(d.get("enabled", True)),
        source=_enum(d.get("source"), ("llm", "user"), "llm"),
        score=_i(d.get("score"), 0, 0, 100),
        word_start=word_start, word_end=word_end,
        t_start=t_start, t_end=t_end,
        quote=_s(d.get("quote")), reason=_s(d.get("reason")),
        status=_enum(d.get("status"), _STATUSES, ST_OK),
        status_note=_s(d.get("status_note")),
        edited=bool(d.get("edited", False)))


def default_params() -> dict:
    return {"density": "normal",
            "types": {"image": True, "animation": True,
                      "list_card": True, "cta": True},
            "image_source": "auto"}


def sanitize_params(d) -> dict:
    """Whitelist-sanitize the ``params`` block (строгий паттерн B5)."""
    d = d if isinstance(d, dict) else {}
    out = default_params()
    out["density"] = _enum(d.get("density"),
                           ("min", "normal", "aggressive"), "normal")
    t = d.get("types") if isinstance(d.get("types"), dict) else {}
    out["types"] = {k: bool(t.get(k, True)) for k in out["types"]}
    out["image_source"] = _enum(d.get("image_source"),
                                ("auto", "emoji", "user_folder"), "auto")
    return out


@dataclass
class EnrichPlan:
    version: int = 1
    hash: str = ""                     # s.audio_hash — полная инвалидация
    cutlist_rev: str = ""              # sha1 enabled-вырезов — мягкий баннер
    generated_at: str = ""
    model: str = ""
    params: dict = field(default_factory=default_params)
    items: list[EnrichItem] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"version": self.version, "hash": self.hash,
                "cutlist_rev": self.cutlist_rev,
                "generated_at": self.generated_at, "model": self.model,
                "params": sanitize_params(self.params),
                "items": [it.to_dict() for it in self.items]}

    @staticmethod
    def from_dict(d: dict, log: LogFn = _noop) -> "EnrichPlan":
        raw = d.get("items") if isinstance(d.get("items"), list) else []
        items = [it for it in (item_from_dict(r, log) for r in raw)
                 if it is not None]
        return EnrichPlan(
            version=_i(d.get("version"), 1, 1),
            hash=_s(d.get("hash")), cutlist_rev=_s(d.get("cutlist_rev")),
            generated_at=_s(d.get("generated_at")), model=_s(d.get("model")),
            params=sanitize_params(d.get("params")), items=items)


def compute_cutlist_rev(cuts: Union[CutList, Iterable[tuple]]) -> str:
    """sha1 of the canonicalized ENABLED remove-intervals (§1.2 cutlist_rev).

    Canonical form: sorted ``[start, end]`` pairs rounded to 3 decimals —
    stable across segment reordering and unrelated (disabled/censor) edits.
    """
    if isinstance(cuts, CutList):
        iv = cuts.enabled_removes()
    else:
        iv = list(cuts)
    canon = sorted((round(float(a), 3), round(float(b), 3))
                   for a, b in iv if float(b) > float(a))
    blob = json.dumps(canon, separators=(",", ":"))
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()


def load_enrich(path: str | Path, log: LogFn = _noop) -> Optional[EnrichPlan]:
    """Read ``out/<stem>.enrich.json``. Missing/corrupt/wrong shape -> None
    (never raises). Unknown item types are skipped with a log line, not a
    failure — forward compatibility for plans written by a newer version."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 — missing / unreadable / bad JSON
        return None
    if not isinstance(data, dict) or not isinstance(data.get("items"), list):
        return None
    return EnrichPlan.from_dict(data, log)


def save_enrich(plan: EnrichPlan, path: str | Path) -> None:
    """Persist the plan atomically (.tmp -> os.replace). Raises on failure —
    the caller decides between best-effort (suggest cache) and strict 500
    (/api/enrich/save), как у clips."""
    if not plan.generated_at:
        plan.generated_at = datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ")
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(plan.to_dict(), ensure_ascii=False, indent=2),
                   encoding="utf-8")
    try:
        os.replace(tmp, p)
    except OSError:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise


# --- emoji asset (Tier 0 fallback, §4) -------------------------------------------
# Растеризатор эмодзи — РЕШЕНИЕ P5 (задокументировано в docstring emoji_png_path):
# в .venv этой машины НЕТ svg→png библиотеки (cairosvg/resvg-py проверены живьём —
# отсутствуют), а Pillow SVG не парсит. План §4 предусматривает ровно этот фолбэк:
# «рендер цветного глифа эмодзи системным шрифтом через Pillow ImageFont». Имя
# Noto («u26a1», «u1f1fa_u1f1f8») — это и есть кодпойнт(ы): разворачиваем обратно
# в символ и рисуем цветной глиф Segoe UI Emoji (COLR/CBDT, embedded_color=True) на
# прозрачном 256-px холсте с тёмной скруглённой мини-подложкой (§4). SVG-файл из
# vpipe/data/enrich/emoji/noto (его кладёт P5-АССЕТЫ) нам не требуется — глиф даёт
# шрифт; наличие .svg лишь подтверждает, что эмодзи из вендоренного сабсета.
_EMOJI_FONT_CANDIDATES = (
    Path(r"C:\Windows\Fonts\seguiemj.ttf"),     # Segoe UI Emoji (цветной, Win10/11)
)
EMOJI_BACKPLATE_RGBA = (17, 24, 39, 220)        # #111827 ~86% — тёмная мини-подложка
EMOJI_BACKPLATE_PAD = 10                        # отступ подложки от краёв 256-холста
EMOJI_BACKPLATE_RADIUS = 40                     # скругление подложки
EMOJI_GLYPH_FRAC = 0.74                         # кегль глифа ≈ 74% от 256 (≈190 pt)


def _emoji_to_char(name: str) -> str:
    """Имя Noto-svg -> символ(ы). «u26a1»->'⚡'; «u1f1fa_u1f1f8»->'🇺🇸'
    (ZWJ/флаги = несколько кодпойнтов через «_»). Битый сегмент -> '' (вызов
    отдаст None — рендер честно дропнет оверлей со status_note)."""
    out: list[str] = []
    for part in name.strip().lower().split("_"):
        hexv = part.lstrip("u")
        if not hexv:
            continue
        try:
            out.append(chr(int(hexv, 16)))
        except (ValueError, OverflowError):
            return ""
    return "".join(out)


def _emoji_font_path() -> Optional[Path]:
    for p in _EMOJI_FONT_CANDIDATES:
        if p.is_file():
            return p
    return None


def _rasterize_emoji(char: str, dst: Path) -> bool:
    """Цветной глиф ``char`` -> 256-px PNG с прозрачным фоном на тёмной
    мини-подложке (Pillow + Segoe UI Emoji). Пишем атомарно (.tmp -> replace),
    идемпотентно. True при успехе и НЕпустом PNG, иначе False (нет шрифта /
    глиф не отрисовался / ошибка Pillow)."""
    font_path = _emoji_font_path()
    if font_path is None:
        return False
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception:  # noqa: BLE001 — Pillow обязателен в проекте, но не валим
        return False
    size = EMOJI_PNG_SIZE
    try:
        font = ImageFont.truetype(str(font_path), int(size * EMOJI_GLYPH_FRAC))
        # глиф на собственном слое -> центрируем по его bbox (эмодзи в шрифте
        # не центрированы по метрикам) -> поверх тёмной подложки.
        glyph = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        ImageDraw.Draw(glyph).text((size // 2, size // 2), char, font=font,
                                   anchor="mm", embedded_color=True)
        bbox = glyph.getbbox()
        if not bbox:
            return False                # глиф пуст (нет такого эмодзи в шрифте)
        img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        ImageDraw.Draw(img).rounded_rectangle(
            [EMOJI_BACKPLATE_PAD, EMOJI_BACKPLATE_PAD,
             size - EMOJI_BACKPLATE_PAD, size - EMOJI_BACKPLATE_PAD],
            radius=EMOJI_BACKPLATE_RADIUS, fill=EMOJI_BACKPLATE_RGBA)
        gw, gh = bbox[2] - bbox[0], bbox[3] - bbox[1]
        ox = (size - gw) // 2 - bbox[0]
        oy = (size - gh) // 2 - bbox[1]
        img.alpha_composite(glyph, (ox, oy))
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(".png.tmp")
        img.save(tmp, "PNG")
        os.replace(tmp, dst)
    except Exception:  # noqa: BLE001 — любой сбой растеризации = «ассета нет»
        try:
            dst.with_suffix(".png.tmp").unlink()
        except OSError:
            pass
        return False
    return dst.is_file() and dst.stat().st_size > 0


def emoji_png_path(emoji: str, cache_dir: Optional[Path] = None) -> Optional[Path]:
    """Resolve a Noto emoji name (e.g. ``u26a1``) to a cached 256 px PNG.

    РАСТЕРИЗАЦИЯ (выбор P5, проверен живьём на этой машине). В .venv нет
    svg->png библиотеки (cairosvg/resvg-py отсутствуют — проверено), Pillow
    SVG не парсит. План §4 даёт ровно этот фолбэк: рендер ЦВЕТНОГО глифа
    эмодзи системным шрифтом через Pillow. Имя Noto — это кодпойнт(ы)
    (``u26a1`` = U+26A1, ``u1f1fa_u1f1f8`` = флаг из двух кодпойнтов); код
    разворачивает имя в символ и рисует цветной глиф Segoe UI Emoji
    (``C:\\Windows\\Fonts\\seguiemj.ttf``, COLR/CBDT, ``embedded_color=True``)
    на ПРОЗРАЧНОМ 256-px холсте с тёмной скруглённой мини-подложкой (§4:
    «на тёмной мини-подложке»). Вендоренный SVG-сабсет не требуется — глиф
    даёт шрифт.

    Кэш: ``cache/enrich_emoji/<имя>_256.png``, идемпотентно (есть файл —
    отдаём как есть; нет — растеризуем атомарно). Возврат — путь к НЕпустому
    PNG либо ``None`` (пустое имя / битый кодпойнт / нет шрифта / глиф пуст);
    на ``None`` рендер дропает оверлей со status_note.
    """
    emoji = _s(emoji)
    if not emoji:
        return None
    cache = Path(cache_dir) if cache_dir is not None else EMOJI_CACHE_DIR
    png = cache / f"{emoji}_{EMOJI_PNG_SIZE}.png"
    if png.is_file():
        return png                      # идемпотентно: уже растеризовано
    char = _emoji_to_char(emoji)
    if not char:
        return None                     # имя не разворачивается в кодпойнт(ы)
    return png if _rasterize_emoji(char, png) else None


# --- render-ready plan (§2.1) -----------------------------------------------------
@dataclass
class StillOverlay:
    path: str
    x_expr: str
    y_expr: str
    scale_w: int
    t0: float                          # ФИНАЛЬНЫЕ (после-concat) секунды
    t1: float
    fade_s: float = 0.22
    kenburns: bool = False


@dataclass
class AnimOverlay:
    path: str                          # WebM VP9 yuva420p; декод -c:v libvpx-vp9 ДО -i
    x_expr: str
    y_expr: str
    scale_w: int
    t0: float
    t1: float
    loop: bool = True                  # -stream_loop -1 => overlay c shortest=1


@dataclass
class CardPlanItem:
    text: str
    t: float                           # ФИНАЛЬНЫЕ секунды появления пункта


@dataclass
class CardPlan:
    item_id: str
    title: str
    t0: float
    t1: float
    items: list[CardPlanItem] = field(default_factory=list)
    hold_s: float = CARD_HOLD_DEF      # для floor-а t1 в ASS-генераторе (§2.2)


@dataclass
class CtaTextPlan:
    item_id: str
    text: str                          # вопрос cta_comment (событие CtaText в ASS)
    t0: float
    t1: float


@dataclass
class RenderEnrich:
    """Render-ready план (§2.1): render.py (P1.3) ест stills/anims/cards_ass/
    fonts_dir; ``cards``/``cta_texts`` — вход ASS-генератора enrich_cards.py,
    из которого serve-слой строит enrich_{base}.ass и заполняет cards_ass."""
    stills: list[StillOverlay] = field(default_factory=list)
    anims: list[AnimOverlay] = field(default_factory=list)
    cards_ass: Optional[str] = None
    fonts_dir: str = str(FONTS_DIR)
    cards: list[CardPlan] = field(default_factory=list)
    cta_texts: list[CtaTextPlan] = field(default_factory=list)


# --- planner internals --------------------------------------------------------
@dataclass
class _Cand:
    item: EnrichItem
    f0: float                          # final window
    f1: float
    prio: int
    card_items: list[CardPlanItem] = field(default_factory=list)


def _reject(it: EnrichItem, status: str, note: str) -> None:
    """Планировщик не удаляет — помечает и авто-выключает (юзер может вернуть)."""
    it.status = status
    it.status_note = note
    it.enabled = False


def _is_cta(t: str) -> bool:
    return t.startswith("cta_")


def _word_span(words: Optional[list[Word]], idx: int,
               t_fallback: float) -> tuple[float, float]:
    if words and 0 <= idx < len(words):
        w = words[idx]
        return (w.start, w.end)
    return (t_fallback, t_fallback + _FALLBACK_WORD_S)


def _span_in_cut(tl: Timeline, span: tuple[float, float]) -> bool:
    """>50% длительности в вырезах — правило remap_words (timeline.py:103-128)."""
    a, b = span
    dur = b - a
    if dur <= 0.0:
        return tl.inside(0.5 * (a + b))
    return tl.removed_overlap(a, b) > IN_CUT_DROP_FRAC * dur


def card_tail_s(item_texts: list[str], hold_s: float) -> float:
    """Хвост карточки после последнего пункта: max(hold_s, дочитывание
    13 симв/с x2, §2.2), но не дольше CARD_TAIL_MAX_S. Единственный источник
    формулы — её же использует ASS-генератор (P1.2)."""
    read = sum(len(t) for t in item_texts) / CARD_READ_CPS * CARD_READ_FACTOR
    return max(hold_s, min(CARD_TAIL_MAX_S, read))


def _seam_points(tl: Timeline) -> list[float]:
    """Финальные координаты швов вырезов (оба края выреза схлопываются в точку).
    Швы на самых краях ролика чистыми зонами и так накрыты — исключаем."""
    dur = tl.new_duration()
    pts = []
    for a, _b in tl.removed:
        p = tl.remap_clamped(a)
        if 1e-9 < p < dur - 1e-9:
            pts.append(p)
    return pts


def _nudge_from_seams(f0: float, f1: float,
                      seams: list[float]) -> tuple[float, float]:
    """Края окна не ближе SEAM_GAP_S к шву: подрезаем внутрь (никогда не
    расширяем — расширение могло бы залезть в чужой зазор). Окно, накрывающее
    шов целиком с запасом >=0.5 c с обеих сторон, легально (финальный таймлайн
    непрерывен — оверлей просто живёт через джамп-кат)."""
    for p in seams:
        if abs(f0 - p) < SEAM_GAP_S:
            f0 = p + SEAM_GAP_S
        if abs(f1 - p) < SEAM_GAP_S:
            f1 = p - SEAM_GAP_S
    return f0, f1


def _make_candidate(it: EnrichItem, tl: Timeline,
                    words: Optional[list[Word]]) -> Optional[_Cand]:
    """Ремап ОРИГИНАЛЬНЫЕ -> ФИНАЛЬНЫЕ + дроп-правило >50% в вырезах."""
    if it.type == ENR_LIST_CARD:
        return _card_candidate(it, tl, words)
    a, b = it.t_start, it.t_end
    dur = b - a
    if dur <= 0.0 or tl.removed_overlap(a, b) > IN_CUT_DROP_FRAC * dur:
        _reject(it, ST_IN_CUT, "окно на >50% попало в вырезы")
        return None
    f0 = tl.remap_clamped(a)
    f1 = tl.remap_clamped(b)
    if f1 - f0 <= 1e-9:
        _reject(it, ST_IN_CUT, "окно схлопнулось после вырезов")
        return None
    return _Cand(item=it, f0=f0, f1=f1, prio=_PRIO[it.type])


def _card_candidate(it: EnrichItem, tl: Timeline,
                    words: Optional[list[Word]]) -> Optional[_Cand]:
    """Карточка: ремап каждого t_word пункта; пункт >50% в вырезе — дроп;
    <2 выживших или intro в вырезе — вся карточка in_cut (§1.3)."""
    pl: ListCardPayload = it.payload  # type: ignore[assignment]
    if _span_in_cut(tl, _word_span(words, it.word_start, it.t_start)):
        _reject(it, ST_IN_CUT, "интро карточки в вырезе")
        return None
    surv: list[CardPlanItem] = []
    for ci in pl.items:
        if _span_in_cut(tl, _word_span(words, ci.word_idx, ci.t_word)):
            continue                                # пункт в вырезе — дроп
        surv.append(CardPlanItem(text=ci.text, t=tl.remap_clamped(ci.t_word)))
    if len(surv) < 2:
        _reject(it, ST_IN_CUT,
                "после вырезов выжило меньше 2 пунктов — карточка выключена")
        return None
    surv.sort(key=lambda c: c.t)
    t0 = max(0.0, tl.remap_clamped(it.t_start) - CARD_LEAD_S)
    t1 = surv[-1].t + card_tail_s([c.text for c in surv], pl.hold_s)
    return _Cand(item=it, f0=t0, f1=t1, prio=_PRIO[ENR_LIST_CARD],
                 card_items=surv)


def _pip_xy(position: str, H: int) -> tuple[str, str]:
    pad = max(1, round(PIP_PAD_PX * H / 1080.0))
    xt, yt = _POS_XY.get(position, _POS_XY["top_right"])
    return xt.format(pad=pad), yt.format(pad=pad)


def _cta_xy(H: int) -> tuple[str, str]:
    pad = max(1, round(PIP_PAD_PX * H / 1080.0))
    off = max(1, round(CTA_BOTTOM_PX * H / 1080.0))
    return str(pad), f"H-h-{off}"


def plan_render(plan: EnrichPlan, timeline: Timeline,
                words: Optional[list[Word]], cfg, W: int, H: int,
                *, log: Optional[LogFn] = None) -> RenderEnrich:
    """Превратить план (ОРИГИНАЛЬНЫЕ координаты) в RenderEnrich (ФИНАЛЬНЫЕ).

    Мутирует ``plan.items``: статусы/notes/auto-disable — то, что увидит UI.
    Выключенные юзером items не трогаются и место не занимают. ``cfg`` принят
    для совместимости с §2.1 (density-пресеты придут в P2/P3) — лимиты v1
    ЖЁСТКО зашиты константами модуля (R5-дефолты, не выносить в промпт).
    ``cards_ass`` остаётся None: ASS строит serve-слой через enrich_cards.py
    (P1.2) из ``cards``/``cta_texts`` и сам заполняет путь.
    """
    lg = log or _noop
    tl = timeline
    dur = tl.new_duration()
    seams = _seam_points(tl)

    # 1) ремап + >50%-дроп ----------------------------------------------------
    cands: list[_Cand] = []
    for it in plan.items:
        if not it.enabled:
            continue                   # выключенное юзером не трогаем
        if it.type not in _PRIO:
            continue
        c = _make_candidate(it, tl, words)
        if c is not None:
            cands.append(c)

    # 2) чистые зоны / CTA>=60 c / отступ от швов ------------------------------
    placed: list[_Cand] = []
    for c in cands:
        it = c.item
        if c.f0 < CLEAN_HEAD_S or c.f1 > dur - CLEAN_TAIL_S:
            _reject(it, ST_OFF_LIMITS,
                    f"чистая зона: первые {CLEAN_HEAD_S:.0f} c и последние "
                    f"{CLEAN_TAIL_S:.0f} c без оверлеев")
            continue
        if _is_cta(it.type) and c.f0 < CTA_MIN_T_FINAL:
            _reject(it, ST_OFF_LIMITS,
                    f"CTA раньше {CTA_MIN_T_FINAL:.0f}-й секунды")
            continue
        f0, f1 = _nudge_from_seams(c.f0, c.f1, seams)
        if f1 - f0 < MIN_WINDOW_S:
            _reject(it, ST_OFF_LIMITS, "окно схлопнулось у шва выреза")
            continue
        c.f0, c.f1 = f0, f1
        placed.append(c)

    # 3) конфликты: окна не пересекаются НИКОГДА + зазор >=2 c ------------------
    #    приоритет card > cta > image > animation, внутри типа — score.
    order = sorted(placed, key=lambda c: (-c.prio, -c.item.score, c.f0,
                                          c.item.id))
    accepted: list[_Cand] = []
    for c in order:
        winner = next((a for a in accepted
                       if c.f0 < a.f1 + MIN_GAP_S and c.f1 > a.f0 - MIN_GAP_S),
                      None)
        if winner is not None:
            overlap = c.f0 < winner.f1 and c.f1 > winner.f0
            how = ("пересекается с" if overlap
                   else f"ближе {MIN_GAP_S:.0f} c к")
            _reject(c.item, ST_CONFLICT,
                    f"{how} {winner.item.type} {winner.item.id} — "
                    "выключено (приоритет)")
            continue
        accepted.append(c)

    # 4) потолки плотности ------------------------------------------------------
    # 4a. картинки: не больше 2 в любом 60-с окне (по score).
    keep_img: list[_Cand] = []
    for c in sorted((c for c in accepted if c.item.type == ENR_IMAGE),
                    key=lambda c: (-c.item.score, c.f0)):
        if sum(1 for k in keep_img
               if abs(k.f0 - c.f0) < IMAGES_WINDOW_S) >= IMAGES_PER_MIN:
            _reject(c.item, ST_OFF_LIMITS,
                    f"потолок: не больше {IMAGES_PER_MIN} картинок в минуту")
            accepted.remove(c)
        else:
            keep_img.append(c)
    # 4b. CTA: не больше 2 в любом 10-минутном окне (по score).
    keep_cta: list[_Cand] = []
    for c in sorted((c for c in accepted if _is_cta(c.item.type)),
                    key=lambda c: (-c.item.score, c.f0)):
        if sum(1 for k in keep_cta
               if abs(k.f0 - c.f0) < CTA_WINDOW_S) >= CTA_PER_10_MIN:
            _reject(c.item, ST_OFF_LIMITS,
                    f"потолок CTA: не больше {CTA_PER_10_MIN} на 10 минут")
            accepted.remove(c)
        else:
            keep_cta.append(c)
    # 4c. общая плотность: <=2.5 оверлея/мин (жёсткий потолок 4) — трим по score.
    budget = int(min(OVERLAYS_PER_MIN, OVERLAYS_PER_MIN_HARD) * dur / 60.0)
    if len(accepted) > budget:
        by_score = sorted(accepted, key=lambda c: (-c.item.score, c.f0,
                                                   c.item.id))
        for c in by_score[budget:]:
            _reject(c.item, ST_OFF_LIMITS,
                    f"потолок плотности: не больше {OVERLAYS_PER_MIN} "
                    "оверлеев в минуту")
        accepted = by_score[:budget]
    # 4d. экранное время: суммарно <=25% финального ролика — трим по score.
    cap = SCREEN_TIME_FRAC_MAX * dur
    by_score = sorted(accepted, key=lambda c: (-c.item.score, c.f0, c.item.id))
    total, kept = 0.0, []
    for c in by_score:
        if total + (c.f1 - c.f0) > cap + 1e-9:
            _reject(c.item, ST_OFF_LIMITS,
                    "потолок: экранное время оверлеев больше 25% ролика")
            continue
        total += c.f1 - c.f0
        kept.append(c)
    accepted = kept

    # принятые — честный ok (сбрасываем устаревшие пометки прошлых прогонов)
    for c in accepted:
        c.item.status = ST_OK
        c.item.status_note = ""

    # 5) выбор ассетов + сборка RenderEnrich ------------------------------------
    re_plan = RenderEnrich(fonts_dir=str(FONTS_DIR))
    stills: list[tuple[int, StillOverlay, EnrichItem]] = []  # (score, piece, owner)
    anims: list[tuple[int, AnimOverlay, EnrichItem]] = []
    for c in sorted(accepted, key=lambda c: c.f0):
        it = c.item
        if it.type == ENR_LIST_CARD:
            pl: ListCardPayload = it.payload  # type: ignore[assignment]
            re_plan.cards.append(CardPlan(item_id=it.id, title=pl.title,
                                          t0=c.f0, t1=c.f1,
                                          items=c.card_items,
                                          hold_s=pl.hold_s))
            continue
        if _is_cta(it.type):
            if it.type == ENR_CTA_COMMENT:
                # вопрос — событие CtaText в enrich.ass; живёт и без иконки
                re_plan.cta_texts.append(CtaTextPlan(item_id=it.id,
                                                     text=it.payload.question,
                                                     t0=c.f0, t1=c.f1))
            variant = (it.payload.variant if it.type == ENR_CTA_SUBSCRIBE
                       else ("comment" if it.type == ENR_CTA_COMMENT
                             else "like"))
            f = CTA_ASSET_DIR / _CTA_FILES.get(variant, "subscribe_like.webm")
            if not f.is_file():
                it.status_note = f"CTA-ассет не найден: {f.name}"
                lg(f"  enrich: {it.status_note} ({it.id})")
                continue                # дроп из рендера, item остаётся в плане
            x, y = _cta_xy(H)
            anims.append((it.score, AnimOverlay(
                path=str(f), x_expr=x, y_expr=y,
                scale_w=max(1, round(W * CTA_WIDTH_FRAC)),
                t0=c.f0, t1=c.f1, loop=True), it))
            continue
        # image / animation (PiP в верхнем углу)
        pl2 = it.payload
        path: Optional[str] = None
        if pl2.asset_kind == "user":
            if pl2.asset_path and Path(pl2.asset_path).is_file():
                path = pl2.asset_path
            else:
                it.status_note = f"ассет не найден: {pl2.asset_path or '(пусто)'}"
                lg(f"  enrich: {it.status_note} ({it.id})")
                continue                # дроп из рендера, item остаётся в плане
        elif pl2.asset_kind == "emoji":
            p = emoji_png_path(pl2.emoji, EMOJI_CACHE_DIR)
            if p is None:
                it.status_note = (f"эмодзи-ассет {pl2.emoji or '(пусто)'} "
                                  "не растеризовался (битый кодпойнт / нет "
                                  "шрифта эмодзи)")
                lg(f"  enrich: {it.status_note} ({it.id})")
                continue
            path = str(p)
        else:                           # asset_kind=none: предложение без ассета
            continue
        x, y = _pip_xy(pl2.position, H)
        scale_w = max(1, round(W * pl2.width_frac))
        if it.type == ENR_ANIMATION and path.lower().endswith(".webm"):
            anims.append((it.score, AnimOverlay(
                path=path, x_expr=x, y_expr=y, scale_w=scale_w,
                t0=c.f0, t1=c.f1, loop=(pl2.preset == "pulse")), it))
        else:
            # animation с png/emoji деградирует в still c fade — webm-пресеты
            # поверх эмодзи генерит P5; kenburns — только для type=image.
            stills.append((it.score, StillOverlay(
                path=path, x_expr=x, y_expr=y, scale_w=scale_w,
                t0=c.f0, t1=c.f1, fade_s=pl2.fade_ms / 1000.0,
                kenburns=bool(getattr(pl2, "kenburns", False))), it))

    # 6) engine-лимиты §2.1 п.5: <=6 still + <=3 anim — трим по score.
    #    Страховочный (рендерный) лимит: item остаётся enabled/ok, причина —
    #    в status_note + warning-лог (render.py дублирует этот трим).
    def _engine_trim(pieces, cap, what):
        if len(pieces) <= cap:
            return pieces
        pieces.sort(key=lambda sp: (-sp[0], sp[1].t0))
        for _score, piece, owner in pieces[cap:]:
            owner.status_note = (f"лимит движка: больше {cap} {what} "
                                 "на рендер — трим по score")
            lg(f"  enrich: {owner.status_note} ({owner.id}, "
               f"окно {piece.t0:.2f}-{piece.t1:.2f} c)")
        return pieces[:cap]

    stills = _engine_trim(stills, MAX_STILLS, "PNG-оверлеев")
    anims = _engine_trim(anims, MAX_ANIMS, "WebM-оверлеев")
    re_plan.stills = [p for _s2, p, _o in sorted(stills,
                                                 key=lambda sp: sp[1].t0)]
    re_plan.anims = [p for _s2, p, _o in sorted(anims,
                                                key=lambda sp: sp[1].t0)]
    return re_plan
