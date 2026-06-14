# -*- coding: utf-8 -*-
"""(P4, ОПЦИЯ) Анимация появления код-схемы через ASS-события.

ПРИОРИТЕТ — статичный PNG-оверлей (``render.render_schematic``): он работает и
вставляется рендером ролика как still. Это — необязательный слой «вау»: вместо
мгновенного появления карточки её ЛОГИЧЕСКИЕ блоки (строки stat / пункты list /
шаги process / ветки tree) проявляются ступенчато (staggered fade+slide-up),
синхронно с речью — приём из репо (``enrich_cards._card_events_panel``).

Этот хелпер НЕ рисует карточку сам (её рисует Chrome → PNG). Он лишь считает
ТАЙМИНГИ ступеней для уже отрисованного оверлея: на сколько блоков делить и в
какие моменты включать каждый — чтобы внешний аниматор (libass ``\\fad`` /
``\\move`` поверх PNG-слоёв, или N-кейфреймов→WebM) показал «сборку» схемы.

Возврат — список ``RevealStep`` (индекс блока + t_start/t_end в секундах). Это
данные, не файл: вызывающий (рендер ролика) превращает их в реальные события.
НЕ зависит от Chrome/ffmpeg, чистый расчёт — безопасно вызывать всегда.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from . import validate

# сколько ЛОГИЧЕСКИХ блоков у интента (для деления на ступени). 0/None → не
# анимируем поблочно (quote/callout/code — единый блок, проявляется целиком).
_BLOCK_KEY = {
    "stat": "stats",
    "list": "items",
    "process": "steps",
    "timeline": "events",
    "compare": "rows",
    "map": "points",
}

DEFAULT_STAGGER_S = 0.28          # шаг между блоками (приём enrich_cards)
DEFAULT_FADE_S = 0.22            # длительность fade-in одного блока
MIN_HOLD_S = 0.6                 # минимум, сколько схема видна после полной сборки


@dataclass(frozen=True)
class RevealStep:
    """Одна ступень анимации появления: блок ``index`` (0-based; -1 = head/title)
    проявляется в ``t_start`` и держится до ``t_end`` (общий конец оверлея)."""
    index: int
    t_start: float
    t_end: float


def block_count(intent: str, fields: dict) -> int:
    """Сколько ступеней-блоков даст интент с такими полями (0 — поблочно не
    анимируется). Считает по ВАЛИДИРОВАННОМУ payload (те же лимиты, что рендер —
    анимация не разойдётся с картинкой)."""
    payload = validate.build_payload(intent, fields if isinstance(fields, dict)
                                     else {}, validate.STYLE_DEFAULT)
    if not payload:
        return 0
    key = _BLOCK_KEY.get((intent or "").strip().lower())
    if not key:
        return 0
    seq = payload.get(key)
    return len(seq) if isinstance(seq, (list, tuple)) else 0


def reveal_steps(intent: str, fields: dict, *, t_start: float, t_end: float,
                 stagger_s: float = DEFAULT_STAGGER_S,
                 fade_s: float = DEFAULT_FADE_S,
                 reveal_head: bool = True) -> List[RevealStep]:
    """Рассчитать ступени появления для оверлея, живущего ``[t_start, t_end]``.

    * ``reveal_head`` — первой проявить шапку (eyebrow/title), index ``-1``.
    * блоки идут с шагом ``stagger_s``, но автоматически ужимаются, чтобы ВСЕ
      успели проявиться и осталось хотя бы ``MIN_HOLD_S`` показа целой схемы
      (не убегаем за ``t_end``).
    * интент без поблочной анимации (quote/callout/code) → одна ступень index 0.

    Возврат пуст, если окно невалидно (``t_end <= t_start``) или данных нет."""
    if not (t_end > t_start):
        return []
    n = block_count(intent, fields)
    if n <= 0:
        # единый блок: проявить целиком в начале окна
        payload = validate.build_payload(intent, fields if isinstance(fields, dict)
                                         else {}, validate.STYLE_DEFAULT)
        return [RevealStep(0, round(t_start, 3), round(t_end, 3))] if payload else []

    steps: List[RevealStep] = []
    idx0 = -1 if reveal_head else 0
    total = (n + (1 if reveal_head else 0))
    window = t_end - t_start
    # ужать шаг, чтобы (total-1) шагов + хвост MIN_HOLD влезли в окно
    budget = max(0.0, window - MIN_HOLD_S)
    if total > 1 and stagger_s * (total - 1) > budget:
        stagger_s = budget / (total - 1)
    fade_s = min(fade_s, max(0.05, stagger_s))

    cursor = t_start
    order = ([idx0] if reveal_head else []) + list(range(n))
    for i in order:
        steps.append(RevealStep(i, round(cursor, 3), round(t_end, 3)))
        cursor += stagger_s
    return steps


def fade_tag(step: RevealStep, fade_in_ms: int = int(DEFAULT_FADE_S * 1000)) -> str:
    """ASS override-тег появления блока (libass): задержка до ``step.t_start``
    делается через начало строки события (внешний аниматор ставит Start), а
    ``\\fad`` даёт мягкий fade-in. Утилита для P4-генератора событий."""
    fade_in_ms = max(0, int(fade_in_ms))
    return "{\\fad(%d,0)}" % fade_in_ms
