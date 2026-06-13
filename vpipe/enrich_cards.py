"""ASS-генератор карточек-перечислений и CTA-текста (ENRICH_PLAN §2.2).

``build_enrich_ass(cards, ctas, W, H)`` превращает render-ready план
(``RenderEnrich.cards`` / ``RenderEnrich.cta_texts`` из vpipe/enrich.py, все
времена УЖЕ ФИНАЛЬНЫЕ) в текст отдельного ASS-файла ``enrich_{base}.ass``.
Он идёт ПЕРВЫМ subtitles-фильтром, burn.ass — ПОСЛЕДНИМ (§2.2: скрим карточки
не должен притемнять караоке-сабы; karaoke-инвариант render.py не трогаем).

Слои одной карточки (вердикт R2 §4 — ASS, не drawbox/drawtext):

* слой 0 — скрим всего кадра ``{\\p1}``-прямоугольником, 60% непрозрачности
  (вариативно 55/65 для гейта G1 через ``style_overrides``), ``\\fad`` —
  затемнение плавно приходит и уходит (drawbox так не умеет);
* слой 1 — заголовок (события нет, если title пуст);
* слой 2 — пункты ОТДЕЛЬНЫМИ событиями: старт = финальное время произнесения
  слова (ремап сделал планировщик), конец = t1 карточки; появление
  ``\\fad(220,0)`` + slide-up ``\\move``; «мягкое караоке списком» — при старте
  следующего пункта предыдущий гаснет до ~70% белого через ``\\t`` с офсетами
  от начала СВОЕГО события.

Два сознательных отступления от буквы примера §2.2 (оба — потому что libass
исполняет первый позиционный/последний цветовой тег):

* в событии пункта НЕТ ``\\pos`` перед ``\\move`` — libass игнорирует второй
  позиционный тег (первый выигрывает), ``\\pos`` первым убил бы slide-up;
  конечная точка ``\\move`` и есть позиция пункта;
* ``\\t(...\\1c...)`` стоит ПОСЛЕ статических ``\\1c`` маркера/текста, а не до
  них — статический ``\\1c`` после ``\\t`` перезаписал бы анимацию, и пункт
  никогда бы не гас.

Шрифты — статические Inter (§2.4, libass не ест Variable-WOFF2): имена стилей
ровно «Inter SemiBold» / «Inter Medium», файлы придут в vpipe/data/enrich/fonts,
фильтру их отдаёт ``subtitles=...:fontsdir=`` (render.py, P1.3).
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Sequence

from .enrich import (CTA_WIDTH_FRAC, PIP_PAD_PX, CardPlan, CtaTextPlan,
                     card_tail_s)
from .subtitles import _ass_ts

# --- кегли @1080 (§2.2, R5) — масштаб ×PlayResY/1080 ---------------------------
FS_SHAPE_1080 = 20      # CardShape — носитель {\p1}-фигур, шрифт не рисует текст
FS_TITLE_1080 = 56      # CardTitle, Inter SemiBold
FS_ITEM_1080 = 44       # CardItem,  Inter Medium
FS_CTA_1080 = 40        # CtaText,   Inter Medium

# --- геометрия @1080 (§2.2) -----------------------------------------------------
CARD_X_1080 = 360            # левый край пунктов
CARD_Y0_1080 = 340           # верх первого пункта
CARD_STEP_1080 = 88          # шаг пунктов (44×1.35 + 28 ≈ 88)
CARD_TITLE_Y_1080 = 170      # \pos(W/2,170) заголовка (Alignment 8)
ITEM_LINE_H_1080 = 59        # высота строки пункта (44×1.35)
SLIDE_PX_1080 = 8            # slide-up: y+8 -> y в \move
MAX_CARD_ITEMS = 5           # ≤5 пунктов на карточке (§2.2)

# Верхняя кромка зоны burn-сабов @1080: низ кадра − margin_v 40 − 2 строки
# × (size 52 × ~1.2 межстрочного) ≈ 915 (дефолты AssStyleCfg: 52/40/bottom/2).
SUBS_TOP_1080 = 915
CARD_SUBS_CLEAR_1080 = 60    # низ блока пунктов не ниже кромки сабов − 60 px
CTA_GAP_OVER_SUBS_1080 = 70  # CtaText: 60–80 px над зоной сабов — середина
CTA_ICON_GAP_1080 = 24       # зазор между иконкой-облачком (§2.3) и текстом

# --- анимации, мс (§2.2) ----------------------------------------------------------
SCRIM_FAD = (250, 300)
TITLE_FAD = (250, 300)
ITEM_FAD_IN_MS = 220
ITEM_MOVE_MS = 220
ITEM_DIM_MS = 200            # длительность \t-затухания предыдущего пункта
CTA_FAD = (220, 220)

# --- цвета (ASS BGR) ---------------------------------------------------------------
ACCENT_BGR = "0B9EF5"        # маркер-номер: #f59e0b -> BGR (единство с UI/караоке)
DIM_BGR = "B8B8B8"           # «погасший» пункт — ~70% белого
SCRIM_OPACITY_DEF = 60.0     # % непрозрачности скрима; G1 рендерит 55/60/65


def ass_escape(text: str) -> str:
    r"""Явный escape ASS-текста (конвенция репо — subtitles._ass_text_escape).

    ``{`` открывает override-блок, ``\`` — вводит escape, поэтому оба
    нейтрализуем; буквальные переводы строк становятся жёстким ``\N``.
    """
    text = text if isinstance(text, str) else ""
    text = text.replace("\\", "\\\\")        # бэкслеши первыми
    text = text.replace("{", "\\{").replace("}", "\\}")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\n", "\\N")
    return text


def scrim_alpha_hex(opacity_pct: float) -> str:
    """% непрозрачности скрима -> ASS-альфа ``AA`` (00=непрозрачно, FF=прозрачно).

    60% -> ``66`` (как в §2.2), 55% -> ``73`` (R2-проба), 65% -> ``59``.
    """
    try:
        op = float(opacity_pct)
    except (TypeError, ValueError):
        op = SCRIM_OPACITY_DEF
    if not math.isfinite(op):
        op = SCRIM_OPACITY_DEF
    op = min(100.0, max(0.0, op))
    return f"{int(round(255.0 * (1.0 - op / 100.0))):02X}"


def _px(v: float, k: float) -> int:
    return int(round(v * k))


def _style_line(name: str, font: str, fs: int, *, outline: int, align: int,
                ml: int = 0, mr: int = 0, mv: int = 0) -> str:
    return (f"Style: {name},{font},{max(1, fs)},&H00FFFFFF,&H00FFFFFF,"
            f"&H00000000,&H64000000,0,0,0,0,100,100,0,0,1,{outline},0,"
            f"{align},{ml},{mr},{mv},1")


def _card_events(card: CardPlan, W: int, H: int, k: float, alpha_hex: str,
                 accent: str, dim: str) -> list[str]:
    """События одной карточки: слой 0 скрим / 1 заголовок / 2 пункты."""
    items = sorted((it for it in card.items if (it.text or "").strip()),
                   key=lambda it: it.t)
    if not items:
        return []                       # карточка без пунктов не рисуется

    # Геометрия: от y=340 шагом ~88 @1080; низ блока не ниже кромки сабов −60;
    # ≤5 пунктов (оба ограничения §2.2 — действует более строгое).
    x = _px(CARD_X_1080, k)
    y0 = _px(CARD_Y0_1080, k)
    step = max(1, _px(CARD_STEP_1080, k))
    line_h = _px(ITEM_LINE_H_1080, k)
    y_limit = _px(SUBS_TOP_1080 - CARD_SUBS_CLEAR_1080, k)
    n_geom = (y_limit - line_h - y0) // step + 1
    n = max(1, min(MAX_CARD_ITEMS, n_geom, len(items)))
    shown = items[:n]

    # t1 = последний пункт + max(hold_s, дочитывание Σсимв/13×2) — формула живёт
    # в enrich.card_tail_s (единственный источник); планировочный t1 — не меньше.
    t0 = max(0.0, card.t0)
    t1 = max(card.t1,
             shown[-1].t + card_tail_s([it.text for it in shown], card.hold_s))
    starts = [min(max(it.t, t0), t1) for it in shown]

    a, b = _ass_ts(t0), _ass_ts(t1)
    ev: list[str] = []
    # слой 0 — скрим всего кадра (\fad-дим бесплатен внутри burn — R2 §4)
    ev.append(f"Dialogue: 0,{a},{b},CardShape,,0,0,0,,"
              f"{{\\an7\\pos(0,0)\\p1\\1c&H000000&\\1a&H{alpha_hex}&"
              f"\\fad({SCRIM_FAD[0]},{SCRIM_FAD[1]})}}"
              f"m 0 0 l {W} 0 l {W} {H} l 0 {H}{{\\p0}}")
    # слой 1 — заголовок (если есть)
    title = (card.title or "").strip()
    if title:
        ev.append(f"Dialogue: 1,{a},{b},CardTitle,,0,0,0,,"
                  f"{{\\pos({W // 2},{_px(CARD_TITLE_Y_1080, k)})"
                  f"\\fad({TITLE_FAD[0]},{TITLE_FAD[1]})}}{ass_escape(title)}")
    # слой 2 — пункты отдельными событиями
    dy = max(1, _px(SLIDE_PX_1080, k))
    for i, (it, ti) in enumerate(zip(shown, starts)):
        y = y0 + i * step
        head = (f"{{\\an7\\move({x},{y + dy},{x},{y},0,{ITEM_MOVE_MS})"
                f"\\fad({ITEM_FAD_IN_MS},0)}}")
        if i < n - 1:                   # гаснет при старте СЛЕДУЮЩЕГО пункта
            dt = max(0, int(round((starts[i + 1] - ti) * 1000.0)))
            t_tag = f"\\t({dt},{dt + ITEM_DIM_MS},\\1c&H{dim}&)"
        else:                           # последний пункт активен до конца
            t_tag = ""
        marker = (f"{{\\1c&H{accent}&{t_tag}}}{i + 1}"
                  f"{{\\1c&HFFFFFF&{t_tag}}}")
        ev.append(f"Dialogue: 2,{_ass_ts(ti)},{b},CardItem,,0,0,0,,"
                  f"{head}{marker}  {ass_escape(it.text)}")
    return ev


def build_enrich_ass(cards: Sequence[CardPlan], ctas: Sequence[CtaTextPlan],
                     W: int, H: int,
                     style_overrides: Optional[dict] = None) -> str:
    """Собрать текст ``enrich_{base}.ass`` (PlayResX/Y = финальное разрешение).

    ``cards``/``ctas`` — ``RenderEnrich.cards``/``RenderEnrich.cta_texts``
    из планировщика: все времена ФИНАЛЬНЫЕ (после-concat), пункты карточек уже
    пережили ремап и дроп >50%-в-вырезах. ``style_overrides`` — точечные ручки
    для гейта G1: ``scrim_opacity`` (55/60/65 — % непрозрачности скрима),
    ``accent`` / ``dim`` (BGR-hex маркера и погасшего пункта).
    """
    W, H = int(W), int(H)
    if W <= 0 or H <= 0:
        raise ValueError(f"build_enrich_ass: некорректное разрешение {W}x{H}")
    k = H / 1080.0
    ov = style_overrides if isinstance(style_overrides, dict) else {}
    alpha_hex = scrim_alpha_hex(ov.get("scrim_opacity", SCRIM_OPACITY_DEF))
    accent = str(ov.get("accent", ACCENT_BGR))
    dim = str(ov.get("dim", DIM_BGR))

    # CtaText: низ-лево правее иконки-облачка (§2.3: x=48, ширина ~220 @1920),
    # низ текста — на CTA_GAP_OVER_SUBS px выше верхней кромки зоны сабов.
    cta_ml = (_px(PIP_PAD_PX, k) + int(round(W * CTA_WIDTH_FRAC))
              + _px(CTA_ICON_GAP_1080, k))
    cta_mv = _px(1080 - SUBS_TOP_1080 + CTA_GAP_OVER_SUBS_1080, k)

    lines = [
        "[Script Info]",
        "ScriptType: v4.00+",
        "Collisions: Normal",
        "WrapStyle: 0",
        "ScaledBorderAndShadow: yes",
        f"PlayResX: {W}",
        f"PlayResY: {H}",
        "",
        "[V4+ Styles]",
        ("Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
         "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
         "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
         "Alignment, MarginL, MarginR, MarginV, Encoding"),
        _style_line("CardShape", "Inter", _px(FS_SHAPE_1080, k),
                    outline=0, align=7),
        _style_line("CardTitle", "Inter SemiBold", _px(FS_TITLE_1080, k),
                    outline=0, align=8),
        _style_line("CardItem", "Inter Medium", _px(FS_ITEM_1080, k),
                    outline=0, align=7),
        # CtaText живёт поверх видео БЕЗ скрима — нужен контур для читаемости.
        _style_line("CtaText", "Inter Medium", _px(FS_CTA_1080, k),
                    outline=max(1, _px(2, k)), align=1,
                    ml=cta_ml, mr=_px(PIP_PAD_PX, k), mv=cta_mv),
        "",
        "[Events]",
        ("Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, "
         "Effect, Text"),
    ]

    for card in cards:
        lines.extend(_card_events(card, W, H, k, alpha_hex, accent, dim))
    for cta in ctas:
        text = ass_escape((cta.text or "").strip())
        if not text:
            continue                    # пустой вопрос — события нет
        lines.append(f"Dialogue: 0,{_ass_ts(max(0.0, cta.t0))},"
                     f"{_ass_ts(max(0.0, cta.t1))},CtaText,,0,0,0,,"
                     f"{{\\fad({CTA_FAD[0]},{CTA_FAD[1]})}}{text}")
    return "\n".join(lines) + "\n"


def write_enrich_ass(cards: Sequence[CardPlan], ctas: Sequence[CtaTextPlan],
                     W: int, H: int, path: str | Path,
                     style_overrides: Optional[dict] = None) -> Path:
    """Записать ASS в work_dir (UTF-8 c BOM — как write_ass: libass на Windows
    надёжнее всего читает кириллицу с BOM). Возвращает путь файла."""
    p = Path(path)
    p.write_text(build_enrich_ass(cards, ctas, W, H, style_overrides),
                 encoding="utf-8-sig")
    return p
