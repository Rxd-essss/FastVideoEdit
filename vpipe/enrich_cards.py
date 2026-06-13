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

from .enrich import (CTA_WIDTH_FRAC, PIP_PAD_PX, CardPlan, CardPlanItem,
                     CtaTextPlan, card_tail_s)
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
PANEL_BG_BGR = "1A140D"      # тело панели: тёмно-тёплый near-black (BGR), §3
GLASS_BGR = "FFFFFF"         # верхняя кромка-блик панели (стекло)

# --- стиль A «панель» (§3, прототип D:\tmp\enrich2\cards\make_cards.py) ---------
# Все числа — кегли/геометрия @1080, масштабируются ×PlayResY/1080 (как scrim-путь).
# Панель — правая ~46% кадра, центрированная вертикально; слои строятся \p1-фигурами
# (drop-shadow / тело-frosted / glass-блик / акцент-полоса) + текст (Inter).
PANEL_W_1080 = 860           # ширина панели @1080
PANEL_H_1080 = 760           # высота панели @1080
PANEL_RIGHT_PAD_1080 = 90    # отступ панели от правого края @1080
PANEL_RADIUS_1080 = 36       # радиус скругления углов панели
PANEL_SHADOW_DX_1080 = 10    # смещение drop-shadow по X
PANEL_SHADOW_DY_1080 = 18    # смещение drop-shadow по Y
PANEL_SHADOW_BLUR = 18       # \blur drop-shadow
PANEL_ACCENT_W_1080 = 8      # ширина левой акцент-полосы у панели
PANEL_TITLE_DX_1080 = 64     # отступ заголовка от левого края тела панели
PANEL_TITLE_DY_1080 = 56     # отступ заголовка от верха тела панели
PANEL_UNDERLINE_DY_1080 = 78 # подчёркивание-свип под заголовком (от ty)
PANEL_UNDERLINE_W_1080 = 220 # ширина подчёркивания @1080
PANEL_UNDERLINE_H_1080 = 7   # толщина подчёркивания
PANEL_ITEMS_DY_1080 = 120    # верх первого пункта от заголовка (от ty)
PANEL_STEP_1080 = 112        # шаг пунктов @1080
PANEL_PILL_1080 = 56         # сторона номер-пилюли (скруглённый квадрат)
PANEL_PILL_R_1080 = 14       # радиус скругления пилюли
PANEL_PILL_GAP_1080 = 28     # зазор пилюля -> текст пункта
PANEL_ITEM_SLIDE_1080 = 14   # slide-up пункта (y+14 -> y), §3
# Альфы панели (ASS AA: 00=непрозрачно, FF=прозрачно).
PANEL_BODY_ALPHA = "2E"      # тело ~82% — размытое видео слегка просвечивает (стекло)
PANEL_SHADOW_ALPHA = "78"    # тень мягкая
PANEL_GLASS_ALPHA = "E0"     # блик еле виден
PANEL_ACCENT_ALPHA = "10"    # акцент-полоса яркая
PANEL_PILL_DIM_ALPHA = "80"  # пилюля прошлого пункта гаснет
PANEL_DIM_ALPHA = "30"       # текст прошлого пункта приглушается по альфе
# Анимации панели, мс (§3, R2).
PANEL_FAD = (280, 240)       # появление/уход панели
PANEL_TITLE_FAD = (300, 240)
PANEL_UNDER_FAD = (360, 240)
PANEL_UNDER_SWEEP = (360, 720)   # свип ширины подчёркивания (t0,t1 от события)
PANEL_PILL_FAD_IN = 180
PANEL_ITEM_FAD_IN = 220
PANEL_ITEM_MOVE_MS = 260
PANEL_DIM_MS = 220           # длительность затухания прошлого пункта
PANEL_STAGGER_FRAC = 0.55    # пункты влетают по первым 55% окна (§3)
PANEL_STAGGER_LEAD_S = 0.35  # задержка первого пункта от t0
PANEL_EASE_OUT = 0.6         # accel<1 в \t = ease-out (свип/рост)


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


def rrect(x: int, y: int, w: int, h: int, r: int) -> str:
    r"""ASS ``\p1`` drawing path for a rounded rectangle (кубик-безье углы).

    Углы строятся безье-сегментами с каппой ``0.5523`` (прототип R2
    ``make_cards.rrect``). ``r`` клампится в ``[0, min(w,h)//2]`` чтобы при
    маленькой фигуре (пилюля) скругление не вывернулось. Координаты — целые
    (libass рисует фигуры в PlayRes-пикселях).
    """
    w, h = max(0, int(w)), max(0, int(h))
    r = max(0, min(int(r), w // 2, h // 2))
    x, y = int(x), int(y)
    x2, y2 = x + w, y + h
    if r <= 0:                          # вырожденный радиус -> прямоугольник
        return f"m {x} {y} l {x2} {y} l {x2} {y2} l {x} {y2}"
    c = int(round(r * 0.5523))          # bezier kappa
    return (
        f"m {x + r} {y} "
        f"l {x2 - r} {y} b {x2 - r + c} {y} {x2} {y + r - c} {x2} {y + r} "
        f"l {x2} {y2 - r} b {x2} {y2 - r + c} {x2 - r + c} {y2} {x2 - r} {y2} "
        f"l {x + r} {y2} b {x + r - c} {y2} {x} {y2 - r + c} {x} {y2 - r} "
        f"l {x} {y + r} b {x} {y + r - c} {x + r - c} {y} {x + r} {y}"
    )


def _style_line(name: str, font: str, fs: int, *, outline: int, align: int,
                ml: int = 0, mr: int = 0, mv: int = 0) -> str:
    return (f"Style: {name},{font},{max(1, fs)},&H00FFFFFF,&H00FFFFFF,"
            f"&H00000000,&H64000000,0,0,0,0,100,100,0,0,1,{outline},0,"
            f"{align},{ml},{mr},{mv},1")


def _card_window(card: CardPlan, shown: Sequence[CardPlanItem]) -> tuple[float, float]:
    r"""Финальное окно карточки (t0,t1) — общий источник для ASS-времён и для
    blur-backplate в render (планировщик кладёт это же окно в card_windows).

    t1 = ``card.t1`` (план) ИЛИ floor по формуле дочитывания (``card_tail_s``),
    что больше — ровно как в ASS-генераторе ниже (единый источник правды §2.2).
    """
    t0 = max(0.0, card.t0)
    t1 = max(card.t1,
             shown[-1].t + card_tail_s([it.text for it in shown], card.hold_s))
    return t0, t1


def card_windows_for(cards: Sequence[CardPlan]) -> list[tuple[float, float]]:
    r"""Окна (t0,t1) карточек для ``RenderEnrich.card_windows`` (blur-backplate).

    Дроп карточек без рисуемых пунктов (как ASS-генератор) — окно появляется
    ТОЛЬКО для реально нарисованной карточки, иначе blur повис бы в пустоте.
    Источник окна — :func:`_card_window` (тот же, что у ASS-событий), так blur
    и карточка совпадают по времени бит-в-бит.
    """
    out: list[tuple[float, float]] = []
    for card in cards:
        shown = _shown_items(card)
        if shown:
            out.append(_card_window(card, shown))
    return out


def _shown_items(card: CardPlan) -> list[CardPlanItem]:
    r"""Непустые пункты карточки в порядке времени, ≤``MAX_CARD_ITEMS`` (§2.2)."""
    items = sorted((it for it in card.items if (it.text or "").strip()),
                   key=lambda it: it.t)
    return items[:MAX_CARD_ITEMS]


def _stagger(t0: float, t1: float, n: int) -> list[float]:
    r"""Раскидать старты пунктов по первым ``PANEL_STAGGER_FRAC`` окна (§3).

    Один пункт -> чуть после t0; пустой -> []. Старты клампятся в окно
    карточки (как scrim-путь), чтобы пункт не стартовал позже своего конца.
    """
    if n <= 0:
        return []
    if n == 1:
        return [min(t0 + PANEL_STAGGER_LEAD_S, t1)]
    span = (t1 - t0) * PANEL_STAGGER_FRAC
    gap = span / n
    return [min(t0 + PANEL_STAGGER_LEAD_S + i * gap, t1) for i in range(n)]


def _resolve_mode(card: CardPlan, override: Optional[str]) -> str:
    r"""Режим рисовки карточки: panel (стиль A, дефолт V11/§3) | scrim (фолбэк).

    Источник по приоритету: ``style_overrides["mode"]`` (глобальный тумблер
    serve/гейта) -> ``card.mode`` (если поле появится в схеме v1.1) -> ДЕФОЛТ
    panel. Невалидное значение -> panel. ``CardPlan`` сейчас поля ``mode`` не
    несёт (схема V11 флипнула дефолт только в ``ListCardPayload``) — поэтому
    ``getattr`` с фолбэком, без жёсткой зависимости от схемы."""
    m = override if override in ("panel", "scrim") else None
    if m is None:
        cm = getattr(card, "mode", None)
        m = cm if cm in ("panel", "scrim") else "panel"
    return m


def _card_events(card: CardPlan, W: int, H: int, k: float, alpha_hex: str,
                 accent: str, dim: str, mode: str = "panel") -> list[str]:
    r"""События одной карточки — диспетчер по ``mode`` (§3).

    ``panel`` (стиль A, дефолт V11) -> :func:`_card_events_panel`; ``scrim``
    (фолбэк) -> :func:`_card_events_scrim` (исторический плоский скрим-список).
    """
    shown = _shown_items(card)
    if not shown:
        return []                       # карточка без пунктов не рисуется
    if mode == "scrim":
        return _card_events_scrim(card, W, H, k, alpha_hex, accent, dim)
    return _card_events_panel(card, W, H, k, accent, dim)


def _card_events_scrim(card: CardPlan, W: int, H: int, k: float, alpha_hex: str,
                       accent: str, dim: str) -> list[str]:
    """Стиль B-фолбэк «scrim»: слой 0 скрим / 1 заголовок / 2 пункты (плоский
    список — исторический путь, оставлен как fallback за тумблером mode=scrim)."""
    items = _shown_items(card)
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


def _card_events_panel(card: CardPlan, W: int, H: int, k: float,
                       accent: str, dim: str) -> list[str]:
    r"""Стиль A «панель» (§3, прототип R2 ``card_events_panel``) — дефолт V11.

    Слои (всё ``\p1``-фигуры + текст Inter, НЕ drawbox):
      0 — drop-shadow: смещённый ``\blur``-rrect (панель «отрывается» от фона);
      1 — тело панели: frosted rrect (``\1a&H2E&`` ~82% — размытое видео слегка
          просвечивает) + glass-блик по верхней кромке;
      2 — левая акцент-полоса панели (бренд-строка, ``ACCENT``);
      3 — заголовок (Inter SemiBold) + подчёркивание-свип (рост ширины ``\t``);
      4 — номер-пилюли (rrect с акцент-заливкой; активная яркая, прошлые гаснут);
      5 — текст пунктов (Inter Medium), staggered slide-up + ``\fad``; активный
          белый, прошлые в ``DIM`` (~69% белого — ЧИТАЕМО, не пропадает).

    Размытый видеофон-подложку под панелью добавляет render
    (``_enrich_video_chain``, blur-backplate) по тому же окну (``card_windows``)
    — здесь только ASS-слои поверх него.
    """
    items = _shown_items(card)
    if not items:
        return []
    n = len(items)
    t0, t1 = _card_window(card, items)
    a, b = _ass_ts(t0), _ass_ts(t1)
    starts = _stagger(t0, t1, n)

    # Геометрия панели — правая ~46% кадра, центрирована по вертикали (масштаб k).
    pw, ph = _px(PANEL_W_1080, k), _px(PANEL_H_1080, k)
    px = W - pw - _px(PANEL_RIGHT_PAD_1080, k)
    py = (H - ph) // 2
    r = _px(PANEL_RADIUS_1080, k)
    sdx, sdy = _px(PANEL_SHADOW_DX_1080, k), _px(PANEL_SHADOW_DY_1080, k)
    aw = max(1, _px(PANEL_ACCENT_W_1080, k))
    fad = f"\\fad({PANEL_FAD[0]},{PANEL_FAD[1]})"

    ev: list[str] = []
    # слой 0 — drop-shadow (смещён + \blur)
    ev.append(f"Dialogue: 0,{a},{b},CardShape,,0,0,0,,"
              f"{{\\an7\\pos(0,0)\\1c&H000000&\\1a&H{PANEL_SHADOW_ALPHA}&"
              f"\\blur{PANEL_SHADOW_BLUR}{fad}\\p1}}"
              f"{rrect(px + sdx, py + sdy, pw, ph, r)}{{\\p0}}")
    # слой 1 — тело панели (frosted) + glass-блик
    ev.append(f"Dialogue: 1,{a},{b},CardShape,,0,0,0,,"
              f"{{\\an7\\pos(0,0)\\1c&H{PANEL_BG_BGR}&\\1a&H{PANEL_BODY_ALPHA}&"
              f"{fad}\\p1}}{rrect(px, py, pw, ph, r)}{{\\p0}}")
    ev.append(f"Dialogue: 1,{a},{b},CardShape,,0,0,0,,"
              f"{{\\an7\\pos(0,0)\\1c&H{GLASS_BGR}&\\1a&H{PANEL_GLASS_ALPHA}&"
              f"{fad}\\p1}}{rrect(px + 2, py + 2, pw - 4, max(2, _px(4, k)), 2)}"
              f"{{\\p0}}")
    # слой 2 — левая акцент-полоса панели
    ev.append(f"Dialogue: 2,{a},{b},CardShape,,0,0,0,,"
              f"{{\\an7\\pos(0,0)\\1c&H{accent}&\\1a&H{PANEL_ACCENT_ALPHA}&"
              f"{fad}\\p1}}"
              f"m {px} {py + r} l {px + aw} {py + r} "
              f"l {px + aw} {py + ph - r} l {px} {py + ph - r}{{\\p0}}")
    # слой 3 — заголовок + подчёркивание-свип
    tx = px + _px(PANEL_TITLE_DX_1080, k)
    ty = py + _px(PANEL_TITLE_DY_1080, k)
    title = (card.title or "").strip()
    if title:
        ev.append(f"Dialogue: 3,{a},{b},CardTitle,,0,0,0,,"
                  f"{{\\an7\\pos({tx},{ty})"
                  f"\\fad({PANEL_TITLE_FAD[0]},{PANEL_TITLE_FAD[1]})}}"
                  f"{ass_escape(title)}")
        uy = ty + _px(PANEL_UNDERLINE_DY_1080, k)
        uw = _px(PANEL_UNDERLINE_W_1080, k)
        uh = max(1, _px(PANEL_UNDERLINE_H_1080, k))
        s0, s1 = PANEL_UNDER_SWEEP
        # \fscx10 -> 100 свипом слева-направо (ease-out): подчёркивание «растёт».
        ev.append(f"Dialogue: 3,{a},{b},CardShape,,0,0,0,,"
                  f"{{\\an7\\pos({tx},{uy})\\1c&H{accent}&\\1a&H08&"
                  f"\\fad({PANEL_UNDER_FAD[0]},{PANEL_UNDER_FAD[1]})"
                  f"\\t({s0},{s1},{PANEL_EASE_OUT},\\fscx100)\\fscx10\\p1}}"
                  f"m 0 0 l {uw} 0 l {uw} {uh} l 0 {uh}{{\\p0}}")
    # слои 4/5 — пункты: номер-пилюля + текст, staggered
    iy0 = ty + _px(PANEL_ITEMS_DY_1080, k)
    step = max(1, _px(PANEL_STEP_1080, k))
    pill = max(1, _px(PANEL_PILL_1080, k))
    pill_r = _px(PANEL_PILL_R_1080, k)
    pill_fs = max(1, int(round(pill * 0.64)))    # номер ~64% пилюли
    slide = max(1, _px(PANEL_ITEM_SLIDE_1080, k))
    gap = _px(PANEL_PILL_GAP_1080, k)
    for i, (it, ti) in enumerate(zip(items, starts)):
        y = iy0 + i * step
        sa = _ass_ts(ti)
        if i < n - 1:                   # гаснет при старте СЛЕДУЮЩЕГО пункта
            dt = max(0, int(round((starts[i + 1] - ti) * 1000.0)))
            d0, d1 = dt, dt + PANEL_DIM_MS
            dim_txt = f"\\t({d0},{d1},\\1c&H{dim}&\\1a&H{PANEL_DIM_ALPHA}&)"
            dim_pill = f"\\t({d0},{d1},\\1a&H{PANEL_PILL_DIM_ALPHA}&)"
        else:                           # последний активен до конца
            dim_txt = ""
            dim_pill = ""
        # слой 4 — пилюля (rrect, акцент-заливка; гаснет по альфе у прошлых)
        ev.append(f"Dialogue: 4,{sa},{b},CardShape,,0,0,0,,"
                  f"{{\\an7\\pos({tx},{y})\\1c&H{accent}&\\1a&H10&"
                  f"\\fad({PANEL_PILL_FAD_IN},0){dim_pill}\\p1}}"
                  f"{rrect(0, 0, pill, pill, pill_r)}{{\\p0}}")
        # слой 5 — номер по центру пилюли (тёмный текст на акценте)
        ev.append(f"Dialogue: 5,{sa},{b},CardItem,,0,0,0,,"
                  f"{{\\an5\\pos({tx + pill // 2},{y + pill // 2})"
                  f"\\fs{pill_fs}\\1c&H101010&"
                  f"\\fad({PANEL_PILL_FAD_IN},0){dim_pill}}}{i + 1}")
        # слой 5 — текст пункта: slide-up + fad, активный белый / прошлый DIM.
        # \pos НЕ ставим (\move первый позиционный тег — иначе slide-up убит);
        # \t ПОСЛЕ статического \1c (статический после \t перебил бы анимацию).
        txx = tx + pill + gap
        tyi = y + pill // 2
        ev.append(f"Dialogue: 5,{sa},{b},CardItem,,0,0,0,,"
                  f"{{\\an4\\move({txx},{tyi + slide},{txx},{tyi},0,"
                  f"{PANEL_ITEM_MOVE_MS})\\fad({PANEL_ITEM_FAD_IN},0)"
                  f"\\1c&HFFFFFF&{dim_txt}}}{ass_escape(it.text)}")
    return ev


def build_enrich_ass(cards: Sequence[CardPlan], ctas: Sequence[CtaTextPlan],
                     W: int, H: int,
                     style_overrides: Optional[dict] = None) -> str:
    """Собрать текст ``enrich_{base}.ass`` (PlayResX/Y = финальное разрешение).

    ``cards``/``ctas`` — ``RenderEnrich.cards``/``RenderEnrich.cta_texts``
    из планировщика: все времена ФИНАЛЬНЫЕ (после-concat), пункты карточек уже
    пережили ремап и дроп >50%-в-вырезах. ``style_overrides`` — точечные ручки
    для гейта: ``mode`` (``panel`` дефолт V11/§3 | ``scrim`` фолбэк — глобальный
    тумблер стиля карточек), ``scrim_opacity`` (55/60/65 — % непрозрачности
    скрима, только режим scrim), ``accent`` / ``dim`` (BGR-hex акцента и
    погасшего пункта).
    """
    W, H = int(W), int(H)
    if W <= 0 or H <= 0:
        raise ValueError(f"build_enrich_ass: некорректное разрешение {W}x{H}")
    k = H / 1080.0
    ov = style_overrides if isinstance(style_overrides, dict) else {}
    alpha_hex = scrim_alpha_hex(ov.get("scrim_opacity", SCRIM_OPACITY_DEF))
    accent = str(ov.get("accent", ACCENT_BGR))
    dim = str(ov.get("dim", DIM_BGR))
    mode_override = ov.get("mode")       # глобальный тумблер стиля (panel|scrim)

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
        mode = _resolve_mode(card, mode_override)
        lines.extend(_card_events(card, W, H, k, alpha_hex, accent, dim, mode))
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
