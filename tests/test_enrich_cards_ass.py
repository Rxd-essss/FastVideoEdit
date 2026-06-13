# -*- coding: utf-8 -*-
"""P1.2: ASS-генератор карточек/CTA-текста (ENRICH_PLAN §2.2, тесты §7-P1).

Покрытие: PlayRes = финальное разрешение, стили CardShape/CardTitle/CardItem/
CtaText (Inter Medium/SemiBold, кегли ×PlayResY/1080), скрим-слой 0 с
\\1a&H66&+\\fad на весь кадр (и варианты 55/65 для G1), события пунктов по
финальным word-временам с концом = t1 карточки, \\t-затухание предыдущего
пункта с офсетами от начала своего события, ≤5 пунктов, экранирование текста,
формула дочитывания 13 симв/с ×2, CtaText-событие (\\fad(220,220), пустой
вопрос — нет события), пустой title — нет события заголовка, масштаб на 720p.
"""
import pytest

from vpipe.enrich import CardPlan, CardPlanItem, CtaTextPlan, card_tail_s
from vpipe.enrich_cards import (ass_escape, build_enrich_ass, card_windows_for,
                                rrect, scrim_alpha_hex, write_enrich_ass)
from vpipe.subtitles import _ass_ts


# --- helpers --------------------------------------------------------------------
def make_card(**over):
    """Карточка как из планировщика: финальные времена, t1 уже с дочитыванием."""
    items = [CardPlanItem(text="Централизованный", t=100.0),
             CardPlanItem(text="Быстрый", t=104.1),
             CardPlanItem(text="Типизированный", t=109.3)]
    hold = 1.2
    d = {"item_id": "enr_card01", "title": "Чем хорош реестр",
         "t0": 99.7,
         "t1": items[-1].t + card_tail_s([i.text for i in items], hold),
         "items": items, "hold_s": hold}
    d.update(over)
    return CardPlan(**d)


def make_cta(**over):
    d = {"item_id": "enr_cta001",
         "text": "Какой дистрибутив выбрал и для каких целей?",
         "t0": 300.0, "t1": 305.0}
    d.update(over)
    return CtaTextPlan(**d)


def dialogues(ass, style=None):
    out = [l for l in ass.splitlines() if l.startswith("Dialogue:")]
    if style is not None:
        out = [l for l in out if l.split(",")[3] == style]
    return out


def layer(line):
    """Слой Dialogue-строки (целое после 'Dialogue: ')."""
    return int(line[len("Dialogue: "):].split(",", 1)[0])


# V11: дефолт карточки — стиль A «панель». scrim-путь живёт за тумблером.
SCRIM = {"mode": "scrim"}


def scrim_ass(cards, ctas, W, H, ov=None):
    o = dict(SCRIM)
    if ov:
        o.update(ov)
    return build_enrich_ass(cards, ctas, W, H, o)


def style_fields(ass, name):
    """Поля Style-строки по Format-порядку (0=Name ... 22=Encoding)."""
    for l in ass.splitlines():
        if l.startswith(f"Style: {name},"):
            return l[len("Style: "):].split(",")
    raise AssertionError(f"нет стиля {name}")


# --- PlayRes / стили ---------------------------------------------------------------
def test_playres_is_final_resolution():
    ass = build_enrich_ass([make_card()], [], 1920, 1080)
    assert "PlayResX: 1920" in ass and "PlayResY: 1080" in ass
    ass = build_enrich_ass([], [make_cta()], 1280, 720)
    assert "PlayResX: 1280" in ass and "PlayResY: 720" in ass


def test_styles_fonts_and_sizes_1080():
    ass = build_enrich_ass([make_card()], [make_cta()], 1920, 1080)
    shape = style_fields(ass, "CardShape")
    title = style_fields(ass, "CardTitle")
    item = style_fields(ass, "CardItem")
    cta = style_fields(ass, "CtaText")
    # шрифты §2.4 — статические Inter, имена дословно
    assert shape[1] == "Inter" and shape[2] == "20"
    assert title[1] == "Inter SemiBold" and title[2] == "56"
    assert item[1] == "Inter Medium" and item[2] == "44"
    assert cta[1] == "Inter Medium" and cta[2] == "40"
    # выравнивания §2.2: фигуры/пункты 7, заголовок 8, CTA-текст 1
    assert shape[18] == "7" and title[18] == "8"
    assert item[18] == "7" and cta[18] == "1"
    # CardShape/Title/Item без контура (скрим даёт контраст), CtaText — с контуром
    assert shape[16] == "0" and item[16] == "0" and int(cta[16]) >= 1


def test_styles_scale_to_720p():
    ass = build_enrich_ass([make_card()], [make_cta()], 1280, 720)
    assert style_fields(ass, "CardShape")[2] == "13"     # 20×720/1080
    assert style_fields(ass, "CardTitle")[2] == "37"     # 56×720/1080
    assert style_fields(ass, "CardItem")[2] == "29"      # 44×720/1080
    assert style_fields(ass, "CtaText")[2] == "27"       # 40×720/1080


def test_cta_style_margins_sit_over_subs_zone():
    cta = style_fields(build_enrich_ass([], [make_cta()], 1920, 1080),
                       "CtaText")
    # MarginL = 48 (pad) + 220 (иконка-облачко §2.3) + 24 (зазор) = 292
    assert int(cta[19]) == 292
    # MarginV = 1080−915 (кромка сабов) + 70 (60–80 px над зоной сабов) = 235
    assert int(cta[21]) == 235
    cta = style_fields(build_enrich_ass([], [make_cta()], 1280, 720),
                       "CtaText")
    assert int(cta[19]) == 32 + 147 + 16                 # масштаб 720p
    assert int(cta[21]) == 157


# --- скрим (слой 0, фолбэк mode=scrim) ----------------------------------------------
def test_scrim_layer0_full_frame_with_fad():
    card = make_card()
    ass = scrim_ass([card], [], 1920, 1080)
    scrims = [l for l in dialogues(ass, "CardShape") if layer(l) == 0]
    assert len(scrims) == 1
    s = scrims[0]
    assert s.startswith(f"Dialogue: 0,{_ass_ts(card.t0)},{_ass_ts(card.t1)},")
    assert "\\p1" in s and "{\\p0}" in s
    assert "\\1c&H000000&" in s and "\\1a&H66&" in s    # 60% по умолчанию
    assert "\\fad(250,300)" in s
    assert "m 0 0 l 1920 0 l 1920 1080 l 0 1080" in s   # ВЕСЬ кадр


def test_scrim_full_frame_scales_with_resolution():
    ass = scrim_ass([make_card()], [], 1280, 720)
    s0 = [l for l in dialogues(ass, "CardShape") if layer(l) == 0][0]
    assert "m 0 0 l 1280 0 l 1280 720 l 0 720" in s0


def test_scrim_opacity_variants_for_g1():
    card = [make_card()]
    assert "\\1a&H73&" in scrim_ass(card, [], 1920, 1080,
                                    {"scrim_opacity": 55})
    assert "\\1a&H66&" in scrim_ass(card, [], 1920, 1080,
                                    {"scrim_opacity": 60})
    assert "\\1a&H59&" in scrim_ass(card, [], 1920, 1080,
                                    {"scrim_opacity": 65})
    # хелпер: мусор -> дефолт 60%
    assert scrim_alpha_hex("мусор") == "66"
    assert scrim_alpha_hex(float("nan")) == "66"


# --- заголовок (слой 1, scrim-путь) ----------------------------------------------------
def test_title_event_layer1_centered():
    card = make_card()
    ass = scrim_ass([card], [], 1920, 1080)
    titles = dialogues(ass, "CardTitle")
    assert len(titles) == 1
    t = titles[0]
    assert t.startswith(f"Dialogue: 1,{_ass_ts(card.t0)},{_ass_ts(card.t1)},")
    assert "\\pos(960,170)" in t and "\\fad(250,300)" in t
    assert "Чем хорош реестр" in t
    # масштаб 720p: \pos(W/2, 170×720/1080)
    ass = scrim_ass([card], [], 1280, 720)
    assert "\\pos(640,113)" in dialogues(ass, "CardTitle")[0]


def test_empty_title_no_title_event():
    ass = scrim_ass([make_card(title="")], [], 1920, 1080)
    assert dialogues(ass, "CardTitle") == []
    # скрим (слой 0) и пункты на месте
    assert len([l for l in dialogues(ass, "CardShape") if layer(l) == 0]) == 1
    assert len(dialogues(ass, "CardItem")) == 3


# --- пункты (слой 2, scrim-путь) ----------------------------------------------------------
def test_item_events_start_at_word_times_end_at_card_t1():
    card = make_card()
    ass = scrim_ass([card], [], 1920, 1080)
    items = dialogues(ass, "CardItem")
    assert len(items) == 3                              # ОТДЕЛЬНЫМИ событиями
    end = _ass_ts(card.t1)
    for line, it in zip(items, card.items):
        f = line.split(",")
        assert f[0] == "Dialogue: 2"                    # слой 2
        assert f[1] == _ass_ts(it.t)                    # старт = время слова
        assert f[2] == end                              # конец = t1 карточки
        assert "\\fad(220,0)" in line


def test_item_slide_up_move_geometry():
    ass = scrim_ass([make_card()], [], 1920, 1080)
    items = dialogues(ass, "CardItem")
    # от y=340 шагом 88 @1080; slide-up y+8 -> y за 220 мс
    assert "\\move(360,348,360,340,0,220)" in items[0]
    assert "\\move(360,436,360,428,0,220)" in items[1]
    assert "\\move(360,524,360,516,0,220)" in items[2]
    # позиционный тег один — \pos рядом с \move убил бы slide-up (libass:
    # первый позиционный тег выигрывает)
    assert all("\\pos(" not in l for l in items)
    # масштаб 720p
    ass = scrim_ass([make_card()], [], 1280, 720)
    assert "\\move(240,232,240,227,0,220)" in dialogues(ass, "CardItem")[0]


def test_item_dim_transition_offsets_from_own_event():
    ass = scrim_ass([make_card()], [], 1920, 1080)
    items = dialogues(ass, "CardItem")
    # пункт 1 гаснет при старте пункта 2: dt = 104.1−100.0 = 4.1 c = 4100 мс
    assert "\\t(4100,4300,\\1c&HB8B8B8&)" in items[0]
    # пункт 2: dt = 109.3−104.1 = 5.2 c = 5200 мс
    assert "\\t(5200,5400,\\1c&HB8B8B8&)" in items[1]
    # последний пункт не гаснет до конца карточки
    assert "\\t(" not in items[2]
    # \t стоит ПОСЛЕ статических \1c (статический тег после \t перебил бы
    # анимацию) — и для маркера, и для текста пункта
    assert "{\\1c&H0B9EF5&\\t(4100,4300,\\1c&HB8B8B8&)}1" in items[0]
    assert "{\\1c&HFFFFFF&\\t(4100,4300,\\1c&HB8B8B8&)}" in items[0]


def test_item_marker_accent_bgr():
    ass = scrim_ass([make_card()], [], 1920, 1080)
    items = dialogues(ass, "CardItem")
    # маркер-номер акцентом #f59e0b -> BGR &H0B9EF5&, текст — белым
    assert "{\\1c&H0B9EF5&}3{\\1c&HFFFFFF&}  Типизированный" in items[2]
    for n, line in enumerate(items, start=1):
        assert "{\\1c&H0B9EF5&" in line                 # маркер — акцентом
        assert f"}}{n}{{\\1c&HFFFFFF&" in line          # номер, затем белый текст


def test_max_5_items_rendered():
    items = [CardPlanItem(text=f"Пункт номер {i}", t=100.0 + 2.0 * i)
             for i in range(6)]
    card = make_card(items=items,
                     t1=items[-1].t + card_tail_s([i.text for i in items], 1.2))
    ass = scrim_ass([card], [], 1920, 1080)
    lines = dialogues(ass, "CardItem")
    assert len(lines) == 5                              # ≤5 пунктов (§2.2)
    assert "Пункт номер 5" not in ass                   # шестой не отрисован


def test_item_start_clamped_into_card_window():
    card = make_card()
    card.items[0].t = 90.0                              # раньше t0=99.7
    ass = scrim_ass([card], [], 1920, 1080)
    assert dialogues(ass, "CardItem")[0].split(",")[1] == _ass_ts(card.t0)


def test_card_without_items_is_skipped():
    # ни в panel, ни в scrim карточка без пунктов не рисуется
    assert dialogues(build_enrich_ass([make_card(items=[])], [], 1920, 1080)) == []
    assert dialogues(scrim_ass([make_card(items=[])], [], 1920, 1080)) == []


# --- формула дочитывания (§2.2: max(hold, Σсимв/13×2)) -------------------------------------
def test_readout_formula_floor_when_t1_unset():
    # 26 символов / 13 симв/с × 2 = 4.0 c > hold 1.0 -> конец = 52.0 + 4.0.
    # Формула дочитывания едина для panel и scrim — проверяем оба.
    items = [CardPlanItem(text="abcdefghijklm", t=50.0),
             CardPlanItem(text="nopqrstuvwxyz", t=52.0)]
    card = make_card(items=items, t0=49.7, t1=0.0, hold_s=1.0, title="")
    for ass in (build_enrich_ass([card], [], 1920, 1080),
                scrim_ass([card], [], 1920, 1080)):
        for line in dialogues(ass):
            assert line.split(",")[2] == _ass_ts(56.0)


def test_readout_hold_dominates_short_items():
    # 6 символов -> дочитывание 0.92 c < hold 1.5 -> конец = 52.0 + 1.5
    items = [CardPlanItem(text="abc", t=50.0),
             CardPlanItem(text="def", t=52.0)]
    card = make_card(items=items, t0=49.7, t1=0.0, hold_s=1.5, title="")
    for ass in (build_enrich_ass([card], [], 1920, 1080),
                scrim_ass([card], [], 1920, 1080)):
        for line in dialogues(ass):
            assert line.split(",")[2] == _ass_ts(53.5)


def test_planner_t1_is_honored_when_larger():
    card = make_card(t1=120.0)                          # больше floor-а формулы
    for ass in (build_enrich_ass([card], [], 1920, 1080),
                scrim_ass([card], [], 1920, 1080)):
        for line in dialogues(ass):
            assert line.split(",")[2] == _ass_ts(120.0)


# --- CtaText ---------------------------------------------------------------------------------
def test_cta_comment_event():
    cta = make_cta()
    ass = build_enrich_ass([], [cta], 1920, 1080)
    lines = dialogues(ass, "CtaText")
    assert len(lines) == 1
    l = lines[0]
    assert l.startswith(f"Dialogue: 0,{_ass_ts(300.0)},{_ass_ts(305.0)},")
    assert "{\\fad(220,220)}" in l
    assert "Какой дистрибутив выбрал и для каких целей?" in l


def test_cta_empty_question_no_event():
    ass = build_enrich_ass([], [make_cta(text=""), make_cta(text="   ")],
                           1920, 1080)
    assert dialogues(ass) == []


# --- экранирование -----------------------------------------------------------------------------
def test_ass_escape_braces_backslashes_newlines():
    assert ass_escape("a{b}c\\d\ne") == "a\\{b\\}c\\\\d\\Ne"
    assert ass_escape("x\r\ny\rz") == "x\\Ny\\Nz"
    assert ass_escape("чисто") == "чисто"
    assert ass_escape(None) == ""                       # мусор -> пусто


def test_escaping_applied_to_card_and_cta_texts():
    card = make_card(title="Заголовок {смело}")
    card.items[0].text = "Пункт\\первый"
    cta = make_cta(text="Вопрос {спорный}\nвторая строка")
    ass = build_enrich_ass([card], [cta], 1920, 1080)
    assert "Заголовок \\{смело\\}" in ass
    assert "Пункт\\\\первый" in ass
    assert "Вопрос \\{спорный\\}\\Nвторая строка" in ass
    assert "{смело}" not in ass and "{спорный}" not in ass


# === стиль A «панель» (§3, дефолт V11) ==========================================
def panel_shapes(ass):
    return dialogues(ass, "CardShape")


def test_panel_is_default_mode():
    # Без override карточка рисуется ПАНЕЛЬЮ, не плоским скримом: нет full-frame
    # прямоугольника «m 0 0 l W 0 ...», есть frosted-тело и drop-shadow с \blur.
    ass = build_enrich_ass([make_card()], [], 1920, 1080)
    assert "m 0 0 l 1920 0 l 1920 1080 l 0 1080" not in ass   # НЕ плоский скрим
    shapes = panel_shapes(ass)
    # drop-shadow (слой 0, \blur), тело панели (слой 1, frosted \1a&H2E&)
    assert any(layer(s) == 0 and "\\blur18" in s for s in shapes)
    assert any(layer(s) == 1 and "\\1a&H2E&" in s and "\\1c&H1A140D&" in s
               for s in shapes)


def test_panel_layers_present_shadow_body_glass_accent():
    ass = build_enrich_ass([make_card()], [], 1920, 1080)
    s = panel_shapes(ass)
    # drop-shadow: чёрный, мягкая тень, \blur
    assert any("\\1c&H000000&" in x and "\\1a&H78&" in x and "\\blur18" in x
               for x in s)
    # тело панели: тёмно-тёплое near-black + ~82% непрозрачности (frosted)
    assert any("\\1c&H1A140D&" in x and "\\1a&H2E&" in x for x in s)
    # glass-блик: белая тонкая полоса по верхней кромке, еле видна
    assert any("\\1c&HFFFFFF&" in x and "\\1a&HE0&" in x for x in s)
    # левая акцент-полоса панели: акцент BGR, яркая
    assert any("\\1c&H0B9EF5&" in x and "\\1a&H10&" in x for x in s)


def test_panel_title_and_underline_sweep():
    ass = build_enrich_ass([make_card()], [], 1920, 1080)
    titles = dialogues(ass, "CardTitle")
    assert len(titles) == 1 and "Чем хорош реестр" in titles[0]
    assert "\\fad(300,240)" in titles[0]
    # подчёркивание-свип: акцент-rrect, рост ширины \fscx10 -> 100 через \t
    sweeps = [x for x in panel_shapes(ass)
              if "\\fscx10" in x and "\\t(360,720,0.6,\\fscx100)" in x]
    assert len(sweeps) == 1
    assert "\\1c&H0B9EF5&" in sweeps[0]


def test_panel_empty_title_no_title_no_underline():
    ass = build_enrich_ass([make_card(title="")], [], 1920, 1080)
    assert dialogues(ass, "CardTitle") == []
    assert not any("\\fscx10" in x for x in panel_shapes(ass))   # свипа нет
    # тело/тень/акцент панели и пункты на месте
    assert any("\\1a&H2E&" in x for x in panel_shapes(ass))
    assert len(dialogues(ass, "CardItem")) == 3 * 2     # пилюля+текст на пункт


def test_panel_number_pills_active_bright_past_dim():
    ass = build_enrich_ass([make_card()], [], 1920, 1080)
    # Пилюли — rrect-фигуры (CardShape) с акцент-заливкой; номера/текст — CardItem.
    pills = [x for x in panel_shapes(ass)
             if "\\1c&H0B9EF5&" in x and "\\fad(180,0)" in x]
    assert len(pills) == 3                              # по пилюле на пункт
    # активная пилюля (последняя) НЕ гаснет, прошлые гаснут по альфе \1a&H80&
    assert sum("\\t(" in p and "\\1a&H80&" in p for p in pills) == 2
    items = dialogues(ass, "CardItem")
    # номер по центру пилюли (\an5, тёмный текст на акценте) + текст пункта (\an4)
    nums = [x for x in items if "\\an5" in x and "\\1c&H101010&" in x]
    texts = [x for x in items if "\\an4" in x]
    assert len(nums) == 3 and len(texts) == 3
    assert nums[0].rstrip().endswith("}1")
    assert "Централизованный" in texts[0]


def test_panel_items_staggered_slide_up_no_pos():
    ass = build_enrich_ass([make_card()], [], 1920, 1080)
    texts = [x for x in dialogues(ass, "CardItem") if "\\an4" in x]
    # slide-up через \move (y+14 -> y), \fad — staggered появление
    assert all("\\move(" in x and ",0,260)" in x for x in texts)
    assert all("\\fad(220,0)" in x for x in texts)
    # \pos рядом с \move убил бы slide-up (первый позиционный тег выигрывает)
    assert all("\\pos(" not in x for x in texts)
    # активный пункт белый; прошлые гаснут в DIM (~69% белого) через \t
    assert sum("\\t(" in x and "\\1c&HB8B8B8&" in x for x in texts) == 2
    assert "\\t(" not in texts[-1]                      # последний не гаснет


def test_panel_geometry_anchored_right_and_scales():
    # Панель — правая часть кадра: её фигуры стартуют далеко правее центра.
    ass = build_enrich_ass([make_card()], [], 1920, 1080)
    body = [x for x in panel_shapes(ass) if "\\1a&H2E&" in x][0]
    # тело панели: rrect начинается с «m <x> ...», x = W-pw-pad = 1920-860-90=970
    import re as _re
    mx = int(_re.search(r"\}m (\d+) ", body).group(1))
    assert mx > 1920 // 2                               # правее центра кадра
    # 720p: всё масштабируется (панель уже)
    ass2 = build_enrich_ass([make_card()], [], 1280, 720)
    body2 = [x for x in panel_shapes(ass2) if "\\1a&H2E&" in x][0]
    mx2 = int(_re.search(r"\}m (\d+) ", body2).group(1))
    assert mx2 < mx                                     # меньше разрешение -> ближе


def test_panel_card_without_items_skipped():
    assert dialogues(build_enrich_ass([make_card(items=[])], [], 1920, 1080)) == []


def test_panel_max_5_items():
    items = [CardPlanItem(text=f"Пункт {i}", t=100.0 + 2.0 * i) for i in range(6)]
    card = make_card(items=items,
                     t1=items[-1].t + card_tail_s([i.text for i in items], 1.2))
    ass = build_enrich_ass([card], [], 1920, 1080)
    texts = [x for x in dialogues(ass, "CardItem") if "\\an4" in x]
    assert len(texts) == 5                              # ≤5 пунктов (§2.2)
    assert "Пункт 5" not in ass


# --- rrect-хелпер -------------------------------------------------------------------------------
def test_rrect_rounded_rectangle_path():
    p = rrect(0, 0, 100, 60, 10)
    assert p.startswith("m 10 0 ") and " b " in p       # безье-углы
    # вырожденный радиус -> прямой прямоугольник
    assert rrect(0, 0, 40, 40, 0) == "m 0 0 l 40 0 l 40 40 l 0 40"
    # радиус клампится в min(w,h)//2 (пилюля не выворачивается)
    assert " b " in rrect(0, 0, 20, 20, 999)


# --- card_windows_for (окна для blur-backplate в render) ---------------------------------------
def test_card_windows_match_ass_times():
    card = make_card()
    wins = card_windows_for([card])
    assert len(wins) == 1
    t0, t1 = wins[0]
    # окно совпадает с временами ASS-карточки (единый источник _card_window)
    ass = build_enrich_ass([card], [], 1920, 1080)
    body = [x for x in dialogues(ass, "CardShape") if "\\1a&H2E&" in x][0]
    f = body.split(",")
    assert f[1] == _ass_ts(t0) and f[2] == _ass_ts(t1)


def test_card_windows_skip_empty_cards():
    # карточка без пунктов не даёт окна (blur не повисает в пустоте)
    wins = card_windows_for([make_card(items=[]), make_card()])
    assert len(wins) == 1


# --- запись файла / краевые ---------------------------------------------------------------------
def test_write_enrich_ass_utf8_bom(tmp_path):
    p = write_enrich_ass([make_card()], [make_cta()], 1920, 1080,
                         tmp_path / "enrich_test.ass")
    raw = p.read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf")              # BOM — как write_ass
    assert "Чем хорош реестр" in raw.decode("utf-8-sig")


def test_empty_plan_valid_header_no_events():
    ass = build_enrich_ass([], [], 1920, 1080)
    assert "[Script Info]" in ass and "[Events]" in ass
    assert dialogues(ass) == []


def test_bad_resolution_raises():
    with pytest.raises(ValueError):
        build_enrich_ass([], [], 0, 1080)
