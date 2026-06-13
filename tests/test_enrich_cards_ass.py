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
from vpipe.enrich_cards import (ass_escape, build_enrich_ass, scrim_alpha_hex,
                                write_enrich_ass)
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


# --- скрим (слой 0) -----------------------------------------------------------------
def test_scrim_layer0_full_frame_with_fad():
    card = make_card()
    ass = build_enrich_ass([card], [], 1920, 1080)
    scrims = dialogues(ass, "CardShape")
    assert len(scrims) == 1
    s = scrims[0]
    assert s.startswith(f"Dialogue: 0,{_ass_ts(card.t0)},{_ass_ts(card.t1)},")
    assert "\\p1" in s and "{\\p0}" in s
    assert "\\1c&H000000&" in s and "\\1a&H66&" in s    # 60% по умолчанию
    assert "\\fad(250,300)" in s
    assert "m 0 0 l 1920 0 l 1920 1080 l 0 1080" in s   # ВЕСЬ кадр


def test_scrim_full_frame_scales_with_resolution():
    ass = build_enrich_ass([make_card()], [], 1280, 720)
    assert "m 0 0 l 1280 0 l 1280 720 l 0 720" in dialogues(ass, "CardShape")[0]


def test_scrim_opacity_variants_for_g1():
    card = [make_card()]
    assert "\\1a&H73&" in build_enrich_ass(card, [], 1920, 1080,
                                           {"scrim_opacity": 55})
    assert "\\1a&H66&" in build_enrich_ass(card, [], 1920, 1080,
                                           {"scrim_opacity": 60})
    assert "\\1a&H59&" in build_enrich_ass(card, [], 1920, 1080,
                                           {"scrim_opacity": 65})
    # хелпер: мусор -> дефолт 60%
    assert scrim_alpha_hex("мусор") == "66"
    assert scrim_alpha_hex(float("nan")) == "66"


# --- заголовок (слой 1) ----------------------------------------------------------------
def test_title_event_layer1_centered():
    card = make_card()
    ass = build_enrich_ass([card], [], 1920, 1080)
    titles = dialogues(ass, "CardTitle")
    assert len(titles) == 1
    t = titles[0]
    assert t.startswith(f"Dialogue: 1,{_ass_ts(card.t0)},{_ass_ts(card.t1)},")
    assert "\\pos(960,170)" in t and "\\fad(250,300)" in t
    assert "Чем хорош реестр" in t
    # масштаб 720p: \pos(W/2, 170×720/1080)
    ass = build_enrich_ass([card], [], 1280, 720)
    assert "\\pos(640,113)" in dialogues(ass, "CardTitle")[0]


def test_empty_title_no_title_event():
    ass = build_enrich_ass([make_card(title="")], [], 1920, 1080)
    assert dialogues(ass, "CardTitle") == []
    assert len(dialogues(ass, "CardShape")) == 1        # скрим и пункты на месте
    assert len(dialogues(ass, "CardItem")) == 3


# --- пункты (слой 2) ----------------------------------------------------------------------
def test_item_events_start_at_word_times_end_at_card_t1():
    card = make_card()
    ass = build_enrich_ass([card], [], 1920, 1080)
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
    ass = build_enrich_ass([make_card()], [], 1920, 1080)
    items = dialogues(ass, "CardItem")
    # от y=340 шагом 88 @1080; slide-up y+8 -> y за 220 мс
    assert "\\move(360,348,360,340,0,220)" in items[0]
    assert "\\move(360,436,360,428,0,220)" in items[1]
    assert "\\move(360,524,360,516,0,220)" in items[2]
    # позиционный тег один — \pos рядом с \move убил бы slide-up (libass:
    # первый позиционный тег выигрывает)
    assert all("\\pos(" not in l for l in items)
    # масштаб 720p
    ass = build_enrich_ass([make_card()], [], 1280, 720)
    assert "\\move(240,232,240,227,0,220)" in dialogues(ass, "CardItem")[0]


def test_item_dim_transition_offsets_from_own_event():
    ass = build_enrich_ass([make_card()], [], 1920, 1080)
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
    ass = build_enrich_ass([make_card()], [], 1920, 1080)
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
    ass = build_enrich_ass([card], [], 1920, 1080)
    lines = dialogues(ass, "CardItem")
    assert len(lines) == 5                              # ≤5 пунктов (§2.2)
    assert "Пункт номер 5" not in ass                   # шестой не отрисован


def test_item_start_clamped_into_card_window():
    card = make_card()
    card.items[0].t = 90.0                              # раньше t0=99.7
    ass = build_enrich_ass([card], [], 1920, 1080)
    assert dialogues(ass, "CardItem")[0].split(",")[1] == _ass_ts(card.t0)


def test_card_without_items_is_skipped():
    ass = build_enrich_ass([make_card(items=[])], [], 1920, 1080)
    assert dialogues(ass) == []


# --- формула дочитывания (§2.2: max(hold, Σсимв/13×2)) -------------------------------------
def test_readout_formula_floor_when_t1_unset():
    # 26 символов / 13 симв/с × 2 = 4.0 c > hold 1.0 -> конец = 52.0 + 4.0
    items = [CardPlanItem(text="abcdefghijklm", t=50.0),
             CardPlanItem(text="nopqrstuvwxyz", t=52.0)]
    card = make_card(items=items, t0=49.7, t1=0.0, hold_s=1.0, title="")
    ass = build_enrich_ass([card], [], 1920, 1080)
    for line in dialogues(ass):
        assert line.split(",")[2] == _ass_ts(56.0)


def test_readout_hold_dominates_short_items():
    # 6 символов -> дочитывание 0.92 c < hold 1.5 -> конец = 52.0 + 1.5
    items = [CardPlanItem(text="abc", t=50.0),
             CardPlanItem(text="def", t=52.0)]
    card = make_card(items=items, t0=49.7, t1=0.0, hold_s=1.5, title="")
    ass = build_enrich_ass([card], [], 1920, 1080)
    for line in dialogues(ass):
        assert line.split(",")[2] == _ass_ts(53.5)


def test_planner_t1_is_honored_when_larger():
    card = make_card(t1=120.0)                          # больше floor-а формулы
    ass = build_enrich_ass([card], [], 1920, 1080)
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
