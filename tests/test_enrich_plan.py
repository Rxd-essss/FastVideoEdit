# -*- coding: utf-8 -*-
"""P1: планировщик plan_render (ENRICH_PLAN §1.3, §2.1, §7-P1, §9).

Покрытие из §7-P1: ремап-кейсы у швов, >50%-дроп, карточка с пунктом в
вырезе, конфликт-приоритеты card>cta>image>animation, зазоры >=2 c, потолки
(2.5/мин, 2 image/мин, CTA 2/10мин, экранное время 25%), CTA<60 c, чистые
зоны (первые 30 / последние 20 c), engine-лимиты 6 still / 3 anim, выбор
ассетов (user-файл/эмодзи-кэш/CTA-webm) и статусы/notes в плане.

Timeline и words — синтетические (паттерн tests/test_timeline.py).
"""
import pytest

from vpipe import enrich
from vpipe.enrich import (AnimationPayload, CardItem, CtaCommentPayload,
                          CtaLikePayload, CtaSubscribePayload, EnrichItem,
                          EnrichPlan, ImagePayload, ListCardPayload,
                          plan_render)
from vpipe.models import Word
from vpipe.timeline import Timeline

W, H = 1920, 1080


# --- builders (прямое конструирование, как после item_from_dict) ----------------
def img(iid, t, score=70, dur=3.0, kind="none", path="", pos="top_right",
        wf=0.32, kenburns=False, emoji="", enabled=True):
    return EnrichItem(
        id=iid, type=enrich.ENR_IMAGE, score=score, enabled=enabled,
        t_start=t, t_end=t + dur,
        payload=ImagePayload(asset_kind=kind, asset_path=path, emoji=emoji,
                             position=pos, width_frac=wf, kenburns=kenburns))


def anim(iid, t, score=70, dur=3.0, kind="none", path="", preset="pulse"):
    return EnrichItem(
        id=iid, type=enrich.ENR_ANIMATION, score=score,
        t_start=t, t_end=t + dur,
        payload=AnimationPayload(preset=preset, asset_kind=kind,
                                 asset_path=path))


def card(iid, t, items, score=70, hold=1.2, word_start=-1, title=""):
    """items: список (text, word_idx, t_word)."""
    return EnrichItem(
        id=iid, type=enrich.ENR_LIST_CARD, score=score,
        t_start=t, t_end=0.0, word_start=word_start,
        payload=ListCardPayload(
            title=title, hold_s=hold,
            items=[CardItem(text=tx, word_idx=ix, t_word=tw)
                   for tx, ix, tw in items]))


def cta_sub(iid, t, score=70, dur=4.0):
    return EnrichItem(id=iid, type=enrich.ENR_CTA_SUBSCRIBE, score=score,
                      t_start=t, t_end=t + dur, payload=CtaSubscribePayload())


def cta_like(iid, t, score=70, dur=3.0):
    return EnrichItem(id=iid, type=enrich.ENR_CTA_LIKE, score=score,
                      t_start=t, t_end=t + dur, payload=CtaLikePayload())


def cta_comment(iid, t, score=70, dur=5.0, q="Какой дистрибутив выбрал?"):
    return EnrichItem(id=iid, type=enrich.ENR_CTA_COMMENT, score=score,
                      t_start=t, t_end=t + dur,
                      payload=CtaCommentPayload(question=q))


def run(items, tl, words=None):
    plan = EnrichPlan(items=items)
    return plan, plan_render(plan, tl, words, None, W, H)


@pytest.fixture
def png(tmp_path):
    p = tmp_path / "asset.png"
    p.write_bytes(b"\x89PNG fake")
    return str(p)


@pytest.fixture
def webm(tmp_path):
    p = tmp_path / "asset.webm"
    p.write_bytes(b"\x1aE\xdf\xa3 fake")
    return str(p)


@pytest.fixture
def cta_dir(tmp_path, monkeypatch):
    d = tmp_path / "cta"
    d.mkdir()
    monkeypatch.setattr(enrich, "CTA_ASSET_DIR", d)
    return d


def words_grid(n=900, step=0.5, dur=0.4):
    return [Word(f"w{i}", i * step, i * step + dur) for i in range(n)]


# --- ремап у швов (§1.3) ---------------------------------------------------------
def test_remap_window_snapped_to_seam_is_nudged(png):
    tl = Timeline([(50, 60)], duration=210)        # шов в финальной 50.0
    # окно [59.5..62.5]: 0.5 c из 3 в вырезе (16%) -> живёт; t0 прилипает к шву
    plan, re = run([img("a", 59.5, path=png, kind="user")], tl)
    assert len(re.stills) == 1
    st = re.stills[0]
    assert st.t0 == pytest.approx(50.0 + enrich.SEAM_GAP_S)   # 50.5, не 50.0
    assert st.t1 == pytest.approx(52.5)
    assert plan.items[0].status == enrich.ST_OK
    assert plan.items[0].enabled is True


def test_remap_window_ending_at_seam_trimmed_back(png):
    tl = Timeline([(50, 60)], duration=210)
    plan, re = run([img("b", 47.0, dur=2.8, path=png, kind="user")], tl)
    st = re.stills[0]
    assert st.t0 == pytest.approx(47.0)
    assert st.t1 == pytest.approx(50.0 - enrich.SEAM_GAP_S)   # 49.8 -> 49.5


def test_window_collapsed_at_seam_goes_off_limits(png):
    tl = Timeline([(50, 60)], duration=210)
    # [49.2..50.2]: 20% в вырезе -> живёт ремап, но после отступов от шва
    # остаётся 0.3 c < MIN_WINDOW_S -> off_limits
    plan, re = run([img("c", 49.2, dur=1.0, path=png, kind="user")], tl)
    assert re.stills == []
    assert plan.items[0].status == enrich.ST_OFF_LIMITS
    assert plan.items[0].enabled is False
    assert "шва" in plan.items[0].status_note


# --- >50%-дроп (правило remap_words) ----------------------------------------------
def test_in_cut_drop_rule(png):
    tl = Timeline([(50, 60)], duration=210)
    plan, re = run([
        img("full", 52.0, path=png, kind="user"),            # целиком в вырезе
        img("major", 49.0, path=png, kind="user"),           # 2/3 в вырезе
        img("half", 48.0, dur=4.0, path=png, kind="user"),   # ровно 50% — живёт
    ], tl)
    assert plan.items[0].status == enrich.ST_IN_CUT
    assert plan.items[1].status == enrich.ST_IN_CUT
    assert plan.items[0].enabled is False and plan.items[1].enabled is False
    assert ">50%" in plan.items[0].status_note
    assert plan.items[2].status == enrich.ST_OK
    assert len(re.stills) == 1
    assert re.stills[0].t0 == pytest.approx(48.0)
    assert re.stills[0].t1 == pytest.approx(49.5)             # шов 50 - 0.5


# --- карточки: пункт в вырезе, intro, <2 выживших (§1.3) ----------------------------
def test_card_item_inside_cut_dropped_card_survives():
    tl = Timeline([(100, 110)], duration=400)
    words = words_grid()
    c = card("crd", 90.0, [("Раз", 184, 92.0),
                           ("Два", 210, 105.0),     # слово 105.0-105.4 в вырезе
                           ("Три", 240, 120.0)], word_start=180)
    plan, re = run([c], tl, words)
    assert plan.items[0].status == enrich.ST_OK
    assert len(re.cards) == 1
    cp = re.cards[0]
    assert [i.text for i in cp.items] == ["Раз", "Три"]
    assert cp.items[1].t == pytest.approx(110.0)              # 120 - 10 выреза
    assert cp.t0 == pytest.approx(90.0 - enrich.CARD_LEAD_S)
    # хвост: max(hold 1.2, 6 симв /13*2 = 0.92) = 1.2
    assert cp.t1 == pytest.approx(110.0 + 1.2)


def test_card_with_less_than_two_survivors_in_cut():
    tl = Timeline([(100, 110)], duration=400)
    words = words_grid()
    c = card("crd2", 90.0, [("Раз", 184, 92.0),
                            ("Два", 204, 102.0),
                            ("Три", 210, 105.0)], word_start=180)
    plan, re = run([c], tl, words)
    assert plan.items[0].status == enrich.ST_IN_CUT
    assert plan.items[0].enabled is False
    assert "пунктов" in plan.items[0].status_note
    assert re.cards == []


def test_card_intro_inside_cut():
    tl = Timeline([(100, 110)], duration=400)
    words = words_grid()
    c = card("crd3", 105.0, [("Раз", 240, 120.0), ("Два", 250, 125.0)],
             word_start=210)                       # интро-слово 105.0 в вырезе
    plan, re = run([c], tl, words)
    assert plan.items[0].status == enrich.ST_IN_CUT
    assert "интро" in plan.items[0].status_note
    assert re.cards == []


# --- конфликты: приоритет card > cta > image > animation (§9) -----------------------
def test_conflict_card_beats_cta_despite_score():
    tl = Timeline([], duration=300)
    c = card("crd", 60.0, [("Раз", -1, 60.5), ("Два", -1, 61.0)], score=50)
    s = cta_sub("cta", 61.0, score=99)
    plan, re = run([c, s], tl)
    assert len(re.cards) == 1
    assert s.status == enrich.ST_CONFLICT and s.enabled is False
    assert "list_card" in s.status_note and "crd" in s.status_note
    assert "приоритет" in s.status_note


def test_conflict_cta_beats_image_and_image_beats_animation():
    tl = Timeline([], duration=300)
    k = cta_comment("cta", 100.0, score=40)
    i1 = img("im1", 102.0, score=99)
    i2 = img("im2", 150.0, score=40)
    a1 = anim("an1", 151.5, score=99)
    plan, re = run([k, i1, i2, a1], tl)
    assert k.status == enrich.ST_OK                 # cta выиграл у image
    assert i1.status == enrich.ST_CONFLICT and "cta_comment" in i1.status_note
    assert i2.status == enrich.ST_OK                # image выиграл у animation
    assert a1.status == enrich.ST_CONFLICT and "image" in a1.status_note


def test_conflict_same_type_higher_score_wins():
    tl = Timeline([], duration=300)
    hi = img("hi", 200.0, score=90)
    lo = img("lo", 201.0, score=50)
    plan, re = run([hi, lo], tl)
    assert hi.status == enrich.ST_OK
    assert lo.status == enrich.ST_CONFLICT and "hi" in lo.status_note


# --- зазор >= 2 c между любыми окнами ------------------------------------------------
def test_gap_under_two_seconds_is_conflict(png):
    tl = Timeline([], duration=300)
    a = img("a", 60.0, score=90, path=png, kind="user")       # [60..63]
    b = img("b", 64.5, score=50, path=png, kind="user")       # зазор 1.5 c
    plan, re = run([a, b], tl)
    assert len(re.stills) == 1
    assert b.status == enrich.ST_CONFLICT
    assert "ближе 2" in b.status_note


def test_gap_exactly_two_seconds_ok(png):
    tl = Timeline([], duration=300)
    a = img("a", 60.0, score=90, path=png, kind="user")       # [60..63]
    b = img("b", 65.0, score=50, path=png, kind="user")       # зазор ровно 2 c
    plan, re = run([a, b], tl)
    assert len(re.stills) == 2


# --- чистые зоны и CTA >= 60 c ---------------------------------------------------------
def test_clean_zones_head_and_tail(png):
    tl = Timeline([], duration=300)
    head = img("head", 25.0, path=png, kind="user")
    tail = img("tail", 285.0, path=png, kind="user")          # f1=288 > 280
    mid = img("mid", 100.0, path=png, kind="user")
    plan, re = run([head, tail, mid], tl)
    assert head.status == enrich.ST_OFF_LIMITS
    assert tail.status == enrich.ST_OFF_LIMITS
    assert "чистая зона" in head.status_note
    assert mid.status == enrich.ST_OK
    assert len(re.stills) == 1 and re.stills[0].t0 == pytest.approx(100.0)


def test_cta_not_before_60s_final():
    tl = Timeline([], duration=300)
    early = cta_sub("early", 45.0)
    fine = cta_sub("fine", 65.0)
    plan, re = run([early, fine], tl)
    assert early.status == enrich.ST_OFF_LIMITS
    assert "60" in early.status_note
    assert fine.status == enrich.ST_OK


def test_clean_zone_measured_on_final_timeline(png):
    # вырез 0..40: оригинальная 65-я секунда — финальная 25-я -> чистая зона
    tl = Timeline([(0, 40)], duration=340)
    plan, re = run([img("a", 65.0, path=png, kind="user")], tl)
    assert plan.items[0].status == enrich.ST_OFF_LIMITS


# --- потолки плотности (§9, R5) ---------------------------------------------------------
def test_images_max_two_per_minute(png):
    tl = Timeline([], duration=300)
    items = [img("i1", 60.0, score=90, path=png, kind="user"),
             img("i2", 80.0, score=85, path=png, kind="user"),
             img("i3", 100.0, score=80, path=png, kind="user")]
    plan, re = run(items, tl)
    assert len(re.stills) == 2
    assert items[2].status == enrich.ST_OFF_LIMITS            # минимальный score
    assert "картинок в минуту" in items[2].status_note


def test_cta_max_two_per_ten_minutes(cta_dir):
    (cta_dir / "subscribe_like.webm").write_bytes(b"webm")
    tl = Timeline([], duration=700)
    items = [cta_sub("c1", 100.0, score=90),
             cta_sub("c2", 200.0, score=80),
             cta_sub("c3", 300.0, score=70)]
    plan, re = run(items, tl)
    assert len(re.anims) == 2
    assert items[2].status == enrich.ST_OFF_LIMITS
    assert "10 минут" in items[2].status_note


def test_total_density_budget_trims_lowest_score():
    # 120 c финала -> int(2.5 * 2) = 5 окон максимум
    tl = Timeline([], duration=120)
    items = [card(f"c{i}", 35.0 + 5 * i,
                  [("Раз", -1, 35.5 + 5 * i), ("Два", -1, 36.0 + 5 * i)],
                  score=90 - 5 * i, hold=1.2)
             for i in range(6)]
    plan, re = run(items, tl)
    assert len(re.cards) == 5
    loser = items[5]                                          # score 65
    assert loser.status == enrich.ST_OFF_LIMITS
    assert "плотности" in loser.status_note


def test_screen_time_capped_at_quarter():
    # 200 c финала -> суммарное экранное время <= 50 c; карточки по ~7.3 c
    tl = Timeline([], duration=200)
    items = [card(f"c{i}", 35.0 + 10 * i,
                  [("х" * 60, -1, 35.5 + 10 * i),
                   ("у" * 60, -1, 36.0 + 10 * i)],
                  score=90 - i, hold=1.2)
             for i in range(7)]
    plan, re = run(items, tl)
    assert len(re.cards) == 6
    assert items[6].status == enrich.ST_OFF_LIMITS
    assert "25%" in items[6].status_note
    total = sum(c.t1 - c.t0 for c in re.cards)
    assert total <= 0.25 * 200 + 1e-6


# --- выбор ассетов (задача 3) -------------------------------------------------------------
def test_user_asset_missing_drops_from_render_keeps_item(tmp_path):
    tl = Timeline([], duration=300)
    a = img("a", 100.0, kind="user", path=str(tmp_path / "нет.png"))
    plan, re = run([a], tl)
    assert re.stills == []
    assert a.enabled is True and a.status == enrich.ST_OK     # остаётся в плане
    assert "не найден" in a.status_note


def test_user_still_geometry_and_fade(png):
    tl = Timeline([], duration=300)
    a = img("a", 100.0, kind="user", path=png, pos="top_right", kenburns=True)
    b = img("b", 150.0, kind="user", path=png, pos="top_left")
    plan, re = run([a, b], tl)
    st_a, st_b = re.stills
    assert st_a.path == png
    assert st_a.x_expr == "W-w-48" and st_a.y_expr == "48"    # 48 px @1080p
    assert st_b.x_expr == "48" and st_b.y_expr == "48"
    assert st_a.scale_w == round(0.32 * W) == 614             # 30-34% ширины
    assert st_a.fade_s == pytest.approx(0.22)
    assert st_a.kenburns is True and st_b.kenburns is False
    assert st_a.t0 == pytest.approx(100.0)
    assert st_a.t1 == pytest.approx(103.0)


def test_emoji_uses_cache_or_drops(tmp_path, monkeypatch):
    cache = tmp_path / "ecache"
    cache.mkdir()
    monkeypatch.setattr(enrich, "EMOJI_CACHE_DIR", cache)
    cached = cache / f"u26a1_{enrich.EMOJI_PNG_SIZE}.png"
    cached.write_bytes(b"png")                                # уже в кэше
    tl = Timeline([], duration=300)
    ok = img("ok", 100.0, kind="emoji", emoji="u26a1")
    # битый кодпойнт не разворачивается в символ -> дроп из рендера (P5: реальная
    # растеризация; невалидное имя честно даёт status_note, item остаётся в плане)
    miss = img("miss", 150.0, kind="emoji", emoji="zzz")
    plan, re = run([ok, miss], tl)
    assert len(re.stills) == 1
    assert re.stills[0].path == str(cached)                   # из кэша как есть
    assert "zzz" in miss.status_note                          # честная пометка
    assert miss.enabled is True


def test_emoji_png_path_interface(tmp_path):
    cache = tmp_path / "c"
    cache.mkdir()
    assert enrich.emoji_png_path("", cache) is None           # пустое имя
    assert enrich.emoji_png_path("zzz", cache) is None        # битый кодпойнт
    # уже закэшированный файл отдаётся как есть (идемпотентно)
    p = cache / f"u26a1_{enrich.EMOJI_PNG_SIZE}.png"
    p.write_bytes(b"png")
    assert enrich.emoji_png_path("u26a1", cache) == p
    # валидный noto-кодпойнт без кэша -> растеризуется в НЕпустой PNG (P5)
    p2 = enrich.emoji_png_path("u1f5c3", cache)
    assert p2 is not None and p2.is_file() and p2.stat().st_size > 0


def test_asset_kind_none_skipped_silently():
    tl = Timeline([], duration=300)
    a = img("a", 100.0, kind="none")
    plan, re = run([a], tl)
    assert re.stills == [] and a.status_note == ""
    assert a.status == enrich.ST_OK


def test_cta_subscribe_webm_overlay(cta_dir):
    f = cta_dir / "subscribe_like.webm"
    f.write_bytes(b"webm")
    tl = Timeline([], duration=300)
    plan, re = run([cta_sub("c", 100.0)], tl)
    assert len(re.anims) == 1
    an = re.anims[0]
    assert an.path == str(f)
    assert an.x_expr == "48" and an.y_expr == "H-h-160"       # §2.3 низ-лево
    assert an.scale_w == round(W * enrich.CTA_WIDTH_FRAC) == 220
    assert an.loop is True
    assert an.t0 == pytest.approx(100.0) and an.t1 == pytest.approx(104.0)


def test_cta_comment_text_survives_missing_icon(cta_dir):
    tl = Timeline([], duration=300)
    k = cta_comment("c", 100.0, q="Какой дистрибутив выбрал и зачем?")
    plan, re = run([k], tl)
    assert len(re.cta_texts) == 1                             # вопрос — в ASS
    ct = re.cta_texts[0]
    assert ct.text == "Какой дистрибутив выбрал и зачем?"
    assert ct.t0 == pytest.approx(100.0) and ct.t1 == pytest.approx(105.0)
    assert re.anims == []                                     # иконки нет
    assert "не найден" in k.status_note
    assert k.enabled is True and k.status == enrich.ST_OK


def test_cta_comment_with_icon(cta_dir):
    (cta_dir / "comment.webm").write_bytes(b"webm")
    tl = Timeline([], duration=300)
    plan, re = run([cta_comment("c", 100.0)], tl)
    assert len(re.cta_texts) == 1 and len(re.anims) == 1
    assert re.anims[0].path.endswith("comment.webm")


def test_cta_like_uses_like_webm(cta_dir):
    (cta_dir / "like.webm").write_bytes(b"webm")
    tl = Timeline([], duration=300)
    plan, re = run([cta_like("c", 100.0)], tl)
    assert len(re.anims) == 1
    assert re.anims[0].path.endswith("like.webm")


def test_animation_webm_loop_by_preset(webm):
    tl = Timeline([], duration=300)
    pulse = anim("p", 100.0, kind="user", path=webm, preset="pulse")
    pop = anim("q", 150.0, kind="user", path=webm, preset="pop_in")
    plan, re = run([pulse, pop], tl)
    assert [a.loop for a in re.anims] == [True, False]
    assert re.anims[0].scale_w == round(0.18 * W)


def test_animation_png_degrades_to_still(png):
    tl = Timeline([], duration=300)
    a = anim("a", 100.0, kind="user", path=png)
    plan, re = run([a], tl)
    assert re.anims == []
    assert len(re.stills) == 1
    assert re.stills[0].fade_s == pytest.approx(0.18)         # fade_ms анимации


# --- engine-лимиты §2.1 п.5 ------------------------------------------------------------------
def test_engine_limit_six_stills(png):
    tl = Timeline([], duration=600)
    items = [img(f"i{i}", 60.0 + 60 * i, score=90 - i, path=png, kind="user")
             for i in range(8)]
    plan, re = run(items, tl)
    assert len(re.stills) == 6
    for trimmed in items[6:]:
        assert "лимит движка" in trimmed.status_note
        assert trimmed.enabled is True                        # страховка, не вето
    assert [st.t0 for st in re.stills] == sorted(st.t0 for st in re.stills)


def test_engine_limit_three_anims(webm):
    tl = Timeline([], duration=600)
    items = [anim(f"a{i}", 60.0 + 60 * i, score=90 - i, kind="user", path=webm)
             for i in range(5)]
    plan, re = run(items, tl)
    assert len(re.anims) == 3
    assert "лимит движка" in items[4].status_note


# --- прочее поведение планировщика ------------------------------------------------------------
def test_disabled_item_untouched_and_not_blocking(png):
    tl = Timeline([], duration=300)
    off = img("off", 100.0, score=99, path=png, kind="user", enabled=False)
    off.status = enrich.ST_CONFLICT
    off.status_note = "старая пометка"
    on = img("on", 101.0, score=10, path=png, kind="user")
    plan, re = run([off, on], tl)
    assert off.status == enrich.ST_CONFLICT                   # не трогаем
    assert off.status_note == "старая пометка"
    assert on.status == enrich.ST_OK                          # конфликта нет
    assert len(re.stills) == 1 and re.stills[0].t0 == pytest.approx(101.0)


def test_accepted_item_status_reset(png):
    tl = Timeline([], duration=300)
    a = img("a", 100.0, path=png, kind="user")
    a.status = enrich.ST_CONFLICT
    a.status_note = "устарело после правки вырезов"
    plan, re = run([a], tl)
    assert a.status == enrich.ST_OK and a.status_note == ""
    assert len(re.stills) == 1


def test_render_enrich_shape_and_fonts_dir():
    tl = Timeline([], duration=300)
    plan, re = run([], tl)
    assert re.stills == [] and re.anims == []
    assert re.cards == [] and re.cta_texts == []
    assert re.cards_ass is None                               # ASS строит P1.2
    import vpipe
    from pathlib import Path
    expected = Path(vpipe.__file__).resolve().parent / "data" / "enrich" / "fonts"
    assert Path(re.fonts_dir) == expected


def test_card_tail_helper_matches_spec():
    # дочитывание: сумма длин / 13 симв/с * 2, не больше потолка
    assert enrich.card_tail_s(["абв", "где"], 1.2) == pytest.approx(1.2)
    assert enrich.card_tail_s(["х" * 26], 1.0) == pytest.approx(4.0)
    assert enrich.card_tail_s(["х" * 60, "у" * 60], 1.0) == \
        pytest.approx(enrich.CARD_TAIL_MAX_S)
