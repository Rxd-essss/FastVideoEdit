# -*- coding: utf-8 -*-
"""P1: схема enrich.json (ENRICH_PLAN §1.2) — модели, load/save, клампы.

Покрытие из §7-P1: схема/roundtrip, hash/cutlist_rev, клампы score и
длительностей, жёсткие лимиты текстов (items <=6, text <=60, question <=120),
незнакомый type -> скип с логом (не падение), атомарность записи.
"""
import json
import math

import pytest

from vpipe import enrich
from vpipe.enrich import (EnrichItem, EnrichPlan, ImagePayload,
                          compute_cutlist_rev, item_from_dict, load_enrich,
                          save_enrich)
from vpipe.models import (ACTION_CENSOR, ACTION_REMOVE, CutList, CutSegment,
                          TYPE_PAUSE)


# --- raw-dict builders (как пишет detектор/LLM-слой) ---------------------------
def raw_image(**over):
    d = {"id": "enr_img001", "type": "image", "enabled": True, "source": "llm",
         "score": 78, "word_start": 1240, "word_end": 1262,
         "t_start": 512.84, "t_end": 515.84,
         "quote": "реестр это единая точка отказа",
         "reason": "называется конкретная сущность",
         "status": "ok", "status_note": "", "edited": False,
         "payload": {"concept": "структура реестра",
                     "image_query_en": "windows registry diagram",
                     "style_hint": "diagram", "asset_kind": "user",
                     "asset_path": "D:/assets/registry.png", "emoji": "",
                     "position": "top_right", "width_frac": 0.32,
                     "kenburns": False, "fade_ms": 220}}
    d.update(over)
    return d


def raw_animation(**over):
    d = {"id": "enr_anim01", "type": "animation", "score": 60,
         "t_start": 100.0, "t_end": 102.5,
         "payload": {"preset": "pulse", "asset_kind": "emoji",
                     "asset_path": "", "emoji": "u26a1",
                     "position": "top_left", "width_frac": 0.18,
                     "fade_ms": 180}}
    d.update(over)
    return d


def raw_card(**over):
    d = {"id": "enr_card01", "type": "list_card", "score": 85,
         "word_start": 700, "word_end": 760, "t_start": 290.0, "t_end": 305.0,
         "payload": {"title": "Чем хорош реестр", "mode": "scrim",
                     "items": [
                         {"text": "Централизованный", "word_idx": 702,
                          "t_word": 290.12},
                         {"text": "Быстрый", "word_idx": 731,
                          "t_word": 301.40}],
                     "hold_s": 1.2}}
    d.update(over)
    return d


def raw_cta(t="cta_subscribe", **over):
    payload = {"cta_subscribe": {"variant": "sub_like",
                                 "position": "bottom_left", "duration_s": 4.0},
               "cta_like": {"position": "bottom_left", "duration_s": 3.0},
               "cta_comment": {"question": "Какой дистрибутив выбрал?",
                               "position": "bottom_left", "duration_s": 5.0}}[t]
    d = {"id": f"enr_{t}", "type": t, "score": 70,
         "t_start": 600.0, "t_end": 0.0, "payload": dict(payload)}
    d.update(over)
    return d


def full_plan_dict():
    return {"version": 1, "hash": "abc123", "cutlist_rev": "rev777",
            "generated_at": "2026-06-13T12:00:00Z", "model": "qwen3:8b",
            "params": {"density": "normal",
                       "types": {"image": True, "animation": True,
                                 "list_card": True, "cta": True},
                       "image_source": "auto"},
            "items": [raw_image(), raw_animation(), raw_card(),
                      raw_cta("cta_subscribe", id="enr_cta_s"),
                      raw_cta("cta_like", id="enr_cta_l", t_start=700.0),
                      raw_cta("cta_comment", id="enr_cta_c", t_start=800.0)]}


# --- schema roundtrip -----------------------------------------------------------
def test_roundtrip_all_six_types():
    plan = EnrichPlan.from_dict(full_plan_dict())
    assert [it.type for it in plan.items] == [
        "image", "animation", "list_card",
        "cta_subscribe", "cta_like", "cta_comment"]
    d = plan.to_dict()
    assert d["hash"] == "abc123" and d["cutlist_rev"] == "rev777"
    assert d["model"] == "qwen3:8b" and d["version"] == 1
    # второй проход — идемпотентность санитайза
    again = EnrichPlan.from_dict(d).to_dict()
    assert again == d

    img = plan.items[0]
    assert img.id == "enr_img001" and img.score == 78
    assert img.word_start == 1240 and img.word_end == 1262
    assert img.quote.startswith("реестр")
    assert img.payload.style_hint == "diagram"
    assert img.payload.asset_path == "D:/assets/registry.png"

    card = plan.items[2]
    assert card.payload.title == "Чем хорош реестр"
    assert [i.text for i in card.payload.items] == ["Централизованный",
                                                    "Быстрый"]
    assert card.payload.items[0].word_idx == 702

    cta = plan.items[3]
    assert cta.payload.variant == "sub_like"
    assert cta.t_end == pytest.approx(604.0)   # t_end = t_start + duration_s


def test_unknown_type_skipped_with_log():
    logs = []
    d = full_plan_dict()
    d["items"].insert(0, {"id": "enr_bad", "type": "sticker_3d",
                          "t_start": 5.0, "payload": {}})
    d["items"].append("мусор-не-словарь")
    plan = EnrichPlan.from_dict(d, log=logs.append)
    assert len(plan.items) == 6                  # незнакомое скипнуто, не упало
    assert any("sticker_3d" in m for m in logs)


def test_save_load_file_roundtrip(tmp_path):
    plan = EnrichPlan.from_dict(full_plan_dict())
    p = tmp_path / "video.enrich.json"
    save_enrich(plan, p)
    assert p.exists() and not p.with_suffix(".json.tmp").exists()
    raw = json.loads(p.read_text(encoding="utf-8"))
    for key in ("version", "hash", "cutlist_rev", "generated_at", "model",
                "params", "items"):
        assert key in raw
    loaded = load_enrich(p)
    assert loaded is not None
    assert loaded.hash == "abc123" and loaded.cutlist_rev == "rev777"
    assert len(loaded.items) == 6


def test_save_fills_generated_at(tmp_path):
    plan = EnrichPlan(hash="h")
    assert plan.generated_at == ""
    save_enrich(plan, tmp_path / "x.enrich.json")
    assert plan.generated_at != ""


def test_save_atomic_failure_keeps_original(tmp_path, monkeypatch):
    p = tmp_path / "x.enrich.json"
    p.write_text('{"old": true}', encoding="utf-8")

    def boom(src, dst):
        raise OSError("disk full")
    monkeypatch.setattr(enrich.os, "replace", boom)
    with pytest.raises(OSError):
        save_enrich(EnrichPlan(hash="new"), p)
    # оригинал не тронут, временный файл прибран
    assert json.loads(p.read_text(encoding="utf-8")) == {"old": True}
    assert not p.with_suffix(".json.tmp").exists()


def test_load_missing_and_corrupt(tmp_path):
    assert load_enrich(tmp_path / "nope.json") is None
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    assert load_enrich(bad) is None
    shape = tmp_path / "shape.json"
    shape.write_text('["list", "not", "dict"]', encoding="utf-8")
    assert load_enrich(shape) is None
    noitems = tmp_path / "noitems.json"
    noitems.write_text('{"items": "не список"}', encoding="utf-8")
    assert load_enrich(noitems) is None


# --- clamps (всё числовое валидирует код) ----------------------------------------
def test_score_clamped():
    assert item_from_dict(raw_image(score=-5)).score == 0
    assert item_from_dict(raw_image(score=250)).score == 100
    assert item_from_dict(raw_image(score="мусор")).score == 0
    assert item_from_dict(raw_image(score=True)).score == 0   # bool — не число


def test_image_duration_clamped():
    it = item_from_dict(raw_image(t_start=100.0, t_end=110.0))   # 10 c -> 4
    assert it.t_end - it.t_start == pytest.approx(enrich.IMAGE_DUR_MAX)
    it = item_from_dict(raw_image(t_start=100.0, t_end=100.5))   # 0.5 c -> 2.5
    assert it.t_end - it.t_start == pytest.approx(enrich.IMAGE_DUR_MIN)
    it = item_from_dict(raw_image(t_start=100.0, t_end=0.0))     # мусор -> дефолт
    assert it.t_end - it.t_start == pytest.approx(enrich.IMAGE_DUR_DEF)


def test_cta_duration_clamped_and_t_end_derived():
    it = item_from_dict(raw_cta("cta_comment", t_start=100.0, t_end=999.0))
    assert it.payload.duration_s == pytest.approx(5.0)
    assert it.t_end == pytest.approx(105.0)      # t_end игнорирует мусор из файла
    d = raw_cta("cta_comment", t_start=100.0)
    d["payload"]["duration_s"] = 99
    assert item_from_dict(d).t_end == pytest.approx(110.0)   # кламп 10 c
    d["payload"]["duration_s"] = 0.1
    assert item_from_dict(d).t_end == pytest.approx(103.0)   # кламп 3 c


def test_nan_inf_guards():
    it = item_from_dict(raw_image(t_start=float("nan"), t_end=float("inf")))
    assert it.t_start == 0.0
    assert math.isfinite(it.t_end)
    d = raw_card()
    d["payload"]["items"].append({"text": "Битый", "word_idx": 3,
                                  "t_word": float("nan")})
    assert len(item_from_dict(d).payload.items) == 2   # NaN-пункт выброшен


def test_word_index_guards():
    it = item_from_dict(raw_image(word_start=-5, word_end=-7))
    assert it.word_start == 0 and it.word_end == 0
    it = item_from_dict(raw_image(word_start=50, word_end=10))
    assert it.word_end == 50                      # word_end >= word_start


def test_payload_whitelists():
    d = raw_image()
    d["payload"].update({"style_hint": "anime", "asset_kind": "stock",
                         "position": "bottom_right", "width_frac": 0.9,
                         "fade_ms": -50})
    p = item_from_dict(d).payload
    assert p.style_hint == "photo" and p.asset_kind == "none"
    assert p.position == "top_right"
    assert p.width_frac == pytest.approx(enrich.IMG_WIDTH_MAX)
    assert p.fade_ms == 0
    d = raw_animation()
    d["payload"].update({"preset": "explode", "width_frac": 0.01})
    p = item_from_dict(d).payload
    assert p.preset == "pop_in"
    assert p.width_frac == pytest.approx(enrich.ANIM_WIDTH_MIN)
    d = raw_card()
    d["payload"]["mode"] = "fullscreen"
    assert item_from_dict(d).payload.mode == "scrim"
    d = raw_cta("cta_subscribe")
    d["payload"].update({"variant": "youtube_logo", "position": "top_right"})
    p = item_from_dict(d).payload
    assert p.variant == "sub_like" and p.position == "bottom_left"


def test_status_whitelist_and_id_autogen():
    it = item_from_dict(raw_image(status="exploded", id=""))
    assert it.status == "ok"
    assert it.id.startswith("enr_") and len(it.id) == 10
    it2 = item_from_dict(raw_image(id=None))
    assert it2.id != it.id                        # уникальные автогены


# --- жёсткие лимиты текстов §1.2 ---------------------------------------------------
def test_card_items_capped_at_six():
    d = raw_card()
    d["payload"]["items"] = [{"text": f"Пункт {i}", "word_idx": i,
                              "t_word": 100.0 + i} for i in range(8)]
    items = item_from_dict(d).payload.items
    assert len(items) == enrich.CARD_ITEMS_MAX == 6


def test_card_item_text_trimmed_at_word_boundary():
    d = raw_card()
    long = "слово " * 20                            # 119 символов
    d["payload"]["items"] = [{"text": long, "word_idx": 1, "t_word": 100.0},
                             {"text": "Ок", "word_idx": 2, "t_word": 101.0}]
    items = item_from_dict(d).payload.items
    assert len(items[0].text) <= enrich.CARD_ITEM_TEXT_MAX == 60
    assert set(items[0].text.split()) == {"слово"}   # без обрубков слова


def test_card_empty_items_dropped_and_sorted():
    d = raw_card()
    d["payload"]["items"] = [{"text": "Второй", "word_idx": 9, "t_word": 200.0},
                             {"text": "   ", "word_idx": 5, "t_word": 150.0},
                             {"text": "Первый", "word_idx": 2, "t_word": 100.0}]
    items = item_from_dict(d).payload.items
    assert [i.text for i in items] == ["Первый", "Второй"]   # сорт по t_word


def test_card_hold_clamped():
    d = raw_card()
    d["payload"]["hold_s"] = 9.0
    assert item_from_dict(d).payload.hold_s == pytest.approx(enrich.CARD_HOLD_MAX)
    d["payload"]["hold_s"] = 0.2
    assert item_from_dict(d).payload.hold_s == pytest.approx(enrich.CARD_HOLD_MIN)


def test_question_trimmed_to_120():
    d = raw_cta("cta_comment")
    d["payload"]["question"] = "почему " * 40        # 279 символов
    q = item_from_dict(d).payload.question
    assert len(q) <= enrich.CTA_QUESTION_MAX == 120
    assert set(q.split()) == {"почему"}


# --- params sanitize ----------------------------------------------------------------
def test_params_whitelisting():
    p = enrich.sanitize_params({"density": "turbo",
                                "types": {"image": False, "мусор": True},
                                "image_source": "stocks", "lol": 1})
    assert p == {"density": "normal",
                 "types": {"image": False, "animation": True,
                           "list_card": True, "cta": True},
                 "image_source": "auto"}
    assert enrich.sanitize_params(None) == enrich.default_params()


# --- cutlist_rev -----------------------------------------------------------------------
def _cl(segs):
    return CutList(source="x.mp4", duration=600.0, segments=segs)


def _seg(i, a, b, enabled=True, action=ACTION_REMOVE):
    return CutSegment(id=f"c{i}", start=a, end=b, type=TYPE_PAUSE,
                      action=action, enabled=enabled)


def test_cutlist_rev_stable_and_canonical():
    r1 = compute_cutlist_rev(_cl([_seg(1, 10, 12), _seg(2, 50, 55)]))
    # порядок сегментов не влияет
    r2 = compute_cutlist_rev(_cl([_seg(2, 50, 55), _seg(1, 10, 12)]))
    assert r1 == r2
    # выключенные и censor-вырезы не участвуют
    r3 = compute_cutlist_rev(_cl([_seg(1, 10, 12), _seg(2, 50, 55),
                                  _seg(3, 70, 80, enabled=False),
                                  _seg(4, 90, 91, action=ACTION_CENSOR)]))
    assert r3 == r1
    # тоггл выреза -> другой rev
    r4 = compute_cutlist_rev(_cl([_seg(1, 10, 12),
                                  _seg(2, 50, 55, enabled=False)]))
    assert r4 != r1
    # сырые интервалы дают тот же канон
    assert compute_cutlist_rev([(50, 55), (10, 12)]) == r1
    assert len(r1) == 40                          # sha1 hex


def test_cutlist_rev_rounding():
    assert compute_cutlist_rev([(10.0001, 12.0004)]) == \
        compute_cutlist_rev([(10.0, 12.0)])
