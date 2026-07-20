# -*- coding: utf-8 -*-
"""P1: схема enrich.json (ENRICH_PLAN §1.2) — модели, load/save, клампы.

Покрытие из §7-P1: схема/roundtrip, hash/cutlist_rev, клампы score и
длительностей, жёсткие лимиты текстов (items <=6, text <=60, question <=120),
незнакомый type -> скип с логом (не падение), атомарность записи.
"""
import json
import math
import os

import pytest

from vpipe import enrich
from vpipe.enrich import (EnrichItem, EnrichPlan, ImagePayload, ListCardPayload,
                          RenderEnrich, ZoomWindow, compute_cutlist_rev,
                          item_from_dict, load_enrich, save_enrich)
from vpipe.models import (ACTION_CENSOR, ACTION_REMOVE, CutList, CutSegment,
                          TYPE_PAUSE)

# Absolute-path prefix valid on THIS OS. enrich._abs_path keeps only
# Path(s).is_absolute() strings, so a Windows drive literal ("D:/…") is dropped
# to "" on POSIX/CI — build the prefix from the platform so the fixtures stay
# absolute (hence preserved) on both Windows and Linux.
_A = "D:/" if os.name == "nt" else "/"


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
                     "asset_path": _A + "assets/registry.png", "emoji": "",
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
    assert img.payload.asset_path == _A + "assets/registry.png"

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


def test_asset_path_relative_rejected():
    """Path-traversal guard (код-ревью P2): относительный asset_path из
    правленного руками JSON / тела save не должен попадать во входы
    ffmpeg-графа — санитайзер оставляет только абсолютные пути."""
    for evil in ("../../config.yaml", "..\\..\\secret.png",
                 "work/sneaky.png", "   "):
        d = raw_image()
        d["payload"]["asset_path"] = evil
        assert item_from_dict(d).payload.asset_path == ""
        a = raw_animation()
        a["payload"]["asset_path"] = evil
        assert item_from_dict(a).payload.asset_path == ""
    # абсолютный путь живёт как раньше (см. test_roundtrip_all_six_types)
    assert item_from_dict(raw_image()).payload.asset_path == \
        _A + "assets/registry.png"


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
    assert item_from_dict(d).payload.mode == "panel"   # V11: дефолт A «панель»
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


# --- V11 §6: аддитивные поля схемы (обратная совместимость с P1-P5) ---------------
def test_image_gen_fields_roundtrip():
    """ImagePayload: asset_kind=generate + gen_seed/gen_prompt_en (round-trip)."""
    d = raw_image()
    d["payload"].update({"asset_kind": "generate", "gen_seed": 12345,
                         "gen_prompt_en": "a clean photo of a server room"})
    p = item_from_dict(d).payload
    assert p.asset_kind == "generate"
    assert p.gen_seed == 12345
    assert p.gen_prompt_en == "a clean photo of a server room"
    # to_dict экспонирует новые поля -> re-sanitize идемпотентен
    back = ImagePayload.sanitize(p.to_dict())
    assert back == p
    assert "gen_seed" in p.to_dict() and "gen_prompt_en" in p.to_dict()


def test_image_gen_defaults_and_back_compat():
    """Старый payload без новых полей санитайзится в дефолты = старое поведение."""
    p = ImagePayload.sanitize({})
    assert p.gen_seed == -1                     # дефолт = хэш query_en на рендере
    assert p.gen_prompt_en == ""
    assert p.asset_kind == "none"               # без generate в enum-фолбэке
    # raw_image() (как пишет детектор P1-P5) не знает новых полей — не падает
    old = item_from_dict(raw_image()).payload
    assert old.gen_seed == -1 and old.gen_prompt_en == ""
    assert old.asset_kind == "user"             # старое значение цело


def test_image_gen_seed_guards():
    """gen_seed: bool/мусор -> -1; нижняя граница -1 (отрицательные сиды = авто)."""
    d = raw_image()
    d["payload"]["gen_seed"] = True             # bool — не число
    assert item_from_dict(d).payload.gen_seed == -1
    d["payload"]["gen_seed"] = "мусор"
    assert item_from_dict(d).payload.gen_seed == -1
    d["payload"]["gen_seed"] = -99              # клампится к -1
    assert item_from_dict(d).payload.gen_seed == -1
    d["payload"]["gen_seed"] = 7
    assert item_from_dict(d).payload.gen_seed == 7


def test_card_mode_default_panel():
    """V11 §3: дефолт карточки -> panel; scrim остаётся валидным."""
    assert ListCardPayload().mode == "panel"
    assert ListCardPayload.sanitize({}).mode == "panel"
    assert ListCardPayload.sanitize({"mode": "scrim"}).mode == "scrim"
    assert ListCardPayload.sanitize({"mode": "panel"}).mode == "panel"
    assert ListCardPayload.sanitize({"mode": "fullscreen"}).mode == "panel"
    # старый план с явным scrim читается без изменений (back-compat)
    d = raw_card()                              # payload mode=scrim
    assert item_from_dict(d).payload.mode == "scrim"


def test_image_source_generate_accepted():
    """V11 §4: image_source += generate; auto/emoji/user_folder без изменений."""
    assert enrich.sanitize_params(
        {"image_source": "generate"})["image_source"] == "generate"
    for src in ("auto", "emoji", "user_folder"):
        assert enrich.sanitize_params(
            {"image_source": src})["image_source"] == src
    # незнакомый источник -> auto (семантика auto не сломана)
    assert enrich.sanitize_params(
        {"image_source": "google"})["image_source"] == "auto"
    assert enrich.default_params()["image_source"] == "auto"


def test_zoom_window_dataclass_defaults():
    """V11 §3/§4a: ZoomWindow — чистый датакласс с дефолтами для трека динамики."""
    z = ZoomWindow(t0=10.0, t1=12.5)
    assert (z.t0, z.t1) == (10.0, 12.5)
    assert z.z_max == 1.08 and z.cx == 0.5 and z.cy == 0.5
    z2 = ZoomWindow(t0=1.0, t1=2.0, z_max=1.05, cx=0.4, cy=0.6)
    assert z2.z_max == 1.05 and z2.cx == 0.4 and z2.cy == 0.6


def test_render_enrich_new_fields_default_empty():
    """V11 §3: punches/card_windows — пустые по умолчанию = старое поведение."""
    re = RenderEnrich()
    assert re.punches == [] and re.card_windows == []
    # заполняет трек динамики/карточек (фаза ПОСЛЕ схемы) — контракт держим
    re.punches.append(ZoomWindow(t0=5.0, t1=7.0))
    re.card_windows.append((20.0, 24.0))
    assert isinstance(re.punches[0], ZoomWindow)
    assert re.card_windows[0] == (20.0, 24.0)
    # старые поля не сломаны
    assert re.stills == [] and re.anims == [] and re.cards == []
    assert re.cta_texts == [] and re.cards_ass is None


def test_plan_roundtrip_preserves_gen_fields():
    """Полный план с generate-картинкой переживает save/load round-trip."""
    d = full_plan_dict()
    d["items"][0]["payload"].update(
        {"asset_kind": "generate", "gen_seed": 99,
         "gen_prompt_en": "abstract conceptual illustration, no text"})
    plan = EnrichPlan.from_dict(d)
    again = EnrichPlan.from_dict(plan.to_dict()).to_dict()
    assert again == plan.to_dict()              # идемпотентность с новыми полями
    img = again["items"][0]["payload"]
    assert img["asset_kind"] == "generate" and img["gen_seed"] == 99
    assert img["gen_prompt_en"].startswith("abstract")


# --- «Монтаж V2» §6: визуал-кандидаты + выбор (аддитивно, back-compat) -----------
def _schematic_cand(**over):
    d = {"source": "schematic", "intent": "compare", "style": "minimal",
         "fields": {"title": "Linux vs Windows", "col_a": "Linux",
                    "col_b": "Windows",
                    "rows": [{"feature": "Доля серверов", "a": "96 %",
                              "b": "4 %", "winner": "a"}]},
         "preview": _A + "cache/codegfx/abc.png"}
    d.update(over)
    return d


def _diffusion_cand(**over):
    d = {"source": "diffusion", "prompt": "modern data center, cinematic",
         "seed": 12345, "asset_path": _A + "cache/enrich_img/s0.png"}
    d.update(over)
    return d


def test_v2_candidates_roundtrip_and_idempotent():
    """ImagePayload c кандидатами: round-trip + идемпотентность санитайза."""
    d = raw_image()
    d["payload"].update({"candidates": [_schematic_cand(), _diffusion_cand()],
                         "selected": 1})
    p = item_from_dict(d).payload
    assert len(p.candidates) == 2
    assert p.selected == 1
    assert p.source == "diffusion"               # source = source выбранного канд.
    assert p.candidates[0]["source"] == "schematic"
    assert p.candidates[0]["intent"] == "compare"
    assert p.candidates[0]["fields"]["col_a"] == "Linux"
    assert p.candidates[0]["fields"]["rows"][0]["winner"] == "a"
    assert p.candidates[1]["seed"] == 12345
    # to_dict -> sanitize идемпотентно
    back = ImagePayload.sanitize(p.to_dict())
    assert back == p
    for key in ("candidates", "selected", "source", "intent",
                "schematic_style"):
        assert key in p.to_dict()


def test_v2_defaults_no_candidates():
    """Пустой payload → дефолты «Монтаж V2», старое поведение цело."""
    p = ImagePayload.sanitize({})
    assert p.candidates == [] and p.selected == 0
    assert p.source == "none" and p.intent == ""
    assert p.schematic_style == "minimal"
    # старые поля не сдвинуты
    assert p.asset_kind == "none" and p.gen_seed == -1


def test_v2_back_compat_old_image_payload():
    """Старый image-payload БЕЗ candidates читается как раньше (back-compat)."""
    old = item_from_dict(raw_image()).payload     # asset_kind=user, без V2-полей
    assert old.candidates == [] and old.selected == 0
    assert old.asset_path == _A + "assets/registry.png"
    assert old.asset_kind == "user"
    # source выводится из плоского asset_kind через resolved_asset()
    ap, src = old.resolved_asset()
    assert ap == _A + "assets/registry.png" and src == "stock"
    # generate → diffusion; emoji → icon; none → none
    gen = ImagePayload.sanitize({"asset_kind": "generate",
                                 "asset_path": _A + "cache/x.png"})
    assert gen.resolved_asset() == (_A + "cache/x.png", "diffusion")
    emj = ImagePayload.sanitize({"asset_kind": "emoji", "emoji": "u26a1"})
    assert emj.resolved_asset() == ("", "icon")
    assert ImagePayload.sanitize({}).resolved_asset() == ("", "none")


def test_v2_selected_clamped_into_range():
    """selected клампится в [0, len-1]; пустой список → 0."""
    base = raw_image()
    base["payload"]["candidates"] = [_schematic_cand(), _diffusion_cand()]
    for raw, exp in ((99, 1), (-3, 0), ("мусор", 0), (True, 0)):
        d = raw_image()
        d["payload"]["candidates"] = [_schematic_cand(), _diffusion_cand()]
        d["payload"]["selected"] = raw
        assert item_from_dict(d).payload.selected == exp
    # selected без кандидатов -> 0
    d = raw_image()
    d["payload"]["selected"] = 5
    assert item_from_dict(d).payload.selected == 0


def test_v2_candidates_capped_at_six():
    """Не больше CANDIDATES_MAX=6 кандидатов (листание)."""
    d = raw_image()
    d["payload"]["candidates"] = [_diffusion_cand(seed=i) for i in range(10)]
    p = item_from_dict(d).payload
    assert len(p.candidates) == enrich.CANDIDATES_MAX == 6


def test_v2_candidate_sanitize_per_source():
    """sanitize_candidate: по source оставляет только осмысленные ключи."""
    sc = enrich.sanitize_candidate(_schematic_cand())
    assert sc["source"] == "schematic" and sc["intent"] == "compare"
    assert sc["style"] == "minimal" and "fields" in sc
    # незнакомый intent/style -> фолбэк ("" / minimal)
    sc2 = enrich.sanitize_candidate(_schematic_cand(intent="hologram",
                                                    style="vaporwave"))
    assert sc2["intent"] == "" and sc2["style"] == "minimal"
    # diffusion: seed bool/мусор -> -1, относительный asset_path выкинут
    df = enrich.sanitize_candidate(
        {"source": "diffusion", "prompt": "x", "seed": True,
         "asset_path": "../evil.png"})
    assert df["seed"] == -1 and "asset_path" not in df
    # icon: emoji ИЛИ abs asset_path
    ic = enrich.sanitize_candidate({"source": "icon", "emoji": "u1f427"})
    assert ic["emoji"] == "u1f427"
    # kinetic: text + style
    kn = enrich.sanitize_candidate({"source": "kinetic", "text": "и тут магия"})
    assert kn["text"] == "и тут магия" and kn["style"] == "minimal"
    # none -> только source
    nn = enrich.sanitize_candidate({"source": "none"})
    assert nn == {"source": "none"}
    # битый/неизвестный source -> None (дроп)
    assert enrich.sanitize_candidate({"source": "tiktok"}) is None
    assert enrich.sanitize_candidate("не словарь") is None
    assert enrich.sanitize_candidate({}) is None


def test_v2_candidate_fields_flattened_and_limited():
    """_clean_fields: глубокая вложенность/не-данные отбрасываются, скаляры цело."""
    cand = {"source": "schematic", "intent": "stat",
            "fields": {"eyebrow": "Ядро Linux",
                       "stats": [{"value": "11 000", "label": "разработчиков"},
                                 {"deep": {"nope": 1}},          # row схлопся -> дроп
                                 "просто строка"],
                       "bar": 96,
                       "junk": {"a": {"b": 1}},                  # dict-значение -> дроп
                       "nan": float("nan")}}                     # NaN -> дроп
    sc = enrich.sanitize_candidate(cand)
    f = sc["fields"]
    assert f["eyebrow"] == "Ядро Linux" and f["bar"] == 96
    assert "junk" not in f and "nan" not in f
    # row из одних вложенных dict схлопывается в пустой -> выбрасывается из списка;
    # выживают валидная строка-dict и скаляр-строка.
    assert f["stats"] == [{"value": "11 000", "label": "разработчиков"},
                          "просто строка"]


def test_v2_resolved_asset_from_chosen_candidate():
    """resolved_asset(): выбранный кандидат -> (asset_path|preview, source)."""
    d = raw_image()
    d["payload"].update({"candidates": [_schematic_cand(), _diffusion_cand()],
                         "selected": 0})
    p = item_from_dict(d).payload
    # schematic: asset_path нет -> preview как путь (PNG движка)
    ap, src = p.resolved_asset()
    assert src == "schematic" and ap == _A + "cache/codegfx/abc.png"
    # переключение selected -> diffusion asset_path
    d["payload"]["selected"] = 1
    p2 = item_from_dict(d).payload
    assert p2.resolved_asset() == (_A + "cache/enrich_img/s0.png", "diffusion")
    # none-кандидат -> ("", "none")
    d2 = raw_image()
    d2["payload"]["candidates"] = [{"source": "none"}]
    assert item_from_dict(d2).payload.resolved_asset() == ("", "none")
    assert item_from_dict(d2).payload.chosen_candidate() == {"source": "none"}


def test_v2_plan_save_load_roundtrip_with_candidates(tmp_path):
    """Полный план с кандидатами переживает save/load + идемпотентен."""
    d = full_plan_dict()
    d["items"][0]["payload"].update(
        {"candidates": [_schematic_cand(), _diffusion_cand()], "selected": 1})
    plan = EnrichPlan.from_dict(d)
    p = tmp_path / "v2.enrich.json"
    save_enrich(plan, p)
    loaded = load_enrich(p)
    assert loaded is not None
    again = EnrichPlan.from_dict(loaded.to_dict()).to_dict()
    assert again == plan.to_dict()
    img = again["items"][0]["payload"]
    assert len(img["candidates"]) == 2 and img["selected"] == 1
    assert img["source"] == "diffusion"


def test_v2_config_codegfx_defaults():
    """config.render.codegfx — дефолты (codegfx_enabled=True, автопоиск Chrome)."""
    from vpipe.config import Config
    cg = Config().render.codegfx
    assert cg.codegfx_enabled is True
    assert cg.codegfx_chrome == ""               # пусто = автопоиск
    assert cg.codegfx_style == "minimal"
    assert cg.codegfx_size == "1920x1080"


def test_v2_config_imagegen_diffusion_defaults():
    """ImageGenCfg += cfg/candidates/vae (§5); старые поля цело."""
    from vpipe.config import Config
    ig = Config().render.imagegen
    assert ig.imagegen_cfg == pytest.approx(1.5)
    assert ig.imagegen_candidates == 4
    assert ig.imagegen_vae == ""
    # старые поля не сдвинуты
    assert ig.imagegen_size == 768 and ig.imagegen_steps == 4
    assert ig.imagegen_enabled is False
