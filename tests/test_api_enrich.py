# -*- coding: utf-8 -*-
"""P2 — API авто-обогащения (ENRICH_PLAN §5, §7-P2): /api/enrich/suggest,
GET /api/enrich, /api/enrich/save, opts.enrich рендера и /api/state.
TestClient + FakeSession (паттерн test_api_clips), без ffmpeg/whisper/Ollama.

Покрывает §7-P2 целиком:
 1. suggest happy (детекторы P3 деградируют на llm без chat_json): задача
    ``enrich``, файл
    out/<stem>.enrich.json создан атомарно с правильными hash/cutlist_rev/
    model/params и ПУСТЫМИ items; настройки персистятся в cache/enrich_ui.json;
    повторный suggest НЕ теряет работу юзера (мерж: enabled/edited по id +
    ручные source:"user" целиком; чужой hash не мержится);
 2. suggest без LLM → мгновенный 200 {ok:false, reason:'llm_off'} БЕЗ задачи;
 3. 409: нет транскрипта / нет катлиста / занятая задача / очередь / нет
    сессии; 400 на мусор в настройках (строгий B5-sanitize);
 4. GET /api/enrich: нет файла / битый файл / stale по hash /
    cutlist_changed по compute_cutlist_rev;
 5. save: enabled-тоггл (edited НЕ ставится), правка текста пункта карточки и
    вопроса CTA (edited:true, лимиты текстов §1.2), NaN/Infinity → 400,
    неизвестный id → 404, новый id с type → source:"user", нет плана → 409,
    несвежий hash → 409, строгий 500 при сбое записи с ЦЕЛЫМ оригиналом;
 6. рендер: whitelisting opts.enrich в _resolve_render_opts (дефолт ВЫКЛ,
    мусор → выкл, min_score клампится), пайплайн: свежий план → render()
    получает RenderEnrich; несвежий hash / нет плана → рендер БЕЗ обогащения
    с предупреждением в stage; cutlist_override (клипы) обогащения не получает;
    min_score=70 режет предложения ниже порога; карточки → enrich_{base}.ass;
 7. CSRF на оба POST'а; /api/state: enrich_opts + enrich:{count, stale}.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import serve
from vpipe import enrich as enrich_mod
from vpipe.config import ProfanityLists, load_config
from vpipe.detect.profanity import ProfanityMatcher
from vpipe.models import (ACTION_REMOVE, TYPE_PAUSE, CutList, CutSegment,
                          Segment, Transcript, Word)

HASH = "c9" * 20

_SILENT = lambda *a, **k: None  # noqa: E731


def _make_transcript(duration: float) -> Transcript:
    n = int(duration)
    words = [Word(f"сл{i:03d}", i + 0.1, i + 0.9) for i in range(n)]
    return Transcript(language="ru", duration=duration, model="t",
                      audio_hash=HASH,
                      segments=[Segment(0.0, duration,
                                        " ".join(w.word for w in words),
                                        words)])


class FakeSession:
    """Минимальная сессия с НАСТОЯЩЕЙ task-механикой serve.Session (паттерн
    test_api_clips) — фоновые задачи/409/прогресс как в проде, без ffmpeg."""

    start_task = serve.Session.start_task
    set_progress = serve.Session.set_progress
    stage = serve.Session.stage

    def __init__(self, tmp_path: Path, *, duration: float = 40.0, llm=None,
                 cuts=()):
        self.cfg = load_config("config.yaml")
        self.inp = Path("fake.mp4")
        self.media = SimpleNamespace(path="fake.mp4", duration=duration,
                                     width=1920, height=1080, fps=30.0)
        self.ff = None
        self.work_dir = tmp_path / "work"
        self.out_dir = tmp_path / "out"
        self.cache_dir = tmp_path / "cache"
        for d in (self.work_dir, self.out_dir, self.cache_dir):
            d.mkdir(parents=True, exist_ok=True)
        self.cfg.paths.cache_dir = str(self.cache_dir)
        self.last_out_dir = str(self.out_dir.resolve())
        self.audio_hash = HASH
        self.matcher = ProfanityMatcher(ProfanityLists(roots=[], allow=[]))
        self.llm = llm
        self.transcript = _make_transcript(duration)
        self.cutlist = CutList(source="fake.mp4", duration=duration,
                               segments=list(cuts))
        self.task = {"name": None, "running": False, "percent": 0.0,
                     "stage": "", "error": None, "done": False, "results": None}


def _wait_done(sess, timeout: float = 5.0) -> None:
    t0 = time.time()
    while sess.task["running"]:
        if time.time() - t0 > timeout:
            raise AssertionError(f"задача не завершилась: {sess.task}")
        time.sleep(0.01)


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setattr(serve, "_queue_running", False)
    monkeypatch.setattr(serve, "SESSION", None)
    return TestClient(serve.app)


def _install(monkeypatch, sess) -> None:
    monkeypatch.setattr(serve, "SESSION", sess)
    # enrich_ui.json живёт в APP-конфиге (паттерн detect_ui.json)
    monkeypatch.setitem(serve.APP, "cfg", sess.cfg)


# --- builders плана (прямое конструирование, как в test_enrich_plan) -----------
def _img(iid="enr_img001", t=40.0, score=80, enabled=True, kind="none",
         path=""):
    return enrich_mod.EnrichItem(
        id=iid, type=enrich_mod.ENR_IMAGE, score=score, enabled=enabled,
        t_start=t, t_end=t + 3.0,
        payload=enrich_mod.ImagePayload(concept="реестр", asset_kind=kind,
                                        asset_path=path))


def _card(iid="enr_card01", t=60.0):
    return enrich_mod.EnrichItem(
        id=iid, type=enrich_mod.ENR_LIST_CARD, score=75, t_start=t, t_end=0.0,
        payload=enrich_mod.ListCardPayload(
            title="Чем хорош реестр",
            items=[enrich_mod.CardItem(text="Централизованный", word_idx=-1,
                                       t_word=t + 1.0),
                   enrich_mod.CardItem(text="Быстрый", word_idx=-1,
                                       t_word=t + 3.0)]))


def _cta(iid="enr_cta001", t=70.0, q="Какой дистрибутив выбрал?"):
    return enrich_mod.EnrichItem(
        id=iid, type=enrich_mod.ENR_CTA_COMMENT, score=90,
        t_start=t, t_end=t + 5.0,
        payload=enrich_mod.CtaCommentPayload(question=q))


def _write_plan(sess, items, *, hash=None, rev=None):
    plan = enrich_mod.EnrichPlan(
        hash=hash if hash is not None else sess.audio_hash,
        cutlist_rev=(rev if rev is not None
                     else enrich_mod.compute_cutlist_rev(sess.cutlist)),
        model="qwen3:8b", items=list(items))
    enrich_mod.save_enrich(plan, serve._enrich_json_path(sess))
    return plan


def _plan_file(sess) -> dict:
    return json.loads(serve._enrich_json_path(sess).read_text(encoding="utf-8"))


# === 1. suggest: happy path с заглушкой P3 ======================================
def test_suggest_happy_creates_plan_with_hash_and_rev(client, monkeypatch,
                                                      tmp_path):
    cut = CutSegment(id="p1", start=7.0, end=8.0, type=TYPE_PAUSE,
                     action=ACTION_REMOVE, enabled=True)
    sess = FakeSession(tmp_path, llm=object(), cuts=[cut])
    _install(monkeypatch, sess)
    r = client.post("/api/enrich/suggest", json={
        "types": {"image": False}, "density": "min",
        "image_source": "emoji", "user_folder": "  D:/assets  "})
    assert r.status_code == 200 and r.json() == {"ok": True}
    assert sess.task["name"] == "enrich"
    _wait_done(sess)
    assert sess.task["error"] is None and sess.task["done"] is True

    # файл создан, атомарно, с правильными hash/cutlist_rev/model/params
    p = serve._enrich_json_path(sess)
    assert p.name == "fake.enrich.json" and p.exists()
    assert not p.with_suffix(".json.tmp").exists()
    data = _plan_file(sess)
    assert data["version"] == 1
    assert data["hash"] == HASH
    assert data["cutlist_rev"] == enrich_mod.compute_cutlist_rev(sess.cutlist)
    assert data["model"] == sess.cfg.llm.model
    assert data["generated_at"]
    assert data["params"]["density"] == "min"
    assert data["params"]["types"] == {"image": False, "animation": True,
                                       "list_card": True, "cta": True}
    assert data["params"]["image_source"] == "emoji"
    # llm=object() без chat_json: каждый вызов детектора падает и честно
    # пропускается (one-bad-window/детектор не валит пасс) → план пуст
    assert data["items"] == []
    assert sess.task["results"]["enrich"]["items"] == []

    # настройки запуска персистятся в cache/enrich_ui.json (канонический вид)
    saved = json.loads((sess.cache_dir / "enrich_ui.json")
                       .read_text(encoding="utf-8"))
    assert saved["density"] == "min"
    assert saved["types"]["image"] is False
    assert saved["user_folder"] == "D:/assets"
    assert saved["stocks"] == {"enabled": False}


def test_suggest_routes_to_detect_all_with_sanitized_params(client, monkeypatch,
                                                            tmp_path):
    """P3: заглушка заменена на enrich_llm.detect_all — задача передаёт ему
    транскрипт/катлист/whitelist-params/llm сессии и прогресс-хук."""
    sess = FakeSession(tmp_path, llm=object())
    _install(monkeypatch, sess)
    seen: dict = {}

    def fake_detect_all(transcript, cutlist, params, llm, log=None,
                        on_progress=None):
        seen.update(transcript=transcript, cutlist=cutlist, params=params,
                    llm=llm, log=log, on_progress=on_progress)
        return []

    monkeypatch.setattr(serve.enrich_llm, "detect_all", fake_detect_all)
    client.post("/api/enrich/suggest",
                json={"types": {"cta": False}, "density": "min",
                      "user_folder": "D:/assets"})    # не-params ключ отсечён
    _wait_done(sess)
    assert sess.task["error"] is None
    assert seen["transcript"] is sess.transcript
    assert seen["cutlist"] is sess.cutlist
    assert seen["llm"] is sess.llm
    assert callable(seen["log"]) and callable(seen["on_progress"])
    # детекторам уходит ровно whitelist-подмножество (sanitize_params)
    assert seen["params"] == {
        "density": "min",
        "types": {"image": True, "animation": True,
                  "list_card": True, "cta": False},
        "image_source": "auto"}


def test_suggest_rerun_preserves_user_edits(client, monkeypatch, tmp_path):
    """CRITICAL код-ревью P2: повторный suggest НЕ уничтожает работу юзера —
    ручные предложения (source:"user") и правки enabled/edited переживают
    новый анализ (llm=object() — детекторы деградируют в [])."""
    sess = FakeSession(tmp_path, llm=object(), duration=120.0)
    _install(monkeypatch, sess)
    client.post("/api/enrich/suggest", json={})
    _wait_done(sess)
    assert _plan_file(sess)["items"] == []       # P2: первый план честно пуст

    # юзер руками добавил CTA и картинку, поправил вопрос, выключил картинку
    r = client.post("/api/enrich/save", json={"items": [
        {"id": "enr_manual1", "type": "cta_comment", "t_start": 70.0,
         "payload": {"question": "Какой дистрибутив выбрал?"}},
        {"id": "enr_manual2", "type": "image", "t_start": 40.0}]})
    assert r.status_code == 200
    r = client.post("/api/enrich/save", json={"items": [
        {"id": "enr_manual1", "payload": {"question": "Какой у тебя сетап?"}},
        {"id": "enr_manual2", "enabled": False}]})
    assert r.status_code == 200

    client.post("/api/enrich/suggest", json={})  # повторный анализ
    _wait_done(sess)
    assert sess.task["error"] is None
    by_id = {it["id"]: it for it in _plan_file(sess)["items"]}
    assert set(by_id) == {"enr_manual1", "enr_manual2"}
    assert by_id["enr_manual1"]["source"] == "user"
    assert by_id["enr_manual1"]["payload"]["question"] == "Какой у тебя сетап?"
    assert by_id["enr_manual1"]["edited"] is True    # правка пережила re-suggest
    assert by_id["enr_manual2"]["enabled"] is False  # тоггл пережил re-suggest
    assert by_id["enr_manual2"]["edited"] is False


def test_suggest_rerun_matched_id_carries_enabled_and_edits(client, monkeypatch,
                                                            tmp_path):
    """Протокол мержа для P3 (совпавшие id от детекторов): enabled-решение
    юзера — закон; edited переносит payload+тайминги; свежие поля детектора
    (score) остаются новыми; исчезнувшие LLM-предложения уходят."""
    sess = FakeSession(tmp_path, llm=object(), duration=120.0)
    _install(monkeypatch, sess)
    edited = _img("enr_img002", t=80.0, score=55)
    edited.payload.concept = "правленый концепт"
    edited.edited = True
    edited.t_start, edited.t_end = 81.0, 84.0
    _write_plan(sess, [_img("enr_img001", t=40.0, enabled=False), edited,
                       _img("enr_gone01", t=95.0)])   # детектор его не вернёт
    # «детекторы P3» снова предложили те же id — свежие дефолты, новый score
    monkeypatch.setattr(
        serve, "_run_enrich_detectors",
        lambda s, params, log: [_img("enr_img001", t=40.0, score=90),
                                _img("enr_img002", t=80.0, score=90)])
    client.post("/api/enrich/suggest", json={})
    _wait_done(sess)
    assert sess.task["error"] is None
    by_id = {it["id"]: it for it in _plan_file(sess)["items"]}
    assert set(by_id) == {"enr_img001", "enr_img002"}   # gone — честно ушёл
    assert by_id["enr_img001"]["enabled"] is False      # решение юзера — закон
    assert by_id["enr_img001"]["score"] == 90           # свежая инфа детектора
    assert by_id["enr_img001"]["edited"] is False
    assert by_id["enr_img002"]["edited"] is True
    assert by_id["enr_img002"]["payload"]["concept"] == "правленый концепт"
    assert by_id["enr_img002"]["t_start"] == pytest.approx(81.0)


def test_suggest_rerun_stale_plan_not_merged(client, monkeypatch, tmp_path):
    """План от ДРУГОГО видео не мержится — полная hash-инвалидация (как в
    GET/save): чужие user-items не протекают в новый план."""
    sess = FakeSession(tmp_path, llm=object())
    _install(monkeypatch, sess)
    manual = _cta("enr_manual1", t=70.0)
    manual.source = "user"
    _write_plan(sess, [manual], hash="ff" * 20)
    client.post("/api/enrich/suggest", json={})
    _wait_done(sess)
    assert sess.task["error"] is None
    data = _plan_file(sess)
    assert data["items"] == []                   # полная инвалидация
    assert data["hash"] == HASH


# === 2. llm_off → мгновенный 200 без задачи =====================================
def test_suggest_llm_off_no_task_no_files(client, monkeypatch, tmp_path):
    sess = FakeSession(tmp_path, llm=None)
    _install(monkeypatch, sess)
    r = client.post("/api/enrich/suggest", json={"density": "min"})
    assert r.status_code == 200
    assert r.json() == {"ok": False, "reason": "llm_off"}
    assert sess.task["name"] is None and sess.task["running"] is False
    assert not serve._enrich_json_path(sess).exists()
    assert not (sess.cache_dir / "enrich_ui.json").exists()


# === 3. 409 / 400 ================================================================
def test_suggest_requires_transcript_and_cutlist(client, monkeypatch, tmp_path):
    sess = FakeSession(tmp_path, llm=object())
    sess.transcript = None
    _install(monkeypatch, sess)
    assert client.post("/api/enrich/suggest", json={}).status_code == 409
    sess.transcript = _make_transcript(40.0)
    sess.cutlist = None
    assert client.post("/api/enrich/suggest", json={}).status_code == 409


def test_suggest_busy_task_409(client, monkeypatch, tmp_path):
    sess = FakeSession(tmp_path, llm=object())
    sess.task["running"] = True
    _install(monkeypatch, sess)
    assert client.post("/api/enrich/suggest", json={}).status_code == 409


def test_suggest_queue_running_409(client, monkeypatch, tmp_path):
    sess = FakeSession(tmp_path, llm=object())
    _install(monkeypatch, sess)
    monkeypatch.setattr(serve, "_queue_running", True)
    assert client.post("/api/enrich/suggest", json={}).status_code == 409


def test_suggest_no_session_409(client):
    assert client.post("/api/enrich/suggest", json={}).status_code == 409


@pytest.mark.parametrize("body", [
    {"density": "turbo"},                        # вне белого списка
    {"image_source": "google"},
    {"types": "all"},                            # не объект
    {"types": {"image": "да"}},                  # не bool
    {"user_folder": 42},                         # не строка
    {"stocks": "on"},                            # не объект
])
def test_suggest_bad_opts_400_no_task_no_persist(client, monkeypatch, tmp_path,
                                                 body):
    sess = FakeSession(tmp_path, llm=object())
    _install(monkeypatch, sess)
    r = client.post("/api/enrich/suggest", json=body)
    assert r.status_code == 400
    assert sess.task["name"] is None             # задача не создана
    assert not (sess.cache_dir / "enrich_ui.json").exists()


# === 4. GET /api/enrich ==========================================================
def test_get_enrich_empty_when_no_file(client, monkeypatch, tmp_path):
    sess = FakeSession(tmp_path)
    _install(monkeypatch, sess)
    assert client.get("/api/enrich").json() == {
        "items": [], "params": None, "stale": False, "cutlist_changed": False}


def test_get_enrich_corrupt_file_like_no_file(client, monkeypatch, tmp_path):
    sess = FakeSession(tmp_path)
    _install(monkeypatch, sess)
    serve._enrich_json_path(sess).write_text("{мусор", encoding="utf-8")
    assert client.get("/api/enrich").json()["items"] == []
    serve._enrich_json_path(sess).write_text('{"items": "не список"}',
                                             encoding="utf-8")
    assert client.get("/api/enrich").json()["items"] == []


def test_get_enrich_returns_fresh_plan(client, monkeypatch, tmp_path):
    sess = FakeSession(tmp_path)
    _install(monkeypatch, sess)
    _write_plan(sess, [_img(), _cta()])
    j = client.get("/api/enrich").json()
    assert j["stale"] is False and j["cutlist_changed"] is False
    assert [it["id"] for it in j["items"]] == ["enr_img001", "enr_cta001"]
    assert j["params"]["density"] == "normal"
    assert j["model"] == "qwen3:8b" and j["generated_at"]


def test_get_enrich_stale_on_hash_mismatch(client, monkeypatch, tmp_path):
    sess = FakeSession(tmp_path)
    _install(monkeypatch, sess)
    _write_plan(sess, [_img()])
    sess.audio_hash = "ff" * 20                  # «другое» видео, то же имя
    j = client.get("/api/enrich").json()
    assert j == {"items": [], "params": None, "stale": True,
                 "cutlist_changed": False}


def test_get_enrich_cutlist_changed_soft_banner(client, monkeypatch, tmp_path):
    sess = FakeSession(tmp_path)
    _install(monkeypatch, sess)
    _write_plan(sess, [_img()])                  # rev снят с ПУСТОГО катлиста
    # юзер дорезал ролик после анализа → rev разъехался, но items живы
    sess.cutlist.segments.append(CutSegment(
        id="m1", start=5.0, end=6.0, type=TYPE_PAUSE,
        action=ACTION_REMOVE, enabled=True))
    j = client.get("/api/enrich").json()
    assert j["stale"] is False
    assert j["cutlist_changed"] is True
    assert len(j["items"]) == 1                  # предложения НЕ прячутся


def test_get_enrich_disabled_cut_does_not_change_rev(client, monkeypatch,
                                                     tmp_path):
    sess = FakeSession(tmp_path)
    _install(monkeypatch, sess)
    _write_plan(sess, [_img()])
    # выключенный вырез не входит в канон enabled-вырезов → rev стабилен
    sess.cutlist.segments.append(CutSegment(
        id="m1", start=5.0, end=6.0, type=TYPE_PAUSE,
        action=ACTION_REMOVE, enabled=False))
    assert client.get("/api/enrich").json()["cutlist_changed"] is False


def test_get_enrich_no_session_409(client):
    assert client.get("/api/enrich").status_code == 409


# === 5. POST /api/enrich/save ====================================================
def test_save_enabled_toggle_does_not_mark_edited(client, monkeypatch,
                                                  tmp_path):
    sess = FakeSession(tmp_path)
    _install(monkeypatch, sess)
    _write_plan(sess, [_img(), _cta()])
    r = client.post("/api/enrich/save", json={
        "items": [{"id": "enr_img001", "enabled": False}]})
    assert r.status_code == 200
    j = r.json()
    assert j["ok"] is True and len(j["items"]) == 1
    assert j["items"][0]["enabled"] is False
    assert j["items"][0]["edited"] is False      # тоггл — не правка
    data = _plan_file(sess)
    by_id = {it["id"]: it for it in data["items"]}
    assert by_id["enr_img001"]["enabled"] is False
    assert by_id["enr_img001"]["edited"] is False
    assert by_id["enr_img001"]["score"] == 80    # остальное не тронуто
    assert by_id["enr_cta001"]["enabled"] is True   # сосед цел
    # топ-уровень файла не сломан (hash-валидация GET жива)
    assert data["hash"] == HASH
    assert client.get("/api/enrich").json()["stale"] is False
    assert not serve._enrich_json_path(sess).with_suffix(".json.tmp").exists()


def test_save_card_item_text_edit_marks_edited_and_trims(client, monkeypatch,
                                                         tmp_path):
    sess = FakeSession(tmp_path)
    _install(monkeypatch, sess)
    _write_plan(sess, [_card()])
    long_text = "Очень " * 20 + "длинно"          # > 60 симв — жёсткий лимит §1.2
    r = client.post("/api/enrich/save", json={"items": [{
        "id": "enr_card01",
        "payload": {"items": [
            {"text": "Новый пункт", "word_idx": 5, "t_word": 61.0},
            {"text": long_text, "word_idx": -1, "t_word": 63.0}]}}]})
    assert r.status_code == 200
    it = r.json()["items"][0]
    assert it["edited"] is True
    texts = [x["text"] for x in it["payload"]["items"]]
    assert texts[0] == "Новый пункт"
    assert len(texts[1]) <= 60                   # обрезан по слову
    assert it["payload"]["title"] == "Чем хорош реестр"   # мерж не потерял title
    assert _plan_file(sess)["items"][0]["edited"] is True


def test_save_cta_question_edit_marks_edited_and_trims(client, monkeypatch,
                                                       tmp_path):
    sess = FakeSession(tmp_path)
    _install(monkeypatch, sess)
    _write_plan(sess, [_cta()])
    r = client.post("/api/enrich/save", json={"items": [{
        "id": "enr_cta001", "payload": {"question": "Какой у тебя сетап?"}}]})
    assert r.status_code == 200
    it = r.json()["items"][0]
    assert it["payload"]["question"] == "Какой у тебя сетап?"
    assert it["edited"] is True
    # лимит вопроса 120 симв (§1.2)
    r = client.post("/api/enrich/save", json={"items": [{
        "id": "enr_cta001", "payload": {"question": "оч " * 80}}]})
    assert len(r.json()["items"][0]["payload"]["question"]) <= 120


def test_save_t_start_shift_marks_edited_and_clamps(client, monkeypatch,
                                                    tmp_path):
    sess = FakeSession(tmp_path, duration=40.0)
    _install(monkeypatch, sess)
    _write_plan(sess, [_img(t=20.0)])
    r = client.post("/api/enrich/save", json={"items": [{
        "id": "enr_img001", "t_start": 20.5, "t_end": 23.5}]})
    assert r.status_code == 200
    it = r.json()["items"][0]
    assert it["t_start"] == pytest.approx(20.5)
    assert it["edited"] is True
    # кламп к ролику: за границы не выехать
    r = client.post("/api/enrich/save", json={"items": [{
        "id": "enr_img001", "t_start": -5.0}]})
    assert r.json()["items"][0]["t_start"] == 0.0


@pytest.mark.parametrize("body", [
    {},                                          # нет items
    {"items": []},                               # пустой список
    {"items": "не список"},
    {"items": ["не объект"]},
    {"items": [{"enabled": True}]},              # нет id
    {"items": [{"id": 42}]},                     # id не строка
    {"items": [{"id": "enr_img001", "enabled": "да"}]},
    {"items": [{"id": "enr_img001", "payload": "мусор"}]},
    {"items": [{"id": "enr_img001", "t_start": "пять"}]},
    # NaN/±Infinity валидны для json.loads — сырое тело (паттерн clips/save)
    '{"items": [{"id": "enr_img001", "t_start": NaN}]}',
    '{"items": [{"id": "enr_img001", "t_end": Infinity}]}',
    '{"items": [{"id": "enr_img001", "t_start": -Infinity}]}',
])
def test_save_validation_400_file_untouched(client, monkeypatch, tmp_path,
                                            body):
    sess = FakeSession(tmp_path)
    _install(monkeypatch, sess)
    _write_plan(sess, [_img()])
    before = _plan_file(sess)
    if isinstance(body, str):
        r = client.post("/api/enrich/save", content=body,
                        headers={"Content-Type": "application/json"})
    else:
        r = client.post("/api/enrich/save", json=body)
    assert r.status_code == 400
    assert _plan_file(sess) == before            # файл не изменён


def test_save_unknown_id_404(client, monkeypatch, tmp_path):
    sess = FakeSession(tmp_path)
    _install(monkeypatch, sess)
    _write_plan(sess, [_img()])
    r = client.post("/api/enrich/save", json={
        "items": [{"id": "enr_чужой", "enabled": False}]})
    assert r.status_code == 404


def test_save_new_id_with_type_becomes_user_item(client, monkeypatch,
                                                 tmp_path):
    sess = FakeSession(tmp_path)
    _install(monkeypatch, sess)
    _write_plan(sess, [_img()])
    r = client.post("/api/enrich/save", json={"items": [{
        "id": "enr_manual1", "type": "cta_like", "t_start": 25.0}]})
    assert r.status_code == 200
    it = r.json()["items"][0]
    assert it["id"] == "enr_manual1"
    assert it["source"] == "user"                # ручное предложение (§5)
    assert it["enabled"] is True
    assert it["t_end"] == pytest.approx(28.0)    # t_start + дефолт cta_like 3 c
    data = _plan_file(sess)
    assert [x["id"] for x in data["items"]] == ["enr_img001", "enr_manual1"]


def test_save_no_plan_409(client, monkeypatch, tmp_path):
    sess = FakeSession(tmp_path)
    _install(monkeypatch, sess)
    r = client.post("/api/enrich/save", json={
        "items": [{"id": "enr_img001", "enabled": False}]})
    assert r.status_code == 409


def test_save_stale_hash_409(client, monkeypatch, tmp_path):
    sess = FakeSession(tmp_path)
    _install(monkeypatch, sess)
    _write_plan(sess, [_img()])
    sess.audio_hash = "ff" * 20
    r = client.post("/api/enrich/save", json={
        "items": [{"id": "enr_img001", "enabled": False}]})
    assert r.status_code == 409


def test_save_busy_task_409(client, monkeypatch, tmp_path):
    sess = FakeSession(tmp_path)
    sess.task["running"] = True
    _install(monkeypatch, sess)
    _write_plan(sess, [_img()])
    r = client.post("/api/enrich/save", json={
        "items": [{"id": "enr_img001", "enabled": False}]})
    assert r.status_code == 409


def test_save_write_failure_500_keeps_original(client, monkeypatch, tmp_path):
    sess = FakeSession(tmp_path)
    _install(monkeypatch, sess)
    _write_plan(sess, [_img()])
    before = _plan_file(sess)

    real_replace = enrich_mod.os.replace

    def boom(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(enrich_mod.os, "replace", boom)
    r = client.post("/api/enrich/save", json={
        "items": [{"id": "enr_img001", "enabled": False}]})
    assert r.status_code == 500
    monkeypatch.setattr(enrich_mod.os, "replace", real_replace)
    # atomic: оригинал цел и валиден, .tmp подчищен
    assert _plan_file(sess) == before
    assert not serve._enrich_json_path(sess).with_suffix(".json.tmp").exists()
    assert client.get("/api/enrich").json()["stale"] is False


# === 6. рендер: whitelisting opts.enrich + применение плана =====================
@pytest.mark.parametrize("opts,enabled", [
    ({}, False),                                 # нет ключа = выключено
    ({"enrich": {"enabled": True}}, True),
    ({"enrich": {"enabled": False}}, False),
    ({"enrich": True}, False),                   # не объект — мусор
    ({"enrich": "да"}, False),
    ({"enrich": None}, False),
    ({"enrich": []}, False),
])
def test_resolve_opts_enrich_whitelisting(tmp_path, opts, enabled):
    sess = FakeSession(tmp_path)
    # даже включённый в config.yaml enrich глушится без явного opts.enrich
    sess.cfg.render.enrich.enabled = True
    cfg, *_ = serve._resolve_render_opts(sess, opts)
    assert cfg.render.enrich.enabled is enabled


@pytest.mark.parametrize("ms,expected", [
    (70, 70), ("70", 70), (250, 100), (-5, 0),
    ("мусор", 0), (None, 0), (True, 0), (float("nan"), 0), (float("inf"), 0),
])
def test_resolve_opts_enrich_min_score_clamped(tmp_path, ms, expected):
    sess = FakeSession(tmp_path)
    cfg, *_ = serve._resolve_render_opts(
        sess, {"enrich": {"enabled": True, "min_score": ms}})
    assert cfg.render.enrich.enabled is True
    assert cfg.render.enrich.min_score == expected


def _patch_render(monkeypatch):
    """vpipe.render.render → рекордер (без ffmpeg); ловит и enrich-kwarg."""
    calls: list[dict] = []

    def fake_render(ff, media, cl, cfg, out, work_dir, *, on_progress=None,
                    log=None, scale_h=None, fps=None, ass_path=None,
                    crop_filter=None, edge_fade=0.0, **kw):
        calls.append({"cl": cl, "ass_path": ass_path, "kw": kw})
        return {"out": str(out), "encoder": "fake"}

    monkeypatch.setattr(serve.render_mod, "render", fake_render)
    return calls


_BASE = {"subtitles": False, "chapters": False, "metadata": False}


def _run_pipeline(sess, opts, monkeypatch, **pipe_kw):
    calls = _patch_render(monkeypatch)
    stages: list[str] = []
    cfg, scale_h, fps, out_dir, base = serve._resolve_render_opts(sess, opts)
    serve._run_render_pipeline(sess, cfg, scale_h, fps, out_dir, base,
                               _SILENT, stages.append, **pipe_kw)
    return calls, stages


@pytest.fixture()
def png(tmp_path):
    p = tmp_path / "asset.png"
    p.write_bytes(b"\x89PNG fake")
    return str(p)


def test_render_applies_fresh_plan(monkeypatch, tmp_path, png):
    sess = FakeSession(tmp_path, duration=120.0)
    monkeypatch.setitem(serve.APP, "cfg", sess.cfg)
    _write_plan(sess, [_img(t=60.0, kind="user", path=png)])
    calls, stages = _run_pipeline(
        sess, {**_BASE, "enrich": {"enabled": True}}, monkeypatch)
    enr = calls[0]["kw"].get("enrich")
    assert enr is not None
    assert len(enr.stills) == 1 and enr.anims == []
    assert enr.stills[0].path == png
    assert enr.stills[0].t0 == pytest.approx(60.0)   # вырезов нет — 1:1
    assert enr.cards_ass is None                     # карточек нет — нет ASS


def test_render_without_opts_ignores_plan(monkeypatch, tmp_path, png):
    sess = FakeSession(tmp_path, duration=120.0)
    _write_plan(sess, [_img(t=60.0, kind="user", path=png)])
    calls, _ = _run_pipeline(sess, dict(_BASE), monkeypatch)
    assert "enrich" not in calls[0]["kw"]            # дефолт: выключено


def test_render_stale_hash_warns_and_renders_without_enrich(monkeypatch,
                                                            tmp_path, png):
    sess = FakeSession(tmp_path, duration=120.0)
    _write_plan(sess, [_img(t=60.0, kind="user", path=png)], hash="ff" * 20)
    calls, stages = _run_pipeline(
        sess, {**_BASE, "enrich": {"enabled": True}}, monkeypatch)
    assert "enrich" not in calls[0]["kw"]            # mp4 свят: рендер без
    assert any("рендер без обогащения" in m for m in stages)
    assert any("другого видео" in m for m in stages)


def test_render_missing_plan_warns_and_renders_without_enrich(monkeypatch,
                                                              tmp_path):
    sess = FakeSession(tmp_path, duration=120.0)
    calls, stages = _run_pipeline(
        sess, {**_BASE, "enrich": {"enabled": True}}, monkeypatch)
    assert "enrich" not in calls[0]["kw"]
    assert any("план не найден" in m for m in stages)


def test_render_clip_override_never_enriched(monkeypatch, tmp_path, png):
    # анти-скоуп §9: cutlist_override (клипы Clip Maker/автопака) — без enrich
    sess = FakeSession(tmp_path, duration=120.0)
    _write_plan(sess, [_img(t=60.0, kind="user", path=png)])
    clip_cl = CutList(source="fake.mp4", duration=120.0, segments=[])
    calls, stages = _run_pipeline(
        sess, {**_BASE, "enrich": {"enabled": True}}, monkeypatch,
        cutlist_override=clip_cl)
    assert "enrich" not in calls[0]["kw"]
    assert not any("Обогащение" in m for m in stages)   # даже без warning'а


def test_render_min_score_filters_below_threshold(monkeypatch, tmp_path, png):
    sess = FakeSession(tmp_path, duration=120.0)
    _write_plan(sess, [_img("enr_hi", t=60.0, score=80, kind="user", path=png),
                       _img("enr_lo", t=70.0, score=50, kind="user", path=png)])
    calls, _ = _run_pipeline(
        sess, {**_BASE, "enrich": {"enabled": True, "min_score": 70}},
        monkeypatch)
    enr = calls[0]["kw"]["enrich"]
    assert len(enr.stills) == 1
    assert enr.stills[0].t0 == pytest.approx(60.0)   # выжил только score 80


def test_render_cards_write_enrich_ass_in_work_dir(monkeypatch, tmp_path):
    sess = FakeSession(tmp_path, duration=120.0)
    _write_plan(sess, [_card(t=60.0), _cta(t=80.0)])
    calls, _ = _run_pipeline(
        sess, {**_BASE, "enrich": {"enabled": True}}, monkeypatch)
    enr = calls[0]["kw"]["enrich"]
    ass = Path(enr.cards_ass)
    assert ass.name == "enrich_fake.ass"             # enrich_{base}.ass
    assert ass.parent == sess.work_dir
    text = ass.read_text(encoding="utf-8-sig")
    assert "Чем хорош реестр" in text                # карточка
    assert "Какой дистрибутив выбрал?" in text       # вопрос CTA (CtaText)
    assert "PlayResX: 1920" in text and "PlayResY: 1080" in text


def test_render_disabled_items_not_rendered(monkeypatch, tmp_path, png):
    sess = FakeSession(tmp_path, duration=120.0)
    _write_plan(sess, [_img(t=60.0, kind="user", path=png, enabled=False)])
    calls, stages = _run_pipeline(
        sess, {**_BASE, "enrich": {"enabled": True}}, monkeypatch)
    assert "enrich" not in calls[0]["kw"]            # всё выключено → как пустой
    assert any("нет применимых предложений" in m for m in stages)


# === 7. CSRF + /api/state =======================================================
def test_enrich_posts_csrf_guarded(client, monkeypatch, tmp_path):
    sess = FakeSession(tmp_path, llm=object())
    _install(monkeypatch, sess)
    _write_plan(sess, [_img()])
    evil = {"Origin": "http://evil.example"}
    assert client.post("/api/enrich/suggest", headers=evil,
                       json={}).status_code == 403
    r = client.post("/api/enrich/save", headers=evil, json={
        "items": [{"id": "enr_img001", "enabled": False}]})
    assert r.status_code == 403
    assert sess.task["name"] is None
    assert _plan_file(sess)["items"][0]["enabled"] is True   # не тронуто


def test_state_includes_enrich_summary_and_opts(client, monkeypatch, tmp_path):
    sess = FakeSession(tmp_path)
    _install(monkeypatch, sess)
    j = client.get("/api/state").json()
    assert j["enrich"] == {"count": 0, "stale": False}
    assert j["enrich_opts"] is None                  # дефолты — не настраивали

    _write_plan(sess, [_img(), _cta()])
    serve._write_enrich_opts(serve._sanitize_enrich_opts({"density": "min"}))
    j = client.get("/api/state").json()
    assert j["enrich"] == {"count": 2, "stale": False}
    assert j["enrich_opts"]["density"] == "min"

    sess.audio_hash = "ff" * 20
    j = client.get("/api/state").json()
    assert j["enrich"] == {"count": 0, "stale": True}
