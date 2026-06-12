"""C4 — «Авто-пак» (POST /api/autopack): сырец → готовый ролик + пак Shorts
ОДНОЙ фоновой задачей. TestClient + моки (паттерны test_serve_render /
test_api_clips / test_queue) — без ffmpeg/whisper/Ollama.

Покрывает контракт C4 целиком:
 1. полный happy-path: стадии в правильном порядке (extract → transcribe →
    detect → main → suggest → clip×K), топ-K по порядку re-rank, results
    {main.formats, clips[{id,mp4,hook}], totals, ok, warnings, skipped},
    clips.json закэширован, edge_fade/граничные REMOVE — как у /api/clips/render;
 2. кэшированная транскрипция пропускается (transcribe/extract не зовутся);
 3. существующий НЕпустой катлист НЕ передетекчивается; пустой → детект;
 4. мультиформат основного ролика через _render_formats (имена _9x16 и т.п.);
 5. отмена на КАЖДОЙ стадии (транскрипция / детект / основной рендер /
    suggest / между клипами) — частичные results сохранены, чистый «cancelled»;
 6. suggest упал (Ollama умерла) → частичный успех: ok:true, warnings,
    clips:{error}, основной ролик в results, задача НЕ валится;
 7. LLM выключен и clips=true → warning «ИИ выключен — клипы не предложены»,
    основной рендерится; свежий clips.json реюзается БЕЗ suggest (и без LLM);
 8. упавший клип не валит остальные; top_k клампится в 1..10;
 9. 409 (занятая задача / очередь / нет сессии / нет аудио), 400-валидация
    тела, CSRF (чужой Origin → 403).
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import serve
from vpipe.clips import ClipCandidate
from vpipe.config import ProfanityLists, load_config
from vpipe.detect.profanity import ProfanityMatcher
from vpipe.models import (ACTION_REMOVE, TYPE_PAUSE, CutList, CutSegment,
                          Segment, Transcript, Word)

HASH = "b7" * 20


def _make_transcript(duration: float = 40.0) -> Transcript:
    n = int(duration)
    words = [Word(f"сл{i:02d}", i + 0.1, i + 0.9) for i in range(n)]
    return Transcript(language="ru", duration=duration, model="t",
                      audio_hash=HASH,
                      segments=[Segment(0.0, duration,
                                        " ".join(w.word for w in words), words)])


def _cut(start: float = 7.0, end: float = 8.0) -> CutSegment:
    return CutSegment(id="p1", start=start, end=end, type=TYPE_PAUSE,
                      action=ACTION_REMOVE, enabled=True)


def _cand(cid: str = "c01", start: float = 5.0, end: float = 35.0,
          hook: str = "Сервера обрабатывают") -> ClipCandidate:
    return ClipCandidate(id=cid, seg_start=0, seg_end=3, start=start, end=end,
                         dur_raw=end - start, dur_eff=end - start, score=85,
                         score_window=85, hook_phrase=hook, reason="тест")


class FakeSession:
    """Минимальная сессия с НАСТОЯЩЕЙ task-механикой serve.Session (паттерн
    test_api_clips), плюс записывающий _detect — детекция здесь мокается на
    уровне сессии, а не run_detection (autopack зовёт s._detect())."""

    start_task = serve.Session.start_task
    set_progress = serve.Session.set_progress
    stage = serve.Session.stage

    def __init__(self, tmp_path: Path, *, duration: float = 40.0, llm=None,
                 cuts=(), transcript: bool = True, cutlist: bool = True,
                 has_audio: bool = True, events=None):
        self.cfg = load_config("config.yaml")
        self.inp = Path("fake.mp4")
        self.media = SimpleNamespace(path="fake.mp4", duration=duration,
                                     width=1920, height=1080, fps=30.0,
                                     has_audio=has_audio)
        self.ff = None
        self.work_dir = tmp_path / "work"
        self.out_dir = tmp_path / "out"
        self.cache_dir = tmp_path / "cache"
        for d in (self.work_dir, self.out_dir, self.cache_dir):
            d.mkdir(parents=True, exist_ok=True)
        self.last_out_dir = str(self.out_dir.resolve())
        self.audio_hash = HASH
        self.matcher = ProfanityMatcher(ProfanityLists(roots=[], allow=[]))
        self.llm = llm
        self.transcript = _make_transcript(duration) if transcript else None
        self.cutlist = (CutList(source="fake.mp4", duration=duration,
                                segments=list(cuts)) if cutlist else None)
        self.task = {"name": None, "running": False, "percent": 0.0,
                     "stage": "", "error": None, "done": False, "results": None}
        self.events = events if events is not None else []
        self.detect_calls: list = []
        self.detect_hook = None          # для cancel-теста стадии детекта

    def _detect(self, cfg=None):
        self.detect_calls.append(cfg)
        self.events.append("detect")
        if self.detect_hook is not None:
            self.detect_hook()
        cut = CutSegment(id="d1", start=10.0, end=12.0, type=TYPE_PAUSE,
                         action=ACTION_REMOVE, enabled=True)
        self.cutlist = CutList(source="fake.mp4", duration=self.media.duration,
                               segments=[cut])
        return self.cutlist


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


def _patch_transcribe(monkeypatch, events, duration: float = 40.0, block=None):
    """serve.extract_audio / serve.transcribe_audio → рекордеры без whisper."""

    def fake_extract(ff, inp, wav, total=None, on_progress=None):
        events.append("extract")
        return wav

    def fake_transcribe(wav, tcfg, dur, audio_hash, cache_dir=None,
                        log=None, on_progress=None):
        events.append("transcribe")
        if block is not None:
            block()
        return _make_transcript(duration)

    monkeypatch.setattr(serve, "extract_audio", fake_extract)
    monkeypatch.setattr(serve, "transcribe_audio", fake_transcribe)


def _patch_suggest(monkeypatch, events, result=None, block=None):
    """serve.clips_mod.suggest → рекордер; Exception в result → raise."""

    def fake_suggest(tr, cl, ccfg, lcfg, llm, *, log=print, on_progress=None,
                     on_stage=None):
        events.append("suggest")
        if block is not None:
            block()
        if isinstance(result, Exception):
            raise result
        return list(result or [])

    monkeypatch.setattr(serve.clips_mod, "suggest", fake_suggest)


def _patch_pipeline(monkeypatch, events, behaviors=None):
    """Мок _run_render_pipeline: и основной рендер (_render_formats), и клипы
    идут через него; различаем по cutlist_override. behaviors — по ПОРЯДКУ всех
    вызовов: Exception → raise, callable → вызвать, иначе успех."""
    calls: list[dict] = []
    behaviors = list(behaviors or [])

    def fake_pipeline(s, cfg, scale_h, fps, out_dir, base, on_progress,
                      on_stage, cutlist_override=None, edge_fade=0.0,
                      sidecar_base=None):
        idx = len(calls)
        events.append("clip" if cutlist_override is not None else "main")
        calls.append({"cfg": cfg, "base": base, "out_dir": out_dir,
                      "override": cutlist_override, "edge_fade": edge_fade,
                      "stage_at_entry": s.task["stage"]})
        b = behaviors[idx] if idx < len(behaviors) else None
        if isinstance(b, Exception):
            raise b
        if callable(b):
            b(s, on_progress, on_stage)
        return {"mp4": str(base) + ".mp4", "encoder": "fake",
                "new_duration": 33.0}

    monkeypatch.setattr(serve, "_run_render_pipeline", fake_pipeline)
    return calls


def _blocker():
    """(started, release, block): block() сигналит started и ждёт release."""
    started, release = threading.Event(), threading.Event()

    def block():
        started.set()
        assert release.wait(5.0)

    return started, release, block


# --- 1. полный happy-path: порядок стадий, топ-K, results, кэш clips.json ------
def test_full_happy_path_stage_order_and_results(client, monkeypatch, tmp_path):
    events: list = []
    sess = FakeSession(tmp_path, llm=object(), transcript=False, cutlist=False,
                       events=events)
    _install(monkeypatch, sess)
    _patch_transcribe(monkeypatch, events)
    _patch_suggest(monkeypatch, events, result=[
        _cand("c01", 5.0, 35.0),
        _cand("c02", 36.0, 39.0, hook="Лицензия не требует"),
        _cand("c03", 1.0, 4.0)])
    calls = _patch_pipeline(monkeypatch, events)

    r = client.post("/api/autopack", json={"top_k": 2})
    assert r.status_code == 200
    assert r.json() == {"ok": True, "top_k": 2, "formats": ["source"],
                        "clips": True}
    assert sess.task["name"] == "autopack"
    _wait_done(sess)
    assert sess.task["error"] is None and sess.task["done"] is True
    assert sess.task["percent"] == 100.0

    # стадии строго по порядку; из 3 кандидатов рендерятся топ-2 (порядок re-rank)
    assert events == ["extract", "transcribe", "detect", "main",
                      "suggest", "clip", "clip"]
    res = sess.task["results"]
    assert res["ok"] is True
    assert res["warnings"] == []

    # main: формат source отрендерен, merged-результат на верхнем уровне
    fmts = res["main"]["formats"]
    assert len(fmts) == 1
    assert fmts[0]["format"] == "source" and fmts[0]["ok"] is True
    assert fmts[0]["mp4"].endswith("fake.mp4")

    # clips: id/mp4/hook, имена <stem>_clipNN
    assert [c["id"] for c in res["clips"]] == ["c01", "c02"]
    assert all(c["ok"] for c in res["clips"])
    assert res["clips"][0]["mp4"].endswith("fake_clip01.mp4")
    assert res["clips"][1]["mp4"].endswith("fake_clip02.mp4")
    assert res["clips"][0]["hook"] == "Сервера обрабатывают"
    assert res["clips"][1]["hook"] == "Лицензия не требует"

    # totals: до/после, число вырезов (детект дал 1), отрендеренные клипы
    assert res["totals"] == {"duration_before": 40.0, "duration_after": 33.0,
                             "cuts": 1, "clips_rendered": 2}

    # точный цикл /api/clips/render: основной — БЕЗ override и edge_fade,
    # клипы — с граничными REMOVE и edge_fade из cfg.clips (дефолт 0.025)
    assert calls[0]["override"] is None and calls[0]["edge_fade"] == 0.0
    for i, call in enumerate(calls[1:]):
        assert call["edge_fade"] == pytest.approx(0.025)
        by_id = {seg.id: seg for seg in call["override"].segments}
        assert f"clipA{i}" in by_id and f"clipB{i}" in by_id
        assert "d1" in by_id                 # живой вырез скопирован внутрь
    assert (calls[1]["override"].segments[0] is not
            sess.cutlist.segments[0])        # копия, сессия не мутируется
    assert len(sess.cutlist.segments) == 1

    # suggest закэширован в out/<stem>.clips.json (панель оживёт без LLM)
    data = json.loads((sess.out_dir / "fake.clips.json")
                      .read_text(encoding="utf-8"))
    assert data["hash"] == HASH and len(data["clips"]) == 3


# --- 2. кэшированная транскрипция пропускается ----------------------------------
def test_cached_transcript_and_cutlist_skipped(client, monkeypatch, tmp_path):
    events: list = []
    sess = FakeSession(tmp_path, llm=None, cuts=(_cut(),), events=events)
    _install(monkeypatch, sess)
    _patch_transcribe(monkeypatch, events)       # НЕ должны вызываться
    _patch_pipeline(monkeypatch, events)

    r = client.post("/api/autopack", json={"clips": False})
    assert r.status_code == 200
    _wait_done(sess)
    assert sess.task["error"] is None and sess.task["done"] is True
    assert events == ["main"]                    # ни extract/transcribe, ни detect
    skipped = sess.task["results"]["skipped"]
    assert any("Транскрипция" in x for x in skipped)
    assert any("Детекция" in x for x in skipped)


# --- 3. существующий катлист не передетекчивается; пустой → детект ---------------
def test_existing_cutlist_not_redetected(client, monkeypatch, tmp_path):
    events: list = []
    sess = FakeSession(tmp_path, llm=None, cuts=(_cut(),), events=events)
    _install(monkeypatch, sess)
    _patch_pipeline(monkeypatch, events)
    client.post("/api/autopack", json={"clips": False})
    _wait_done(sess)
    assert sess.detect_calls == []
    assert sess.cutlist.segments[0].id == "p1"   # катлист юзера нетронут


def test_empty_cutlist_triggers_detect(client, monkeypatch, tmp_path):
    events: list = []
    sess = FakeSession(tmp_path, llm=None, cuts=(), events=events)  # пустой
    _install(monkeypatch, sess)
    _patch_pipeline(monkeypatch, events)
    client.post("/api/autopack", json={"clips": False})
    _wait_done(sess)
    assert sess.task["error"] is None
    assert len(sess.detect_calls) == 1
    assert events == ["detect", "main"]


# --- 4. мультиформат основного ролика (C2 переиспользован) -----------------------
def test_main_multiformat_renders_each_format(client, monkeypatch, tmp_path):
    events: list = []
    sess = FakeSession(tmp_path, llm=None, cuts=(_cut(),), events=events)
    _install(monkeypatch, sess)
    calls = _patch_pipeline(monkeypatch, events)
    r = client.post("/api/autopack",
                    json={"formats": ["source", "9x16"], "clips": False})
    assert r.json()["formats"] == ["source", "9x16"]
    _wait_done(sess)
    assert sess.task["error"] is None
    assert [c["base"].name for c in calls] == ["fake", "fake_9x16"]
    fmts = sess.task["results"]["main"]["formats"]
    assert [(f["format"], f["ok"]) for f in fmts] == [("source", True),
                                                      ("9x16", True)]
    # stage с префиксом стадии: «Основной ролик: Формат i/N: …»
    assert calls[1]["stage_at_entry"].startswith("Основной ролик: Формат 2/2")


# --- 5. отмена на каждой стадии ---------------------------------------------------
def test_cancel_during_transcribe(client, monkeypatch, tmp_path):
    events: list = []
    started, release, block = _blocker()
    sess = FakeSession(tmp_path, llm=None, transcript=False, cutlist=False,
                       events=events)
    _install(monkeypatch, sess)
    _patch_transcribe(monkeypatch, events, block=block)
    _patch_pipeline(monkeypatch, events)
    client.post("/api/autopack", json={"clips": False})
    assert started.wait(5.0)
    assert client.post("/api/cancel").json() == {"ok": True}
    release.set()
    _wait_done(sess)
    assert sess.task["error"] == "cancelled" and sess.task["done"] is False
    assert events == ["extract", "transcribe"]   # детект/рендер не начались
    assert "main" not in sess.task["results"]


def test_cancel_during_detect(client, monkeypatch, tmp_path):
    events: list = []
    started, release, block = _blocker()
    sess = FakeSession(tmp_path, llm=None, cutlist=False, events=events)
    sess.detect_hook = block
    _install(monkeypatch, sess)
    _patch_pipeline(monkeypatch, events)
    client.post("/api/autopack", json={"clips": False})
    assert started.wait(5.0)
    client.post("/api/cancel")
    release.set()
    _wait_done(sess)
    assert sess.task["error"] == "cancelled"
    assert events == ["detect"]                  # основной рендер не начался
    assert "main" not in sess.task["results"]


def test_cancel_during_main_render_keeps_main_in_results(client, monkeypatch,
                                                         tmp_path):
    events: list = []
    started, release, block = _blocker()

    def slow_main(s, on_progress, on_stage):
        block()

    sess = FakeSession(tmp_path, llm=object(), cuts=(_cut(),), events=events)
    _install(monkeypatch, sess)
    _patch_suggest(monkeypatch, events, result=[_cand()])
    _patch_pipeline(monkeypatch, events, behaviors=[slow_main])
    client.post("/api/autopack", json={})
    assert started.wait(5.0)
    client.post("/api/cancel")
    release.set()
    _wait_done(sess)
    assert sess.task["error"] == "cancelled" and sess.task["done"] is False
    res = sess.task["results"]
    # уже отрендеренный основной ролик сохранён; suggest/клипы не начались
    assert res["main"]["formats"][0]["ok"] is True
    assert "suggest" not in events and "clip" not in events
    assert res["totals"]["clips_rendered"] == 0


def test_cancel_during_suggest(client, monkeypatch, tmp_path):
    events: list = []
    started, release, block = _blocker()
    sess = FakeSession(tmp_path, llm=object(), cuts=(_cut(),), events=events)
    _install(monkeypatch, sess)
    _patch_suggest(monkeypatch, events, result=[_cand()], block=block)
    _patch_pipeline(monkeypatch, events)
    client.post("/api/autopack", json={})
    assert started.wait(5.0)
    client.post("/api/cancel")
    release.set()
    _wait_done(sess)
    assert sess.task["error"] == "cancelled"
    res = sess.task["results"]
    assert res["main"]["formats"][0]["ok"] is True   # основной сохранён
    assert res["clips"] == []                        # клипы не рендерились
    assert "clip" not in events


def test_cancel_between_clips_keeps_partial(client, monkeypatch, tmp_path):
    events: list = []
    started, release, block = _blocker()

    def slow_clip(s, on_progress, on_stage):
        block()

    sess = FakeSession(tmp_path, llm=object(), cuts=(_cut(),), events=events)
    _install(monkeypatch, sess)
    _patch_suggest(monkeypatch, events,
                   result=[_cand("c01"), _cand("c02", 36.0, 39.0)])
    calls = _patch_pipeline(monkeypatch, events,
                            behaviors=[None, slow_clip])  # main ok, клип 1 висит
    client.post("/api/autopack", json={})
    assert started.wait(5.0)
    client.post("/api/cancel")
    release.set()
    _wait_done(sess)
    assert sess.task["error"] == "cancelled" and sess.task["done"] is False
    assert len(calls) == 2                           # второй клип не запускался
    res = sess.task["results"]
    assert len(res["clips"]) == 1 and res["clips"][0]["ok"] is True
    assert res["totals"]["clips_rendered"] == 1      # частичный результат честен


# --- 6. suggest упал → частичный успех (ok:true + warnings), не падение ----------
def test_suggest_failure_partial_success(client, monkeypatch, tmp_path):
    events: list = []
    sess = FakeSession(tmp_path, llm=object(), cuts=(_cut(),), events=events)
    _install(monkeypatch, sess)
    _patch_suggest(monkeypatch, events, result=RuntimeError("Ollama умерла"))
    _patch_pipeline(monkeypatch, events)
    client.post("/api/autopack", json={})
    _wait_done(sess)
    # задача завершилась УСПЕХОМ, не ошибкой
    assert sess.task["error"] is None and sess.task["done"] is True
    res = sess.task["results"]
    assert res["ok"] is True
    assert res["main"]["formats"][0]["ok"] is True   # основной ролик на месте
    assert res["clips"] == {"error": "Ollama умерла"}
    assert any("Ollama умерла" in w for w in res["warnings"])
    assert "clip" not in events
    assert res["totals"]["clips_rendered"] == 0


def test_suggest_empty_result_warns_and_finishes(client, monkeypatch, tmp_path):
    events: list = []
    sess = FakeSession(tmp_path, llm=object(), cuts=(_cut(),), events=events)
    _install(monkeypatch, sess)
    _patch_suggest(monkeypatch, events, result=[])
    _patch_pipeline(monkeypatch, events)
    client.post("/api/autopack", json={})
    _wait_done(sess)
    assert sess.task["error"] is None
    res = sess.task["results"]
    assert res["ok"] is True and res["clips"] == []
    assert any("не нашёл" in w for w in res["warnings"])
    assert "clip" not in events


# --- 7. LLM выключен / свежий clips.json реюзается --------------------------------
def test_llm_off_clips_requested_warns_and_renders_main(client, monkeypatch,
                                                        tmp_path):
    events: list = []
    sess = FakeSession(tmp_path, llm=None, cuts=(_cut(),), events=events)
    _install(monkeypatch, sess)
    _patch_suggest(monkeypatch, events)          # НЕ должен вызываться
    _patch_pipeline(monkeypatch, events)
    client.post("/api/autopack", json={})        # clips по умолчанию true
    _wait_done(sess)
    assert sess.task["error"] is None and sess.task["done"] is True
    res = sess.task["results"]
    assert res["ok"] is True
    assert "ИИ выключен — клипы не предложены" in res["warnings"]
    assert res["clips"] == []
    assert res["main"]["formats"][0]["ok"] is True
    assert events == ["main"]                    # ни suggest, ни клипов


def test_fresh_clips_json_reused_without_llm(client, monkeypatch, tmp_path):
    # hash-валидация кэша: свежий clips.json рендерится даже при ВЫКЛЮЧЕННОМ
    # LLM, suggest не зовётся, warning'ов нет.
    events: list = []
    sess = FakeSession(tmp_path, llm=None, cuts=(_cut(),), events=events)
    _install(monkeypatch, sess)
    serve._save_clips_json(sess, [_cand("c01"), _cand("c02", 36.0, 39.0)])
    _patch_suggest(monkeypatch, events)          # НЕ должен вызываться
    _patch_pipeline(monkeypatch, events)
    client.post("/api/autopack", json={"top_k": 1})
    _wait_done(sess)
    assert sess.task["error"] is None
    res = sess.task["results"]
    assert res["ok"] is True and res["warnings"] == []
    assert "suggest" not in events
    assert events == ["main", "clip"]            # top_k=1 — один клип
    assert [c["id"] for c in res["clips"]] == ["c01"]
    assert any("clips.json" in x for x in res["skipped"])


def test_stale_clips_json_not_reused(client, monkeypatch, tmp_path):
    # clips.json от другого видео (hash mismatch) игнорируется → suggest.
    events: list = []
    sess = FakeSession(tmp_path, llm=object(), cuts=(_cut(),), events=events)
    _install(monkeypatch, sess)
    serve._save_clips_json(sess, [_cand("старый")])
    sess.audio_hash = "ff" * 20
    _patch_suggest(monkeypatch, events, result=[_cand("новый")])
    _patch_pipeline(monkeypatch, events)
    client.post("/api/autopack", json={})
    _wait_done(sess)
    assert "suggest" in events
    assert [c["id"] for c in sess.task["results"]["clips"]] == ["новый"]


# --- 8. упавший клип не валит остальные; top_k клампится --------------------------
def test_one_clip_fails_others_render(client, monkeypatch, tmp_path):
    events: list = []
    sess = FakeSession(tmp_path, llm=object(), cuts=(_cut(),), events=events)
    _install(monkeypatch, sess)
    _patch_suggest(monkeypatch, events,
                   result=[_cand("c01"), _cand("c02", 36.0, 39.0)])
    _patch_pipeline(monkeypatch, events,
                    behaviors=[None, RuntimeError("boom")])  # клип 1 падает
    client.post("/api/autopack", json={})
    _wait_done(sess)
    assert sess.task["error"] is None and sess.task["done"] is True
    res = sess.task["results"]
    assert res["ok"] is True
    assert res["clips"][0] == {"ok": False, "id": "c01",
                               "filename": "fake_clip01",
                               "hook": "Сервера обрабатывают", "error": "boom"}
    assert res["clips"][1]["ok"] is True
    assert res["clips"][1]["mp4"].endswith("fake_clip02.mp4")
    assert res["totals"]["clips_rendered"] == 1


def test_top_k_clamped(client, monkeypatch, tmp_path):
    events: list = []
    sess = FakeSession(tmp_path, llm=None, cuts=(_cut(),), events=events)
    _install(monkeypatch, sess)
    # 12 свежих кандидатов в кэше — больше любого валидного top_k
    serve._save_clips_json(
        sess, [_cand(f"c{i:02d}", 1.0 + i * 3, 3.0 + i * 3) for i in range(12)])
    _patch_pipeline(monkeypatch, events)

    r = client.post("/api/autopack", json={"top_k": 99})
    assert r.json()["top_k"] == 10               # кламп сверху
    _wait_done(sess)
    assert len(sess.task["results"]["clips"]) == 10

    r = client.post("/api/autopack", json={"top_k": 0})
    assert r.json()["top_k"] == 1                # кламп снизу
    _wait_done(sess)
    assert len(sess.task["results"]["clips"]) == 1


def test_default_top_k_is_3(client, monkeypatch, tmp_path):
    events: list = []
    sess = FakeSession(tmp_path, llm=None, cuts=(_cut(),), events=events)
    _install(monkeypatch, sess)
    serve._save_clips_json(
        sess, [_cand(f"c{i:02d}", 1.0 + i * 3, 3.0 + i * 3) for i in range(5)])
    _patch_pipeline(monkeypatch, events)
    r = client.post("/api/autopack", json={})
    assert r.json()["top_k"] == 3
    _wait_done(sess)
    assert len(sess.task["results"]["clips"]) == 3


# --- 9. 409 / 400 / CSRF -----------------------------------------------------------
def test_busy_task_409(client, monkeypatch, tmp_path):
    sess = FakeSession(tmp_path)
    sess.task["running"] = True
    _install(monkeypatch, sess)
    assert client.post("/api/autopack", json={}).status_code == 409


def test_queue_running_409(client, monkeypatch, tmp_path):
    sess = FakeSession(tmp_path)
    _install(monkeypatch, sess)
    monkeypatch.setattr(serve, "_queue_running", True)
    assert client.post("/api/autopack", json={}).status_code == 409


def test_no_session_409(client):
    assert client.post("/api/autopack", json={}).status_code == 409


def test_no_audio_no_transcript_409(client, monkeypatch, tmp_path):
    sess = FakeSession(tmp_path, transcript=False, has_audio=False)
    _install(monkeypatch, sess)
    r = client.post("/api/autopack", json={})
    assert r.status_code == 409
    assert sess.task["name"] is None             # задача не создана


@pytest.mark.parametrize("body", [
    {"formats": "9x16"},                         # не список
    {"formats": []},                             # пустой список
    {"formats": ["bogus"]},                      # неизвестный формат
    {"top_k": "три"},                            # не число
    {"top_k": True},                             # bool — не число
    {"render_opts": "мусор"},                    # не объект
    {"render_opts": {"scale_h": 99}},            # битые общие опции → 400 сразу
])
def test_validation_400(client, monkeypatch, tmp_path, body):
    sess = FakeSession(tmp_path, cuts=(_cut(),))
    _install(monkeypatch, sess)
    r = client.post("/api/autopack", json=body)
    assert r.status_code == 400
    assert sess.task["name"] is None             # задача не создана


def test_autopack_csrf_guarded(client, monkeypatch, tmp_path):
    sess = FakeSession(tmp_path, cuts=(_cut(),))
    _install(monkeypatch, sess)
    r = client.post("/api/autopack", json={},
                    headers={"Origin": "http://evil.example"})
    assert r.status_code == 403
    assert sess.task["name"] is None


# --- мусорные записи в кэше клипов молча пропускаются ------------------------------
def test_autopack_top_clips_filters_garbage():
    cands = [
        "не объект",
        {"id": "битый", "start": "пять", "end": 10.0},
        {"id": "nan", "start": float("nan"), "end": 10.0},
        {"id": "пустой", "start": 10.0, "end": 10.0},
        {"id": "ok1", "start": 5.0, "end": 15.0, "hook_phrase": "Хук"},
        {"id": "ok2", "start": -3.0, "end": 50.0},   # клампится к [0, 40]
        {"id": "ok3", "start": 20.0, "end": 30.0},
    ]
    top = serve._autopack_top_clips(cands, 40.0, 2)
    assert [c["id"] for c in top] == ["ok1", "ok2"]
    assert top[0]["hook"] == "Хук"
    assert top[1] == {"id": "ok2", "start": 0.0, "end": 40.0, "hook": ""}
