"""F3 — Clip Maker API (план §2.4–2.5): /api/clips/suggest, GET /api/clips,
/api/clips/render + _save/_load_clips_json. TestClient + мок-LLM, без ffmpeg
(рендер-валидация мокает _run_render_pipeline).

Покрывает ожидания плана §5/F3 пп.1-7:
 1. suggest без LLM → 200 {ok:false, reason:'llm_off'}, задача НЕ создана;
    с мок-LLM → задача preview_clips, результат в task.results.clips, файл
    out/<stem>.clips.json создан, валиден (формат §2.5) и записан атомарно;
 2. suggest при занятом слоте задач / работающей очереди → 409;
 3. GET /api/clips отдаёт сохранённые кандидаты; несовпадение hash →
    {clips:[], stale:true}; нет/битый файл → {clips:[]};
 4. render с 2 клипами → ОДНА задача render_clips, в results два элемента,
    имена <stem>_clip01/_clip02, chapters/metadata принудительно false даже
    если клиент прислал true; катлист клипа = живые вырезы + 2 граничных
    REMOVE; сессия не мутируется;
 5. первый клип падает (мок) → второй рендерится; [{ok:false},{ok:true}];
 6. /api/cancel между клипами останавливает цикл; частичные результаты
    сохранены, задача завершается чистым «cancelled»;
 7. прогресс монотонный: stage «Клип 1/2…» → «Клип 2/2…», percent (i+f)/N.
Плюс: CSRF-гард покрывает новые роуты (чужой Origin → 403).
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

HASH = "a8" * 20

# Осмысленные предложения с заглавной буквы (lowercase-guard и hook-цитатность
# из vpipe/clips.py должны пропустить кандидата мок-LLM).
_T = [
    "Прямо сейчас почти весь интернет держится на линуксе.",
    "Сервера обрабатывают миллионы запросов каждую секунду без сбоев.",
    "Давай разберем честно как устроена файловая система.",
    "Пакетный менеджер ставит программы одной короткой командой.",
    "Открытый код позволяет находить уязвимости буквально за часы.",
    "Обновления не перезагружают машину посреди рабочего дня.",
    "Сообщество отвечает на вопросы быстрее платной поддержки.",
    "Лицензия не требует ни копейки за тысячу серверов.",
]


def _seg(start: float, end: float, text: str) -> Segment:
    toks = text.split()
    words, step, t = [], (end - start) / max(1, len(toks)), start
    for tok in toks:
        words.append(Word(tok, round(t, 3), round(min(t + step * 0.8, end), 3)))
        t += step
    return Segment(start, end, text, words)


def _make_transcript(duration: float) -> Transcript:
    seg_dur = duration / len(_T)
    segs = [_seg(i * seg_dur, (i + 1) * seg_dur, t) for i, t in enumerate(_T)]
    return Transcript(language="ru", duration=duration, model="test",
                      audio_hash=HASH, segments=segs)


class MockLLM:
    """Строгий мок (план F3 — никакого Ollama): объект с chat_json()."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0

    def chat_json(self, system, user, schema, keep_alive=None):
        self.calls += 1
        if not self._responses:
            return {"clips": []}
        r = self._responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


class FakeSession:
    """Минимальная сессия с НАСТОЯЩЕЙ task-механикой serve.Session (start_task /
    set_progress / stage), чтобы фоновые задачи, /api/cancel и прогресс вели
    себя как в проде — но без ffmpeg/probe/whisper."""

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
        for d in (self.work_dir, self.out_dir):
            d.mkdir(parents=True, exist_ok=True)
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


def _cand(cid: str = "c01", start: float = 5.0, end: float = 35.0) -> ClipCandidate:
    return ClipCandidate(id=cid, seg_start=1, seg_end=6, start=start, end=end,
                         dur_raw=end - start, dur_eff=end - start, score=85,
                         score_window=85, hook_phrase="Сервера обрабатывают",
                         reason="тест")


_GOOD_LLM_RESPONSE = {"clips": [{
    "start_index": 0, "end_index": 5, "score": 85,
    "hook_phrase": " ".join(_T[0].split()[:5]), "reason": "хук с цифрой"}]}


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setattr(serve, "_queue_running", False)
    monkeypatch.setattr(serve, "SESSION", None)
    return TestClient(serve.app)


def _install(monkeypatch, sess) -> None:
    monkeypatch.setattr(serve, "SESSION", sess)


def _patch_pipeline(monkeypatch, behaviors=None):
    """Мок _run_render_pipeline (план F3: без реального ffmpeg). behaviors —
    список по вызовам: Exception → raise, callable → вызвать, иначе успех."""
    calls: list[dict] = []
    behaviors = list(behaviors or [])

    def fake_pipeline(s, cfg, scale_h, fps, out_dir, base, on_progress,
                      on_stage, cutlist_override=None, edge_fade=0.0):
        idx = len(calls)
        calls.append({"cfg": cfg, "base": base, "out_dir": out_dir,
                      "override": cutlist_override, "edge_fade": edge_fade,
                      "stage_at_entry": s.task["stage"]})
        b = behaviors[idx] if idx < len(behaviors) else None
        if isinstance(b, Exception):
            raise b
        if callable(b):
            b(s, on_progress, on_stage)
        return {"mp4": str(base) + ".mp4", "encoder": "fake"}

    monkeypatch.setattr(serve, "_run_render_pipeline", fake_pipeline)
    return calls


# --- 1. suggest: llm_off / happy path -----------------------------------------
def test_suggest_llm_off_returns_200_without_task(client, monkeypatch, tmp_path):
    sess = FakeSession(tmp_path, llm=None)
    _install(monkeypatch, sess)
    r = client.post("/api/clips/suggest")
    assert r.status_code == 200
    assert r.json() == {"ok": False, "reason": "llm_off"}
    # задача НЕ создана
    assert sess.task["name"] is None and sess.task["running"] is False
    assert not (sess.out_dir / "fake.clips.json").exists()


def test_suggest_requires_transcript_and_cutlist(client, monkeypatch, tmp_path):
    sess = FakeSession(tmp_path, llm=MockLLM([]))
    sess.transcript = None
    _install(monkeypatch, sess)
    assert client.post("/api/clips/suggest").status_code == 409
    sess.transcript = _make_transcript(40.0)
    sess.cutlist = None
    assert client.post("/api/clips/suggest").status_code == 409


def test_suggest_no_session_409(client):
    assert client.post("/api/clips/suggest").status_code == 409


def test_suggest_happy_path_task_results_and_cache_file(client, monkeypatch,
                                                        tmp_path):
    llm = MockLLM([_GOOD_LLM_RESPONSE])
    sess = FakeSession(tmp_path, llm=llm)
    _install(monkeypatch, sess)
    r = client.post("/api/clips/suggest")
    assert r.status_code == 200 and r.json() == {"ok": True}
    assert sess.task["name"] == "preview_clips"
    _wait_done(sess)
    assert sess.task["error"] is None and sess.task["done"] is True
    res = sess.task["results"]["clips"]
    assert len(res) == 1
    c = res[0]
    assert c["id"] == "c01"
    assert (c["seg_start"], c["seg_end"]) == (0, 5)
    assert c["score"] == 85 and c["hook_phrase"].startswith("Прямо сейчас")
    assert 0.0 <= c["start"] < c["end"] <= 40.0
    # файл out/<stem>.clips.json создан и валиден (формат §2.5)
    p = sess.out_dir / "fake.clips.json"
    data = json.loads(p.read_text(encoding="utf-8"))
    assert data["version"] == 1
    assert data["hash"] == HASH
    assert data["model"] == sess.cfg.llm.model
    assert data["generated_at"]
    assert data["clips"] == res
    # атомарность: .tmp не остаётся
    assert not (sess.out_dir / "fake.clips.json.tmp").exists()


# --- F6: rank_source доезжает до results / clips.json / GET ----------------------
_TWO_CAND_RESPONSE = {"clips": [
    {"start_index": 0, "end_index": 3, "score": 90,
     "hook_phrase": " ".join(_T[0].split()[:5]), "reason": "a"},
    {"start_index": 4, "end_index": 7, "score": 80,
     "hook_phrase": " ".join(_T[4].split()[:5]), "reason": "b"}]}


def test_suggest_rerank_rank_source_llm_everywhere(client, monkeypatch, tmp_path):
    llm = MockLLM([_TWO_CAND_RESPONSE,
                   {"ranking": [{"id": 2, "score": 95}, {"id": 1, "score": 60}]}])
    sess = FakeSession(tmp_path, llm=llm)
    _install(monkeypatch, sess)
    assert client.post("/api/clips/suggest").json() == {"ok": True}
    _wait_done(sess)
    assert sess.task["error"] is None
    res = sess.task["results"]
    assert res["rank_source"] == "llm"
    assert [c["seg_start"] for c in res["clips"]] == [4, 0]  # порядок от re-rank
    assert [c["score"] for c in res["clips"]] == [95, 60]    # re-rank-скоры
    assert [c["score_window"] for c in res["clips"]] == [80, 90]
    data = json.loads((sess.out_dir / "fake.clips.json")
                      .read_text(encoding="utf-8"))
    assert data["rank_source"] == "llm"
    assert client.get("/api/clips").json()["rank_source"] == "llm"


def test_suggest_rerank_failure_marks_round_robin(client, monkeypatch, tmp_path):
    llm = MockLLM([_TWO_CAND_RESPONSE, RuntimeError("rerank down")])
    sess = FakeSession(tmp_path, llm=llm)
    _install(monkeypatch, sess)
    client.post("/api/clips/suggest")
    _wait_done(sess)
    assert sess.task["error"] is None                # фолбэк, не падение задачи
    res = sess.task["results"]
    assert res["rank_source"] == "round_robin"
    assert [c["seg_start"] for c in res["clips"]] == [0, 4]  # окно-скор desc
    assert [c["score"] for c in res["clips"]] == [90, 80]
    assert client.get("/api/clips").json()["rank_source"] == "round_robin"


# --- 2. занятый слот задач → 409 ------------------------------------------------
def test_suggest_busy_task_409(client, monkeypatch, tmp_path):
    sess = FakeSession(tmp_path, llm=MockLLM([]))
    sess.task["running"] = True
    _install(monkeypatch, sess)
    assert client.post("/api/clips/suggest").status_code == 409


def test_suggest_queue_running_409(client, monkeypatch, tmp_path):
    sess = FakeSession(tmp_path, llm=MockLLM([]))
    _install(monkeypatch, sess)
    monkeypatch.setattr(serve, "_queue_running", True)
    assert client.post("/api/clips/suggest").status_code == 409


# --- 3. GET /api/clips: кэш / stale / отсутствие --------------------------------
def test_get_clips_returns_saved_candidates(client, monkeypatch, tmp_path):
    sess = FakeSession(tmp_path)
    _install(monkeypatch, sess)
    serve._save_clips_json(sess, [_cand()])
    j = client.get("/api/clips").json()
    assert j["stale"] is False
    assert j["model"] == sess.cfg.llm.model
    assert len(j["clips"]) == 1
    assert j["clips"][0]["id"] == "c01"
    assert j["clips"][0]["hook_phrase"] == "Сервера обрабатывают"


def test_get_clips_stale_on_hash_mismatch(client, monkeypatch, tmp_path):
    sess = FakeSession(tmp_path)
    _install(monkeypatch, sess)
    serve._save_clips_json(sess, [_cand()])
    sess.audio_hash = "ff" * 20            # «другое» видео с тем же именем
    assert client.get("/api/clips").json() == {"clips": [], "stale": True}


def test_get_clips_empty_when_no_file(client, monkeypatch, tmp_path):
    sess = FakeSession(tmp_path)
    _install(monkeypatch, sess)
    assert client.get("/api/clips").json() == {"clips": []}


def test_get_clips_corrupt_file_yields_empty(client, monkeypatch, tmp_path):
    sess = FakeSession(tmp_path)
    _install(monkeypatch, sess)
    (sess.out_dir / "fake.clips.json").write_text("{мусор", encoding="utf-8")
    assert client.get("/api/clips").json() == {"clips": []}
    (sess.out_dir / "fake.clips.json").write_text('{"clips": "не список"}',
                                                  encoding="utf-8")
    assert client.get("/api/clips").json() == {"clips": []}


# --- 4. render: одна задача, имена, принудительные false, граничные REMOVE ------
def test_render_two_clips_names_and_forced_flags(client, monkeypatch, tmp_path):
    cut = CutSegment(id="p1", start=7.0, end=8.0, type=TYPE_PAUSE,
                     action=ACTION_REMOVE, enabled=True)
    sess = FakeSession(tmp_path, cuts=[cut])
    _install(monkeypatch, sess)
    calls = _patch_pipeline(monkeypatch)
    before_snap = json.dumps(sess.cutlist.to_dict(), sort_keys=True)

    r = client.post("/api/clips/render", json={
        "clips": [{"start": 5.0, "end": 15.0}, {"start": 20.0, "end": 30.0}],
        # клиент «врёт» — сервер обязан прибить chapters/metadata к false
        "render_opts": {"chapters": True, "metadata": True, "subtitles": False}})
    assert r.status_code == 200
    assert r.json() == {"ok": True, "count": 2}
    assert sess.task["name"] == "render_clips"
    _wait_done(sess)
    assert sess.task["error"] is None and sess.task["done"] is True

    res = sess.task["results"]["clips"]
    assert len(res) == 2 and all(x["ok"] for x in res)
    # имена <stem>_clip01/_clip02
    assert [c["base"].name for c in calls] == ["fake_clip01", "fake_clip02"]
    assert [x["filename"] for x in res] == ["fake_clip01", "fake_clip02"]
    # chapters/metadata принудительно false на сервере
    for c in calls:
        assert c["cfg"].chapters.enabled is False
        assert c["cfg"].metadata.enabled is False
    # F8: фейды краёв клипа — ЯВНЫЙ параметр из cfg.clips.edge_fade
    # (дефолт 0.025); обычный рендер таким параметром не пользуется.
    for c in calls:
        assert c["edge_fade"] == pytest.approx(0.025)
    # катлист клипа = живые вырезы + 2 граничных REMOVE (§2.4)
    ov = calls[0]["override"]
    by_id = {seg.id: seg for seg in ov.segments}
    assert (by_id["clipA0"].start, by_id["clipA0"].end) == (0.0, 5.0)
    assert (by_id["clipB0"].start, by_id["clipB0"].end) == (15.0, 40.0)
    assert by_id["clipA0"].action == ACTION_REMOVE and by_id["clipA0"].enabled
    assert "p1" in by_id                       # живой вырез скопирован внутрь
    ov2 = {seg.id for seg in calls[1]["override"].segments}
    assert {"clipA1", "clipB1", "p1"} <= ov2
    # сессия не мутирована
    assert json.dumps(sess.cutlist.to_dict(), sort_keys=True) == before_snap
    assert len(sess.cutlist.segments) == 1


def test_render_custom_filename_and_full_range_no_boundary_cuts(
        client, monkeypatch, tmp_path):
    sess = FakeSession(tmp_path)
    _install(monkeypatch, sess)
    calls = _patch_pipeline(monkeypatch)
    r = client.post("/api/clips/render", json={
        "clips": [{"start": 0.0, "end": 40.0, "filename": "мой_шорт"}]})
    assert r.status_code == 200
    _wait_done(sess)
    assert calls[0]["base"].name == "мой_шорт"
    # клип на весь файл — граничные REMOVE не добавляются
    assert [seg.id for seg in calls[0]["override"].segments] == []


def test_render_edge_fade_zero_in_config_passes_zero(client, monkeypatch,
                                                     tmp_path):
    # F8: cfg.clips.edge_fade=0 → выкл — пайплайн получает 0.0 (нет фильтра).
    sess = FakeSession(tmp_path)
    sess.cfg.clips.edge_fade = 0.0
    _install(monkeypatch, sess)
    calls = _patch_pipeline(monkeypatch)
    r = client.post("/api/clips/render", json={
        "clips": [{"start": 5.0, "end": 15.0}]})
    assert r.status_code == 200
    _wait_done(sess)
    assert calls[0]["edge_fade"] == 0.0


# --- 5. упавший клип не валит остальные ------------------------------------------
def test_render_first_clip_fails_second_still_renders(client, monkeypatch,
                                                      tmp_path):
    sess = FakeSession(tmp_path)
    _install(monkeypatch, sess)
    calls = _patch_pipeline(monkeypatch, behaviors=[RuntimeError("boom")])
    r = client.post("/api/clips/render", json={
        "clips": [{"start": 5.0, "end": 15.0}, {"start": 20.0, "end": 30.0}]})
    assert r.status_code == 200
    _wait_done(sess)
    assert sess.task["error"] is None and sess.task["done"] is True
    res = sess.task["results"]["clips"]
    assert len(calls) == 2 and len(res) == 2
    assert res[0]["ok"] is False and res[0]["error"] == "boom"
    assert res[1]["ok"] is True and res[1]["mp4"].endswith("fake_clip02.mp4")


# --- 6. cancel между клипами: цикл остановлен, частичные результаты целы ---------
def test_cancel_between_clips_stops_loop_keeps_partial(client, monkeypatch,
                                                       tmp_path):
    sess = FakeSession(tmp_path)
    _install(monkeypatch, sess)
    started, release = threading.Event(), threading.Event()

    def slow_first(s, on_progress, on_stage):
        started.set()
        assert release.wait(5.0)

    calls = _patch_pipeline(monkeypatch, behaviors=[slow_first])
    r = client.post("/api/clips/render", json={
        "clips": [{"start": 5.0, "end": 15.0}, {"start": 20.0, "end": 30.0}]})
    assert r.status_code == 200
    assert started.wait(5.0)
    # клип 1 «рендерится» — отменяем; цикл должен прерваться ДО клипа 2
    assert client.post("/api/cancel").json() == {"ok": True}
    release.set()
    _wait_done(sess)
    assert len(calls) == 1                       # второй клип не запускался
    res = sess.task["results"]["clips"]
    assert len(res) == 1 and res[0]["ok"] is True   # частичный результат цел
    assert sess.task["error"] == "cancelled"        # чистая отмена, не «exit 1»
    assert sess.task["done"] is False


# --- 7. прогресс/стадии: stage «Клип i/N…», percent (i+f)/N монотонный -----------
def test_render_progress_and_stage_per_clip(client, monkeypatch, tmp_path):
    sess = FakeSession(tmp_path)
    _install(monkeypatch, sess)
    percents: list[float] = []

    def step(s, on_progress, on_stage):
        for f in (0.5, 1.0):
            on_progress(f)
            percents.append(s.task["percent"])
        on_stage("Рендер видео…")
        percents.append(("stage", s.task["stage"]))

    calls = _patch_pipeline(monkeypatch, behaviors=[step, step])
    client.post("/api/clips/render", json={
        "clips": [{"start": 5.0, "end": 15.0}, {"start": 20.0, "end": 30.0}]})
    _wait_done(sess)
    assert [c["stage_at_entry"] for c in calls] == [
        "Клип 1/2: рендер…", "Клип 2/2: рендер…"]
    nums = [p for p in percents if not isinstance(p, tuple)]
    assert nums == [25.0, 50.0, 75.0, 100.0]        # (i+f)/N × 100
    assert nums == sorted(nums)                      # монотонный
    stages = [p[1] for p in percents if isinstance(p, tuple)]
    assert stages == ["Клип 1/2: Рендер видео…", "Клип 2/2: Рендер видео…"]


# --- валидация тела render --------------------------------------------------------
@pytest.mark.parametrize("body", [
    {},                                              # нет clips
    {"clips": []},                                   # пустой список
    {"clips": "не список"},
    {"clips": ["не объект"]},
    {"clips": [{"start": "пять", "end": 15}]},       # не числа
    {"clips": [{"start": None, "end": 15}]},
    {"clips": [{"start": 15.0, "end": 5.0}]},        # пустой диапазон
    {"clips": [{"start": 5.0, "end": 15.0, "filename": 42}]},
    # NaN/±Infinity валидны для json.loads (сервер парсит их в float), но без
    # isfinite-проверки проходят сквозь клампы (max(0, nan)→0, min(dur, nan)→dur)
    # → 200 и клип на весь файл. httpx не сериализует их через json=, поэтому
    # сырые тела — str (тест шлёт их как content с Content-Type: application/json).
    '{"clips": [{"start": NaN, "end": NaN}]}',
    '{"clips": [{"start": NaN, "end": 15.0}]}',
    '{"clips": [{"start": 5.0, "end": NaN}]}',
    '{"clips": [{"start": -Infinity, "end": Infinity}]}',
    '{"clips": [{"start": 5.0, "end": Infinity}]}',
])
def test_render_bad_body_400(client, monkeypatch, tmp_path, body):
    sess = FakeSession(tmp_path)
    _install(monkeypatch, sess)
    _patch_pipeline(monkeypatch)
    if isinstance(body, str):                        # сырое тело (NaN/Infinity)
        r = client.post("/api/clips/render", content=body,
                        headers={"Content-Type": "application/json"})
    else:
        r = client.post("/api/clips/render", json=body)
    assert r.status_code == 400
    assert sess.task["name"] is None                 # задача не создана


def test_render_requires_cutlist_and_transcript(client, monkeypatch, tmp_path):
    sess = FakeSession(tmp_path)
    sess.cutlist = None
    _install(monkeypatch, sess)
    body = {"clips": [{"start": 1.0, "end": 20.0}]}
    assert client.post("/api/clips/render", json=body).status_code == 409
    sess.cutlist = CutList(source="fake.mp4", duration=40.0, segments=[])
    sess.transcript = None
    assert client.post("/api/clips/render", json=body).status_code == 409


def test_render_busy_task_409(client, monkeypatch, tmp_path):
    sess = FakeSession(tmp_path)
    sess.task["running"] = True
    _install(monkeypatch, sess)
    r = client.post("/api/clips/render",
                    json={"clips": [{"start": 1.0, "end": 20.0}]})
    assert r.status_code == 409


def test_render_sets_last_out_dir(client, monkeypatch, tmp_path):
    sess = FakeSession(tmp_path)
    _install(monkeypatch, sess)
    _patch_pipeline(monkeypatch)
    custom = tmp_path / "shorts_out"
    client.post("/api/clips/render", json={
        "clips": [{"start": 5.0, "end": 15.0}],
        "render_opts": {"out_dir": str(custom)}})
    _wait_done(sess)
    assert Path(sess.last_out_dir) == custom.resolve()
    assert custom.is_dir()                           # создан для /api/output


# --- CSRF: новые роуты под общим гардом /api/* -----------------------------------
def test_clips_endpoints_csrf_guarded(client, monkeypatch, tmp_path):
    sess = FakeSession(tmp_path, llm=MockLLM([]))
    _install(monkeypatch, sess)
    evil = {"Origin": "http://evil.example"}
    assert client.post("/api/clips/suggest", headers=evil).status_code == 403
    r = client.post("/api/clips/render", headers=evil,
                    json={"clips": [{"start": 1.0, "end": 20.0}]})
    assert r.status_code == 403
    assert sess.task["name"] is None


# === F7: POST /api/clips/save — правка границ кандидата из UI ====================
def _saved_file(sess) -> dict:
    return json.loads((sess.out_dir / "fake.clips.json")
                      .read_text(encoding="utf-8"))


def test_save_happy_path_recomputes_and_marks_edited(client, monkeypatch,
                                                     tmp_path):
    # enabled-вырез [10,12] внутри нового диапазона → dur_eff = raw − 2;
    # disabled-вырез [20,25] НЕ считается.
    cuts = [CutSegment(id="p1", start=10.0, end=12.0, type=TYPE_PAUSE,
                       action=ACTION_REMOVE, enabled=True),
            CutSegment(id="p2", start=20.0, end=25.0, type=TYPE_PAUSE,
                       action=ACTION_REMOVE, enabled=False)]
    sess = FakeSession(tmp_path, cuts=cuts)
    _install(monkeypatch, sess)
    serve._save_clips_json(sess, [_cand("c01", 5.0, 35.0),
                                  _cand("c02", 36.0, 39.0)])

    r = client.post("/api/clips/save",
                    json={"id": "c01", "start": 6.0, "end": 36.0})
    assert r.status_code == 200
    j = r.json()
    assert j["ok"] is True
    c = j["clip"]
    # пересчитанный dur_eff возвращается в ответе (карточка обновляет цифру)
    assert (c["start"], c["end"]) == (6.0, 36.0)
    assert c["dur_raw"] == pytest.approx(30.0)
    assert c["dur_eff"] == pytest.approx(28.0)
    assert c["edited"] is True
    # авто-границы запомнены при первой правке (для «сбросить к авто»)
    assert (c["auto_start"], c["auto_end"]) == (5.0, 35.0)

    # файл обновлён атомарно, .tmp не остался
    data = _saved_file(sess)
    assert not (sess.out_dir / "fake.clips.json.tmp").exists()
    by_id = {x["id"]: x for x in data["clips"]}
    assert by_id["c01"] == c
    # сосед не тронут
    assert (by_id["c02"]["start"], by_id["c02"]["end"]) == (36.0, 39.0)
    assert "edited" not in by_id["c02"]


def test_save_keeps_top_level_and_stays_valid_for_get(client, monkeypatch,
                                                      tmp_path):
    sess = FakeSession(tmp_path)
    _install(monkeypatch, sess)
    serve._save_clips_json(sess, [_cand()])
    before = _saved_file(sess)
    assert client.post("/api/clips/save", json={
        "id": "c01", "start": 7.0, "end": 30.0}).status_code == 200
    data = _saved_file(sess)
    # hash-валидация кэша НЕ сломана: топ-уровень файла бит-в-бит прежний
    for key in ("version", "hash", "generated_at", "model", "rank_source"):
        assert data[key] == before[key]
    # GET /api/clips видит правленный файл как свежий (не stale) с новыми границами
    j = client.get("/api/clips").json()
    assert j["stale"] is False
    assert j["clips"][0]["start"] == 7.0 and j["clips"][0]["end"] == 30.0
    assert j["clips"][0]["edited"] is True


def test_save_second_edit_keeps_original_auto_bounds(client, monkeypatch,
                                                     tmp_path):
    sess = FakeSession(tmp_path)
    _install(monkeypatch, sess)
    serve._save_clips_json(sess, [_cand("c01", 5.0, 35.0)])
    client.post("/api/clips/save", json={"id": "c01", "start": 6.0, "end": 36.0})
    r = client.post("/api/clips/save", json={"id": "c01", "start": 8.0, "end": 30.0})
    c = r.json()["clip"]
    assert (c["start"], c["end"]) == (8.0, 30.0)
    assert (c["auto_start"], c["auto_end"]) == (5.0, 35.0)   # не перетёрты


def test_save_reset_restores_auto_bounds(client, monkeypatch, tmp_path):
    cuts = [CutSegment(id="p1", start=10.0, end=12.0, type=TYPE_PAUSE,
                       action=ACTION_REMOVE, enabled=True)]
    sess = FakeSession(tmp_path, cuts=cuts)
    _install(monkeypatch, sess)
    serve._save_clips_json(sess, [_cand("c01", 5.0, 35.0)])
    client.post("/api/clips/save", json={"id": "c01", "start": 6.0, "end": 36.0})
    r = client.post("/api/clips/save", json={"id": "c01", "reset": True})
    assert r.status_code == 200
    c = r.json()["clip"]
    assert (c["start"], c["end"]) == (5.0, 35.0)
    assert c["edited"] is False
    assert c["dur_raw"] == pytest.approx(30.0)
    assert c["dur_eff"] == pytest.approx(28.0)   # пересчитан по текущему катлисту
    assert _saved_file(sess)["clips"][0]["edited"] is False


def test_save_reset_unedited_is_idempotent_noop(client, monkeypatch, tmp_path):
    sess = FakeSession(tmp_path)
    _install(monkeypatch, sess)
    serve._save_clips_json(sess, [_cand("c01", 5.0, 35.0)])
    r = client.post("/api/clips/save", json={"id": "c01", "reset": True})
    assert r.status_code == 200
    c = r.json()["clip"]
    assert (c["start"], c["end"]) == (5.0, 35.0)
    assert c["edited"] is False


@pytest.mark.parametrize("body", [
    {},                                               # нет id
    {"id": 42, "start": 5.0, "end": 15.0},            # id не строка
    {"id": "c01"},                                    # нет start/end
    {"id": "c01", "start": "пять", "end": 15.0},
    {"id": "c01", "start": None, "end": 15.0},
    {"id": "c01", "start": 5.0, "end": 9.9},          # короче 5с
    {"id": "c01", "start": 15.0, "end": 5.0},         # перевёрнутый диапазон
    {"id": "c01", "start": -1.0, "end": 15.0},        # за левую границу
    {"id": "c01", "start": 5.0, "end": 41.0},         # за конец ролика (40с)
    '{"id": "c01", "start": NaN, "end": 15.0}',       # NaN сквозь json.loads
    '{"id": "c01", "start": 5.0, "end": Infinity}',
])
def test_save_validation_400(client, monkeypatch, tmp_path, body):
    sess = FakeSession(tmp_path)
    _install(monkeypatch, sess)
    serve._save_clips_json(sess, [_cand("c01", 5.0, 35.0)])
    if isinstance(body, str):
        r = client.post("/api/clips/save", content=body,
                        headers={"Content-Type": "application/json"})
    else:
        r = client.post("/api/clips/save", json=body)
    assert r.status_code == 400
    # файл не изменён
    c = _saved_file(sess)["clips"][0]
    assert (c["start"], c["end"]) == (5.0, 35.0)
    assert "edited" not in c


def test_save_longer_than_90s_rejected(client, monkeypatch, tmp_path):
    sess = FakeSession(tmp_path, duration=200.0)
    _install(monkeypatch, sess)
    serve._save_clips_json(sess, [_cand("c01", 5.0, 35.0)])
    r = client.post("/api/clips/save",
                    json={"id": "c01", "start": 5.0, "end": 96.0})
    assert r.status_code == 400
    # ровно 90с — можно (мягкий предел 60с — забота фронта)
    r = client.post("/api/clips/save",
                    json={"id": "c01", "start": 5.0, "end": 95.0})
    assert r.status_code == 200
    assert r.json()["clip"]["dur_raw"] == pytest.approx(90.0)


def test_save_unknown_id_404(client, monkeypatch, tmp_path):
    sess = FakeSession(tmp_path)
    _install(monkeypatch, sess)
    serve._save_clips_json(sess, [_cand("c01")])
    r = client.post("/api/clips/save",
                    json={"id": "чужой", "start": 5.0, "end": 15.0})
    assert r.status_code == 404


def test_save_no_clips_file_404(client, monkeypatch, tmp_path):
    sess = FakeSession(tmp_path)
    _install(monkeypatch, sess)
    r = client.post("/api/clips/save",
                    json={"id": "c01", "start": 5.0, "end": 15.0})
    assert r.status_code == 404


def test_save_stale_hash_409(client, monkeypatch, tmp_path):
    sess = FakeSession(tmp_path)
    _install(monkeypatch, sess)
    serve._save_clips_json(sess, [_cand()])
    sess.audio_hash = "ff" * 20            # «другое» видео с тем же именем
    r = client.post("/api/clips/save",
                    json={"id": "c01", "start": 5.0, "end": 15.0})
    assert r.status_code == 409


def test_save_busy_task_409(client, monkeypatch, tmp_path):
    sess = FakeSession(tmp_path)
    sess.task["running"] = True
    _install(monkeypatch, sess)
    serve._save_clips_json(sess, [_cand()])
    r = client.post("/api/clips/save",
                    json={"id": "c01", "start": 5.0, "end": 15.0})
    assert r.status_code == 409


def test_save_no_session_409(client):
    r = client.post("/api/clips/save",
                    json={"id": "c01", "start": 5.0, "end": 15.0})
    assert r.status_code == 409


def test_save_write_failure_500_keeps_original_file(client, monkeypatch,
                                                    tmp_path):
    sess = FakeSession(tmp_path)
    _install(monkeypatch, sess)
    serve._save_clips_json(sess, [_cand("c01", 5.0, 35.0)])

    real_replace = serve.os.replace

    def boom(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(serve.os, "replace", boom)
    r = client.post("/api/clips/save",
                    json={"id": "c01", "start": 6.0, "end": 36.0})
    assert r.status_code == 500
    monkeypatch.setattr(serve.os, "replace", real_replace)
    # atomic: оригинальный файл цел и валиден
    c = _saved_file(sess)["clips"][0]
    assert (c["start"], c["end"]) == (5.0, 35.0)
    assert client.get("/api/clips").json()["stale"] is False


def test_save_csrf_guarded(client, monkeypatch, tmp_path):
    sess = FakeSession(tmp_path)
    _install(monkeypatch, sess)
    serve._save_clips_json(sess, [_cand()])
    r = client.post("/api/clips/save", headers={"Origin": "http://evil.example"},
                    json={"id": "c01", "start": 6.0, "end": 36.0})
    assert r.status_code == 403
    c = _saved_file(sess)["clips"][0]
    assert (c["start"], c["end"]) == (5.0, 35.0)   # файл не тронут
