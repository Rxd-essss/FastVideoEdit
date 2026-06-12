# -*- coding: utf-8 -*-
"""B5 — параметры детекции: POST /api/detect c overrides + persist.

Покрывает:
  * _sanitize_detect_opts: клампы числовых диапазонов (min_silence 0.3–2.0,
    padding 0–0.3, sensitivity 0–1), 400 на неверные ТИПЫ (strict), тихий
    дроп мусора при чтении файла (strict=False), фильтрация detectors;
  * _apply_detect_opts: без opts -> ТОТ ЖЕ объект cfg (регрессия: дефолтное
    поведение байт-в-байт); с opts -> deep copy, оригинал не мутируется;
    честный маппинг hesitation_sensitivity (s=0.5 == конфиг, монотонность);
  * выключенные детекторы реально не запускаются: run_detection с
    pauses/fillers off не даёт их сегментов, badtakes=false не трогает LLM
    (llm-стаб взрывается на любом обращении к атрибутам);
  * персист detect_ui.json: atomic запись, roundtrip, испорченный/чужой
    файл -> None;
  * REST: POST /api/detect с телом — сохранение + применение; без тела —
    сохранённые; без тела и без файла — НЕизменённый cfg (identity);
    400 на мусор без записи файла и без запуска задачи; пересохранение
    вторым телом; 409 без транскрипта; detect_opts в GET /api/state
    (null до настройки, dict после) в обеих ветках (с сессией и без).

Никакого ffmpeg/whisper/GPU/Ollama: TestClient + фейковая сессия, которая
выполняет start_task синхронно и записывает cfg, который получил _detect.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import serve                                            # noqa: E402
from vpipe.config import FillerLists, ProfanityLists, load_config  # noqa: E402
from vpipe.detect import run_detection                  # noqa: E402
from vpipe.models import Segment, Transcript, Word      # noqa: E402


# --- fixtures -----------------------------------------------------------------
@pytest.fixture()
def cfg(tmp_path):
    c = load_config("config.yaml")
    c.paths.cache_dir = str(tmp_path / "cache")
    c.paths.out_dir = str(tmp_path / "out")
    c.paths.work_dir = str(tmp_path / "work")
    return c


@pytest.fixture()
def client(cfg, monkeypatch):
    monkeypatch.setitem(serve.APP, "cfg", cfg)
    monkeypatch.setitem(serve.APP, "out_dir", cfg.paths.out_dir)
    monkeypatch.setitem(serve.APP, "use_llm", False)
    monkeypatch.setattr(serve, "SESSION", None)
    monkeypatch.setattr(serve, "_queue_running", False)
    return TestClient(serve.app)


class FakeDetectSession:
    """Минимум для /api/detect: синхронный start_task + запись полученного cfg."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.transcript = object()          # «есть транскрипт»
        self.task = {"name": None, "running": False, "percent": 0.0,
                     "stage": "", "error": None, "done": False, "results": None}
        self.detect_cfgs: list = []

    def stage(self, msg):
        pass

    def start_task(self, name, fn):
        fn()                                # синхронно — без потоков в тестах

    def _detect(self, cfg=None):
        self.detect_cfgs.append(cfg if cfg is not None else self.cfg)


@pytest.fixture()
def sess(client, cfg, monkeypatch):
    s = FakeDetectSession(cfg)
    monkeypatch.setattr(serve, "SESSION", s)
    return s


# --- _sanitize_detect_opts ------------------------------------------------------
def test_sanitize_clamps_ranges():
    o = serve._sanitize_detect_opts({
        "pause_min_silence": 0.05,          # ниже пола 0.3
        "pause_padding": 9,                 # выше потолка 0.3
        "hesitation_sensitivity": -3,       # ниже пола 0
    })
    assert o == {"pause_min_silence": 0.3, "pause_padding": 0.3,
                 "hesitation_sensitivity": 0.0}
    o2 = serve._sanitize_detect_opts({
        "pause_min_silence": 5.0, "hesitation_sensitivity": 2})
    assert o2 == {"pause_min_silence": 2.0, "hesitation_sensitivity": 1.0}


def test_sanitize_passthrough_in_range():
    o = serve._sanitize_detect_opts({
        "pause_min_silence": 0.8, "pause_padding": 0.1,
        "hesitation_sensitivity": 0.25,
        "detectors": {"pauses": True, "badtakes": False}})
    assert o == {"pause_min_silence": 0.8, "pause_padding": 0.1,
                 "hesitation_sensitivity": 0.25,
                 "detectors": {"pauses": True, "badtakes": False}}


@pytest.mark.parametrize("raw", [
    {"pause_min_silence": "fast"},
    {"pause_padding": [0.1]},
    {"hesitation_sensitivity": True},       # bool — не число
    {"pause_min_silence": float("nan")},
    {"detectors": "all"},
    {"detectors": {"pauses": "yes"}},
])
def test_sanitize_strict_rejects_bad_types(raw):
    with pytest.raises(HTTPException) as e:
        serve._sanitize_detect_opts(raw)
    assert e.value.status_code == 400


def test_sanitize_lenient_drops_garbage_keeps_good():
    o = serve._sanitize_detect_opts(
        {"pause_min_silence": "fast", "pause_padding": 0.2,
         "detectors": {"pauses": "yes", "fillers": False},
         "unknown_future_key": 1},
        strict=False)
    assert o == {"pause_padding": 0.2, "detectors": {"fillers": False}}


def test_sanitize_ignores_unknown_detector_names():
    o = serve._sanitize_detect_opts({"detectors": {"pauses": False, "lazers": True}})
    assert o == {"detectors": {"pauses": False}}


def test_sanitize_none_values_skipped():
    assert serve._sanitize_detect_opts({"pause_min_silence": None}) == {}


# --- _apply_detect_opts ----------------------------------------------------------
def test_apply_no_opts_returns_same_object(cfg):
    # РЕГРЕССИЯ: без настроек run_detection обязан получить НЕизменённый cfg —
    # буквально тот же объект, никаких копий/мутаций.
    assert serve._apply_detect_opts(cfg, None) is cfg
    assert serve._apply_detect_opts(cfg, {}) is cfg


def test_apply_overrides_deep_copy_and_values(cfg):
    before = cfg.model_dump()
    eff = serve._apply_detect_opts(cfg, {
        "pause_min_silence": 1.2, "pause_padding": 0.25,
        "detectors": {"badtakes": False, "hesitations": False,
                      "profanity": False}})
    assert eff is not cfg
    assert eff.pauses.min_silence == 1.2
    assert eff.pauses.pad_start == 0.25 and eff.pauses.pad_end == 0.25
    assert eff.bad_takes.enabled is False
    assert eff.hesitations.enabled is False
    assert eff.profanity.enabled is False
    assert eff.pauses.enabled is True       # не упомянут — не тронут
    assert cfg.model_dump() == before       # оригинал цел


def test_sensitivity_mapping_honest(cfg):
    h0 = cfg.hesitations
    lo = serve._apply_detect_opts(cfg, {"hesitation_sensitivity": 0.0}).hesitations
    mid = serve._apply_detect_opts(cfg, {"hesitation_sensitivity": 0.5}).hesitations
    hi = serve._apply_detect_opts(cfg, {"hesitation_sensitivity": 1.0}).hesitations
    # s=0.5 — ровно конфиг (честная середина).
    assert mid.min_duration == pytest.approx(h0.min_duration)
    assert mid.vad_threshold == pytest.approx(h0.vad_threshold)
    assert mid.pad_start == pytest.approx(h0.pad_start)
    # Монотонность: 0 «реже режет» (длиннее порог, ниже VAD-порог, шире пады),
    # 1 «агрессивнее» (короче порог, выше VAD-порог, уже пады).
    assert lo.min_duration > h0.min_duration > hi.min_duration
    assert lo.vad_threshold < h0.vad_threshold < hi.vad_threshold
    assert lo.pad_start > h0.pad_start > hi.pad_start
    # Инварианты детектора целы: max_duration не тронут, min < max.
    assert lo.max_duration == hi.max_duration == h0.max_duration
    assert hi.min_duration >= 0.04
    assert lo.min_duration <= h0.max_duration


# --- выключенные детекторы не запускаются ------------------------------------------
class _ExplodingLLM:
    """Любое обращение к атрибуту — провал теста: LLM не должен быть вызван."""

    def __getattr__(self, name):
        raise AssertionError(f"LLM touched ({name}) while badtakes disabled")


def _tr() -> Transcript:
    words = [Word(" раз", 0.0, 0.4), Word(" вот", 0.5, 0.9),
             Word(" два", 3.0, 3.4)]
    seg = Segment(0.0, 3.4, "раз вот два", words)
    return Transcript(language="ru", duration=10.0, model="t",
                      audio_hash="h", segments=[seg])


def test_disabled_detectors_do_not_run(cfg):
    lists = FillerLists(words=["вот"])
    eff = serve._apply_detect_opts(cfg, {"detectors": {
        "pauses": False, "fillers": True, "profanity": False,
        "hesitations": False, "badtakes": False}})
    cl = run_detection(_tr(), eff, lists, ProfanityLists(), source="x",
                       llm=_ExplodingLLM(), log=lambda *_: None)
    assert {s.type for s in cl.segments} == {"filler"}   # ни одной паузы

    eff2 = serve._apply_detect_opts(cfg, {"detectors": {
        "pauses": True, "fillers": False, "profanity": False,
        "hesitations": False, "badtakes": False}})
    cl2 = run_detection(_tr(), eff2, lists, ProfanityLists(), source="x",
                        llm=_ExplodingLLM(), log=lambda *_: None)
    assert {s.type for s in cl2.segments} == {"pause"}   # ни одного филлера


# --- персист detect_ui.json ---------------------------------------------------------
def test_persist_roundtrip(client, cfg):
    opts = {"pause_min_silence": 0.8, "detectors": {"pauses": False}}
    serve._write_detect_opts(opts)
    p = Path(cfg.paths.cache_dir) / "detect_ui.json"
    assert p.exists()
    assert json.loads(p.read_text(encoding="utf-8")) == opts
    assert serve._read_detect_opts() == opts
    assert not p.with_suffix(".json.tmp").exists()       # atomic: tmp подменён


def test_read_missing_and_corrupt_give_none(client, cfg):
    assert serve._read_detect_opts() is None             # файла нет
    p = Path(cfg.paths.cache_dir) / "detect_ui.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("{ not json", encoding="utf-8")
    assert serve._read_detect_opts() is None             # битый JSON
    p.write_text("[1, 2]", encoding="utf-8")
    assert serve._read_detect_opts() is None             # не объект


def test_read_sanitizes_hand_edited_file(client, cfg):
    p = Path(cfg.paths.cache_dir) / "detect_ui.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"pause_min_silence": 99,    # кламп при чтении
                             "pause_padding": "junk",    # тихий дроп
                             "detectors": {"fillers": False}}),
                 encoding="utf-8")
    assert serve._read_detect_opts() == {
        "pause_min_silence": 2.0, "detectors": {"fillers": False}}


# --- POST /api/detect ----------------------------------------------------------------
def test_detect_with_body_saves_and_applies(client, cfg, sess):
    r = client.post("/api/detect", json={
        "pause_min_silence": 1.5, "pause_padding": 0.0,
        "detectors": {"badtakes": False}})
    assert r.status_code == 200
    assert r.json()["detect_opts"]["pause_min_silence"] == 1.5
    # сохранено на диск
    saved = json.loads((Path(cfg.paths.cache_dir) / "detect_ui.json")
                       .read_text(encoding="utf-8"))
    assert saved == {"pause_min_silence": 1.5, "pause_padding": 0.0,
                     "detectors": {"badtakes": False}}
    # применено к КОПИИ cfg, оригинал сессии не мутирован
    eff = sess.detect_cfgs[0]
    assert eff is not sess.cfg
    assert eff.pauses.min_silence == 1.5
    assert eff.pauses.pad_start == 0.0 and eff.pauses.pad_end == 0.0
    assert eff.bad_takes.enabled is False
    assert sess.cfg.pauses.min_silence == 0.6
    assert sess.cfg.bad_takes.enabled is True


def test_detect_no_body_uses_saved(client, cfg, sess):
    client.post("/api/detect", json={"pause_min_silence": 1.1})
    sess.detect_cfgs.clear()
    r = client.post("/api/detect")                       # без тела
    assert r.status_code == 200
    assert r.json()["detect_opts"] == {"pause_min_silence": 1.1}
    assert sess.detect_cfgs[0].pauses.min_silence == 1.1


def test_detect_no_body_no_saved_is_identity(client, sess):
    # РЕГРЕССИЯ: без тела и без detect_ui.json _detect получает ТОТ ЖЕ cfg.
    r = client.post("/api/detect")
    assert r.status_code == 200
    assert r.json()["detect_opts"] is None
    assert sess.detect_cfgs[0] is sess.cfg


def test_detect_body_resaves(client, cfg, sess):
    client.post("/api/detect", json={"pause_min_silence": 1.1})
    client.post("/api/detect", json={"pause_min_silence": 0.7,
                                     "hesitation_sensitivity": 1.0})
    saved = json.loads((Path(cfg.paths.cache_dir) / "detect_ui.json")
                       .read_text(encoding="utf-8"))
    assert saved == {"pause_min_silence": 0.7, "hesitation_sensitivity": 1.0}


def test_detect_bad_body_400_no_save_no_task(client, cfg, sess):
    r = client.post("/api/detect", json={"pause_min_silence": "максимум"})
    assert r.status_code == 400
    assert not (Path(cfg.paths.cache_dir) / "detect_ui.json").exists()
    assert sess.detect_cfgs == []                        # задача не стартовала


def test_detect_requires_transcript(client, sess):
    sess.transcript = None
    assert client.post("/api/detect").status_code == 409


# --- GET /api/state: detect_opts ------------------------------------------------------
def test_state_detect_opts_null_then_dict_no_session(client):
    assert client.get("/api/state").json()["detect_opts"] is None
    serve._write_detect_opts({"pause_padding": 0.05})
    assert client.get("/api/state").json()["detect_opts"] == {"pause_padding": 0.05}


def test_state_detect_opts_with_session(client, cfg, monkeypatch):
    media = SimpleNamespace(duration=10.0, fps=25.0, width=1920, height=1080)
    Path(cfg.paths.out_dir).mkdir(parents=True, exist_ok=True)
    sess = SimpleNamespace(
        inp=Path("clip.mp4"), media=media, audio_hash="a" * 12,
        transcript=None, cutlist=None, llm=None, cfg=cfg,
        out_dir=Path(cfg.paths.out_dir),
        task={"name": None, "running": False})
    monkeypatch.setattr(serve, "SESSION", sess)
    assert client.get("/api/state").json()["detect_opts"] is None
    serve._write_detect_opts({"detectors": {"hesitations": False}})
    assert client.get("/api/state").json()["detect_opts"] == {
        "detectors": {"hesitations": False}}
