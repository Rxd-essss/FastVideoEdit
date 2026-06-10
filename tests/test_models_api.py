"""P2-#5 — swappable local models (Whisper recognition + LLM) from the UI.

Covers the parts that don't need ffmpeg/whisper/GPU/Ollama:
  * OllamaClient.list_models — parses /api/tags, graceful [] on network failure.
  * models.json persistence: atomic write, corrupt/missing tolerance, and the
    startup _apply_saved_models(cfg) gate (whitelist + non-empty checks).
  * the REST surface via FastAPI TestClient:
      - GET  /api/models  (presets, whitelist, graceful LLM snapshot when off)
      - POST /api/models  (whisper switch + whitelist 400; llm switch + persist;
                           empty-body 400; SESSION.llm rebuild)
      - POST /api/transcribe one-shot {model} override validation (400 on a
        non-whitelisted name) WITHOUT actually transcribing.
      - /api/state exposes whisper_model / llm_model / transcript_model.

Ollama is never contacted: available()/list_models()/has_model() are
monkeypatched, so the tests are deterministic and offline.
"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import serve                                       # noqa: E402
from vpipe.config import LlmCfg, load_config       # noqa: E402
from vpipe.llm import OllamaClient                 # noqa: E402


# --- OllamaClient.list_models -----------------------------------------------
def test_list_models_parses_tags(monkeypatch):
    c = OllamaClient(LlmCfg(model="qwen3:8b"))
    monkeypatch.setattr(c, "_get", lambda path, timeout=5.0: {
        "models": [{"name": "qwen3:8b"}, {"name": "llama3:8b"}, {"name": ""}]})
    # empty names are filtered out; order preserved
    assert c.list_models() == ["qwen3:8b", "llama3:8b"]


def test_list_models_graceful_when_off(monkeypatch):
    c = OllamaClient(LlmCfg(model="qwen3:8b"))

    def boom(path, timeout=5.0):
        raise OSError("connection refused")

    monkeypatch.setattr(c, "_get", boom)
    assert c.list_models() == []          # never raises -> empty list


def test_list_models_missing_models_key(monkeypatch):
    c = OllamaClient(LlmCfg(model="qwen3:8b"))
    monkeypatch.setattr(c, "_get", lambda path, timeout=5.0: {})
    assert c.list_models() == []


# --- fixtures ----------------------------------------------------------------
@pytest.fixture()
def cfg(tmp_path):
    c = load_config("config.yaml")
    c.paths.cache_dir = str(tmp_path / "cache")
    c.paths.out_dir = str(tmp_path / "out")
    return c


@pytest.fixture()
def client(cfg, monkeypatch):
    """TestClient with APP wired, no session, Ollama stubbed OFF by default."""
    monkeypatch.setitem(serve.APP, "cfg", cfg)
    monkeypatch.setitem(serve.APP, "out_dir", cfg.paths.out_dir)
    monkeypatch.setitem(serve.APP, "use_llm", True)
    monkeypatch.setattr(serve, "SESSION", None)
    # Default: Ollama unreachable (available -> False). Individual tests override.
    monkeypatch.setattr(OllamaClient, "available", lambda self: False)
    monkeypatch.setattr(OllamaClient, "list_models", lambda self: [])
    return TestClient(serve.app)


# --- models.json persistence -------------------------------------------------
def test_write_read_models_roundtrip(client):
    serve._write_models("medium", "llama3:8b")
    data = serve._read_models()
    assert data == {"whisper": "medium", "llm": "llama3:8b"}
    p = Path(serve.APP["cfg"].paths.cache_dir) / "models.json"
    assert p.exists()
    assert json.loads(p.read_text(encoding="utf-8")) == {"whisper": "medium", "llm": "llama3:8b"}


def test_read_models_corrupt_defaults_empty(client):
    p = Path(serve.APP["cfg"].paths.cache_dir)
    p.mkdir(parents=True, exist_ok=True)
    (p / "models.json").write_text("{ not json", encoding="utf-8")
    assert serve._read_models() == {}


def test_read_models_missing_defaults_empty(client):
    assert serve._read_models() == {}          # file never written


def test_apply_saved_models_whitelist(client, cfg):
    # valid whisper + llm applied
    serve._write_models("small", "mistral:7b")
    serve._apply_saved_models(cfg)
    assert cfg.transcribe.model == "small"
    assert cfg.llm.model == "mistral:7b"


def test_apply_saved_models_rejects_bad_whisper(client, cfg):
    before = cfg.transcribe.model
    serve._write_models("totally-bogus-model", "")
    serve._apply_saved_models(cfg)
    assert cfg.transcribe.model == before      # bad name ignored
    # empty llm string ignored too (keeps cfg default)


def test_apply_saved_models_missing_file_keeps_defaults(client, cfg):
    before_w, before_l = cfg.transcribe.model, cfg.llm.model
    serve._apply_saved_models(cfg)             # no models.json on disk
    assert cfg.transcribe.model == before_w
    assert cfg.llm.model == before_l


# --- GET /api/models ---------------------------------------------------------
def test_get_models_shape_ollama_off(client):
    r = client.get("/api/models")
    assert r.status_code == 200
    j = r.json()
    # whisper block
    assert j["whisper"]["current"] == "large-v3"
    assert isinstance(j["whisper"]["presets"], list) and j["whisper"]["presets"]
    models = {p["model"] for p in j["whisper"]["presets"]}
    assert models == set(j["whisper"]["allowed"])
    assert "large-v3" in models and "large-v3-turbo" in models
    for p in j["whisper"]["presets"]:
        assert {"key", "label", "model", "hint"} <= set(p)
    assert j["whisper"]["transcript"] is None   # no session
    # llm block — graceful when Ollama off
    assert j["llm"]["current"] == "qwen3:8b"
    assert j["llm"]["available"] is False
    assert j["llm"]["installed"] == []
    assert j["llm"]["ready"] is False


def test_get_models_lists_installed_when_ollama_on(client, monkeypatch):
    monkeypatch.setattr(OllamaClient, "available", lambda self: True)
    monkeypatch.setattr(OllamaClient, "list_models",
                        lambda self: ["qwen3:8b", "llama3:8b"])
    j = client.get("/api/models").json()
    assert j["llm"]["available"] is True
    assert j["llm"]["installed"] == ["qwen3:8b", "llama3:8b"]


def test_get_models_transcript_model_from_session(client, monkeypatch):
    # The «⚙ Модели» modal shows «Текущий транскрипт: <model>» — GET must carry
    # the loaded transcript's model so the UI can flag that a new choice applies
    # only on the NEXT run.
    sess = SimpleNamespace(llm=None, transcript=SimpleNamespace(model="small"))
    monkeypatch.setattr(serve, "SESSION", sess)
    j = client.get("/api/models").json()
    assert j["whisper"]["transcript"] == "small"


# --- POST /api/models : whisper ---------------------------------------------
def test_post_models_whisper_only_no_llm_reason(client):
    # Whisper-only save must NOT report an LLM reason — the modal would otherwise
    # show a spurious «ИИ-модель…» warning on a pure recognition change.
    r = client.post("/api/models", json={"whisper": "small"})
    j = r.json()
    assert j["whisper"] == "small"
    assert j["llm_reason"] is None



def test_post_models_switch_whisper_persists(client):
    r = client.post("/api/models", json={"whisper": "medium"})
    assert r.status_code == 200
    j = r.json()
    assert j["ok"] is True and j["whisper"] == "medium"
    assert serve.APP["cfg"].transcribe.model == "medium"   # shared cfg mutated
    # persisted
    p = Path(serve.APP["cfg"].paths.cache_dir) / "models.json"
    assert json.loads(p.read_text(encoding="utf-8"))["whisper"] == "medium"


def test_post_models_bad_whisper_400(client):
    r = client.post("/api/models", json={"whisper": "gpt-4o"})
    assert r.status_code == 400
    assert serve.APP["cfg"].transcribe.model == "large-v3"  # unchanged


def test_post_models_empty_body_400(client):
    r = client.post("/api/models", json={})
    assert r.status_code == 400


def test_post_models_empty_llm_400(client):
    r = client.post("/api/models", json={"llm": "   "})
    assert r.status_code == 400


# --- POST /api/models : llm + SESSION.llm rebuild ---------------------------
def test_post_models_switch_llm_persists_no_session(client):
    r = client.post("/api/models", json={"llm": "llama3:8b"})
    assert r.status_code == 200
    j = r.json()
    assert j["llm"] == "llama3:8b"
    assert serve.APP["cfg"].llm.model == "llama3:8b"
    p = Path(serve.APP["cfg"].paths.cache_dir) / "models.json"
    assert json.loads(p.read_text(encoding="utf-8"))["llm"] == "llama3:8b"
    # no session -> ready stays False, no crash
    assert j["llm_ready"] is False


def test_post_models_llm_rebuild_ready_when_installed(client, monkeypatch):
    # fake an open session whose llm starts None (idle: set_models guard passes)
    sess = SimpleNamespace(llm=None, transcript=None, task={"running": False})
    monkeypatch.setattr(serve, "SESSION", sess)
    monkeypatch.setattr(OllamaClient, "available", lambda self: True)
    monkeypatch.setattr(OllamaClient, "has_model", lambda self, model=None: True)
    monkeypatch.setattr(OllamaClient, "list_models", lambda self: ["llama3:8b"])
    r = client.post("/api/models", json={"llm": "llama3:8b"})
    j = r.json()
    assert j["llm_ready"] is True
    assert j["llm_reason"] is None
    assert sess.llm is not None                # rebuilt and wired


def test_post_models_llm_model_missing_reason(client, monkeypatch):
    sess = SimpleNamespace(llm=object(), transcript=None,   # had a working llm
                           task={"running": False})
    monkeypatch.setattr(serve, "SESSION", sess)
    monkeypatch.setattr(OllamaClient, "available", lambda self: True)
    monkeypatch.setattr(OllamaClient, "has_model", lambda self, model=None: False)
    monkeypatch.setattr(OllamaClient, "list_models", lambda self: ["qwen3:8b"])
    r = client.post("/api/models", json={"llm": "not-pulled:7b"})
    j = r.json()
    assert j["llm_ready"] is False
    assert j["llm_reason"] == "model_missing"   # UI -> «ollama pull not-pulled:7b»
    assert sess.llm is None                      # zeroed gracefully


def test_post_models_llm_ollama_off_reason(client, monkeypatch):
    sess = SimpleNamespace(llm=object(), transcript=None,
                           task={"running": False})
    monkeypatch.setattr(serve, "SESSION", sess)
    monkeypatch.setattr(OllamaClient, "available", lambda self: False)
    monkeypatch.setattr(OllamaClient, "has_model", lambda self, model=None: False)
    r = client.post("/api/models", json={"llm": "qwen3:8b"})
    j = r.json()
    assert j["llm_ready"] is False
    assert j["llm_reason"] == "ollama_off"
    assert sess.llm is None


# --- POST /api/transcribe one-shot override ---------------------------------
def test_transcribe_override_bad_model_400(client, monkeypatch):
    # a fake session that passes the has_audio guard; the bad model must 400
    # BEFORE any task starts (so we never touch ffmpeg/whisper).
    sess = SimpleNamespace(
        media=SimpleNamespace(has_audio=True),
        task={"running": False},
    )
    monkeypatch.setattr(serve, "SESSION", sess)
    monkeypatch.setattr(serve, "_queue_running", False)
    r = client.post("/api/transcribe", json={"model": "whisper-tiny-bogus"})
    assert r.status_code == 400


# --- /api/state model fields (no-session path is fine for shape) ------------
def test_state_exposes_model_defaults(client, monkeypatch):
    # craft a minimal session-like object for /api/state's S() path
    media = SimpleNamespace(duration=10.0, fps=25.0, width=1920, height=1080)
    cfg = serve.APP["cfg"]
    sess = SimpleNamespace(
        inp=Path("clip.mp4"), media=media, audio_hash="a" * 12,
        transcript=None, cutlist=None, llm=None, cfg=cfg,
        out_dir=Path(cfg.paths.out_dir),
        task={"name": None, "running": False},
    )
    Path(cfg.paths.out_dir).mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(serve, "SESSION", sess)
    j = client.get("/api/state").json()
    assert j["defaults"]["whisper_model"] == cfg.transcribe.model
    assert j["defaults"]["llm_model"] == cfg.llm.model
    assert j["transcript_model"] is None
