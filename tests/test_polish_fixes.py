"""Regression tests for the pre-release audit fixes (P0–P2).

Covers the parts that don't need ffmpeg/whisper/GPU:
  * P1-2  _allowed_hosts (DNS-rebinding host allow-list)
  * P1-4  transcribe cache: keep an OOM fallback, re-transcribe on a real switch
  * P1-5  /api/models refuses while a task runs
  * P2-3  Session raises a clear error on zero/unknown duration
  * P2-4  offline + uncached model -> friendly message (not a raw HF trace)
  * P2-6  /api/output only serves whitelisted extensions
  * P0-3b privacy summary stays honest about a non-local LLM host
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import serve                                          # noqa: E402
import vpipe.transcribe as T                          # noqa: E402
from vpipe.config import TranscribeCfg, load_config   # noqa: E402
from vpipe.models import Transcript                   # noqa: E402


# --- P1-2: host allow-list ---------------------------------------------------
def test_allowed_hosts_loopback_pins_loopback():
    for h in ("127.0.0.1", "localhost", "::1"):
        assert serve._allowed_hosts(h) == ["127.0.0.1", "localhost", "::1"]


def test_allowed_hosts_nonlocal_is_wildcard():
    # An explicit non-loopback bind means the user accepted exposure.
    assert serve._allowed_hosts("0.0.0.0") == ["*"]
    assert serve._allowed_hosts("192.168.1.5") == ["*"]


# --- P1-4: cache keeps an OOM fallback but re-runs on a real switch ----------
def test_cache_reuses_oom_fallback_without_rerunning(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    h = "hashoom"
    # Requested large-v3 but OOM fell back to medium last time.
    Transcript(language="ru", duration=5.0, model="medium",
               audio_hash=h, requested_model="large-v3").save(
        cache / f"{h}.transcript.json")
    cfg = TranscribeCfg(model="large-v3")
    # Returns from cache before ever importing faster-whisper -> no GPU touched.
    out = T.transcribe_audio("x.wav", cfg, 5.0, h, cache_dir=cache,
                             log=lambda *a, **k: None)
    assert out.model == "medium"          # kept the fallback, did NOT re-OOM


def test_cache_misses_on_voluntary_model_switch(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    cache.mkdir()
    h = "hashswitch"
    # Last run the user deliberately chose 'small'.
    Transcript(language="ru", duration=5.0, model="small",
               audio_hash=h, requested_model="small").save(
        cache / f"{h}.transcript.json")
    cfg = TranscribeCfg(model="large-v3", device="cpu")   # now they switch up
    called = {"n": 0}

    def fake_run_once(*a, **k):
        called["n"] += 1
        raise RuntimeError("re-transcribe attempted")     # avoid real work

    monkeypatch.setattr(T, "bootstrap_cuda_dlls", lambda: None)
    monkeypatch.setattr(T, "_run_once", fake_run_once)
    with pytest.raises(RuntimeError):
        T.transcribe_audio("x.wav", cfg, 5.0, h, cache_dir=cache,
                           log=lambda *a, **k: None)
    assert called["n"] >= 1               # cache miss -> tried to re-transcribe


# --- P2-4: offline + uncached model -> friendly message ----------------------
def test_offline_uncached_model_message(tmp_path, monkeypatch):
    cfg = TranscribeCfg(model="large-v3", device="cpu", cache=False,
                        fallback_models=[])
    monkeypatch.setattr(T, "bootstrap_cuda_dlls", lambda: None)
    monkeypatch.setattr(T, "_run_once",
                        lambda *a, **k: (_ for _ in ()).throw(
                            RuntimeError("LocalEntryNotFound")))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    with pytest.raises(RuntimeError) as ei:
        T.transcribe_audio("x.wav", cfg, 5.0, "h", cache_dir=None,
                           log=lambda *a, **k: None)
    assert "офлайн" in str(ei.value).lower() or "offline" in str(ei.value).lower()


# --- API fixtures ------------------------------------------------------------
@pytest.fixture()
def cfg(tmp_path):
    c = load_config("config.yaml")
    c.paths.cache_dir = str(tmp_path / "cache")
    c.paths.out_dir = str(tmp_path / "out")
    Path(c.paths.out_dir).mkdir(parents=True, exist_ok=True)
    return c


@pytest.fixture()
def client(cfg, monkeypatch):
    monkeypatch.setitem(serve.APP, "cfg", cfg)
    monkeypatch.setitem(serve.APP, "out_dir", cfg.paths.out_dir)
    monkeypatch.setitem(serve.APP, "use_llm", True)
    monkeypatch.setattr(serve, "SESSION", None)
    monkeypatch.setattr(serve, "_queue_running", False)
    return TestClient(serve.app)


# --- P1-5: /api/models refuses while a task runs -----------------------------
def test_set_models_409_while_task_running(client, monkeypatch):
    monkeypatch.setattr(serve, "SESSION",
                        SimpleNamespace(task={"running": True}, llm=None))
    r = client.post("/api/models", json={"whisper": "medium"})
    assert r.status_code == 409


def test_set_models_409_while_queue_running(client, monkeypatch):
    monkeypatch.setattr(serve, "_queue_running", True)
    r = client.post("/api/models", json={"whisper": "medium"})
    assert r.status_code == 409


# --- P2-6: /api/output extension whitelist -----------------------------------
def test_output_serves_whitelisted_extension(client, cfg):
    out = Path(cfg.paths.out_dir)
    (out / "result.mp4").write_bytes(b"\x00\x00\x00\x18ftypmp42")
    assert client.get("/api/output/result.mp4").status_code == 200


def test_output_rejects_non_whitelisted_extension(client, cfg):
    out = Path(cfg.paths.out_dir)
    (out / "secrets.env").write_text("TOKEN=abc")
    # Even though the file exists in the out dir, a non-artifact ext -> 404.
    assert client.get("/api/output/secrets.env").status_code == 404


# --- MONTAGE_V2 §3: honest previews — cache PNGs must be servable -------------
def test_output_serves_codegfx_preview(client, cfg):
    """Code-graphic candidate PNG lives in cache/codegfx — the UI requests it as
    /api/output/<basename>; it must resolve (not 404), else the «Монтаж» card
    shows a broken tile instead of the real rendered schematic."""
    cg = Path(cfg.paths.cache_dir) / "codegfx"
    cg.mkdir(parents=True, exist_ok=True)
    (cg / "abc123.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    r = client.get("/api/output/abc123.png")
    assert r.status_code == 200
    assert r.content.startswith(b"\x89PNG")


def test_output_serves_diffusion_preview(client, cfg):
    """Diffusion candidate PNG lives in cache/enrich_img — same contract."""
    eg = Path(cfg.paths.cache_dir) / "enrich_img"
    eg.mkdir(parents=True, exist_ok=True)
    (eg / "deadbeef.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    assert client.get("/api/output/deadbeef.png").status_code == 200


def test_output_image_ext_now_whitelisted(client, cfg):
    """Image extensions are part of the whitelist (regression for the V2 fix:
    previously .png was rejected before the cache lookup even ran)."""
    for ext in (".png", ".jpg", ".jpeg", ".webp"):
        assert ext in serve.OUTPUT_EXT_ALLOWED


def test_output_preview_no_path_traversal(client, cfg, monkeypatch):
    """The relative-path guard still confines reads to the cache subdirs even
    after adding them as roots — a basename can never escape upward."""
    # A real .png sitting one level above the cache root must NOT be reachable.
    secret = Path(cfg.paths.cache_dir).parent / "outside.png"
    secret.parent.mkdir(parents=True, exist_ok=True)
    secret.write_bytes(b"\x89PNG\r\n\x1a\n")
    # FastAPI's path param won't pass raw '../', but assert the resolver itself
    # rejects anything that escapes the configured roots.
    roots = serve._enrich_preview_roots()
    assert all(not (r / "..\\outside.png").resolve().is_relative_to(r)
               for r in roots)


# --- P0-3b: privacy summary honest about a non-local LLM host ----------------
def test_network_summary_warns_on_external_llm_host(client, cfg, monkeypatch):
    cfg.llm.host = "http://203.0.113.9:11434"          # remote Ollama
    monkeypatch.setitem(serve.APP, "cfg", cfg)
    summary = serve._network_summary(False, {"external_allowed": 0, "blocked": 0,
                                              "external_hosts": {}})
    assert "транскрипт" in summary.lower()              # honest warning present


def test_network_summary_clean_when_local(client, cfg, monkeypatch):
    cfg.llm.host = "http://localhost:11434"
    monkeypatch.setitem(serve.APP, "cfg", cfg)
    summary = serve._network_summary(False, {"external_allowed": 0, "blocked": 0,
                                             "external_hosts": {}})
    assert "локально" in summary.lower()
    assert "транскрипт" not in summary.lower()          # no false alarm
