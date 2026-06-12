# -*- coding: utf-8 -*-
"""A7 — backend reliability: /api/health + the ConnectionReset log filter.

Covers:
  * GET /api/health schema: ok / ffmpeg{found,path} / ffprobe{found} /
    ollama{found} / whisper_model_cached.
  * ffmpeg/ffprobe missing (resolve_bin raises FFmpegError) -> found:false,
    ok:false — and the endpoint still answers 200, never 500.
  * everything present -> found:true with the resolved path, ok:true.
  * unexpected explosions in every probe -> still 200 (dirt-proof).
  * _whisper_model_cached against a mocked HF cache layout (no network).
  * _ConnectionResetFilter: keeps ordinary errors, drops ConnectionResetError
    records both by exc_info and by message text; installer wires both loggers.

No ffmpeg / Ollama / network is ever touched: resolve_bin and
OllamaClient.available are monkeypatched.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import serve                                            # noqa: E402
from vpipe.config import load_config                    # noqa: E402
from vpipe.ffmpeg_utils import FFmpegError              # noqa: E402
from vpipe.llm import OllamaClient                      # noqa: E402

# Captured at import time, BEFORE any monkeypatching, so a test can restore the
# genuine helper after the client fixture stubbed it.
_REAL_WHISPER_CACHED = serve._whisper_model_cached


# --- fixtures ------------------------------------------------------------------
@pytest.fixture()
def cfg(tmp_path):
    c = load_config("config.yaml")
    c.paths.cache_dir = str(tmp_path / "cache")
    c.paths.out_dir = str(tmp_path / "out")
    return c


@pytest.fixture()
def client(cfg, monkeypatch):
    """TestClient with APP wired, no session, all external probes stubbed OFF."""
    monkeypatch.setitem(serve.APP, "cfg", cfg)
    monkeypatch.setitem(serve.APP, "out_dir", cfg.paths.out_dir)
    monkeypatch.setitem(serve.APP, "use_llm", True)
    monkeypatch.setattr(serve, "SESSION", None)
    monkeypatch.setattr(OllamaClient, "available",
                        lambda self, timeout=5.0: False)
    monkeypatch.setattr(serve, "_whisper_model_cached", lambda model: False)
    return TestClient(serve.app)


def _no_bin(configured, name):
    raise FFmpegError(f"Could not find '{name}'.")


# --- /api/health: schema + missing binaries --------------------------------------
def test_health_missing_ffmpeg_is_200_not_ok(client, monkeypatch):
    monkeypatch.setattr(serve.ffmpeg_utils, "resolve_bin", _no_bin)
    r = client.get("/api/health")
    assert r.status_code == 200
    j = r.json()
    # full schema, exact keys
    assert set(j) == {"ok", "ffmpeg", "ffprobe", "ollama", "whisper_model_cached"}
    assert j["ok"] is False
    assert j["ffmpeg"] == {"found": False, "path": None}
    assert j["ffprobe"] == {"found": False}
    assert j["ollama"] == {"found": False}
    assert j["whisper_model_cached"] is False


def test_health_everything_present(client, monkeypatch):
    monkeypatch.setattr(serve.ffmpeg_utils, "resolve_bin",
                        lambda configured, name: f"C:\\fake\\bin\\{name}.exe")
    monkeypatch.setattr(OllamaClient, "available",
                        lambda self, timeout=5.0: True)
    monkeypatch.setattr(serve, "_whisper_model_cached", lambda model: True)
    r = client.get("/api/health")
    assert r.status_code == 200
    j = r.json()
    assert j["ok"] is True
    assert j["ffmpeg"] == {"found": True, "path": "C:\\fake\\bin\\ffmpeg.exe"}
    assert j["ffprobe"] == {"found": True}
    assert j["ollama"] == {"found": True}
    assert j["whisper_model_cached"] is True


def test_health_ok_requires_both_ffmpeg_and_ffprobe(client, monkeypatch):
    def only_ffmpeg(configured, name):
        if name == "ffmpeg":
            return "C:\\fake\\ffmpeg.exe"
        raise FFmpegError("no ffprobe")

    monkeypatch.setattr(serve.ffmpeg_utils, "resolve_bin", only_ffmpeg)
    j = client.get("/api/health").json()
    assert j["ffmpeg"]["found"] is True
    assert j["ffprobe"]["found"] is False
    assert j["ok"] is False                  # ok = ffmpeg AND ffprobe


def test_health_never_500_even_when_probes_explode(client, monkeypatch):
    """Unexpected (non-FFmpegError) explosions must not surface as 500."""
    def boom(*a, **k):
        raise RuntimeError("totally unexpected")

    monkeypatch.setattr(serve.ffmpeg_utils, "resolve_bin", boom)
    monkeypatch.setattr(OllamaClient, "available", boom)
    # Put the REAL _whisper_model_cached back (the client fixture stubbed it)
    # and blow up the transcribe helper underneath — it must swallow the error.
    monkeypatch.setattr(serve, "_whisper_model_cached", _REAL_WHISPER_CACHED)
    monkeypatch.setattr(serve.transcribe_mod, "_hf_repo_id", boom)
    r = client.get("/api/health")
    assert r.status_code == 200
    j = r.json()
    assert j["ok"] is False
    assert j["ffmpeg"] == {"found": False, "path": None}
    assert j["ollama"] == {"found": False}
    assert j["whisper_model_cached"] is False


def test_health_no_cfg_still_answers(client, monkeypatch):
    monkeypatch.setitem(serve.APP, "cfg", None)
    monkeypatch.setattr(serve.ffmpeg_utils, "resolve_bin", _no_bin)
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["ok"] is False


# --- _whisper_model_cached: real helper against a mocked HF cache ----------------
def _fake_cache(tmp_path, repo_id, complete=True):
    repo = tmp_path / ("models--" + repo_id.replace("/", "--"))
    snap = repo / "snapshots" / "abc"
    snap.mkdir(parents=True)
    if complete:
        (snap / "model.bin").write_bytes(b"\x00" * 8)
    return repo


def test_whisper_model_cached_true_on_complete_snapshot(tmp_path, monkeypatch):
    monkeypatch.setattr(
        serve.transcribe_mod, "_hf_model_dir",
        lambda repo_id, cache_root=None:
            tmp_path / ("models--" + repo_id.replace("/", "--")))
    _fake_cache(tmp_path, "Systran/faster-whisper-small")
    assert serve._whisper_model_cached("small") is True


def test_whisper_model_cached_false_when_absent_or_unknown(tmp_path, monkeypatch):
    monkeypatch.setattr(
        serve.transcribe_mod, "_hf_model_dir",
        lambda repo_id, cache_root=None:
            tmp_path / ("models--" + repo_id.replace("/", "--")))
    assert serve._whisper_model_cached("small") is False      # nothing on disk
    _fake_cache(tmp_path, "Systran/faster-whisper-medium", complete=False)
    assert serve._whisper_model_cached("medium") is False     # partial download
    assert serve._whisper_model_cached("no-such-size") is False  # unknown size


# --- _ConnectionResetFilter -------------------------------------------------------
def _record(msg, args=None, exc_info=None, name="uvicorn.error"):
    return logging.LogRecord(name=name, level=logging.ERROR, pathname=__file__,
                             lineno=1, msg=msg, args=args, exc_info=exc_info)


def _exc_info(exc):
    try:
        raise exc
    except type(exc):
        return sys.exc_info()


def test_filter_keeps_ordinary_errors():
    flt = serve._ConnectionResetFilter()
    assert flt.filter(_record("Exception in ASGI application")) is True
    assert flt.filter(_record("boom %s", args=("x",))) is True
    # A different exception type in exc_info is NOT muted.
    rec = _record("Exception in callback",
                  exc_info=_exc_info(ValueError("nope")))
    assert flt.filter(rec) is True
    # Other WinError codes / other reset-ish wording stay visible too.
    assert flt.filter(_record("OSError [WinError 10038] not a socket")) is True


def test_filter_drops_connection_reset_by_exc_info():
    flt = serve._ConnectionResetFilter()
    rec = _record(
        "Exception in callback _ProactorBasePipeTransport._call_connection_lost",
        exc_info=_exc_info(
            ConnectionResetError(10054, "Удалённый хост разорвал соединение")))
    assert flt.filter(rec) is False


def test_filter_drops_connection_reset_by_message_text():
    flt = serve._ConnectionResetFilter()
    assert flt.filter(
        _record("ConnectionResetError: [WinError 10054] ...")) is False
    assert flt.filter(
        _record("socket send returned [WinError 10054]")) is False


def test_filter_survives_unformattable_record():
    flt = serve._ConnectionResetFilter()
    # args don't match the format string -> getMessage raises -> keep the record.
    assert flt.filter(_record("%d %d", args=("only-one",))) is True


def test_installer_wires_both_loggers():
    before = {n: list(logging.getLogger(n).filters)
              for n in ("uvicorn.error", "asyncio")}
    serve._install_connection_reset_filter()
    try:
        for n in ("uvicorn.error", "asyncio"):
            assert any(isinstance(f, serve._ConnectionResetFilter)
                       for f in logging.getLogger(n).filters)
    finally:
        for n, flts in before.items():       # don't leak filters into other tests
            logging.getLogger(n).filters = flts
