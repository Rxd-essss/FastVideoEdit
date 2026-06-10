# -*- coding: utf-8 -*-
"""First-run Whisper model-download notice (vpipe.transcribe).

faster-whisper silently downloads models from HF on first use; we announce it
(in Russian, with an approximate size) and report progress from a watcher
thread. Covered here:

  * size -> HF repo-id resolution (matches faster_whisper.utils._MODELS)
  * cached-model detection against a mocked HF cache directory layout
  * cached model  -> None, not a single extra message (behaviour unchanged)
  * uncached model -> Russian announcement with the approximate size
  * watcher thread reports downloaded GB and stops via the Event
  * offline mode (HF_HUB_OFFLINE=1) -> silent, no download promise
  * any internal error -> swallowed, returns None (dirt-proof)
  * _run_once stops the watcher even when WhisperModel() raises
"""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import vpipe.transcribe as T                        # noqa: E402
from vpipe.config import TranscribeCfg              # noqa: E402


# --- helpers ------------------------------------------------------------------
def _make_cached_repo(cache_root: Path, repo_id: str) -> Path:
    """Build the HF cache layout of a fully downloaded model."""
    repo = cache_root / ("models--" + repo_id.replace("/", "--"))
    snap = repo / "snapshots" / "abc123"
    snap.mkdir(parents=True)
    (snap / "model.bin").write_bytes(b"\x00" * 16)
    return repo


# --- repo-id resolution ---------------------------------------------------------
def test_repo_id_resolution_matches_faster_whisper():
    assert T._hf_repo_id("small") == "Systran/faster-whisper-small"
    assert T._hf_repo_id("large-v3") == "Systran/faster-whisper-large-v3"
    assert (T._hf_repo_id("large-v3-turbo")
            == "mobiuslabsgmbh/faster-whisper-large-v3-turbo")
    # A full repo id passes through untouched (WhisperModel accepts those too).
    assert T._hf_repo_id("Org/custom-model") == "Org/custom-model"
    # Unknown size -> None -> the feature silently steps aside.
    assert T._hf_repo_id("no-such-size") is None


# --- cached-model detection (mocked cache directory) ----------------------------
def test_model_in_cache_detects_complete_snapshot(tmp_path):
    repo = _make_cached_repo(tmp_path, "Systran/faster-whisper-small")
    assert T._model_in_cache(repo) is True


def test_model_in_cache_misses_absent_and_partial(tmp_path):
    # Repo dir does not exist at all -> not cached.
    assert T._model_in_cache(tmp_path / "models--Systran--x") is False
    # Snapshot exists but model.bin has not landed yet -> not cached.
    repo = tmp_path / "models--Systran--faster-whisper-small"
    (repo / "snapshots" / "abc123").mkdir(parents=True)
    (repo / "snapshots" / "abc123" / "config.json").write_text("{}")
    assert T._model_in_cache(repo) is False
    # None (cache path could not be resolved) -> assume cached, stay silent.
    assert T._model_in_cache(None) is True


# --- cached model: zero extra messages ------------------------------------------
def test_cached_model_emits_nothing(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    _make_cached_repo(tmp_path, "Systran/faster-whisper-small")
    msgs: list[str] = []
    stop = T._start_download_watch("small", msgs.append, cache_root=tmp_path)
    assert stop is None
    assert msgs == []                       # behaviour byte-for-byte unchanged


# --- uncached model: Russian announcement with size ------------------------------
def test_uncached_model_announces_in_russian_with_size(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    msgs: list[str] = []
    stop = T._start_download_watch("large-v3", msgs.append,
                                   interval=60.0, cache_root=tmp_path)
    try:
        assert stop is not None
        assert len(msgs) == 1
        m = msgs[0]
        assert "Скачиваю модель Whisper «large-v3»" in m
        assert "~3.1 ГБ" in m
        assert "однократно" in m            # one-time, then offline
    finally:
        if stop:
            stop()


def test_unknown_size_announces_without_size_figure(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    msgs: list[str] = []
    stop = T._start_download_watch("Org/custom-model", msgs.append,
                                   interval=60.0, cache_root=tmp_path)
    try:
        assert stop is not None
        assert "Скачиваю модель Whisper «Org/custom-model»" in msgs[0]
        assert "ГБ," not in msgs[0]         # no made-up size figure
    finally:
        if stop:
            stop()


# --- watcher thread: reports progress, stops on the Event ------------------------
def test_watcher_reports_progress_and_stops(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.setattr(T, "_dir_gb", lambda p: 1.23)
    msgs: list[str] = []
    stop = T._start_download_watch("medium", msgs.append,
                                   interval=0.02, cache_root=tmp_path)
    assert stop is not None
    deadline = time.time() + 2.0
    while time.time() < deadline and len(msgs) < 3:
        time.sleep(0.01)
    progress = [m for m in msgs if "1.2 ГБ" in m]
    assert progress, f"no progress messages in {msgs!r}"
    assert "из ~1.5 ГБ" in progress[0]      # downloaded X of ~total

    stop()                                  # Event set + join
    assert not any(th.name.startswith("whisper-download-watch")
                   for th in threading.enumerate())
    n = len(msgs)
    time.sleep(0.1)
    assert len(msgs) == n                   # nothing logged after stop


def test_watcher_skips_zero_size_and_swallows_errors(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    calls = {"n": 0}

    def flaky(_p):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("transient fs error")
        return 0.0                          # nothing on disk yet -> skip

    monkeypatch.setattr(T, "_dir_gb", flaky)
    msgs: list[str] = []
    stop = T._start_download_watch("small", msgs.append,
                                   interval=0.02, cache_root=tmp_path)
    assert stop is not None
    deadline = time.time() + 2.0
    while time.time() < deadline and calls["n"] < 3:
        time.sleep(0.01)
    stop()
    assert calls["n"] >= 3                  # survived the error, kept ticking
    assert len(msgs) == 1                   # only the announcement, no "0.0 ГБ"


# --- offline mode: never promise a download ---------------------------------------
def test_offline_mode_stays_silent(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    msgs: list[str] = []
    stop = T._start_download_watch("large-v3", msgs.append, cache_root=tmp_path)
    assert stop is None
    assert msgs == []


# --- dirt-proofing: any check error -> silently continue as before ----------------
def test_check_errors_are_swallowed(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.setattr(T, "_hf_repo_id",
                        lambda s: (_ for _ in ()).throw(RuntimeError("boom")))
    msgs: list[str] = []
    stop = T._start_download_watch("small", msgs.append, cache_root=tmp_path)
    assert stop is None
    assert msgs == []

    monkeypatch.undo()
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.setattr(T, "_model_in_cache",
                        lambda d: (_ for _ in ()).throw(OSError("denied")))
    stop = T._start_download_watch("small", msgs.append, cache_root=tmp_path)
    assert stop is None
    assert msgs == []


def test_hf_model_dir_resolves_default_cache():
    # No cache_root -> resolved from huggingface_hub.constants (no network).
    d = T._hf_model_dir("Systran/faster-whisper-small")
    assert d is not None
    assert d.name == "models--Systran--faster-whisper-small"


# --- _run_once: watcher is stopped even when the constructor raises ---------------
def test_run_once_stops_watch_on_constructor_error(monkeypatch):
    import faster_whisper

    stopped = {"n": 0}

    def fake_watch(size, log, **kw):
        def stopper():
            stopped["n"] += 1
        return stopper

    def boom(*a, **k):
        raise RuntimeError("download failed mid-flight")

    monkeypatch.setattr(T, "_start_download_watch", fake_watch)
    monkeypatch.setattr(faster_whisper, "WhisperModel", boom)
    with pytest.raises(RuntimeError, match="mid-flight"):
        T._run_once("x.wav", "small", "cpu", "int8",
                    TranscribeCfg(), 1.0, "h", lambda *a, **k: None)
    assert stopped["n"] == 1                # try/finally released the watcher
