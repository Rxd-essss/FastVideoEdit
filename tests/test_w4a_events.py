# -*- coding: utf-8 -*-
"""Wave-4a #77: GET /api/events must not KeyError('cancelled') on a session that
never started a task.

Two guards: (1) the Session ctor task dict now carries 'cancelled' (aligned with
start_task's shape); (2) the SSE _payload serializes defensively with .get() so
even a legacy 7-key task dict can't abort the stream mid-flight.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import serve                                              # noqa: E402
from vpipe.config import load_config                      # noqa: E402


def _patch_probe(monkeypatch, audio_hash="EVHASH", duration=20.0):
    monkeypatch.setattr(serve, "hash_input", lambda *a, **k: audio_hash)
    monkeypatch.setattr(serve, "probe_media",
                        lambda *a, **k: SimpleNamespace(
                            duration=duration, has_audio=True,
                            width=1920, height=1080, fps=30.0))


def test_ctor_task_dict_has_cancelled_key(monkeypatch, tmp_path):
    _patch_probe(monkeypatch)
    cfg = load_config("config.yaml")
    cfg.paths.cache_dir = str(tmp_path / "cache")
    cfg.paths.work_dir = str(tmp_path / "work")
    video = tmp_path / "v.mp4"
    video.write_bytes(b"x")
    s = serve.Session(str(video), cfg, str(tmp_path / "out"), False)
    assert s.task["cancelled"] is False        # aligned with start_task's shape


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setattr(serve, "_queue_running", False)
    return TestClient(serve.app)


def test_events_before_any_task_no_keyerror(client, monkeypatch):
    # A session whose task dict is the LEGACY 7-key shape (no 'cancelled'):
    # _payload must serialize it without KeyError thanks to the .get() softening.
    legacy = SimpleNamespace(task={"name": None, "running": False,
                                   "percent": 0.0, "stage": "", "error": None,
                                   "done": False, "results": None})
    monkeypatch.setattr(serve, "SESSION", legacy)
    r = client.get("/api/events")
    assert r.status_code == 200
    lines = [ln for ln in r.text.splitlines() if ln.startswith("data: ")]
    assert lines
    data = json.loads(lines[0][len("data: "):])
    assert "cancelled" in data and data["cancelled"] is None
