# -*- coding: utf-8 -*-
"""Wave-1 (queue-stop-guard): POST /api/queue/stop must only fire the
PROCESS-WIDE ffmpeg_utils.cancel_all() when the queue is actually running. A
stale Stop button (queue done, editor mid-render) or an out-of-band call must
NOT murder the editor's ffmpeg with a raw exit-1 dump.
"""
from __future__ import annotations

import sys
import threading
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import serve                                              # noqa: E402
from vpipe import ffmpeg_utils                            # noqa: E402


@pytest.fixture()
def client():
    return TestClient(serve.app)


def test_queue_stop_idle_does_not_cancel_ffmpeg(client, monkeypatch):
    calls = []
    monkeypatch.setattr(ffmpeg_utils, "cancel_all", lambda: calls.append(1))
    monkeypatch.setattr(serve, "_queue_running", False)
    monkeypatch.setattr(serve, "_queue_cancel", threading.Event())
    r = client.post("/api/queue/stop")
    assert r.status_code == 200
    assert calls == []                     # idle queue -> editor render untouched


def test_queue_stop_running_cancels(client, monkeypatch):
    calls = []
    monkeypatch.setattr(ffmpeg_utils, "cancel_all", lambda: calls.append(1))
    monkeypatch.setattr(serve, "_queue_running", True)
    monkeypatch.setattr(serve, "_queue_cancel", threading.Event())
    r = client.post("/api/queue/stop")
    assert r.status_code == 200
    assert calls == [1]                    # running queue -> ffmpeg killed as before
