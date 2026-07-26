"""Locks Session.start_task's worker branches and the /api/events SSE stream.

start_task's worker (serve.py) and the /api/events terminal-snapshot flush are
the UI's only progress/done/error channel, yet nothing exercised them. These
tests drive the worker synchronously (Thread patched to run inline) and stream
/api/events via TestClient, asserting the terminal frame the UI depends on.
All fully mocked — no ffmpeg/GPU/network.
"""
import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import serve


def _run_worker(fake, fn, monkeypatch):
    """Drive start_task's worker synchronously: patch Thread to run the target
    inline (SimpleNamespace.start == the worker closure)."""
    monkeypatch.setattr(serve, "_queue_running", False)
    monkeypatch.setattr(
        serve.threading, "Thread",
        lambda target, daemon=None: SimpleNamespace(start=target))
    serve.Session.start_task(fake, "render", fn)


# 1) genuine success -> percent 100, done True, no error
def test_start_task_success(monkeypatch):
    fake = SimpleNamespace(task={"running": False})
    _run_worker(fake, lambda: None, monkeypatch)
    assert fake.task["done"] is True
    assert fake.task["percent"] == 100.0
    assert fake.task["error"] is None
    assert fake.task["running"] is False


# 2) genuine failure, NOT cancelled -> the real message is surfaced verbatim
def test_start_task_reports_real_error(monkeypatch):
    fake = SimpleNamespace(task={"running": False})

    def fn():
        raise RuntimeError("ffmpeg exit 1: no such filter")

    _run_worker(fake, fn, monkeypatch)
    assert fake.task["error"] == "ffmpeg exit 1: no such filter"
    assert fake.task["running"] is False
    assert fake.task["done"] is False


# 3) DOCUMENTED masking: a real failure AFTER a cancel flag is set is reported as
#    'cancelled', hiding the true error. xfail(strict) locks the bug and flips
#    green the moment the worker is fixed to distinguish a clean cancel from a
#    real post-cancel crash.
@pytest.mark.xfail(strict=True,
                   reason="worker masks a genuine post-cancel error as "
                          "'cancelled'; fix should surface the real crash")
def test_post_cancel_real_error_not_masked(monkeypatch):
    fake = SimpleNamespace(task={"running": False})

    def fn():
        fake.task["cancelled"] = True          # user cancel flips the flag...
        raise RuntimeError("disk full")        # ...but a REAL crash follows

    _run_worker(fake, fn, monkeypatch)
    assert "disk full" in (fake.task["error"] or "")


# 4) SSE terminal-snapshot flush: an already-finished task yields exactly one
#    terminal frame carrying done/results, then the stream closes (no hang).
def test_events_flushes_terminal_snapshot(monkeypatch):
    fake = SimpleNamespace(task={
        "name": "render", "running": False, "percent": 100.0, "stage": "",
        "error": None, "done": True, "results": {"mp4": "o.mp4"},
        "cancelled": False})
    monkeypatch.setattr(serve, "SESSION", fake)
    client = TestClient(serve.app)
    frames = []
    with client.stream("GET", "/api/events") as r:
        assert r.status_code == 200
        for line in r.iter_lines():
            if line.startswith("data: "):
                frames.append(json.loads(line[6:]))
            if frames:
                break
    assert frames
    assert frames[-1]["done"] is True
    assert frames[-1]["results"] == {"mp4": "o.mp4"}
    assert frames[-1]["error"] is None


# 5) SSE reports a terminal ERROR state too (done False, error set).
def test_events_flushes_terminal_error(monkeypatch):
    fake = SimpleNamespace(task={
        "name": "render", "running": False, "percent": 12.0, "stage": "encode",
        "error": "boom", "done": False, "results": None, "cancelled": False})
    monkeypatch.setattr(serve, "SESSION", fake)
    client = TestClient(serve.app)
    frames = []
    with client.stream("GET", "/api/events") as r:
        for line in r.iter_lines():
            if line.startswith("data: "):
                frames.append(json.loads(line[6:]))
            if frames:
                break
    assert frames
    assert frames[-1]["running"] is False
    assert frames[-1]["error"] == "boom"
