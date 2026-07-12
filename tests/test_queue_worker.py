"""Batch-queue worker coverage: serve._queue_process_one + serve._queue_worker.

The heavy pipeline stages (Session ctor, detect, render) are faked so NO
ffmpeg/whisper/GPU runs. Every module global the worker reads or writes
(QUEUE, _queue_running, _queue_cancel, _save_queue, APP) is isolated per-test
via the autouse fixture, so the suite never writes the real ./cache/queue.json
and never leaves the editor<->queue GPU flag stuck (the exact wedge these tests
guard against). The worker loop runs to exhaustion synchronously with fakes, so
no thread/join is needed — we call _queue_worker() directly (not the daemon
spawner _start_queue_worker).
"""
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

import serve


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    """Isolate every module global the worker touches (no real cache/GPU)."""
    monkeypatch.setitem(serve.APP, "cfg", object())
    monkeypatch.setitem(serve.APP, "use_llm", False)
    monkeypatch.setattr(serve, "QUEUE", [])
    monkeypatch.setattr(serve, "_queue_running", False)
    monkeypatch.setattr(serve, "_queue_cancel", threading.Event())
    monkeypatch.setattr(serve, "_save_queue", lambda: None)


def _fake_session(monkeypatch, *, transcript=True, fresh=True,
                  ctor_exc=None, detect_calls=None):
    """Replace serve.Session with a throwaway namespace (no media probe/GPU)."""
    def _ctor(path, cfg, out_dir, use_llm):
        if ctor_exc is not None:
            raise ctor_exc
        ls = SimpleNamespace(
            path=path, out_dir=out_dir, cfg=cfg,
            media=SimpleNamespace(has_audio=True, duration=10.0,
                                  width=1920, height=1080, fps=30.0),
            ff=None, inp=SimpleNamespace(stem="clip"),
            transcript=(object() if transcript else None),
            audio_hash="h", cache_dir="c", work_dir="w",
            _ctor_fresh_detect=fresh)

        def _detect():
            if detect_calls is not None:
                detect_calls.append(1)

        ls._detect = _detect
        return ls

    monkeypatch.setattr(serve, "Session", _ctor)


def _patch_pipeline(monkeypatch, *, raise_in_render=None, on_render=None):
    """Fake the render plumbing so no ffmpeg runs."""
    monkeypatch.setattr(
        serve, "_resolve_render_opts",
        lambda s, opts: (getattr(s, "cfg", None), None, None,
                         Path("o"), Path("o/clip")))

    def _rrp(*a, **k):
        if on_render is not None:
            on_render()
        if raise_in_render is not None:
            raise raise_in_render
        return {"mp4": "out.mp4"}

    monkeypatch.setattr(serve, "_run_render_pipeline", _rrp)


# --- _queue_process_one (direct) --------------------------------------------
def test_process_one_success(monkeypatch):
    _fake_session(monkeypatch, transcript=True, fresh=True)
    _patch_pipeline(monkeypatch)
    job = serve.QueueJob(id="j", path="clip.mp4", out_dir="/o")
    serve._queue_process_one(job)
    assert job.status == "done"
    assert job.result == {"mp4": "out.mp4"}
    assert job.percent == 100.0


def test_process_one_skips_detect_when_fresh(monkeypatch):
    calls = []
    _fake_session(monkeypatch, transcript=True, fresh=True, detect_calls=calls)
    _patch_pipeline(monkeypatch)
    job = serve.QueueJob(id="a", path="clip.mp4", out_dir="/o")
    serve._queue_process_one(job)
    assert calls == []                     # fresh ctor detect -> no re-detect
    assert job.status == "done"


def test_process_one_redetects_when_not_fresh(monkeypatch):
    calls = []
    _fake_session(monkeypatch, transcript=True, fresh=False, detect_calls=calls)
    _patch_pipeline(monkeypatch)
    job = serve.QueueJob(id="a", path="clip.mp4", out_dir="/o")
    serve._queue_process_one(job)
    assert calls == [1]                    # stale on-disk cutlist -> re-detect
    assert job.status == "done"


def test_process_one_cancel_at_stage_boundary(monkeypatch):
    _fake_session(monkeypatch)
    _patch_pipeline(monkeypatch)
    ev = threading.Event()
    ev.set()
    monkeypatch.setattr(serve, "_queue_cancel", ev)
    job = serve.QueueJob(id="a", path="clip.mp4", out_dir="/o")
    with pytest.raises(RuntimeError, match="Очередь остановлена"):
        serve._queue_process_one(job)


# --- _queue_worker (loop) ----------------------------------------------------
def test_worker_happy_path_and_persists(monkeypatch):
    _fake_session(monkeypatch)
    _patch_pipeline(monkeypatch)
    saves = []
    monkeypatch.setattr(
        serve, "_save_queue",
        lambda: saves.append([(j.id, j.status) for j in serve.QUEUE]))
    j = serve.QueueJob(id="a", path="clip.mp4", out_dir="/o", status="pending")
    monkeypatch.setattr(serve, "QUEUE", [j])
    monkeypatch.setattr(serve, "_queue_running", True)
    serve._queue_worker()
    assert j.status == "done" and j.result == {"mp4": "out.mp4"}
    assert serve._queue_running is False
    # persistence ordering: pending->running snapshotted before work, done after
    assert ("a", "running") in saves[0]
    assert ("a", "done") in saves[-1]


def test_worker_marks_error_releases_flag_and_continues(monkeypatch):
    _fake_session(monkeypatch)
    _patch_pipeline(monkeypatch, raise_in_render=RuntimeError("boom"))
    saves = []
    monkeypatch.setattr(
        serve, "_save_queue",
        lambda: saves.append([(j.id, j.status) for j in serve.QUEUE]))
    j1 = serve.QueueJob(id="a", path="clip.mp4", out_dir="/o", status="pending")
    j2 = serve.QueueJob(id="b", path="clip.mp4", out_dir="/o", status="pending")
    monkeypatch.setattr(serve, "QUEUE", [j1, j2])
    monkeypatch.setattr(serve, "_queue_running", True)
    serve._queue_worker()
    assert j1.status == "error" and j1.error == "boom"
    assert j2.status == "error" and j2.error == "boom"   # next job still runs
    assert serve._queue_running is False                 # GPU flag released
    assert any(("a", "running") in snap for snap in saves)


def test_worker_ctor_failure_becomes_job_error(monkeypatch):
    _fake_session(monkeypatch, ctor_exc=ValueError("corrupt"))
    _patch_pipeline(monkeypatch)
    j = serve.QueueJob(id="a", path="bad.mp4", out_dir="/o", status="pending")
    monkeypatch.setattr(serve, "QUEUE", [j])
    monkeypatch.setattr(serve, "_queue_running", True)
    serve._queue_worker()
    assert j.status == "error"
    assert j.error is not None and "corrupt" in j.error
    assert serve._queue_running is False                 # worker survived, flag freed


def test_worker_maps_cancel_to_stopped_message(monkeypatch):
    ev = threading.Event()
    monkeypatch.setattr(serve, "_queue_cancel", ev)
    _fake_session(monkeypatch)

    def _cancel_then_raise():
        ev.set()                                         # stop requested mid-render
        raise RuntimeError("boom")

    _patch_pipeline(monkeypatch, on_render=_cancel_then_raise)
    j = serve.QueueJob(id="a", path="clip.mp4", out_dir="/o", status="pending")
    monkeypatch.setattr(serve, "QUEUE", [j])
    monkeypatch.setattr(serve, "_queue_running", True)
    serve._queue_worker()
    # cancel flag set -> error mapped to the Russian stop message, not str(exc)
    assert j.status == "error" and j.error == "Очередь остановлена"
    assert serve._queue_running is False
