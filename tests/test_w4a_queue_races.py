# -*- coding: utf-8 -*-
"""Wave-4a queue/guard: #75 guard TASK_LOCK + drain re-check, #62 same-stem
output collision, #64 queue/clear statuses filter."""
from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import serve
from vpipe.config import load_config


@pytest.fixture()
def client(tmp_path, monkeypatch):
    cfg = load_config("config.yaml")
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    cfg.paths.cache_dir = str(cache_dir)
    monkeypatch.setitem(serve.APP, "cfg", cfg)
    monkeypatch.setitem(serve.APP, "out_dir", str(out_dir))
    monkeypatch.setitem(serve.APP, "use_llm", False)
    monkeypatch.setattr(serve, "QUEUE", [])
    monkeypatch.setattr(serve, "_queue_running", False)
    monkeypatch.setattr(serve, "SESSION", None)
    return TestClient(serve.app)


# --- #75(a): guard reads flags under TASK_LOCK — outcome unchanged (smoke) ----
def test_guard_no_task_outcome_unchanged(monkeypatch):
    from fastapi import HTTPException
    monkeypatch.setattr(serve, "SESSION", None)
    monkeypatch.setattr(serve, "_queue_running", False)
    assert serve._guard_no_task() is None

    monkeypatch.setattr(serve, "_queue_running", True)
    with pytest.raises(HTTPException) as ei:
        serve._guard_no_task()
    assert ei.value.status_code == 409

    monkeypatch.setattr(serve, "_queue_running", False)
    monkeypatch.setattr(serve, "SESSION", SimpleNamespace(task={"running": True}))
    with pytest.raises(HTTPException) as ei2:
        serve._guard_no_task()
    assert ei2.value.status_code == 409


# --- #75(b): worker drains a job that appears while the last one runs ---------
def test_worker_drains_job_added_mid_run(monkeypatch):
    monkeypatch.setattr(serve, "QUEUE", [])
    monkeypatch.setattr(serve, "_queue_running", True)
    monkeypatch.setattr(serve, "_queue_cancel", threading.Event())
    monkeypatch.setattr(serve, "_save_queue", lambda: None)
    j1 = serve.QueueJob(id="a", path="c.mp4", out_dir="/o", status="pending")
    serve.QUEUE.append(j1)

    def fake_process(job):
        if job.id == "a":                       # queue/add arrives mid-run
            serve.QUEUE.append(serve.QueueJob(id="b", path="c.mp4",
                                              out_dir="/o", status="pending"))
        job.status = "done"

    monkeypatch.setattr(serve, "_queue_process_one", fake_process)
    serve._queue_worker()
    # both drained; the worker did NOT exit leaving a pending job
    assert [j.status for j in serve.QUEUE] == ["done", "done"]
    assert serve._queue_running is False


# --- #62: two same-stem jobs from DIFFERENT videos don't clobber each other ---
def test_same_stem_different_source_no_clobber(tmp_path, monkeypatch):
    out = tmp_path / "out"
    out.mkdir()
    monkeypatch.setattr(serve, "_queue_cancel", threading.Event())
    monkeypatch.setitem(serve.APP, "cfg", object())      # _queue_process_one reads APP["cfg"]
    monkeypatch.setitem(serve.APP, "use_llm", False)     # ... and APP["use_llm"]
    hashes = iter(["aaaaaa11", "bbbbbb22"])

    def _ctor(path, cfg, out_dir, use_llm):
        return SimpleNamespace(
            path=path, out_dir=out_dir, cfg=object(),
            media=SimpleNamespace(has_audio=True), ff=None,
            inp=SimpleNamespace(stem="clip"), transcript=object(),
            audio_hash=next(hashes), cache_dir="c", work_dir="w",
            _ctor_fresh_detect=True, _detect=lambda: None)

    monkeypatch.setattr(serve, "Session", _ctor)
    base = out / "clip"
    monkeypatch.setattr(serve, "_resolve_render_opts",
                        lambda ls, opts: (object(), None, None, out, base))

    def _rrp(s, cfg, scale_h, fps, out_dir, base, on_progress, on_stage, **k):
        serve._with_ext(base, ".mp4").write_bytes(b"x")      # touch the mp4
        return {"mp4": str(serve._with_ext(base, ".mp4"))}

    monkeypatch.setattr(serve, "_run_render_pipeline", _rrp)
    j1 = serve.QueueJob(id="a", path="v1.mp4", out_dir=str(out))
    j2 = serve.QueueJob(id="b", path="v2.mp4", out_dir=str(out))
    serve._queue_process_one(j1)
    serve._queue_process_one(j2)

    mp4s = sorted(p.name for p in out.glob("*.mp4"))
    assert len(mp4s) == 2                       # two distinct outputs, no clobber
    assert "clip.mp4" in mp4s
    assert any(n.startswith("clip-") for n in mp4s)
    assert j2.result.get("renamed", "").startswith("clip-")


# --- #64: queue/clear statuses filter spares pending jobs ---------------------
def _jobs():
    return [
        serve.QueueJob(id="p", path="c.mp4", out_dir="/o", status="pending"),
        serve.QueueJob(id="r", path="c.mp4", out_dir="/o", status="running"),
        serve.QueueJob(id="d", path="c.mp4", out_dir="/o", status="done"),
        serve.QueueJob(id="e", path="c.mp4", out_dir="/o", status="error"),
    ]


def test_queue_clear_statuses_filter_keeps_pending(client, monkeypatch):
    monkeypatch.setattr(serve, "QUEUE", _jobs())
    r = client.post("/api/queue/clear", json={"statuses": ["done", "error"]})
    assert r.status_code == 200 and r.json()["removed"] == 2
    assert [j.id for j in serve.QUEUE] == ["p", "r"]


def test_queue_clear_no_body_still_drops_all_finished(client, monkeypatch):
    monkeypatch.setattr(serve, "QUEUE", _jobs())
    r = client.post("/api/queue/clear")
    assert r.json()["removed"] == 3
    assert [j.id for j in serve.QUEUE] == ["r"]
