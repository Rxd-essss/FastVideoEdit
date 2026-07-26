"""Perf #93 — /api/upload offloads the blocking f.write to a thread. Correctness
regression: the streamed bytes must land on disk intact and in order (the anyio
offload must not reorder/corrupt the stream). No ffmpeg.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import anyio
import serve
from fastapi.testclient import TestClient


def _client(tmp_path, monkeypatch):
    monkeypatch.setattr(serve, "SESSION", None)
    monkeypatch.setattr(serve, "_queue_running", False)
    cfg = SimpleNamespace(paths=SimpleNamespace(work_dir=str(tmp_path)))
    monkeypatch.setitem(serve.APP, "cfg", cfg)
    return TestClient(serve.app)


def test_upload_streams_bytes_intact(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    body = bytes(range(256)) * 800                # multi-chunk-ish payload
    r = client.post("/api/upload?name=clip.mp4", content=body)
    assert r.status_code == 200
    dest = Path(r.json()["path"])
    assert dest.read_bytes() == body              # intact + in order


def test_upload_offloads_each_write_to_thread(tmp_path, monkeypatch):
    calls = {"n": 0}
    orig = anyio.to_thread.run_sync

    async def counting(func, *a, **k):
        calls["n"] += 1
        return await orig(func, *a, **k)

    monkeypatch.setattr(serve.anyio.to_thread, "run_sync", counting)
    client = _client(tmp_path, monkeypatch)
    r = client.post("/api/upload?name=clip.mp4", content=b"hello world")
    assert r.status_code == 200
    assert calls["n"] >= 1                         # write went through the threadpool
