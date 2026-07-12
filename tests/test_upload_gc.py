"""Perf #92 — work/_uploads GC janitor. _sweep_uploads deletes only genuinely
stale uploads: files older than keep_days that are neither the open session
input nor referenced by any queue job; *.part scraps and recent files are never
touched. No ffmpeg; module globals wired via monkeypatch (like the queue tests).
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from types import SimpleNamespace

import serve


def _cfg(work: Path):
    return SimpleNamespace(paths=SimpleNamespace(
        work_dir=str(work), cache_dir=str(work / "cache"),
        out_dir=str(work / "out")))


def test_sweep_uploads_keeps_referenced_and_recent(tmp_path, monkeypatch):
    work = tmp_path / "work"
    updir = work / "_uploads"
    updir.mkdir(parents=True)
    old_unref = updir / "old.mp4"
    queued = updir / "queued.mp4"
    opened = updir / "opened.mp4"
    recent = updir / "recent.mp4"
    part = updir / "half.part"
    for p in (old_unref, queued, opened, recent, part):
        p.write_bytes(b"x")
    old_t = time.time() - 30 * 86400
    for p in (old_unref, queued, opened, part):   # all "old" except recent
        os.utime(p, (old_t, old_t))

    job = serve.QueueJob(id="1", path=str(queued.resolve()),
                         out_dir=str(tmp_path))
    monkeypatch.setattr(serve, "QUEUE", [job])
    monkeypatch.setattr(serve, "SESSION", SimpleNamespace(inp=opened))

    serve._sweep_uploads(_cfg(work), keep_days=7)

    survivors = {p.name for p in updir.iterdir()}
    # (a) old unreferenced -> deleted; (b) queued, (c) open session, (d) recent,
    # (e) *.part -> all kept.
    assert survivors == {"queued.mp4", "opened.mp4", "recent.mp4", "half.part"}
    assert not old_unref.exists()


def test_sweep_uploads_missing_dir_is_noop(tmp_path, monkeypatch):
    monkeypatch.setattr(serve, "QUEUE", [])
    monkeypatch.setattr(serve, "SESSION", None)
    # No work/_uploads dir at all -> must not raise.
    serve._sweep_uploads(_cfg(tmp_path / "nope"), keep_days=7)
