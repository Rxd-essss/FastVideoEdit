# -*- coding: utf-8 -*-
"""Wave-4a #76: watch-folder must NOT promote a size-stable-but-still-locked file.

A shared-read open() probe keeps a still-locked candidate pending for a free
retry next scan, instead of promoting it into a job that errors on probe and is
never re-enqueued. Fail-before: the file was promoted on the second stable scan
regardless of the lock; pass-after: it stays pending until open() succeeds.
"""
from __future__ import annotations

import builtins
import os
from pathlib import Path

import pytest

import serve


def _mk(path: Path, size: int = 100, mtime: float = 1000.0) -> Path:
    path.write_bytes(b"x" * size)
    os.utime(path, (mtime, mtime))
    return path


def _key(path: Path) -> str:
    return os.path.normcase(str(path.resolve()))


@pytest.fixture()
def in_dir(tmp_path):
    d = tmp_path / "inbox"
    d.mkdir()
    return d


def test_locked_stable_file_stays_pending_then_promotes(in_dir, monkeypatch):
    f = _mk(in_dir / "rec.mp4")
    registry, pending = {}, {}

    # scan 1: candidate, not promoted
    assert serve.scan_once(in_dir, registry, pending) == []
    assert _key(f) in pending

    # scan 2: size stable BUT an exclusive writer lock makes open() raise ->
    # must stay pending, NOT enter registry / new_files.
    real_open = builtins.open

    def locked_open(file, *a, **k):
        if Path(file).name == "rec.mp4":
            raise PermissionError("still locked")
        return real_open(file, *a, **k)

    monkeypatch.setattr(builtins, "open", locked_open)
    assert serve.scan_once(in_dir, registry, pending) == []
    assert _key(f) in pending
    assert _key(f) not in registry

    # scan 3: lock released -> promoted exactly once
    monkeypatch.setattr(builtins, "open", real_open)
    got = serve.scan_once(in_dir, registry, pending)
    assert [p.name for p in got] == ["rec.mp4"]
    assert _key(f) in registry and _key(f) not in pending
