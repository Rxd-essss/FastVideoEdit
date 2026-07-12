# -*- coding: utf-8 -*-
"""Wave-1 (serve-shutdown-cancel-all): the server must kill in-flight ffmpeg on
exit so a parent quit/crash can't orphan ffmpeg.exe (holding the GPU + locking
out.part). Covers the ASGI shutdown hook (behavioural) and the atexit
registration (source guard — main() can't be run in-process without mutating the
global app with TrustedHostMiddleware/mounts, which would break other tests).
"""
from __future__ import annotations

import sys
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import serve                                              # noqa: E402
from vpipe import ffmpeg_utils                            # noqa: E402


def test_shutdown_calls_cancel_all(monkeypatch):
    calls = []
    monkeypatch.setattr(ffmpeg_utils, "cancel_all", lambda: calls.append(1))
    # Entering/exiting the TestClient context runs the ASGI startup+shutdown
    # lifespan, which fires @app.on_event("shutdown").
    with TestClient(serve.app):
        pass
    assert calls, "shutdown hook did not call ffmpeg_utils.cancel_all()"


def test_main_registers_atexit_cancel_all():
    src = Path(serve.__file__).read_text(encoding="utf-8")
    assert "import atexit" in src
    assert "atexit.register(ffmpeg_utils.cancel_all)" in src
    # the ASGI-shutdown teardown hook is present too
    assert 'on_event("shutdown")' in src
