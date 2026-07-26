# -*- coding: utf-8 -*-
"""Wave-4a #61: PUT /api/cutlist must validate segments BEFORE CutList.from_dict.

An unguarded NaN/Infinity (json.loads accepts the literals) was re-serialized
into cutlist.json and then bricked every later GET /api/cutlist with a permanent
500 (Starlette JSONResponse uses allow_nan=False); a missing key raised a raw
KeyError-500. This mirrors the /api/clips/render guard: fail-before (garbage got
a 200 / poisoned the file), pass-after (clean per-index 400, no poisoning).
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import serve


class _FakeSession:
    def __init__(self, tmp_path: Path):
        out = tmp_path / "out"
        out.mkdir(parents=True, exist_ok=True)
        self.media = SimpleNamespace(duration=40.0)
        self.inp = Path("fake.mp4")
        self.out_dir = out
        self.cutlist_path = out / "fake.cutlist.json"
        self.audio_hash = "h" * 8
        self.cutlist = None
        self.task = {"name": None, "running": False, "percent": 0.0,
                     "stage": "", "error": None, "done": False,
                     "results": None, "cancelled": False}


@pytest.fixture()
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(serve, "_queue_running", False)
    monkeypatch.setattr(serve, "SESSION", _FakeSession(tmp_path))
    return TestClient(serve.app)


_JSON = {"Content-Type": "application/json"}


def _seg(**over):
    d = {"id": "x", "start": 1.0, "end": 10.0, "type": "manual",
         "action": "remove"}
    d.update(over)
    return d


# --- raw JSON literals json.loads accepts but JSONResponse later rejects ------
@pytest.mark.parametrize("raw", [
    '{"segments":[{"id":"x","start":NaN,"end":10,"type":"manual","action":"remove"}]}',
    '{"segments":[{"id":"x","start":-Infinity,"end":10,"type":"manual","action":"remove"}]}',
    '{"segments":[{"id":"x","start":0,"end":Infinity,"type":"manual","action":"remove"}]}',
])
def test_put_cutlist_rejects_nonfinite(client, raw):
    r = client.put("/api/cutlist", content=raw, headers=_JSON)
    assert r.status_code == 400


@pytest.mark.parametrize("payload", [
    {"segments": [{"start": 1, "end": 10, "type": "manual", "action": "remove"}]},
    {"segments": [{"id": "x", "start": 1, "end": 10, "action": "remove"}]},
    {"segments": [{"id": "x", "start": 10, "end": 5, "type": "manual", "action": "remove"}]},
    {"segments": [{"id": "x", "start": -1, "end": 10, "type": "manual", "action": "remove"}]},
    # start ЗА пределами ролика (не просто хвост end>duration) — по-прежнему 400
    {"segments": [{"id": "x", "start": 41, "end": 100, "type": "manual", "action": "remove"}]},
    {"segments": "notalist"},
])
def test_put_cutlist_rejects_bad_segments(client, payload):
    r = client.put("/api/cutlist", json=payload)
    assert r.status_code == 400


def test_put_cutlist_valid_then_get_ok(client):
    r = client.put("/api/cutlist", json={"segments": [_seg()]})
    assert r.status_code == 200 and r.json()["segments"] == 1
    # No poisoning: the follow-up GET serializes via JSONResponse(allow_nan=False)
    g = client.get("/api/cutlist")
    assert g.status_code == 200


def test_put_cutlist_clamps_detector_tail_past_duration(client):
    """W6 #2: детекторный хвост (end > clamped duration — Whisper видел ПОЛНОЕ
    аудио) КЛАМПИТСЯ к duration, а не 400. fail-before: один такой сегмент
    навсегда ломал автосохранение («границы вне ролика» на каждый PUT)."""
    r = client.put("/api/cutlist",
                   json={"segments": [_seg(start=35.0, end=100.0)]})
    assert r.status_code == 200
    g = client.get("/api/cutlist")
    assert g.status_code == 200
    seg = g.json()["segments"][0]
    assert seg["start"] == 35.0 and seg["end"] == 40.0    # clamped to duration
