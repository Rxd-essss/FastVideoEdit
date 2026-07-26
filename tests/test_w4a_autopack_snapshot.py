# -*- coding: utf-8 -*-
"""Wave-4a #78: autopack must publish a FROZEN snapshot of results after each
top-level insert.

Previously run() published the LIVE results dict once (by reference) and kept
inserting new top-level keys from the worker thread while GET /api/state
serialized it — an insert during jsonable_encoder's dict iteration raised
RuntimeError('dictionary changed size during iteration') -> 500. Fail-before:
s.task['results'] identity is constant across stages (one live dict); pass-after:
each _set_result republishes dict(results), so the observed identity changes.
"""
from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import serve
from vpipe.clips import ClipCandidate
from vpipe.config import ProfanityLists, load_config
from vpipe.detect.profanity import ProfanityMatcher
from vpipe.models import (ACTION_REMOVE, TYPE_PAUSE, CutList, CutSegment,
                          Segment, Transcript, Word)

HASH = "c9" * 20


def _make_transcript(duration=40.0):
    n = int(duration)
    words = [Word(f"сл{i:02d}", i + 0.1, i + 0.9) for i in range(n)]
    return Transcript(language="ru", duration=duration, model="t",
                      audio_hash=HASH,
                      segments=[Segment(0.0, duration,
                                        " ".join(w.word for w in words), words)])


def _cand(cid, start, end, hook="Сервера обрабатывают"):
    return ClipCandidate(id=cid, seg_start=0, seg_end=3, start=start, end=end,
                         dur_raw=end - start, dur_eff=end - start, score=85,
                         score_window=85, hook_phrase=hook, reason="t")


class FakeSession:
    start_task = serve.Session.start_task
    set_progress = serve.Session.set_progress
    stage = serve.Session.stage

    def __init__(self, tmp_path):
        self.cfg = load_config("config.yaml")
        self.inp = Path("fake.mp4")
        self.media = SimpleNamespace(path="fake.mp4", duration=40.0,
                                     width=1920, height=1080, fps=30.0,
                                     has_audio=True)
        self.ff = None
        self.work_dir = tmp_path / "work"
        self.out_dir = tmp_path / "out"
        self.cache_dir = tmp_path / "cache"
        for d in (self.work_dir, self.out_dir, self.cache_dir):
            d.mkdir(parents=True, exist_ok=True)
        self.last_out_dir = str(self.out_dir.resolve())
        self.audio_hash = HASH
        self.matcher = ProfanityMatcher(ProfanityLists(roots=[], allow=[]))
        self.llm = object()
        self.transcript = _make_transcript()
        self.cutlist = CutList(source="fake.mp4", duration=40.0, segments=[
            CutSegment(id="p1", start=7.0, end=8.0, type=TYPE_PAUSE,
                       action=ACTION_REMOVE, enabled=True)])
        self.task = {"name": None, "running": False, "percent": 0.0,
                     "stage": "", "error": None, "done": False,
                     "results": None, "cancelled": False}


def _wait_done(sess, timeout=5.0):
    t0 = time.time()
    while sess.task["running"]:
        if time.time() - t0 > timeout:
            raise AssertionError(f"task hung: {sess.task}")
        time.sleep(0.01)


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setattr(serve, "_queue_running", False)
    monkeypatch.setattr(serve, "SESSION", None)
    return TestClient(serve.app)


def test_autopack_publishes_fresh_snapshots(client, monkeypatch, tmp_path):
    sess = FakeSession(tmp_path)
    monkeypatch.setattr(serve, "SESSION", sess)

    def fake_suggest(tr, cl, ccfg, lcfg, llm, *, log=print, on_progress=None,
                     on_stage=None):
        return [_cand("c01", 5.0, 35.0), _cand("c02", 36.0, 39.0)]

    monkeypatch.setattr(serve.clips_mod, "suggest", fake_suggest)

    seen_ids = []

    def fake_pipeline(s, cfg, scale_h, fps, out_dir, base, on_progress,
                      on_stage, cutlist_override=None, edge_fade=0.0,
                      sidecar_base=None):
        seen_ids.append(id(s.task["results"]))
        return {"mp4": str(base) + ".mp4", "encoder": "fake",
                "new_duration": 33.0}

    monkeypatch.setattr(serve, "_run_render_pipeline", fake_pipeline)

    r = client.post("/api/autopack", json={"top_k": 2})
    assert r.status_code == 200
    _wait_done(sess)
    assert sess.task["error"] is None
    # main + 2 clips captured; each top-level insert republishes a FRESH dict,
    # so the results identity observed mid-run is NOT constant.
    assert len(seen_ids) >= 3
    assert len(set(seen_ids)) > 1           # fail-before: live dict -> single id
    res = sess.task["results"]
    assert res["ok"] is True
    assert [c["id"] for c in res["clips"]] == ["c01", "c02"]
