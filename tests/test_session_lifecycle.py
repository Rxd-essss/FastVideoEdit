"""Locks Session.__init__ / open_session lifecycle and /api/open, /api/upload.

Every other API test uses a SimpleNamespace stand-in for the session; nothing
constructed a real Session or drove open_session. These tests exercise:
  * fresh open (no cache)          -> transcript/cutlist None, no detect
  * cached transcript, no cutlist  -> transcript loaded, ctor does NOT detect
  * matching on-disk cutlist       -> loaded, no detect
  * foreign/stale cutlist          -> dropped (cross-video guard)
  * _detect                        -> preserves manual cuts, re-sorted by start
  * open_session                   -> launches a detect TASK only when a
                                      transcript exists but no valid cutlist
plus the /api/open and /api/upload validation surfaces.

The ctor's heavy collaborators (FFmpeg, probe_media, hash_input, get_client,
run_detection) are monkeypatched so no ffmpeg/probe/GPU/LLM ever runs.
"""
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import serve
from vpipe.config import Config
from vpipe.models import (ACTION_REMOVE, TYPE_MANUAL, TYPE_PAUSE, CutList,
                          CutSegment, Segment, Transcript, Word)


@pytest.fixture()
def ctor_env(tmp_path, monkeypatch):
    cfg = Config()
    cfg.paths.cache_dir = str(tmp_path / "cache")
    cfg.paths.work_dir = str(tmp_path / "work")
    cfg.paths.out_dir = str(tmp_path / "out")
    monkeypatch.setattr(serve, "FFmpeg", lambda c: SimpleNamespace())
    monkeypatch.setattr(serve, "probe_media",
                        lambda ff, p: SimpleNamespace(duration=100.0, width=1920,
                                                      height=1080, fps=30.0,
                                                      has_audio=True))
    monkeypatch.setattr(serve, "hash_input", lambda p: "deadbeefcafe")
    monkeypatch.setattr(serve, "get_client", lambda c: None)
    vid = tmp_path / "clip.mp4"
    vid.write_bytes(b"\x00")
    return cfg, vid, tmp_path


def _tr():
    words = [Word("раз", 0.0, 0.4), Word("два", 0.5, 0.9)]
    return Transcript(language="ru", duration=100.0, model="t",
                      audio_hash="deadbeefcafe",
                      segments=[Segment(0.0, 0.9, "раз два", words)])


# --- Session.__init__ --------------------------------------------------------

def test_ctor_blank(ctor_env):
    cfg, vid, tmp = ctor_env
    s = serve.Session(str(vid), cfg, str(tmp / "out"), False)
    assert s.transcript is None
    assert s.cutlist is None
    assert s._ctor_fresh_detect is False


def test_ctor_loads_cached_transcript_without_detecting(ctor_env, monkeypatch):
    """A cached transcript with no valid cutlist is loaded, but the ctor must
    NOT run detection (that used to freeze /api/open for minutes)."""
    cfg, vid, tmp = ctor_env
    Path(cfg.paths.cache_dir).mkdir(parents=True, exist_ok=True)
    _tr().save(Path(cfg.paths.cache_dir) / "deadbeefcafe.transcript.json")
    monkeypatch.setattr(serve, "run_detection",
                        lambda *a, **k: pytest.fail("ctor must not detect"))
    s = serve.Session(str(vid), cfg, str(tmp / "out"), False)
    assert s.transcript is not None
    assert s.cutlist is None
    assert s._ctor_fresh_detect is False


def test_ctor_loads_matching_cutlist(ctor_env, monkeypatch):
    cfg, vid, tmp = ctor_env
    out = Path(tmp / "out")
    out.mkdir(parents=True, exist_ok=True)
    CutList(source=str(vid), duration=100.0, audio_hash="deadbeefcafe",
            segments=[CutSegment(id="m0", start=10.0, end=12.0, type=TYPE_MANUAL,
                                 action=ACTION_REMOVE, enabled=True)]
            ).save_json(out / "clip.cutlist.json")
    monkeypatch.setattr(serve, "run_detection",
                        lambda *a, **k: pytest.fail("must not detect"))
    s = serve.Session(str(vid), cfg, str(tmp / "out"), False)
    assert s.cutlist is not None
    assert s._ctor_fresh_detect is False
    assert any(seg.type == TYPE_MANUAL for seg in s.cutlist.segments)


def test_ctor_drops_foreign_cutlist(ctor_env):
    """A same-stem cutlist whose audio_hash belongs to a DIFFERENT video must be
    dropped (self.cutlist stays None) so a later _detect can't merge stale cuts."""
    cfg, vid, tmp = ctor_env
    out = Path(tmp / "out")
    out.mkdir(parents=True, exist_ok=True)
    CutList(source="other.mp4", duration=100.0, audio_hash="not-this-video",
            segments=[CutSegment(id="m0", start=10.0, end=12.0, type=TYPE_MANUAL,
                                 action=ACTION_REMOVE, enabled=True)]
            ).save_json(out / "clip.cutlist.json")
    s = serve.Session(str(vid), cfg, str(tmp / "out"), False)
    assert s.cutlist is None


# --- Session._detect manual-cut preservation ---------------------------------

def test_detect_preserves_manual_cuts(ctor_env, monkeypatch):
    cfg, vid, tmp = ctor_env
    s = serve.Session(str(vid), cfg, str(tmp / "out"), False)
    s.transcript = _tr()
    s.cutlist = CutList(source=str(vid), duration=100.0, segments=[
        CutSegment(id="m0", start=50.0, end=52.0, type=TYPE_MANUAL,
                   action=ACTION_REMOVE, enabled=True)])
    monkeypatch.setattr(serve, "run_detection", lambda *a, **k: CutList(
        source=str(vid), duration=100.0, segments=[
            CutSegment(id="p0", start=5.0, end=6.0, type=TYPE_PAUSE,
                       action=ACTION_REMOVE, enabled=True)]))
    monkeypatch.setattr(serve, "save_txt", lambda *a, **k: None)
    cl = s._detect()
    types = {seg.type for seg in cl.segments}
    assert TYPE_MANUAL in types and TYPE_PAUSE in types   # manual survived
    starts = [seg.start for seg in cl.segments]
    assert starts == sorted(starts)                       # re-sorted by start
    assert s.cutlist is cl                                # session updated


# --- open_session: fresh-detect-on-open --------------------------------------

def test_open_session_launches_detect_when_transcript_but_no_cutlist(monkeypatch):
    started = []
    fake = SimpleNamespace(cutlist=None, transcript=object())
    fake.start_task = lambda name, fn: started.append(name)
    monkeypatch.setattr(serve, "SESSION", None)
    monkeypatch.setattr(serve, "Session", lambda *a, **k: fake)
    monkeypatch.setattr(serve, "APP",
                        {"cfg": Config(), "out_dir": "out", "use_llm": False})
    s = serve.open_session("clip.mp4")
    assert s is fake
    assert started == ["detect"]


def test_open_session_no_detect_when_cutlist_present(monkeypatch):
    started = []
    fake = SimpleNamespace(cutlist=object(), transcript=object())
    fake.start_task = lambda name, fn: started.append(name)
    monkeypatch.setattr(serve, "SESSION", None)
    monkeypatch.setattr(serve, "Session", lambda *a, **k: fake)
    monkeypatch.setattr(serve, "APP",
                        {"cfg": Config(), "out_dir": "out", "use_llm": False})
    serve.open_session("clip.mp4")
    assert started == []


# --- /api/open and /api/upload validation ------------------------------------

@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setattr(serve, "SESSION", None)
    return TestClient(serve.app)


def test_api_open_404_missing_path(client):
    r = client.post("/api/open", json={"path": str(Path("no_such_dir") / "x.mp4")})
    assert r.status_code == 404


def test_api_open_400_non_video(client, tmp_path):
    f = tmp_path / "notes.txt"
    f.write_text("x")
    r = client.post("/api/open", json={"path": str(f)})
    assert r.status_code == 400


def test_api_open_ok_calls_open_session(client, tmp_path, monkeypatch):
    vid = tmp_path / "clip.mp4"
    vid.write_bytes(b"\x00")
    called = {}
    monkeypatch.setattr(serve, "open_session",
                        lambda p: called.setdefault("path", p))
    r = client.post("/api/open", json={"path": str(vid)})
    assert r.status_code == 200
    assert r.json()["filename"] == "clip.mp4"
    assert called["path"].endswith("clip.mp4")


def test_api_upload_415_bad_ext(client):
    r = client.post("/api/upload?name=notes.txt", content=b"hello")
    assert r.status_code == 415


def test_api_upload_ok_writes_file(client, tmp_path, monkeypatch):
    cfg = Config()
    cfg.paths.work_dir = str(tmp_path / "work")
    monkeypatch.setitem(serve.APP, "cfg", cfg)
    r = client.post("/api/upload?name=clip.mp4", content=b"\x00\x01\x02")
    assert r.status_code == 200
    p = Path(r.json()["path"])
    assert p.exists()
    assert p.read_bytes() == b"\x00\x01\x02"
