"""F3 — batch render queue.

Covers the parts that don't need ffmpeg/whisper/GPU:
  * QueueJob defaults + _make_job_dict serialization.
  * The queue REST surface (add / list / remove / clear) via FastAPI TestClient.
  * _resolve_render_opts: the SHARED render-option logic the editor and the queue
    both use — verifies overrides + identity-resize/fps fast-paths still apply.

The worker itself (_queue_process_one) drives real ffmpeg/whisper, so it is not
exercised here; instead we assert the queue plumbing around it.
"""
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import serve
from vpipe.config import load_config


# --- fixtures ---------------------------------------------------------------
@pytest.fixture()
def client(tmp_path, monkeypatch):
    """A TestClient with APP wired up and a clean, isolated QUEUE."""
    cfg = load_config("config.yaml")
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    # Isolate cache_dir at a tmp path: the queue endpoints (add/remove/clear) now
    # call _save_queue(), which must NOT write queue.json into the real ./cache
    # (that left a phantom job in a live server after running the suite).
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    cfg.paths.cache_dir = str(cache_dir)
    monkeypatch.setitem(serve.APP, "cfg", cfg)
    monkeypatch.setitem(serve.APP, "out_dir", str(out_dir))
    monkeypatch.setitem(serve.APP, "use_llm", False)
    monkeypatch.setattr(serve, "QUEUE", [])
    monkeypatch.setattr(serve, "_queue_running", False)
    return TestClient(serve.app)


@pytest.fixture()
def sample_video(tmp_path):
    """A real-on-disk file with a video extension (content is irrelevant for the
    add/list/remove/clear endpoints — they validate path + suffix only)."""
    p = tmp_path / "clip.mp4"
    p.write_bytes(b"\x00\x00\x00\x18ftypmp42")
    return p


# --- model + serialization --------------------------------------------------
def test_queuejob_defaults():
    j = serve.QueueJob(id="abc", path="/x/clip.mp4", out_dir="/out")
    assert j.status == "pending"
    assert j.percent == 0.0
    assert j.stage == ""
    assert j.result is None
    assert j.error is None
    assert j.render_opts == {}


def test_make_job_dict_exposes_all_fields():
    j = serve.QueueJob(id="id1", path="/dir/My Clip.mp4", out_dir="/out",
                       render_opts={"encoder": "x264"})
    j.status = "running"
    j.percent = 42.345
    j.stage = "Рендер видео…"
    d = serve._make_job_dict(j)
    assert d["id"] == "id1"
    assert d["name"] == "My Clip.mp4"          # basename derived from path
    assert d["path"] == "/dir/My Clip.mp4"
    assert d["out_dir"] == "/out"
    assert d["render_opts"] == {"encoder": "x264"}
    assert d["status"] == "running"
    assert d["percent"] == 42.3                 # rounded to 1 dp
    assert d["stage"] == "Рендер видео…"
    assert d["result"] is None
    assert d["error"] is None


# --- REST surface -----------------------------------------------------------
def test_queue_add_list_remove_clear(client, sample_video):
    # empty to start
    r = client.get("/api/queue")
    assert r.status_code == 200
    body = r.json()
    assert body == {"jobs": [], "running": False}

    # add two jobs
    r1 = client.post("/api/queue/add", json={"path": str(sample_video),
                                             "render_opts": {"encoder": "x264"}})
    assert r1.status_code == 200 and r1.json()["ok"]
    id1 = r1.json()["id"]
    r2 = client.post("/api/queue/add", json={"path": str(sample_video)})
    id2 = r2.json()["id"]
    assert id1 != id2

    jobs = client.get("/api/queue").json()["jobs"]
    assert [j["id"] for j in jobs] == [id1, id2]
    assert jobs[0]["status"] == "pending"
    assert jobs[0]["name"] == "clip.mp4"
    assert jobs[0]["render_opts"] == {"encoder": "x264"}
    # out_dir defaulted to APP['out_dir']
    assert jobs[1]["out_dir"] == serve.APP["out_dir"]

    # remove the first
    rr = client.post("/api/queue/remove", json={"id": id1})
    assert rr.json() == {"ok": True, "removed": 1}
    assert [j["id"] for j in client.get("/api/queue").json()["jobs"]] == [id2]

    # clear the rest
    rc = client.post("/api/queue/clear")
    assert rc.json()["ok"] and rc.json()["removed"] == 1
    assert client.get("/api/queue").json()["jobs"] == []


def test_queue_add_rejects_missing_file(client, tmp_path):
    r = client.post("/api/queue/add", json={"path": str(tmp_path / "nope.mp4")})
    assert r.status_code == 404


def test_queue_add_rejects_non_video(client, tmp_path):
    p = tmp_path / "notes.txt"
    p.write_text("hi", encoding="utf-8")
    r = client.post("/api/queue/add", json={"path": str(p)})
    assert r.status_code == 400


def test_queue_remove_skips_running_job(client, sample_video, monkeypatch):
    # a running job must NOT be removable
    job = serve.QueueJob(id="run1", path=str(sample_video),
                         out_dir=serve.APP["out_dir"], status="running")
    monkeypatch.setattr(serve, "QUEUE", [job])
    r = client.post("/api/queue/remove", json={"id": "run1"})
    assert r.json() == {"ok": True, "removed": 0}
    assert len(serve.QUEUE) == 1


def test_queue_clear_keeps_running_only(client, sample_video, monkeypatch):
    jobs = [
        serve.QueueJob(id="p", path=str(sample_video), out_dir="/o", status="pending"),
        serve.QueueJob(id="r", path=str(sample_video), out_dir="/o", status="running"),
        serve.QueueJob(id="d", path=str(sample_video), out_dir="/o", status="done"),
        serve.QueueJob(id="e", path=str(sample_video), out_dir="/o", status="error"),
    ]
    monkeypatch.setattr(serve, "QUEUE", jobs)
    r = client.post("/api/queue/clear")
    assert r.json()["removed"] == 3
    assert [j.id for j in serve.QUEUE] == ["r"]


def test_queue_start_blocked_while_editor_busy(client, monkeypatch):
    busy = SimpleNamespace(task={"running": True})
    monkeypatch.setattr(serve, "SESSION", busy)
    r = client.post("/api/queue/start")
    assert r.status_code == 409


def test_editor_start_task_blocked_while_queue_running(monkeypatch):
    """The editor's start_task must refuse (clean 409) while the queue runs, so
    they never compete for GPU — and without blocking on TASK_LOCK."""
    from fastapi import HTTPException

    monkeypatch.setattr(serve, "_queue_running", True)
    fake = SimpleNamespace(task={"running": False})
    with pytest.raises(HTTPException) as ei:
        serve.Session.start_task(fake, "render", lambda: None)
    assert ei.value.status_code == 409


# --- shared render-options logic (reused by editor AND queue) ----------------
def _fake_session():
    """Minimal stand-in carrying just what _resolve_render_opts touches."""
    cfg = load_config("config.yaml")
    media = SimpleNamespace(height=1080, fps=30.0)
    return SimpleNamespace(cfg=cfg, media=media, out_dir="/out",
                           inp=SimpleNamespace(stem="clip"))


def test_resolve_render_opts_applies_overrides(tmp_path):
    s = _fake_session()
    s.out_dir = tmp_path / "out"
    opts = {"encoder": "x264", "quality": 20, "audio_bitrate": "256k",
            "censor_method": "pitch", "subtitles": False, "chapters": False,
            "scale_h": 720, "fps": 24, "filename": "result"}
    cfg, scale_h, fps, out_dir, base = serve._resolve_render_opts(s, opts)
    assert cfg.render.encoder == "x264"
    assert cfg.render.x264.crf == 20
    assert cfg.render.nvenc.qp == 20 and cfg.render.nvenc.cq == 20
    assert cfg.render.audio_bitrate == "256k"
    assert cfg.censor.method == "pitch"
    assert cfg.subtitles.enabled is False
    assert cfg.chapters.enabled is False
    assert scale_h == 720
    assert fps == 24.0
    assert base.name == "result"


def test_resolve_render_opts_cut_fade(tmp_path):
    s = _fake_session()
    s.out_dir = tmp_path / "out"
    # UI sends seconds; value is clamped to [0, 0.06].
    cfg, *_ = serve._resolve_render_opts(s, {"cut_fade": 0.02})
    assert cfg.render.cut_fade == 0.02
    cfg, *_ = serve._resolve_render_opts(s, {"cut_fade": 5.0})   # clamped
    assert cfg.render.cut_fade == 0.06
    cfg, *_ = serve._resolve_render_opts(s, {"cut_fade": 0.0})   # hard cuts
    assert cfg.render.cut_fade == 0.0


def test_resolve_render_opts_identity_resize_and_fps_fastpath(tmp_path):
    s = _fake_session()
    s.out_dir = tmp_path / "out"
    # scale_h == source height and fps == source fps -> both dropped to None so
    # the lossless copy fast-path wins (mirrors do_render's old behavior).
    cfg, scale_h, fps, out_dir, base = serve._resolve_render_opts(
        s, {"scale_h": 1080, "fps": 30})
    assert scale_h is None
    assert fps is None
    assert base.name == "clip"   # filename omitted -> source stem


def test_resolve_render_opts_bad_scale_h_raises(tmp_path):
    from fastapi import HTTPException
    s = _fake_session()
    s.out_dir = tmp_path / "out"
    with pytest.raises(HTTPException):
        serve._resolve_render_opts(s, {"scale_h": 99})   # below 144


# --- #43: queue must not re-detect over the user's curated cutlist ------------
def _fake_queue_session(cutlist_source, inp_path):
    import threading
    from pathlib import Path
    from unittest.mock import MagicMock
    return SimpleNamespace(
        transcript=object(),
        cutlist=SimpleNamespace(source=cutlist_source),
        inp=Path(str(inp_path)),
        _ctor_fresh_detect=False,
        _detect=MagicMock(),
        media=SimpleNamespace(has_audio=True),
        percent=0.0, stage="", result=None, status="pending", error=None)


def _wire_queue_process(monkeypatch, fake):
    import threading
    monkeypatch.setattr(serve, "_queue_cancel", threading.Event())
    monkeypatch.setattr(serve, "Session",
                        lambda path, cfg, out, use_llm: fake)
    monkeypatch.setattr(serve, "_resolve_render_opts",
                        lambda ls, opts: (None, None, None, None, None))
    monkeypatch.setattr(serve, "_run_render_pipeline",
                        lambda *a, **k: {"mp4": "x"})


def test_queue_own_cutlist_not_re_detected(client, sample_video, monkeypatch):
    """Катлист ЭТОГО входа (source == inp) — курированный юзером; _detect()
    пропускается, иначе он сохранил бы свежую детекцию поверх ревью."""
    from pathlib import Path
    inp = sample_video
    out_dir = Path(serve.APP["out_dir"])
    cl_path = out_dir / (inp.stem + ".cutlist.json")
    cl_path.write_text('{"curated": true}', encoding="utf-8")
    before = cl_path.read_bytes()
    fake = _fake_queue_session(str(inp), inp)
    _wire_queue_process(monkeypatch, fake)
    serve._queue_process_one(
        serve.QueueJob(id="j1", path=str(inp), out_dir=str(out_dir)))
    fake._detect.assert_not_called()               # курированный катлист не тронут
    assert cl_path.read_bytes() == before          # файл на диске не перезаписан


def test_queue_foreign_cutlist_still_re_detected(client, sample_video,
                                                 monkeypatch):
    """Катлист с ЧУЖИМ source (копия/легаси) по-прежнему переопределяется —
    защита от протечки устаревшей детекции сохранена."""
    from pathlib import Path
    inp = sample_video
    out_dir = Path(serve.APP["out_dir"])
    fake = _fake_queue_session("D:/other/foreign.mp4", inp)
    _wire_queue_process(monkeypatch, fake)
    serve._queue_process_one(
        serve.QueueJob(id="j2", path=str(inp), out_dir=str(out_dir)))
    fake._detect.assert_called_once()
