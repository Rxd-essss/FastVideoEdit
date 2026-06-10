"""P2-#6 — batch-queue persistence (queue.json).

Pure stdlib/unit coverage for the persist layer added to serve.py:
  * _save_queue: atomic snapshot of QUEUE -> cache_dir/queue.json.
  * _load_queue: restore QUEUE from json (running->pending reset, done/error
    preserved, corrupt/missing -> empty, bad entries skipped, unknown keys
    ignored, status normalized).
  * Round-trip through the REST surface (add/remove/clear) and the round-trip
    invariant save->load.

No ffmpeg/whisper/GPU/server/browser — just the JSON + list plumbing.
"""
import json

import pytest
from fastapi.testclient import TestClient

import serve
from vpipe.config import load_config


# --- fixtures ---------------------------------------------------------------
@pytest.fixture()
def cfg_with_cache(tmp_path, monkeypatch):
    """Wire APP with cache_dir pointed at an isolated tmp dir + a clean QUEUE."""
    cfg = load_config("config.yaml")
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    cfg.paths.cache_dir = str(cache_dir)
    monkeypatch.setitem(serve.APP, "cfg", cfg)
    monkeypatch.setitem(serve.APP, "out_dir", str(out_dir))
    monkeypatch.setitem(serve.APP, "use_llm", False)
    monkeypatch.setattr(serve, "QUEUE", [])
    monkeypatch.setattr(serve, "_queue_running", False)
    return cfg, cache_dir, out_dir


@pytest.fixture()
def client(cfg_with_cache):
    return TestClient(serve.app)


@pytest.fixture()
def sample_video(tmp_path):
    p = tmp_path / "clip.mp4"
    p.write_bytes(b"\x00\x00\x00\x18ftypmp42")
    return p


def _read_json(cache_dir):
    return json.loads((cache_dir / "queue.json").read_text(encoding="utf-8"))


# --- _queue_path ------------------------------------------------------------
def test_queue_path_under_cache_dir(cfg_with_cache):
    _, cache_dir, _ = cfg_with_cache
    assert serve._queue_path() == cache_dir / "queue.json"


# --- _save_queue ------------------------------------------------------------
def test_save_queue_writes_all_fields(cfg_with_cache):
    cfg, cache_dir, _ = cfg_with_cache
    job = serve.QueueJob(id="j1", path="/v/clip.mp4", out_dir="/out",
                         render_opts={"encoder": "x264"})
    job.status = "done"
    job.percent = 100.0
    job.stage = "Готово"
    job.result = {"output_path": "/out/clip.mp4"}
    serve.QUEUE.append(job)

    serve._save_queue()

    data = _read_json(cache_dir)
    assert isinstance(data, list) and len(data) == 1
    rec = data[0]
    assert rec["id"] == "j1"
    assert rec["path"] == "/v/clip.mp4"
    assert rec["out_dir"] == "/out"
    assert rec["render_opts"] == {"encoder": "x264"}
    assert rec["status"] == "done"
    assert rec["result"] == {"output_path": "/out/clip.mp4"}


def test_save_queue_is_atomic_no_tmp_left(cfg_with_cache):
    cfg, cache_dir, _ = cfg_with_cache
    serve.QUEUE.append(serve.QueueJob(id="a", path="/v/a.mp4", out_dir="/o"))
    serve._save_queue()
    # The .tmp scratch file must have been os.replace'd away.
    assert not (cache_dir / "queue.json.tmp").exists()
    assert (cache_dir / "queue.json").exists()


def test_save_queue_preserves_cyrillic(cfg_with_cache):
    cfg, cache_dir, _ = cfg_with_cache
    job = serve.QueueJob(id="c", path="/v/c.mp4", out_dir="/o")
    job.stage = "Транскрипция…"
    job.status = "error"
    job.error = "Очередь остановлена"
    serve.QUEUE.append(job)
    serve._save_queue()
    rec = _read_json(cache_dir)[0]
    assert rec["stage"] == "Транскрипция…"
    assert rec["error"] == "Очередь остановлена"


def test_save_queue_best_effort_on_bad_cache_dir(cfg_with_cache, monkeypatch):
    cfg, _, _ = cfg_with_cache
    serve.QUEUE.append(serve.QueueJob(id="x", path="/v/x.mp4", out_dir="/o"))
    # Point cache_dir at something that can't be created (a path under a file).
    bad = serve.Path(serve.APP["out_dir"]) / "afile"
    bad.write_text("not a dir")
    cfg.paths.cache_dir = str(bad / "sub")
    serve._save_queue()  # must NOT raise


# --- _load_queue ------------------------------------------------------------
def test_load_missing_file_empty_queue(cfg_with_cache):
    serve._load_queue()
    assert serve.QUEUE == []


def test_load_corrupt_json_empty_queue(cfg_with_cache):
    _, cache_dir, _ = cfg_with_cache
    (cache_dir / "queue.json").write_text("{not json", encoding="utf-8")
    serve._load_queue()
    assert serve.QUEUE == []


def test_load_non_list_json_empty_queue(cfg_with_cache):
    _, cache_dir, _ = cfg_with_cache
    (cache_dir / "queue.json").write_text('{"id": "x"}', encoding="utf-8")
    serve._load_queue()
    assert serve.QUEUE == []


def test_load_running_reset_to_pending(cfg_with_cache):
    _, cache_dir, _ = cfg_with_cache
    payload = [{"id": "r", "path": "/v/r.mp4", "out_dir": "/o",
                "render_opts": {}, "status": "running",
                "percent": 73.2, "stage": "Рендер…", "result": None, "error": None}]
    (cache_dir / "queue.json").write_text(json.dumps(payload), encoding="utf-8")
    serve._load_queue()
    assert len(serve.QUEUE) == 1
    j = serve.QUEUE[0]
    assert j.status == "pending"
    assert j.percent == 0.0
    assert j.stage == ""


def test_load_done_and_error_preserved(cfg_with_cache):
    _, cache_dir, _ = cfg_with_cache
    payload = [
        {"id": "d", "path": "/v/d.mp4", "out_dir": "/o", "render_opts": {},
         "status": "done", "percent": 100.0, "stage": "Готово",
         "result": {"output_path": "/o/d.mp4"}, "error": None},
        {"id": "e", "path": "/v/e.mp4", "out_dir": "/o", "render_opts": {},
         "status": "error", "percent": 12.0, "stage": "",
         "result": None, "error": "boom"},
    ]
    (cache_dir / "queue.json").write_text(json.dumps(payload), encoding="utf-8")
    serve._load_queue()
    by_id = {j.id: j for j in serve.QUEUE}
    assert by_id["d"].status == "done"
    assert by_id["d"].result == {"output_path": "/o/d.mp4"}
    assert by_id["e"].status == "error"
    assert by_id["e"].error == "boom"


def test_load_invalid_status_becomes_pending(cfg_with_cache):
    _, cache_dir, _ = cfg_with_cache
    payload = [{"id": "s", "path": "/v/s.mp4", "out_dir": "/o",
                "status": "weird"}]
    (cache_dir / "queue.json").write_text(json.dumps(payload), encoding="utf-8")
    serve._load_queue()
    assert serve.QUEUE[0].status == "pending"


def test_load_unknown_keys_ignored_and_defaults(cfg_with_cache):
    _, cache_dir, _ = cfg_with_cache
    # 'name' is derived (excluded); 'future_field' is unknown -> both dropped.
    # Missing percent/stage/result/error -> dataclass defaults.
    payload = [{"id": "u", "path": "/v/u.mp4", "out_dir": "/o",
                "name": "u.mp4", "future_field": 123, "status": "pending"}]
    (cache_dir / "queue.json").write_text(json.dumps(payload), encoding="utf-8")
    serve._load_queue()
    assert len(serve.QUEUE) == 1
    j = serve.QUEUE[0]
    assert j.id == "u"
    assert j.percent == 0.0
    assert j.stage == ""
    assert j.result is None
    assert j.error is None
    assert not hasattr(j, "name")  # derived, never an attribute


def test_load_skips_entries_missing_required_fields(cfg_with_cache):
    _, cache_dir, _ = cfg_with_cache
    payload = [
        {"path": "/v/no-id.mp4", "out_dir": "/o"},          # no id
        {"id": "", "path": "/v/empty-id.mp4", "out_dir": "/o"},  # empty id
        {"id": "noout", "path": "/v/x.mp4"},                 # no out_dir
        {"id": "nopath", "out_dir": "/o"},                   # no path
        "a string, not a dict",                              # wrong type
        {"id": "ok", "path": "/v/ok.mp4", "out_dir": "/o"},  # valid
    ]
    (cache_dir / "queue.json").write_text(json.dumps(payload), encoding="utf-8")
    serve._load_queue()
    assert [j.id for j in serve.QUEUE] == ["ok"]


def test_load_coerces_bad_field_types(cfg_with_cache):
    _, cache_dir, _ = cfg_with_cache
    payload = [{"id": "t", "path": "/v/t.mp4", "out_dir": "/o",
                "render_opts": "not-a-dict", "percent": "oops",
                "stage": 123, "result": "not-a-dict", "error": 5}]
    (cache_dir / "queue.json").write_text(json.dumps(payload), encoding="utf-8")
    serve._load_queue()
    j = serve.QUEUE[0]
    assert j.render_opts == {}
    assert j.percent == 0.0
    assert j.stage == ""
    assert j.result is None
    assert j.error is None


def test_load_appends_not_replaces(cfg_with_cache):
    _, cache_dir, _ = cfg_with_cache
    serve.QUEUE.append(serve.QueueJob(id="pre", path="/v/pre.mp4", out_dir="/o"))
    payload = [{"id": "new", "path": "/v/new.mp4", "out_dir": "/o"}]
    (cache_dir / "queue.json").write_text(json.dumps(payload), encoding="utf-8")
    serve._load_queue()
    assert [j.id for j in serve.QUEUE] == ["pre", "new"]


# --- round-trip -------------------------------------------------------------
def test_round_trip_save_then_load(cfg_with_cache, monkeypatch):
    cfg, cache_dir, _ = cfg_with_cache
    j1 = serve.QueueJob(id="one", path="/v/one.mp4", out_dir="/o",
                        render_opts={"quality": 20})
    j2 = serve.QueueJob(id="two", path="/v/two.mp4", out_dir="/o")
    j2.status = "done"
    j2.result = {"output_path": "/o/two.mp4"}
    serve.QUEUE.extend([j1, j2])
    serve._save_queue()

    # Wipe in-memory queue, reload from disk.
    monkeypatch.setattr(serve, "QUEUE", [])
    serve._load_queue()

    by_id = {j.id: j for j in serve.QUEUE}
    assert set(by_id) == {"one", "two"}
    assert by_id["one"].render_opts == {"quality": 20}
    assert by_id["one"].status == "pending"
    assert by_id["two"].status == "done"
    assert by_id["two"].result == {"output_path": "/o/two.mp4"}


# --- endpoints persist on mutation ------------------------------------------
def test_add_persists_to_disk(client, cfg_with_cache, sample_video):
    _, cache_dir, _ = cfg_with_cache
    r = client.post("/api/queue/add", json={"path": str(sample_video)})
    assert r.status_code == 200
    data = _read_json(cache_dir)
    assert len(data) == 1
    assert data[0]["id"] == r.json()["id"]
    assert data[0]["status"] == "pending"


def test_remove_persists_to_disk(client, cfg_with_cache, sample_video):
    _, cache_dir, _ = cfg_with_cache
    jid = client.post("/api/queue/add", json={"path": str(sample_video)}).json()["id"]
    client.post("/api/queue/remove", json={"id": jid})
    assert _read_json(cache_dir) == []


def test_clear_persists_to_disk(client, cfg_with_cache, sample_video):
    _, cache_dir, _ = cfg_with_cache
    client.post("/api/queue/add", json={"path": str(sample_video)})
    client.post("/api/queue/add", json={"path": str(sample_video)})
    client.post("/api/queue/clear")
    assert _read_json(cache_dir) == []


def test_add_then_reload_restores_via_list_endpoint(client, cfg_with_cache, sample_video):
    """Add a job, simulate a restart (wipe memory + _load_queue), and confirm
    the GET /api/queue surface shows the restored job."""
    _, cache_dir, _ = cfg_with_cache
    jid = client.post("/api/queue/add", json={"path": str(sample_video)}).json()["id"]
    # Simulate server restart.
    serve.QUEUE.clear()
    serve._load_queue()
    listed = client.get("/api/queue").json()["jobs"]
    assert [j["id"] for j in listed] == [jid]
    assert listed[0]["status"] == "pending"


# --- queue badge bootstrap: /api/state exposes the pending count ------------
def test_pending_count_helper_counts_pending_and_running(cfg_with_cache):
    """_queue_pending_count() = pending + running; done/error don't count."""
    serve.QUEUE.extend([
        serve.QueueJob(id="p1", path="/v/p1.mp4", out_dir="/o"),               # pending
        serve.QueueJob(id="p2", path="/v/p2.mp4", out_dir="/o"),               # pending
    ])
    running = serve.QueueJob(id="r", path="/v/r.mp4", out_dir="/o")
    running.status = "running"
    done = serve.QueueJob(id="d", path="/v/d.mp4", out_dir="/o")
    done.status = "done"
    err = serve.QueueJob(id="e", path="/v/e.mp4", out_dir="/o")
    err.status = "error"
    serve.QUEUE.extend([running, done, err])
    # 2 pending + 1 running = 3; done/error excluded.
    assert serve._queue_pending_count() == 3


def test_pending_count_empty_queue_is_zero(cfg_with_cache):
    assert serve._queue_pending_count() == 0


def test_state_no_session_exposes_queue_pending(client, cfg_with_cache, monkeypatch):
    """The no-session bootstrap branch of /api/state must carry queue_pending so
    the badge lights up for a restored queue even with no clip open + worker
    stopped (the UX-round-2 fix)."""
    monkeypatch.setattr(serve, "SESSION", None)
    serve.QUEUE.append(serve.QueueJob(id="b", path="/v/b.mp4", out_dir="/o"))
    s = client.get("/api/state").json()
    assert s.get("no_session") is True
    assert s["queue_pending"] == 1
    assert s["queue_running"] is False


def test_state_no_session_zero_pending_when_empty(client, cfg_with_cache, monkeypatch):
    monkeypatch.setattr(serve, "SESSION", None)
    s = client.get("/api/state").json()
    assert s.get("no_session") is True
    assert s["queue_pending"] == 0


def test_restored_queue_is_visible_via_state_after_reload(client, cfg_with_cache,
                                                          sample_video, monkeypatch):
    """End-to-end of the badge fix: add → simulate restart (_load_queue) →
    /api/state (no session) reports queue_pending>0, so the frontend lights the
    badge without the worker running."""
    monkeypatch.setattr(serve, "SESSION", None)
    client.post("/api/queue/add", json={"path": str(sample_video)})
    # Simulate server restart: wipe memory, reload from queue.json.
    serve.QUEUE.clear()
    serve._load_queue()
    s = client.get("/api/state").json()
    assert s["queue_pending"] == 1
    assert s["queue_running"] is False
