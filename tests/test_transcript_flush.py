"""Perf #94 (UNVERIFIED, opt-in DEFAULT-OFF) — debounced transcript flush.

Default (env unset) keeps the exact inline atomic flush on EVERY edit
(byte-for-byte). Opt-in (FVE_TRANSCRIPT_FLUSH_DEBOUNCE_SEC>0) coalesces a burst
into one write; a forced flush (task-start / shutdown hook) persists the last
edit. In-memory state (what GET reads) reflects each edit immediately either
way. No ffmpeg — FastAPI TestClient + a fake session (like test_transcript_edit).
"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import serve                                        # noqa: E402
from vpipe.models import Segment, Transcript, Word  # noqa: E402

HASH = "ab" * 20


def _make_transcript() -> Transcript:
    return Transcript(
        language="ru", duration=5.0, model="large-v3", audio_hash=HASH,
        segments=[Segment(0.0, 2.5, "Привет мир", [
            Word(" Привет", 0.0, 1.0, 0.99),
            Word(" мир", 1.0, 2.5, 0.98)])])


@pytest.fixture()
def sess(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    tr = _make_transcript()
    tr.save(cache / f"{HASH}.transcript.json")     # starting cache (post-transcribe)
    cfg = SimpleNamespace(transcribe=SimpleNamespace(device="cuda"))
    return SimpleNamespace(transcript=tr, cache_dir=cache, audio_hash=HASH,
                           cfg=cfg, task={"running": False})


@pytest.fixture()
def client(sess, monkeypatch):
    monkeypatch.setattr(serve, "SESSION", sess)
    monkeypatch.setattr(serve, "_queue_running", False)
    return TestClient(serve.app)


@pytest.fixture(autouse=True)
def _cancel_pending_timer():
    """Never let a debounce timer outlive a test (would touch a gone tmp_path)."""
    yield
    with serve._TR_FLUSH_LOCK:
        if serve._tr_flush_timer is not None:
            serve._tr_flush_timer.cancel()
            serve._tr_flush_timer = None
        serve._tr_flush_session = None


def _put(client, text, si=0, wi=1):
    return client.put("/api/transcript/word", json={"si": si, "wi": wi,
                                                    "text": text})


def _count_flushes(monkeypatch):
    n = {"c": 0}
    orig = serve._flush_transcript_now

    def counting(s):
        n["c"] += 1
        orig(s)

    monkeypatch.setattr(serve, "_flush_transcript_now", counting)
    return n


def test_default_flushes_inline_every_edit(client, sess, monkeypatch):
    monkeypatch.delenv("FVE_TRANSCRIPT_FLUSH_DEBOUNCE_SEC", raising=False)
    n = _count_flushes(monkeypatch)
    for t in ("a", "bb", "ccc"):
        assert _put(client, t).status_code == 200
    assert n["c"] == 3                              # one disk flush per edit (default)
    disk = json.loads((sess.cache_dir / f"{HASH}.transcript.json")
                      .read_text(encoding="utf-8"))
    assert disk["segments"][0]["words"][1]["w"] == " ccc"


def test_optin_debounce_collapses_and_force_flush_persists(client, sess,
                                                           monkeypatch):
    monkeypatch.setenv("FVE_TRANSCRIPT_FLUSH_DEBOUNCE_SEC", "30")
    monkeypatch.setattr(serve, "_tr_flush_timer", None)
    n = _count_flushes(monkeypatch)
    p = sess.cache_dir / f"{HASH}.transcript.json"

    for t in ("a", "bb", "ccc"):
        assert _put(client, t).status_code == 200

    # Burst coalesced: no disk write yet (timer pending, 30s).
    assert n["c"] == 0
    disk = json.loads(p.read_text(encoding="utf-8"))
    assert disk["segments"][0]["words"][1]["w"] == " мир"   # original still on disk
    # In-memory (what GET returns) reflects the edit immediately.
    assert sess.transcript.segments[0].words[1].word == " ccc"

    # Force-flush (task-start / shutdown hook) writes exactly once, last edit.
    serve._force_flush_pending_transcript(sess)
    assert n["c"] == 1
    disk2 = json.loads(p.read_text(encoding="utf-8"))
    assert disk2["segments"][0]["words"][1]["w"] == " ccc"


def test_force_flush_is_noop_when_nothing_pending(sess, monkeypatch):
    monkeypatch.setattr(serve, "SESSION", sess)
    monkeypatch.setattr(serve, "_tr_flush_timer", None)
    n = _count_flushes(monkeypatch)
    serve._force_flush_pending_transcript()          # no timer pending
    assert n["c"] == 0                                # default hooks change nothing


# --- W6 #10: flush is keyed by the CAPTURED session ---------------------------
def test_force_flush_other_session_skips_and_preserves_pending(client, sess,
                                                               monkeypatch):
    """Смена клипа в окне дебаунса: force-flush НОВОЙ сессии не должен гасить
    таймер СТАРОЙ и флашить не тот транскрипт (fail-before: таймер А отменялся,
    флашилась нетронутая Б, правка А терялась навсегда)."""
    monkeypatch.setenv("FVE_TRANSCRIPT_FLUSH_DEBOUNCE_SEC", "30")
    monkeypatch.setattr(serve, "_tr_flush_timer", None)
    monkeypatch.setattr(serve, "_tr_flush_session", None, raising=False)
    flushed = []
    monkeypatch.setattr(serve, "_flush_transcript_now",
                        lambda s: flushed.append(s))
    assert _put(client, "правка").status_code == 200   # таймер взведён для sess

    other = SimpleNamespace(transcript=None, cache_dir=sess.cache_dir,
                            audio_hash="cd" * 20, task={"running": False})
    serve._force_flush_pending_transcript(other)       # start_task клипа Б
    assert flushed == []                               # Б не флашится
    with serve._TR_FLUSH_LOCK:
        assert serve._tr_flush_timer is not None       # правка А ещё в очереди
    serve._force_flush_pending_transcript(sess)        # сама А — флашится
    assert flushed == [sess]


def test_shutdown_force_flush_targets_captured_session(client, sess,
                                                       monkeypatch):
    """W6 #10: shutdown-хук (s=None) флашит ЗАХВАЧЕННУЮ таймером сессию, а не
    глобальную SESSION (после смены клипа это разные объекты)."""
    monkeypatch.setenv("FVE_TRANSCRIPT_FLUSH_DEBOUNCE_SEC", "30")
    monkeypatch.setattr(serve, "_tr_flush_timer", None)
    monkeypatch.setattr(serve, "_tr_flush_session", None, raising=False)
    flushed = []
    monkeypatch.setattr(serve, "_flush_transcript_now",
                        lambda s: flushed.append(s))
    assert _put(client, "правка").status_code == 200
    monkeypatch.setattr(serve, "SESSION",
                        SimpleNamespace(transcript=None))   # клип сменили
    serve._force_flush_pending_transcript()                 # shutdown/atexit
    assert flushed == [sess]                                # не новая SESSION


def test_natural_fire_clears_pending_state(sess, monkeypatch):
    """W6 #10: сработавший таймер чистит глобалы — последующий force-flush
    больше не пересериализует мульти-МБ транскрипт впустую (докстринг
    «No-op when no timer is pending»)."""
    import time as _t
    monkeypatch.setattr(serve, "_tr_flush_timer", None)
    monkeypatch.setattr(serve, "_tr_flush_session", None, raising=False)
    flushed = []
    monkeypatch.setattr(serve, "_flush_transcript_now",
                        lambda s: flushed.append(s))
    serve._schedule_transcript_flush(sess, 0.01)
    deadline = _t.time() + 5.0
    while _t.time() < deadline:
        with serve._TR_FLUSH_LOCK:
            if serve._tr_flush_timer is None and flushed:
                break
        _t.sleep(0.01)
    assert flushed == [sess]
    serve._force_flush_pending_transcript(sess)
    assert flushed == [sess]                              # второго флаша нет
