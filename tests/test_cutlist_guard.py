# -*- coding: utf-8 -*-
"""Wave-1 (cutlist-source-guard-and-defer-detect + corrupt-guard, merged).

Session.__init__ must:
  * only trust an on-disk cutlist that belongs to THIS video (audio_hash, or
    legacy source+duration) — a same-stem cutlist from a different/re-recorded
    video is ignored, leaving self.cutlist=None so a later _detect can't merge
    its stale manual cuts;
  * NEVER run detection (LLM+VAD) synchronously in the ctor — open_session
    launches it as a background task instead;
  * quarantine a corrupt transcript/cutlist file aside (never fail /api/open).
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import serve                                              # noqa: E402
from vpipe.config import load_config                      # noqa: E402
from vpipe.models import (ACTION_REMOVE, TYPE_MANUAL, CutList,   # noqa: E402
                          CutSegment, Segment, Transcript, Word)


def _cfg(tmp_path):
    cfg = load_config("config.yaml")
    cfg.paths.cache_dir = str(tmp_path / "cache")
    cfg.paths.work_dir = str(tmp_path / "work")
    return cfg


def _patch_probe(monkeypatch, audio_hash="THISHASH", duration=20.0):
    # Session.__init__ builds a real FFmpeg (resolves the binary eagerly) — stub
    # it so the ctor is hermetic and runs on CI where no ffmpeg is installed.
    monkeypatch.setattr(serve, "FFmpeg", lambda *a, **k: SimpleNamespace())
    monkeypatch.setattr(serve, "hash_input", lambda *a, **k: audio_hash)
    monkeypatch.setattr(serve, "probe_media",
                        lambda *a, **k: SimpleNamespace(
                            duration=duration, has_audio=True,
                            width=1920, height=1080, fps=30.0))


def _seg(seg_id="stale"):
    return CutSegment(id=seg_id, start=1.0, end=2.0, type=TYPE_MANUAL,
                      action=ACTION_REMOVE, enabled=True)


def _write_cutlist(out_dir, stem, cl):
    out_dir.mkdir(parents=True, exist_ok=True)
    cl.save_json(out_dir / f"{stem}.cutlist.json")


# --- (a) foreign-hash cutlist ignored, stale manual cut never merged ---------
def test_stale_cutlist_from_other_video_ignored(monkeypatch, tmp_path):
    _patch_probe(monkeypatch)
    out_dir = tmp_path / "out"
    cl = CutList(source="somewhere.mp4", duration=20.0, segments=[_seg()])
    cl.audio_hash = "OTHER"                        # a DIFFERENT video
    _write_cutlist(out_dir, "лекция", cl)

    video = tmp_path / "лекция.mp4"
    video.write_bytes(b"x")
    s = serve.Session(str(video), _cfg(tmp_path), str(out_dir), False)
    assert s.cutlist is None                       # foreign cutlist not trusted

    # a fresh detect must NOT resurrect the stale manual cut
    s.transcript = Transcript(language="ru", duration=20.0, model="t",
                              audio_hash="THISHASH",
                              segments=[Segment(0.0, 20.0, "a", [Word("a", 0.1, 0.5)])])
    monkeypatch.setattr(serve, "run_detection",
                        lambda *a, **k: CutList(source=str(video), duration=20.0, segments=[]))
    s._detect()
    assert all(seg.id != "stale" for seg in s.cutlist.segments)


# --- (b) matching-hash cutlist loaded, no detection in the ctor --------------
def test_matching_hash_cutlist_loaded(monkeypatch, tmp_path):
    _patch_probe(monkeypatch)
    out_dir = tmp_path / "out"
    cl = CutList(source="whatever.mp4", duration=20.0, segments=[_seg("keep")])
    cl.audio_hash = "THISHASH"                      # THIS video
    _write_cutlist(out_dir, "лекция", cl)

    def _boom(*a, **k):
        raise AssertionError("run_detection must not run in the ctor")
    monkeypatch.setattr(serve, "run_detection", _boom)

    video = tmp_path / "лекция.mp4"
    video.write_bytes(b"x")
    s = serve.Session(str(video), _cfg(tmp_path), str(out_dir), False)
    assert s.cutlist is not None
    assert any(seg.id == "keep" for seg in s.cutlist.segments)


# --- (c) legacy (no hash) cutlist: source/duration mismatch ignored ----------
def test_legacy_cutlist_source_mismatch_ignored(monkeypatch, tmp_path):
    _patch_probe(monkeypatch)
    out_dir = tmp_path / "out"
    cl = CutList(source="D:/other/x.mp4", duration=999.0, segments=[_seg()])
    _write_cutlist(out_dir, "лекция", cl)          # no audio_hash -> "" legacy

    video = tmp_path / "лекция.mp4"
    video.write_bytes(b"x")
    s = serve.Session(str(video), _cfg(tmp_path), str(out_dir), False)
    assert s.cutlist is None


def test_legacy_cutlist_source_match_loaded(monkeypatch, tmp_path):
    _patch_probe(monkeypatch)
    out_dir = tmp_path / "out"
    video = tmp_path / "лекция.mp4"
    cl = CutList(source=str(video), duration=20.0, segments=[_seg("keep")])
    _write_cutlist(out_dir, "лекция", cl)
    video.write_bytes(b"x")
    s = serve.Session(str(video), _cfg(tmp_path), str(out_dir), False)
    assert s.cutlist is not None
    assert any(seg.id == "keep" for seg in s.cutlist.segments)


# --- (d) corrupt files are quarantined, not fatal ----------------------------
def test_corrupt_cutlist_quarantined_no_detect_in_ctor(monkeypatch, tmp_path):
    _patch_probe(monkeypatch)
    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True)
    (out_dir / "лекция.cutlist.json").write_text("{ not json", encoding="utf-8")

    def _boom(*a, **k):
        raise AssertionError("no detection inside the ctor")
    monkeypatch.setattr(serve, "run_detection", _boom)

    video = tmp_path / "лекция.mp4"
    video.write_bytes(b"x")
    s = serve.Session(str(video), _cfg(tmp_path), str(out_dir), False)  # must not raise
    assert s.cutlist is None
    assert not (out_dir / "лекция.cutlist.json").exists()
    assert list(out_dir.glob("лекция.cutlist.json.corrupt-*"))          # renamed aside


def test_corrupt_transcript_cache_quarantined(monkeypatch, tmp_path):
    _patch_probe(monkeypatch)
    cfg = _cfg(tmp_path)
    cache_dir = Path(cfg.paths.cache_dir)
    cache_dir.mkdir(parents=True)
    (cache_dir / "THISHASH.transcript.json").write_text("{bad", encoding="utf-8")

    video = tmp_path / "лекция.mp4"
    video.write_bytes(b"x")
    s = serve.Session(str(video), cfg, str(tmp_path / "out"), False)    # must not raise
    assert s.transcript is None
    assert list(cache_dir.glob("THISHASH.transcript.json.corrupt-*"))


# --- (e) open_session defers detection to a background task ------------------
def test_open_session_defers_detect(monkeypatch, tmp_path):
    _patch_probe(monkeypatch)
    monkeypatch.setattr(serve, "SESSION", None)
    cfg = _cfg(tmp_path)
    out_dir = tmp_path / "out"
    cache_dir = Path(cfg.paths.cache_dir)
    cache_dir.mkdir(parents=True)
    # cached transcript, NO cutlist file -> cutlist stays None, detect deferred
    tr = Transcript(language="ru", duration=20.0, model="t", audio_hash="THISHASH",
                    segments=[Segment(0.0, 20.0, "a b",
                                      [Word("a", 0.1, 0.5), Word("b", 1.0, 1.5)])])
    tr.save(cache_dir / "THISHASH.transcript.json")

    video = tmp_path / "лекция.mp4"
    video.write_bytes(b"x")

    monkeypatch.setattr(serve, "APP",
                        {"cfg": cfg, "out_dir": str(out_dir), "use_llm": False})
    started = []
    monkeypatch.setattr(serve.Session, "start_task",
                        lambda self, name, fn: started.append(name))
    monkeypatch.setattr(serve.Session, "_detect",
                        lambda self, cfg=None: started.append("_detect_ran"))

    s = serve.open_session(str(video))
    assert s.transcript is not None
    assert s.cutlist is None
    assert "detect" in started              # detection launched as a background task
    assert "_detect_ran" not in started     # NOT run synchronously inside __init__


# --- (f) W6 #1: probe duration-clamp must not destroy the user's own cutlist --
def test_legacy_cutlist_survives_probe_duration_clamp(monkeypatch, tmp_path):
    """probe-клап до КОРОТЧАЙШЕГО стрима сдвигает media.duration на 0.5–2 c
    (аудио живёт дольше видео) — легаси-катлист ЭТОГО ЖЕ файла обязан
    загрузиться (fail-before: толеранс 0.5 c дропал собственную курацию, и
    ленивый _detect() затирал файл)."""
    _patch_probe(monkeypatch, duration=20.0)       # clamped (shortest stream)
    out_dir = tmp_path / "out"
    video = tmp_path / "лекция.mp4"
    cl = CutList(source=str(video), duration=21.3, segments=[_seg("keep")])
    _write_cutlist(out_dir, "лекция", cl)          # legacy: no audio_hash
    video.write_bytes(b"x")
    s = serve.Session(str(video), _cfg(tmp_path), str(out_dir), False)
    assert s.cutlist is not None
    assert any(seg.id == "keep" for seg in s.cutlist.segments)


def test_hash_match_ignores_duration_drift(monkeypatch, tmp_path):
    """W6 #1: совпавший audio_hash ДОСТАТОЧЕН — дрейф длительности от
    probe-клапа не дропает катлист этого же видео."""
    _patch_probe(monkeypatch, duration=20.0)
    out_dir = tmp_path / "out"
    cl = CutList(source="whatever.mp4", duration=21.7, segments=[_seg("keep")])
    cl.audio_hash = "THISHASH"
    _write_cutlist(out_dir, "лекция", cl)
    video = tmp_path / "лекция.mp4"
    video.write_bytes(b"x")
    s = serve.Session(str(video), _cfg(tmp_path), str(out_dir), False)
    assert s.cutlist is not None
    assert any(seg.id == "keep" for seg in s.cutlist.segments)


def test_rejected_cutlist_backed_up_not_lost(monkeypatch, tmp_path):
    """W6 #1: отвергнутый гардом катлист переименовывается в .bak ДО того, как
    ленивый фоновый _detect() перезапишет файл (fail-before: файл оставался под
    живым именем и молча затирался — курация невосстановима)."""
    _patch_probe(monkeypatch)
    out_dir = tmp_path / "out"
    cl = CutList(source="somewhere.mp4", duration=999.0, segments=[_seg()])
    cl.audio_hash = "OTHER"                        # чужое видео
    _write_cutlist(out_dir, "лекция", cl)
    video = tmp_path / "лекция.mp4"
    video.write_bytes(b"x")
    s = serve.Session(str(video), _cfg(tmp_path), str(out_dir), False)
    assert s.cutlist is None
    assert (out_dir / "лекция.cutlist.json.bak").exists()
    assert not (out_dir / "лекция.cutlist.json").exists()
