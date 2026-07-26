"""CutList.save_json / Transcript.save must be crash-atomic: a failure mid-write
leaves the pre-existing valid file intact (the .tmp took the hit), and the
serialized bytes are unchanged from the old direct write.
(Audit 2026-07: vpipe/models.py — corrupt cutlist bricks the video on open.)"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest                                                      # noqa: E402

from vpipe import models                                          # noqa: E402
from vpipe.models import (ACTION_REMOVE, TYPE_MANUAL, CutList,     # noqa: E402
                          CutSegment)


def _sample() -> CutList:
    return CutList(source="in.mp4", duration=12.5, segments=[
        CutSegment(id="r0", start=1.0, end=2.0, type=TYPE_MANUAL,
                   action=ACTION_REMOVE, enabled=True)])


def test_save_json_roundtrip_bytes_unchanged(tmp_path):
    cl = _sample()
    p = tmp_path / "cutlist.json"
    cl.save_json(p)
    # bytes equal the old direct-write serialization
    expected = json.dumps(cl.to_dict(), ensure_ascii=False, indent=2)
    assert p.read_text(encoding="utf-8") == expected
    # and no stray .tmp survives
    assert not (tmp_path / "cutlist.json.tmp").exists()
    # round-trips back
    assert CutList.load_json(p).to_dict() == cl.to_dict()


def test_save_json_atomic_leaves_original_on_failure(tmp_path, monkeypatch):
    p = tmp_path / "cutlist.json"
    good = _sample()
    good.save_json(p)                       # a valid file already on disk
    before = p.read_text(encoding="utf-8")

    def boom(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(models.os, "replace", boom)
    changed = _sample()
    changed.segments[0].end = 9.9
    with pytest.raises(OSError):
        changed.save_json(p)
    # original file untouched and still parseable
    assert p.read_text(encoding="utf-8") == before
    assert CutList.load_json(p).to_dict() == good.to_dict()


def test_save_json_tmp_is_sibling_with_json_suffix(tmp_path, monkeypatch):
    # the temp path must keep the .json base (x.cutlist.json -> .json.tmp), not
    # collapse to a bare .tmp — verified by capturing the os.replace source arg.
    p = tmp_path / "clip.cutlist.json"
    seen = {}
    real = os.replace

    def spy(src, dst):
        seen["src"] = str(src)
        return real(src, dst)

    monkeypatch.setattr(models.os, "replace", spy)
    _sample().save_json(p)
    assert seen["src"].endswith(".json.tmp")
