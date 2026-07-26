# -*- coding: utf-8 -*-
"""Wave-1 (output-filename-double-stem): dotted output stems must survive.

Before the fix `Path(name).stem` was applied to an already extension-less UI/
queue filename, so `лекция.часть1_clip01` collapsed to `лекция` for EVERY clip
(N Clip-Maker clips -> one file); the mp4 and sidecar paths then re-truncated at
the last dot via `Path.with_suffix`. These tests pin the corrected behaviour and
prove the common non-dotted path stays byte-for-byte unchanged.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import serve                                              # noqa: E402
from vpipe import subtitles as subs_mod                   # noqa: E402
from vpipe.config import (MaskingCfg, ProfanityLists, SubsCfg,   # noqa: E402
                          load_config)
from vpipe.detect.profanity import ProfanityMatcher       # noqa: E402
from vpipe.models import Segment, Transcript, Word         # noqa: E402


def _sess(tmp_path, inp="лекция.часть1.mp4"):
    return SimpleNamespace(
        cfg=load_config("config.yaml"),
        inp=Path(inp),
        media=SimpleNamespace(width=1920, height=1080, fps=30.0, duration=20.0),
        out_dir=tmp_path / "out")


# --- _clean_stem / _with_ext units -------------------------------------------
def test_clean_stem_strips_only_media_ext():
    assert serve._clean_stem("output.mp4", "fb") == "output"      # real ext -> stripped
    assert serve._clean_stem("проект.v2", "fb") == "проект.v2"    # .v2 not media -> kept
    assert serve._clean_stem("лекция.часть1", "fb") == "лекция.часть1"
    assert serve._clean_stem("   ", "fallback") == "fallback"     # blank -> fallback
    assert serve._clean_stem("", "fallback") == "fallback"


def test_with_ext_appends_without_truncation():
    base = Path("out") / "лекция.часть1_clip01"
    assert serve._with_ext(base, ".mp4").name == "лекция.часть1_clip01.mp4"
    plain = Path("out") / "output"
    assert serve._with_ext(plain, ".mp4") == plain.with_suffix(".mp4")   # identical for plain


# --- _resolve_render_opts: dotted names survive ------------------------------
def test_dotted_filename_not_truncated(tmp_path):
    s = _sess(tmp_path)
    _, _, _, _, base = serve._resolve_render_opts(s, {"filename": "лекция.часть1_clip01"})
    assert base.name == "лекция.часть1_clip01"
    assert str(serve._with_ext(base, ".mp4")).endswith("лекция.часть1_clip01.mp4")


def test_two_clips_distinct_outputs(tmp_path):
    s = _sess(tmp_path)
    _, _, _, _, b1 = serve._resolve_render_opts(s, {"filename": "x.a_clip01"})
    _, _, _, _, b2 = serve._resolve_render_opts(s, {"filename": "x.a_clip02"})
    assert serve._with_ext(b1, ".mp4") != serve._with_ext(b2, ".mp4")


def test_plain_stem_unchanged(tmp_path):
    s = _sess(tmp_path)
    _, _, _, _, base = serve._resolve_render_opts(s, {"filename": "output"})
    assert base.name == "output"
    assert serve._with_ext(base, ".mp4") == base.with_suffix(".mp4")


# --- subtitles sidecar keeps the dotted tail ---------------------------------
def test_dotted_base_srt_keeps_tail(tmp_path):
    words = [Word("привет", 0.2, 0.6), Word("мир", 0.7, 1.1)]
    tr = Transcript(language="ru", duration=5.0, model="t", audio_hash="h",
                    segments=[Segment(0.2, 1.1, "привет мир", words)])
    m = ProfanityMatcher(ProfanityLists(roots=[], allow=[]))
    res = subs_mod.generate(tr, [], SubsCfg(), MaskingCfg(), m,
                            tmp_path / "лекция.часть1", log=lambda *a, **k: None)
    assert Path(res["srt"]).name == "лекция.часть1.srt"
