"""B — YouTube metadata: pure normalize() logic + generate() graceful paths.

No real LLM / Session is needed: normalize() is deterministic, and generate()'s
fallbacks (llm=None, empty transcript, LLM raising) are covered with a tiny fake
client and a fake llm.cfg. A happy-path generate() with a stub chat_json checks
the prompt wiring (final-timeline projection, chapters block, normalization).
"""
from pathlib import Path

import pytest

from vpipe import metadata as md
from vpipe.config import LlmCfg, MaskingCfg, MetadataCfg, ProfanityLists
from vpipe.detect.profanity import ProfanityMatcher
from vpipe.models import Segment, Transcript


# --- normalize() -------------------------------------------------------------
def test_normalize_trims_title_and_hook():
    cfg = MetadataCfg(max_title_chars=10, max_hook_chars=5, n_tags=15)
    out = md.normalize({"title": "A" * 50, "hook": "X" * 50,
                        "description": "desc", "tags": []}, cfg)
    assert out["title"] == "A" * 10
    assert out["hook"] == "X" * 5
    assert out["description"] == "desc"


def test_normalize_dedups_tags_case_insensitive_preserving_order():
    cfg = MetadataCfg(n_tags=15)
    out = md.normalize({"title": "t", "description": "d", "hook": "h",
                        "tags": ["Питон", "питон", "Код", "#код", "  ", "API"]}, cfg)
    # case-insensitive dedup ("Питон"/"питон", "Код"/"#код"), '#' stripped,
    # blank dropped, original order/casing of FIRST occurrence kept.
    assert out["tags"] == ["Питон", "Код", "API"]


def test_normalize_caps_tag_count():
    cfg = MetadataCfg(n_tags=3)
    out = md.normalize({"title": "t", "description": "d", "hook": "h",
                        "tags": ["a", "b", "c", "d", "e"]}, cfg)
    assert out["tags"] == ["a", "b", "c"]


def test_normalize_handles_missing_and_wrong_types():
    cfg = MetadataCfg()
    out = md.normalize({"title": 123, "tags": "not-a-list"}, cfg)
    assert out == {"title": "", "description": "", "tags": [], "hook": ""}
    # completely empty input is fine too
    assert md.normalize({}, cfg) == {"title": "", "description": "",
                                     "tags": [], "hook": ""}


# --- _read_chapters_block() --------------------------------------------------
def test_read_chapters_block(tmp_path: Path):
    assert md._read_chapters_block(None) == ""
    assert md._read_chapters_block(tmp_path / "missing.txt") == ""
    p = tmp_path / "chapters.txt"
    p.write_text("00:00 Вступление\n01:24 Основная часть\n", encoding="utf-8")
    block = md._read_chapters_block(p)
    assert block.startswith("Главы:\n")
    assert "Основная часть" in block
    assert block.endswith("\n\n")


# --- generate() graceful paths -----------------------------------------------
def _tr() -> Transcript:
    return Transcript(language="ru", duration=30.0, model="t", audio_hash="h",
                      segments=[Segment(0.0, 10.0, "Привет, это тест."),
                                Segment(10.0, 20.0, "Говорим про монтаж."),
                                Segment(20.0, 30.0, "И про финал.")])


def test_generate_llm_none_returns_empty():
    out = md.generate(_tr(), [], MetadataCfg(), llm=None, log=lambda *_: None)
    assert out == {"title": "", "description": "", "tags": [], "hook": ""}


def test_generate_empty_transcript_returns_empty():
    tr = Transcript(language="ru", duration=0.0, model="t", audio_hash="h",
                    segments=[])

    class _Llm:
        cfg = LlmCfg()

        def chat_json(self, *a, **k):  # pragma: no cover - must not be called
            raise AssertionError("LLM must not be called on an empty transcript")

    out = md.generate(tr, [], MetadataCfg(), llm=_Llm(), log=lambda *_: None)
    assert out == {"title": "", "description": "", "tags": [], "hook": ""}


def test_generate_llm_failure_degrades_to_empty():
    class _Llm:
        cfg = LlmCfg()

        def chat_json(self, *a, **k):
            raise RuntimeError("ollama down")

    out = md.generate(_tr(), [], MetadataCfg(), llm=_Llm(), log=lambda *_: None)
    assert out == {"title": "", "description": "", "tags": [], "hook": ""}


def test_generate_happy_path_normalizes_and_embeds_chapters(tmp_path: Path):
    captured = {}

    class _Llm:
        cfg = LlmCfg()

        def chat_json(self, system, user, schema):
            captured["user"] = user
            captured["system"] = system
            return {"title": "T" * 200, "description": "Описание ролика",
                    "tags": ["монтаж", "Монтаж", "видео"], "hook": "Смотри до конца"}

    ch = tmp_path / "chapters.txt"
    ch.write_text("00:00 Вступление\n00:15 Финал\n", encoding="utf-8")
    cfg = MetadataCfg(max_title_chars=100, n_tags=15)
    out = md.generate(_tr(), [], cfg, llm=_Llm(), chapters_path=ch,
                      log=lambda *_: None)

    assert len(out["title"]) == 100                    # trimmed to max_title_chars
    assert out["tags"] == ["монтаж", "видео"]          # case-insensitive dedup
    assert out["description"] == "Описание ролика"
    assert out["hook"] == "Смотри до конца"
    # the prompt embedded the final transcript and the chapters block
    assert "Транскрипт (после монтажа):" in captured["user"]
    assert "Главы:" in captured["user"]
    assert "Финал" in captured["user"]


def test_generate_masks_profanity_in_fields():
    # A matcher that flags 'плохо' as profane; title/desc/hook/tags get masked.
    matcher = ProfanityMatcher(ProfanityLists(roots=["плох"], allow=[]))
    mask = MaskingCfg()

    class _Llm:
        cfg = LlmCfg()

        def chat_json(self, *a, **k):
            return {"title": "это плохо", "description": "очень плохо",
                    "tags": ["плохо"], "hook": "плохо"}

    out = md.generate(_tr(), [], MetadataCfg(), llm=_Llm(), matcher=matcher,
                      mask=mask, log=lambda *_: None)
    # the censored token must no longer appear verbatim in any field
    assert "плохо" not in out["title"]
    assert "плохо" not in out["description"]
    assert "плохо" not in out["hook"]
    assert all("плохо" != t for t in out["tags"])
