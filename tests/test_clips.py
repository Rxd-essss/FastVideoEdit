"""F1 — Clip Maker core (vpipe/clips.py): suggest() with a strictly mocked LLM.

Covers the plan's F1 expectations 1-7 + founder decision №5:
 1. original coordinates, window-local indices offset to GLOBAL;
 2. >60 s trimmed to a sentence boundary; dur_eff 12 s dropped; 17 s kept+marked;
 3. zone-dependent lowercase-guard (drop in punctuated zone, fuzzy otherwise);
 4. dedup: time IoU>=0.5 collapsed (bigger window score wins); text retake
    ratio>=0.7 collapsed (decision №5: fewer enabled cuts inside wins);
 5. made-up hook_phrase demotes the candidate to the tail;
 6. a failed window is skipped, others survive; on_stage/on_progress per window;
 7. bool/str/negative/out-of-range indices from the mock are quietly dropped.
Plus F6 (one-call re-rank, §3.5): the ranking order is applied, prompt ids are
STRICTLY 1-based, broken replies (garbage / unknown ids / duplicates / gaps /
empty / LLM down) repair or fall back to round-robin, cfg.rerank=False skips
the call, rank_source is marked on every candidate.
No real LLM, no Ollama: the mock is a plain object with chat_json().
"""
from __future__ import annotations

import pytest

from vpipe import clips
from vpipe.clips import ClipCandidate, suggest
from vpipe.config import ClipsCfg, LlmCfg
from vpipe.models import (ACTION_REMOVE, TYPE_BADTAKE, TYPE_PAUSE, CutList,
                          CutSegment, Segment, Transcript, Word)

# Distinct sentences (deliberately different vocabularies so the retake text
# dedup ratio>=0.7 never fires between UNRELATED candidates).
_T = [
    "Прямо сейчас почти весь интернет держится на линуксе.",
    "Сервера обрабатывают миллионы запросов каждую секунду без сбоев.",
    "Давай разберем честно как устроена файловая система.",
    "Реестр хранит настройки в одной централизованной базе данных.",
    "Пакетный менеджер ставит программы одной короткой командой.",
    "Открытый код позволяет находить уязвимости буквально за часы.",
    "Графическая оболочка съедает оперативную память и процессор.",
    "Терминал выглядит страшно только первые несколько дней практики.",
    "Драйверы устройств подтягиваются автоматически из ядра системы.",
    "Обновления не перезагружают машину посреди рабочего дня.",
    "Виртуализация запускает чужие программы в изолированной песочнице.",
    "Логи рассказывают историю каждой ошибки очень подробно.",
    "Резервные копии спасали меня от катастрофы дважды.",
    "Сообщество отвечает на вопросы быстрее платной поддержки.",
    "Лицензия не требует ни копейки за тысячу серверов.",
    "Безопасность строится на прозрачности а не на секретности.",
]


def _seg(start: float, end: float, text: str) -> Segment:
    toks = text.split()
    words: list[Word] = []
    if toks:
        step = (end - start) / len(toks)
        t = start
        for tok in toks:
            words.append(Word(tok, round(t, 3), round(min(t + step * 0.8, end), 3)))
            t += step
    return Segment(start, end, text, words)


def _mk_segs(texts: list[str], seg_dur: float = 5.0) -> list[Segment]:
    return [_seg(i * seg_dur, (i + 1) * seg_dur, t) for i, t in enumerate(texts)]


def _tr(segs: list[Segment], duration: float | None = None) -> Transcript:
    d = duration if duration is not None else (segs[-1].end if segs else 0.0)
    return Transcript(language="ru", duration=d, model="test", audio_hash="h",
                      segments=segs)


def _cl(duration: float, cuts: list[CutSegment] = ()) -> CutList:
    return CutList(source="test.mp4", duration=duration, segments=list(cuts))


def _hook(text: str, n: int = 5) -> str:
    return " ".join(text.split()[:n])


class MockLLM:
    """Strict mock: a plain object exposing chat_json (plan F1 — no real LLM)."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls: list[dict] = []

    def chat_json(self, system, user, schema, keep_alive=None):
        self.calls.append({"system": system, "user": user, "schema": schema,
                           "keep_alive": keep_alive})
        if not self._responses:
            return {"clips": []}
        r = self._responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


_SILENT = lambda *a, **k: None  # noqa: E731


# --- 1. basics: original coordinates, global indices ---------------------------
def test_suggest_basic_original_coords():
    segs = _mk_segs(_T[:10])
    tr = _tr(segs)
    llm = MockLLM([{"clips": [{"start_index": 1, "end_index": 5, "score": 85,
                               "hook_phrase": _hook(_T[1]), "reason": "тест"}]}])
    out = suggest(tr, _cl(tr.duration), ClipsCfg(), LlmCfg(), llm, log=_SILENT)
    assert len(out) == 1
    c = out[0]
    assert isinstance(c, ClipCandidate)
    assert c.id == "c01"
    assert (c.seg_start, c.seg_end) == (1, 5)
    # word-snap по хуку + lead-in −0.15 (клампится к концу предыдущего слова)
    assert 4.5 <= c.start <= 5.0
    # хвост +0.25 за последним словом, не дальше начала следующего
    assert 29.0 <= c.end <= 30.25
    assert c.score == 85 and c.score_window == 85   # MVP: score = окно-скор
    assert c.dur_raw == pytest.approx(c.end - c.start)
    assert c.dur_eff == pytest.approx(c.dur_raw)    # вырезов нет
    assert not c.fuzzy_boundary and not c.short
    assert c.source_window == 0


def test_windows_offset_to_global_keep_alive_and_prompt_format():
    segs = _mk_segs(_T[:16])
    tr = _tr(segs)
    llm = MockLLM([
        {"clips": []},
        {"clips": [{"start_index": 1, "end_index": 5, "score": 80,
                    "hook_phrase": _hook(_T[9]), "reason": "r"}]},
    ])
    out = suggest(tr, _cl(tr.duration), ClipsCfg(window_overlap=2),
                  LlmCfg(max_segments_per_call=10), llm, log=_SILENT)
    assert len(llm.calls) == 2
    # keep_alive: cfg.keep_alive_between между окнами, 0 на последнем
    assert llm.calls[0]["keep_alive"] == 300
    assert llm.calls[1]["keep_alive"] == 0
    # окно 2 = segments[8:16] → локальный 1 = глобальный 9
    assert len(out) == 1
    c = out[0]
    assert (c.seg_start, c.seg_end) == (9, 13)
    assert abs(c.start - 45.0) < 0.3
    assert c.source_window == 1
    # формат user-промпта — план §3.2 (номер | mm:ss | текст)
    lines = llm.calls[1]["user"].split("\n")
    assert lines[0] == "Сегменты расшифровки (номер | время начала | текст):"
    assert lines[1] == f"0 | 0:40 | {_T[8]}"
    assert lines[-2] == ""
    assert lines[-1] == ("Верни JSON: {\"clips\": [{\"start_index\": N, "
                         "\"end_index\": N, \"score\": N, \"hook_phrase\": "
                         "\"...\", \"reason\": \"...\"}]}")
    # system/схема уезжают в chat_json как есть
    assert llm.calls[0]["system"] == clips._SYSTEM
    assert llm.calls[0]["schema"] is clips._SCHEMA


def test_system_prompt_and_schema_invariants():
    # ключевые формулировки проверенного v3-промпта (не «улучшать»!)
    assert "от 4 до 8 ПОДРЯД идущих сегментов" in clips._SYSTEM
    assert "начинается с ЗАГЛАВНОЙ буквы" in clips._SYSTEM
    assert "Не больше 3 фрагментов" in clips._SYSTEM
    assert clips._SCHEMA["required"] == ["clips"]
    item = clips._SCHEMA["properties"]["clips"]["items"]
    assert item["required"] == ["start_index", "end_index", "score",
                                "hook_phrase", "reason"]


# --- 2. duration: trim / drop / mark -------------------------------------------
def test_trim_to_sentence_boundary_under_max():
    # 14 сегментов × 5с = 70с; сегмент 11 не оканчивает предложение → трим к 10
    texts = list(_T[:14])
    texts[11] = texts[11].rstrip(".")
    segs = _mk_segs(texts)
    tr = _tr(segs)
    llm = MockLLM([{"clips": [{"start_index": 0, "end_index": 13, "score": 90,
                               "hook_phrase": _hook(texts[0]), "reason": "r"}]}])
    out = suggest(tr, _cl(tr.duration), ClipsCfg(), LlmCfg(), llm, log=_SILENT)
    assert len(out) == 1
    c = out[0]
    assert c.seg_end == 10                  # самый дальний с ≤60с И границей предложения
    assert c.dur_raw <= 60.0 + 1e-6
    assert not c.fuzzy_boundary


def test_trim_without_sentence_boundary_sets_fuzzy():
    texts = [t.rstrip(".") for t in _T[:14]]    # нигде нет конца предложения
    segs = _mk_segs(texts)
    tr = _tr(segs)
    llm = MockLLM([{"clips": [{"start_index": 0, "end_index": 13, "score": 90,
                               "hook_phrase": _hook(texts[0]), "reason": "r"}]}])
    out = suggest(tr, _cl(tr.duration), ClipsCfg(), LlmCfg(), llm, log=_SILENT)
    assert len(out) == 1
    c = out[0]
    assert c.seg_end == 11                  # просто ≤60с
    assert c.dur_raw <= 60.0 + 1e-6
    assert c.fuzzy_boundary is True


def test_drop_below_hard_min_effective_duration():
    segs = _mk_segs(_T[:4])                 # raw 20с
    tr = _tr(segs)
    cut = CutSegment(id="p1", start=5.0, end=13.0, type=TYPE_PAUSE,
                     action=ACTION_REMOVE, enabled=True)   # −8с → eff 12с
    llm = MockLLM([{"clips": [{"start_index": 0, "end_index": 3, "score": 90,
                               "hook_phrase": _hook(_T[0]), "reason": "r"}]}])
    out = suggest(tr, _cl(tr.duration, [cut]), ClipsCfg(), LlmCfg(), llm,
                  log=_SILENT)
    assert out == []


def test_short_mark_between_hard_min_and_target():
    segs = _mk_segs(_T[:4])                 # raw 20с
    tr = _tr(segs)
    cut = CutSegment(id="p1", start=5.0, end=8.0, type=TYPE_PAUSE,
                     action=ACTION_REMOVE, enabled=True)   # −3с → eff 17с
    llm = MockLLM([{"clips": [{"start_index": 0, "end_index": 3, "score": 90,
                               "hook_phrase": _hook(_T[0]), "reason": "r"}]}])
    out = suggest(tr, _cl(tr.duration, [cut]), ClipsCfg(), LlmCfg(), llm,
                  log=_SILENT)
    assert len(out) == 1
    assert out[0].short is True
    assert 16.0 <= out[0].dur_eff <= 18.0


# --- 3. zone-dependent lowercase-guard ------------------------------------------
def test_lowercase_guard_drops_in_punctuated_zone():
    texts = list(_T[:12])
    texts[2] = texts[2][0].lower() + texts[2][1:]
    segs = _mk_segs(texts)
    tr = _tr(segs)
    llm = MockLLM([{"clips": [{"start_index": 2, "end_index": 5, "score": 90,
                               "hook_phrase": _hook(texts[2]), "reason": "r"}]}])
    out = suggest(tr, _cl(tr.duration), ClipsCfg(), LlmCfg(), llm, log=_SILENT)
    assert out == []


def test_lowercase_guard_keeps_fuzzy_in_unpunctuated_zone():
    texts = [t.lower().rstrip(".") for t in _T[:12]]   # Whisper-стиль, без пунктуации
    segs = _mk_segs(texts)
    tr = _tr(segs)
    llm = MockLLM([{"clips": [{"start_index": 2, "end_index": 5, "score": 90,
                               "hook_phrase": _hook(texts[2]), "reason": "r"}]}])
    out = suggest(tr, _cl(tr.duration), ClipsCfg(), LlmCfg(), llm, log=_SILENT)
    assert len(out) == 1
    assert out[0].fuzzy_boundary is True
    assert (out[0].seg_start, out[0].seg_end) == (2, 5)


# --- 4. dedup --------------------------------------------------------------------
def test_dedup_time_overlap_keeps_higher_window_score():
    segs = _mk_segs(_T[:8])
    tr = _tr(segs)
    llm = MockLLM([{"clips": [
        {"start_index": 0, "end_index": 5, "score": 90,
         "hook_phrase": _hook(_T[0]), "reason": "a"},
        {"start_index": 2, "end_index": 7, "score": 80,        # IoU = 20/40 = 0.5
         "hook_phrase": _hook(_T[2]), "reason": "b"},
    ]}])
    out = suggest(tr, _cl(tr.duration), ClipsCfg(), LlmCfg(), llm, log=_SILENT)
    assert len(out) == 1
    assert out[0].seg_start == 0 and out[0].score == 90


def test_retake_text_dedup_prefers_fewer_enabled_cuts():
    # сегменты 6–9 — дословный ретейк сегментов 0–3 (ratio = 1.0, IoU = 0)
    texts = _T[:4] + [_T[12], _T[13]] + _T[:4]
    segs = _mk_segs(texts)
    tr = _tr(segs)
    cut = CutSegment(id="p1", start=2.0, end=3.0, type=TYPE_PAUSE,
                     action=ACTION_REMOVE, enabled=True)   # заминка в ПЕРВОЙ копии
    llm = MockLLM([{"clips": [
        {"start_index": 0, "end_index": 3, "score": 90,
         "hook_phrase": _hook(_T[0]), "reason": "a"},
        {"start_index": 6, "end_index": 9, "score": 80,
         "hook_phrase": _hook(_T[0]), "reason": "b"},
    ]}])
    out = suggest(tr, _cl(tr.duration, [cut]), ClipsCfg(), LlmCfg(), llm,
                  log=_SILENT)
    assert len(out) == 1
    # решение №5: выживает экземпляр с МЕНЬШИМ числом enabled-вырезов внутри,
    # даже при меньшем окно-скоре
    assert out[0].seg_start == 6


def test_retake_text_dedup_tie_prefers_bigger_score():
    texts = _T[:4] + [_T[12], _T[13]] + _T[:4]
    segs = _mk_segs(texts)
    tr = _tr(segs)                                      # вырезов нет — ничья
    llm = MockLLM([{"clips": [
        {"start_index": 0, "end_index": 3, "score": 90,
         "hook_phrase": _hook(_T[0]), "reason": "a"},
        {"start_index": 6, "end_index": 9, "score": 80,
         "hook_phrase": _hook(_T[0]), "reason": "b"},
    ]}])
    out = suggest(tr, _cl(tr.duration), ClipsCfg(), LlmCfg(), llm, log=_SILENT)
    assert len(out) == 1
    assert out[0].seg_start == 0 and out[0].score == 90


# --- 5. made-up hook → tail; anaphoric opener → soft demote ----------------------
def test_made_up_hook_demotes_to_tail():
    segs = _mk_segs(_T[:10])
    tr = _tr(segs)
    llm = MockLLM([{"clips": [
        {"start_index": 0, "end_index": 3, "score": 95,
         "hook_phrase": "пингвины тайно захватили весь космос", "reason": "a"},
        {"start_index": 6, "end_index": 9, "score": 60,
         "hook_phrase": _hook(_T[6]), "reason": "b"},
    ]}])
    out = suggest(tr, _cl(tr.duration), ClipsCfg(), LlmCfg(), llm, log=_SILENT)
    # честный хук выше, выдуманный — в хвосте (несмотря на скор 95)
    assert [c.seg_start for c in out] == [6, 0]
    assert [c.id for c in out] == ["c01", "c02"]


def test_anaphoric_opener_soft_demoted():
    texts = list(_T[:10])
    texts[0] = "То есть никакой магии в этом нет совсем."
    segs = _mk_segs(texts)
    tr = _tr(segs)
    llm = MockLLM([{"clips": [
        {"start_index": 0, "end_index": 3, "score": 90,
         "hook_phrase": _hook(texts[0]), "reason": "a"},
        {"start_index": 6, "end_index": 9, "score": 70,
         "hook_phrase": _hook(texts[6]), "reason": "b"},
    ]}])
    out = suggest(tr, _cl(tr.duration), ClipsCfg(), LlmCfg(), llm, log=_SILENT)
    assert [c.seg_start for c in out] == [6, 0]


# --- 6. failed window skipped; per-window stage/progress -------------------------
def test_failed_window_skipped_with_stage_and_progress():
    segs = _mk_segs(_T[:16])
    tr = _tr(segs)
    llm = MockLLM([
        RuntimeError("boom"),
        {"clips": [{"start_index": 1, "end_index": 5, "score": 75,
                    "hook_phrase": _hook(_T[9]), "reason": "r"}]},
    ])
    stages: list[str] = []
    progress: list[float] = []
    logs: list[str] = []
    out = suggest(tr, _cl(tr.duration), ClipsCfg(window_overlap=2),
                  LlmCfg(max_segments_per_call=10), llm, log=logs.append,
                  on_stage=stages.append, on_progress=progress.append)
    assert len(out) == 1 and out[0].source_window == 1
    assert stages == ["Клипы… 1/2", "Клипы… 2/2"]
    assert progress[:2] == [0.0, 0.5] and progress[-1] == 1.0
    assert any("[0:10]" in m and "boom" in m for m in logs)


# --- 7. junk from the model is quietly dropped -----------------------------------
def test_junk_values_quietly_dropped():
    segs = _mk_segs(_T[:10])
    tr = _tr(segs)
    llm = MockLLM([{"clips": [
        {"start_index": True, "end_index": 3, "score": 90,
         "hook_phrase": "x", "reason": "r"},                   # bool
        {"start_index": "2", "end_index": 5, "score": 90,
         "hook_phrase": "x", "reason": "r"},                   # строка
        {"start_index": -1, "end_index": 3, "score": 90,
         "hook_phrase": "x", "reason": "r"},                   # отрицательный
        {"start_index": 3, "end_index": 99, "score": 90,
         "hook_phrase": "x", "reason": "r"},                   # вне диапазона
        {"start_index": 5, "end_index": 2, "score": 90,
         "hook_phrase": "x", "reason": "r"},                   # start > end
        {"start_index": 0, "end_index": 3, "score": "высокий",
         "hook_phrase": "x", "reason": "r"},                   # скор-строка
        "не словарь",
        {"start_index": 0, "end_index": 3, "score": 80,
         "hook_phrase": _hook(_T[0]), "reason": "ок"},         # валидный
    ]}])
    out = suggest(tr, _cl(tr.duration), ClipsCfg(), LlmCfg(), llm, log=_SILENT)
    assert len(out) == 1
    assert (out[0].seg_start, out[0].seg_end) == (0, 3)


def test_missing_or_non_list_clips_yields_empty():
    segs = _mk_segs(_T[:6])
    tr = _tr(segs)
    out = suggest(tr, _cl(tr.duration), ClipsCfg(), LlmCfg(),
                  MockLLM([{"foo": 1}]), log=_SILENT)
    assert out == []
    out = suggest(tr, _cl(tr.duration), ClipsCfg(), LlmCfg(),
                  MockLLM([{"clips": "мусор"}]), log=_SILENT)
    assert out == []


def test_empty_transcript_or_no_llm_returns_empty():
    assert suggest(_tr([]), _cl(0.0), ClipsCfg(), LlmCfg(), MockLLM([]),
                   log=_SILENT) == []
    segs = _mk_segs(_T[:4])
    tr = _tr(segs)
    assert suggest(tr, _cl(tr.duration), ClipsCfg(), LlmCfg(), None,
                   log=_SILENT) == []


# --- §3.6.6: >50% диапазона в removed → дроп --------------------------------------
def test_candidate_mostly_inside_removed_is_dropped():
    segs = _mk_segs(_T[:8])                 # кандидат 0–7 = 40с
    tr = _tr(segs)
    cut = CutSegment(id="b1", start=2.0, end=26.0, type=TYPE_BADTAKE,
                     action=ACTION_REMOVE, enabled=True)   # 24/40 = 60% > 50%
    llm = MockLLM([{"clips": [{"start_index": 0, "end_index": 7, "score": 90,
                               "hook_phrase": _hook(_T[0]), "reason": "r"}]}])
    out = suggest(tr, _cl(tr.duration, [cut]), ClipsCfg(), LlmCfg(), llm,
                  log=_SILENT)
    assert out == []                        # eff было бы 16с — дропает именно правило 50%


# --- MVP-сортировка: окно-скор desc + round-robin по окнам ------------------------
def test_mvp_sort_round_robin_across_windows():
    segs = _mk_segs(_T[:16])
    tr = _tr(segs)
    llm = MockLLM([
        {"clips": [
            {"start_index": 0, "end_index": 3, "score": 90,
             "hook_phrase": _hook(_T[0]), "reason": "r"},
            {"start_index": 4, "end_index": 7, "score": 85,
             "hook_phrase": _hook(_T[4]), "reason": "r"},
        ]},
        {"clips": [                          # окно 2: segments[8:16]
            {"start_index": 0, "end_index": 3, "score": 88,
             "hook_phrase": _hook(_T[8]), "reason": "r"},
            {"start_index": 4, "end_index": 7, "score": 70,
             "hook_phrase": _hook(_T[12]), "reason": "r"},
        ]},
    ])
    out = suggest(tr, _cl(tr.duration), ClipsCfg(window_overlap=2),
                  LlmCfg(max_segments_per_call=10), llm, log=_SILENT)
    assert [c.score for c in out] == [90, 88, 85, 70]
    assert [c.source_window for c in out] == [0, 1, 0, 1]
    assert [c.id for c in out] == ["c01", "c02", "c03", "c04"]


def test_max_candidates_cap_and_ids():
    segs = _mk_segs(_T[:12])
    tr = _tr(segs)
    llm = MockLLM([{"clips": [
        {"start_index": 0, "end_index": 3, "score": 90,
         "hook_phrase": _hook(_T[0]), "reason": "r"},
        {"start_index": 4, "end_index": 7, "score": 80,
         "hook_phrase": _hook(_T[4]), "reason": "r"},
        {"start_index": 8, "end_index": 11, "score": 70,
         "hook_phrase": _hook(_T[8]), "reason": "r"},
    ]}])
    out = suggest(tr, _cl(tr.duration), ClipsCfg(max_candidates=2), LlmCfg(),
                  llm, log=_SILENT)
    assert [c.id for c in out] == ["c01", "c02"]
    assert [c.score for c in out] == [90, 80]


# --- word-snap + снап к краям pause-сегментов -------------------------------------
def test_boundaries_snap_off_pause_segments():
    s0 = Segment(0.0, 5.0, "Привет это очень важный тест.",
                 [Word("Привет", 2.2, 2.5), Word("это", 2.6, 2.8),
                  Word("очень", 2.85, 3.1), Word("важный", 3.2, 3.6),
                  Word("тест.", 3.7, 4.0)])
    s3 = Segment(15.0, 20.0, "Конец мысли наступает именно здесь.",
                 [Word("Конец", 15.2, 15.6), Word("мысли", 15.7, 16.1),
                  Word("наступает", 16.2, 17.0), Word("именно", 17.1, 17.6),
                  Word("здесь.", 17.8, 19.4)])
    segs = [s0, _seg(5.0, 10.0, _T[1]), _seg(10.0, 15.0, _T[2]), s3]
    tr = _tr(segs, duration=20.0)
    cuts = [CutSegment(id="p1", start=0.0, end=2.1, type=TYPE_PAUSE,
                       action=ACTION_REMOVE, enabled=True),
            CutSegment(id="p2", start=19.45, end=20.0, type=TYPE_PAUSE,
                       action=ACTION_REMOVE, enabled=True)]
    llm = MockLLM([{"clips": [{"start_index": 0, "end_index": 3, "score": 85,
                               "hook_phrase": "Привет это очень важный тест",
                               "reason": "r"}]}])
    out = suggest(tr, _cl(20.0, cuts), ClipsCfg(), LlmCfg(), llm, log=_SILENT)
    assert len(out) == 1
    c = out[0]
    # старт по хуку: 2.2 − 0.15 = 2.05 попал в паузу (0,2.1) → к правому краю
    assert c.start == pytest.approx(2.1, abs=0.01)
    # конец: 19.4 + 0.25 = 19.65 попал в паузу (19.45,20) → к левому краю
    assert c.end == pytest.approx(19.45, abs=0.01)


# --- residual non-overlap pass (live Prod9 finding: c04/c08 shared seg 279) --
def test_residual_overlap_below_dedup_thresholds_is_dropped():
    """Two clips overlapping ~25% of the shorter (IoU<0.5, containment<0.7)
    survive _dedup but violate the product promise — the lower-scored one
    must be dropped by _drop_residual_overlaps."""
    from vpipe.clips import _Raw, _dedup, _drop_residual_overlaps
    a = _Raw(seg_start=273, seg_end=279, t0=1380.5, t1=1425.5, score_window=85,
             hook_phrase="при этом я", reason="", source_window=4)
    b = _Raw(seg_start=279, seg_end=282, t0=1415.1, t1=1456.0, score_window=80,
             hook_phrase="При этом, если", reason="", source_window=4)
    segs = []  # text dedup not exercised here
    kept = _drop_residual_overlaps([a, b])
    assert [k.score_window for k in kept] == [85], "lower-scored overlapper must go"


def test_tiny_boundary_kiss_survives_overlap_pass():
    from vpipe.clips import _Raw, _drop_residual_overlaps
    a = _Raw(seg_start=0, seg_end=5, t0=0.0, t1=40.0, score_window=90,
             hook_phrase="x", reason="", source_window=0)
    b = _Raw(seg_start=5, seg_end=9, t0=38.0, t1=70.0, score_window=80,
             hook_phrase="y", reason="", source_window=0)  # 2s/32s = 6% < 20%
    kept = _drop_residual_overlaps([a, b])
    assert len(kept) == 2


# === F6: one-call re-rank (§3.5) ===================================================
# Setup: 2 windows ([0:10] и [8:16] при max_segments_per_call=10, overlap=2),
# 4 непересекающихся кандидата. Round-robin base order (исходный порядок для
# ремонта ответа): seg_start [0, 8, 4, 12] → re-rank id 1→0, 2→8, 3→4, 4→12.
def _rerank_setup():
    segs = _mk_segs(_T[:16])
    tr = _tr(segs)
    win_responses = [
        {"clips": [
            {"start_index": 0, "end_index": 3, "score": 90,
             "hook_phrase": _hook(_T[0]), "reason": "r"},
            {"start_index": 4, "end_index": 7, "score": 85,
             "hook_phrase": _hook(_T[4]), "reason": "r"},
        ]},
        {"clips": [                          # окно 2: segments[8:16]
            {"start_index": 0, "end_index": 3, "score": 88,
             "hook_phrase": _hook(_T[8]), "reason": "r"},
            {"start_index": 4, "end_index": 7, "score": 70,
             "hook_phrase": _hook(_T[12]), "reason": "r"},
        ]},
    ]
    return tr, win_responses


def _suggest2(tr, llm, **cfg_kw):
    return suggest(tr, _cl(tr.duration), ClipsCfg(window_overlap=2, **cfg_kw),
                   LlmCfg(max_segments_per_call=10), llm, log=_SILENT)


def test_rerank_order_applied_scores_replaced():
    tr, win = _rerank_setup()
    llm = MockLLM(win + [{"ranking": [
        {"id": 4, "score": 97}, {"id": 2, "score": 80},
        {"id": 1, "score": 64}, {"id": 3, "score": 41}]}])
    out = _suggest2(tr, llm)
    assert len(llm.calls) == 3                       # 2 окна + 1 re-rank
    # порядок задаёт re-rank, НЕ окно-скоры и НЕ round-robin
    assert [c.seg_start for c in out] == [12, 8, 0, 4]
    # re-rank-скоры замещают окно-скоры; окно-скор сохранён отдельно
    assert [c.score for c in out] == [97, 80, 64, 41]
    assert [c.score_window for c in out] == [70, 88, 90, 85]
    assert all(c.rank_source == "llm" for c in out)
    assert [c.id for c in out] == ["c01", "c02", "c03", "c04"]


def test_rerank_prompt_ids_are_one_based_with_hook_dur_text():
    from vpipe import clips as clips_mod
    tr, win = _rerank_setup()
    llm = MockLLM(win + [{"ranking": [{"id": 1, "score": 90}]}])
    _suggest2(tr, llm)
    rcall = llm.calls[2]
    # system/схема — §3.5, как в остальных вызовах clips.py; keep_alive=0
    assert rcall["system"] == clips_mod._RERANK_SYSTEM
    assert rcall["schema"] is clips_mod._RERANK_SCHEMA
    assert rcall["keep_alive"] == 0
    body = [ln for ln in rcall["user"].split("\n") if " | " in ln and "хук:" in ln]
    # id СТРОГО с 1 (0-based теряет нулевой элемент — доказанный баг LLM)
    assert [ln.split(" | ")[0] for ln in body] == ["1", "2", "3", "4"]
    assert not any(ln.startswith("0 |") for ln in rcall["user"].split("\n"))
    # id=1 — лучший по round-robin (окно-скор 90, сегмент 0): хук + текст + 20с
    assert _hook(_T[0]) in body[0] and _T[1] in body[0] and "| 20с |" in body[0]
    assert _hook(_T[8]) in body[1]                  # id=2 — сегмент 8 (окно 2)
    assert "\"ranking\"" in rcall["user"]


def test_rerank_unknown_ids_ignored_missing_appended_in_base_order():
    tr, win = _rerank_setup()
    llm = MockLLM(win + [{"ranking": [
        {"id": 99, "score": 99}, {"id": 0, "score": 88},   # неизвестные (0 — не наш!)
        {"id": 3, "score": 77}]}])
    out = _suggest2(tr, llm)
    # валидный id=3 (сегмент 4) — первым; пропущенные 1,2,4 — хвост в исходном
    # round-robin порядке [0, 8, 12]
    assert [c.seg_start for c in out] == [4, 0, 8, 12]
    # хвост без re-rank-скора сохраняет окно-скор
    assert [c.score for c in out] == [77, 90, 88, 70]
    assert all(c.rank_source == "llm" for c in out)


def test_rerank_duplicate_ids_first_occurrence_wins():
    tr, win = _rerank_setup()
    llm = MockLLM(win + [{"ranking": [
        {"id": 2, "score": 91}, {"id": 2, "score": 15},    # дубль схлопнут
        {"id": 1, "score": 50}]}])
    out = _suggest2(tr, llm)
    assert [c.seg_start for c in out] == [8, 0, 4, 12]
    assert out[0].score == 91                              # скор первого вхождения
    assert [c.score for c in out] == [91, 50, 85, 70]


@pytest.mark.parametrize("bad", [
    {"ranking": "мусор"},                                  # не список
    {"foo": 1},                                            # нет ranking
    {"ranking": []},                                       # пустой
    {"ranking": ["x", 5, {"score": 9}, {"id": True, "score": 9},
                 {"id": "1", "score": 9}]},                # ни одного валидного id
])
def test_rerank_garbage_reply_falls_back_to_round_robin(bad):
    tr, win = _rerank_setup()
    out = _suggest2(tr, MockLLM(win + [bad]))
    assert [c.seg_start for c in out] == [0, 8, 4, 12]     # round-robin
    assert [c.score for c in out] == [90, 88, 85, 70]      # окно-скоры
    assert all(c.rank_source == "round_robin" for c in out)


def test_rerank_llm_exception_falls_back_with_log():
    tr, win = _rerank_setup()
    logs: list[str] = []
    llm = MockLLM(win + [RuntimeError("rerank boom")])
    out = suggest(tr, _cl(tr.duration), ClipsCfg(window_overlap=2),
                  LlmCfg(max_segments_per_call=10), llm, log=logs.append)
    assert [c.seg_start for c in out] == [0, 8, 4, 12]
    assert all(c.rank_source == "round_robin" for c in out)
    assert any("re-rank" in m and "rerank boom" in m for m in logs)


def test_rerank_disabled_no_extra_call_round_robin():
    tr, win = _rerank_setup()
    llm = MockLLM(list(win))
    out = _suggest2(tr, llm, rerank=False)
    assert len(llm.calls) == 2                             # только окна
    assert [c.seg_start for c in out] == [0, 8, 4, 12]
    assert [c.score for c in out] == [90, 88, 85, 70]
    assert all(c.rank_source == "round_robin" for c in out)


def test_rerank_skipped_for_single_candidate():
    segs = _mk_segs(_T[:10])
    tr = _tr(segs)
    llm = MockLLM([{"clips": [{"start_index": 1, "end_index": 5, "score": 85,
                               "hook_phrase": _hook(_T[1]), "reason": "r"}]}])
    out = suggest(tr, _cl(tr.duration), ClipsCfg(), LlmCfg(), llm, log=_SILENT)
    assert len(llm.calls) == 1                             # сравнивать нечего
    assert len(out) == 1 and out[0].rank_source == "round_robin"


def test_rerank_pool_capped_tail_keeps_order(monkeypatch):
    from vpipe import clips as clips_mod
    monkeypatch.setattr(clips_mod, "_RERANK_MAX", 2)
    segs = _mk_segs(_T[:12])
    tr = _tr(segs)
    llm = MockLLM([
        {"clips": [
            {"start_index": 0, "end_index": 3, "score": 90,
             "hook_phrase": _hook(_T[0]), "reason": "r"},
            {"start_index": 4, "end_index": 7, "score": 85,
             "hook_phrase": _hook(_T[4]), "reason": "r"},
            {"start_index": 8, "end_index": 11, "score": 80,
             "hook_phrase": _hook(_T[8]), "reason": "r"},
        ]},
        {"ranking": [{"id": 2, "score": 99}, {"id": 1, "score": 55}]},
    ])
    out = suggest(tr, _cl(tr.duration), ClipsCfg(), LlmCfg(), llm, log=_SILENT)
    # в промпт ушли только топ-2 (бюджет токенов §3.5)
    body = [ln for ln in llm.calls[1]["user"].split("\n") if "хук:" in ln]
    assert len(body) == 2
    # третий кандидат — за пулом, идёт хвостом в прежнем порядке
    assert [c.seg_start for c in out] == [4, 0, 8]
    assert [c.score for c in out] == [99, 55, 80]


def test_rerank_long_text_clipped_head_and_tail():
    from vpipe import clips as clips_mod
    roots = ["абрикос", "берёза", "вулкан", "гитара",
             "дельфин", "ландыш", "жираф", "закат"]
    texts = [f"Сегмент {i} " + " ".join(f"{roots[i]}{j}" for j in range(24)) + "."
             for i in range(8)]
    segs = _mk_segs(texts)
    tr = _tr(segs)
    llm = MockLLM([{"clips": [
        {"start_index": 0, "end_index": 3, "score": 90,
         "hook_phrase": _hook(texts[0]), "reason": "r"},
        {"start_index": 4, "end_index": 7, "score": 80,
         "hook_phrase": _hook(texts[4]), "reason": "r"},
    ]}])
    suggest(tr, _cl(tr.duration), ClipsCfg(), LlmCfg(), llm, log=_SILENT)
    line = next(ln for ln in llm.calls[1]["user"].split("\n")
                if ln.startswith("1 |"))
    full = " ".join(texts[0:4])
    assert len(full) > clips_mod._RERANK_TEXT_MAX       # есть что резать
    assert full not in line                             # текст усечён…
    assert " … " in line                                # …голова + … + хвост
    assert "Сегмент 0" in line                          # голова на месте
    assert f"{roots[3]}23." in line                     # хвост (конец мысли) на месте


# --- round-robin сам по себе: 3 окна, чистый юнит на _sort_mvp ---------------------
def test_round_robin_three_windows_unit():
    from vpipe.clips import _Raw, _sort_mvp

    def r(w, score, t0):
        return _Raw(seg_start=0, seg_end=0, t0=t0, t1=t0 + 30.0,
                    score_window=score, hook_phrase="h", reason="",
                    source_window=w)

    cands = [r(0, 90, 0), r(0, 70, 100), r(1, 95, 200), r(1, 60, 300),
             r(2, 80, 400)]
    out = _sort_mvp(cands)
    # по одному лучшему из каждого окна по кругу: ряд 1 [95, 90, 80], ряд 2 [70, 60]
    assert [(c.source_window, c.score_window) for c in out] == [
        (1, 95), (0, 90), (2, 80), (0, 70), (1, 60)]
