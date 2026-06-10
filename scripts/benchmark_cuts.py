#!/usr/bin/env python
"""Бенчмарк качества автоматических резов FastVideoEdit.

Закрывает «нулевую доказуемость» качества монтажа: по уже обработанным клипам
(транскрипт берётся СТРОГО из кэша — GPU/Whisper НЕ запускается; нет кэша →
клип пропускается с пометкой) прогоняется штатная детекция (``run_detection``,
``llm=None``) и считаются измеримые метрики:

SAFETY — целостность слов
    % авто-вырезов (все типы, КРОМЕ ``profanity``/``manual``), которые «клиппят»
    слово транскрипта: пересекают его более чем на 12 мс, НЕ удаляя слово
    целиком. Цель — 0 %. Полное поглощение слова вырезом (работа филлер-
    детектора) нарушением не считается — нарушение только частичный «огрызок».

CLEANLINESS — прокси-метрики чистоты результата
    (а) филлеры: сколько слов транскрипта матчится словарю ``fillers_ru.yaml``
        (mumbles длиной >= 2 после нормализации + words) и какой % из них
        покрыт вырезами (покрыто >= 50 % длительности слова);
    (б) паузы: сколько меж-словных гэпов > 0.9 с осталось фактически
        непокрытыми (непокрытый остаток гэпа всё ещё > 0.9 с);
    (в) заминки: сколько VAD-гэпов 0.2–0.55 с осталось не покрытыми вырезами
        (Silero VAD через ``vpipe.detect.hesitations._get_non_speech_gaps``).

СВОДКА
    Вырезов по типам, удалено секунд / % длительности, итоговая длительность.

Запуск (из корня репозитория)::

    .\\.venv\\Scripts\\python.exe scripts\\benchmark_cuts.py
    .\\.venv\\Scripts\\python.exe scripts\\benchmark_cuts.py 3.mp4 --clip extra.mp4
    .\\.venv\\Scripts\\python.exe scripts\\benchmark_cuts.py --no-vad

Результаты: ``benchmark_results.md`` + ``benchmark_results.json`` в корне
репозитория (UTF-8); в консоль — краткий ASCII-итог (безопасно для cp1251).

Метрики реализованы чистыми функциями (без I/O) — их покрывает
``tests/test_benchmark.py`` на синтетических данных, без ffmpeg/GPU.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vpipe.config import (Config, FillerLists, ProfanityLists, load_config,  # noqa: E402
                          load_fillers, load_profanity)
from vpipe.detect import run_detection                                       # noqa: E402
from vpipe.detect.fillers import (_MIN_MUMBLE_LEN, _compile_mumble_matcher,  # noqa: E402
                                  _compile_words_matcher)
from vpipe.models import (ACTION_REMOVE, TYPE_MANUAL, TYPE_PROFANITY,        # noqa: E402
                          CutSegment, Transcript, Word)
from vpipe.probe import hash_input                                           # noqa: E402
from vpipe.textnorm import normalize                                         # noqa: E402

# --- Параметры метрик (дефолты; вынесены в аргументы функций для тестов) ----
WORD_CLIP_TOLERANCE = 0.012   # 12 мс — допуск пересечения выреза со словом
LONG_GAP_THRESHOLD = 0.9      # меж-словный гэп длиннее этого = «длинная пауза»
HES_GAP_MIN = 0.2             # нижняя граница VAD-гэпа «заминки» (с)
HES_GAP_MAX = 0.55            # верхняя граница (не включается; выше = пауза)
COVER_FRACTION = 0.5          # слово/гэп «покрыт», если вырезано >= этой доли

# Типы, исключённые из SAFETY: цензура режет слово НАМЕРЕННО (это её работа),
# ручные вырезы — осознанное решение пользователя, не авто-детекторов.
SAFETY_EXCLUDED_TYPES = frozenset({TYPE_PROFANITY, TYPE_MANUAL})

DEFAULT_CLIPS = ("3.mp4", "2.mp4", "1.mp4")

_EPS = 1e-9


# =============================================================================
# Чистые функции-метрики (без I/O) — покрыты tests/test_benchmark.py
# =============================================================================

def merge_spans(spans: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    """Слить интервалы в непересекающееся отсортированное объединение.

    Инвертированные/пустые интервалы отбрасываются; касающиеся — сливаются.
    """
    valid = sorted((a, b) for a, b in spans if b > a)
    out: list[tuple[float, float]] = []
    for a, b in valid:
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def covered_length(a: float, b: float,
                   merged: list[tuple[float, float]]) -> float:
    """Длина части ``[a, b]``, покрытой объединением ``merged`` интервалов."""
    if b <= a:
        return 0.0
    total = 0.0
    for s, e in merged:
        if s >= b:
            break
        ov = min(b, e) - max(a, s)
        if ov > 0:
            total += ov
    return total


def remove_spans(cuts: Iterable[CutSegment]) -> list[tuple[float, float]]:
    """Слитые интервалы РЕАЛЬНОГО удаления: enabled + action == remove."""
    return merge_spans((c.start, c.end) for c in cuts
                       if c.enabled and c.action == ACTION_REMOVE)


@dataclass
class WordClipViolation:
    """Один случай «огрызка»: вырез частично откусил слово более чем на 12 мс."""
    cut_type: str
    cut_start: float
    cut_end: float
    word: str
    word_start: float
    word_end: float
    overlap_ms: float

    def to_dict(self) -> dict:
        d = asdict(self)
        for k in ("cut_start", "cut_end", "word_start", "word_end"):
            d[k] = round(d[k], 3)
        d["overlap_ms"] = round(d["overlap_ms"], 1)
        return d


def safety_metrics(words: list[Word], cuts: list[CutSegment],
                   tolerance: float = WORD_CLIP_TOLERANCE) -> dict:
    """SAFETY: авто-вырезы, частично клиппящие слово более чем на ``tolerance``.

    Авто-вырез = enabled + action=remove + тип НЕ в ``SAFETY_EXCLUDED_TYPES``.
    Нарушение: пересечение со словом > tolerance, при этом слово НЕ удалено
    целиком (от слова остаётся «огрызок» длиннее tolerance). Возвращает
    ``{"auto_cuts", "violating_cuts", "violation_pct", "violations"}``.
    """
    auto = [c for c in cuts
            if c.enabled and c.action == ACTION_REMOVE
            and c.type not in SAFETY_EXCLUDED_TYPES]
    violations: list[WordClipViolation] = []
    bad_cuts = 0
    for c in auto:
        cut_violates = False
        for w in words:
            ov = min(c.end, w.end) - max(c.start, w.start)
            if ov - tolerance <= _EPS:
                continue                      # касание в пределах допуска
            residual = (w.end - w.start) - ov
            if residual - tolerance <= _EPS:
                continue                      # слово удалено целиком — намеренно
            cut_violates = True
            violations.append(WordClipViolation(
                cut_type=c.type, cut_start=c.start, cut_end=c.end,
                word=w.word.strip(), word_start=w.start, word_end=w.end,
                overlap_ms=ov * 1000.0))
        if cut_violates:
            bad_cuts += 1
    pct = (100.0 * bad_cuts / len(auto)) if auto else 0.0
    return {"auto_cuts": len(auto), "violating_cuts": bad_cuts,
            "violation_pct": round(pct, 2),
            "violations": [v.to_dict() for v in violations]}


def match_filler_words(words: list[Word], lists: FillerLists) -> list[Word]:
    """Слова транскрипта, матчящиеся словарю филлеров (mumbles >= 2 + words).

    Та же семантика, что у боевого детектора (``vpipe.detect.fillers``):
    нормализация, mumble-регексы только для растянутых форм (len >= 2),
    одиночные слова — точное совпадение. Фразы здесь сознательно не участвуют
    (контекстная метрика, спецификация — mumbles + words).
    """
    mumble_rx = _compile_mumble_matcher(lists)
    words_rx = _compile_words_matcher(lists)
    out: list[Word] = []
    for w in words:
        n = normalize(w.word)
        if not n:
            continue
        is_word = words_rx.match(n) is not None
        is_mumble = len(n) >= _MIN_MUMBLE_LEN and mumble_rx.match(n) is not None
        if is_word or is_mumble:
            out.append(w)
    return out


def filler_coverage(words: list[Word], cuts: list[CutSegment],
                    lists: FillerLists,
                    cover_frac: float = COVER_FRACTION) -> dict:
    """CLEANLINESS (а): какая доля словарных филлеров покрыта вырезами.

    Слово «покрыто», если вырезы (объединение enabled remove) накрывают
    >= ``cover_frac`` его длительности (слово нулевой длины — если его центр
    попал внутрь выреза).
    """
    spans = remove_spans(cuts)
    fillers = match_filler_words(words, lists)
    covered = 0
    for w in fillers:
        dur = w.end - w.start
        if dur <= 0:
            mid = w.start
            if any(s <= mid <= e for s, e in spans):
                covered += 1
            continue
        if covered_length(w.start, w.end, spans) + _EPS >= cover_frac * dur:
            covered += 1
    pct = (100.0 * covered / len(fillers)) if fillers else 100.0
    return {"total": len(fillers), "covered": covered,
            "coverage_pct": round(pct, 2)}


def long_pause_residuals(words: list[Word], cuts: list[CutSegment],
                         gap_threshold: float = LONG_GAP_THRESHOLD) -> dict:
    """CLEANLINESS (б): меж-словные гэпы > порога, оставшиеся длинными.

    Гэп считается «не покрытым», если после вычитания вырезов его непокрытый
    остаток ВСЁ ЕЩЁ длиннее ``gap_threshold`` (внутренние отступы pad_start/
    pad_end детектора пауз — намеренный «воздух», не нарушение).
    """
    spans = remove_spans(cuts)
    ws = sorted(words, key=lambda w: w.start)
    long_gaps = 0
    uncovered = 0
    details: list[dict] = []
    for w0, w1 in zip(ws, ws[1:]):
        gap = w1.start - w0.end
        if gap <= gap_threshold:
            continue
        long_gaps += 1
        residual = gap - covered_length(w0.end, w1.start, spans)
        if residual > gap_threshold + _EPS:
            uncovered += 1
            details.append({"start": round(w0.end, 3), "end": round(w1.start, 3),
                            "gap_s": round(gap, 3),
                            "residual_s": round(residual, 3)})
    return {"long_gaps": long_gaps, "uncovered": uncovered,
            "uncovered_details": details}


def hesitation_residuals(vad_gaps: Iterable[tuple[float, float]],
                         cuts: list[CutSegment],
                         min_duration: float = HES_GAP_MIN,
                         max_duration: float = HES_GAP_MAX,
                         cover_frac: float = COVER_FRACTION) -> dict:
    """CLEANLINESS (в): VAD-гэпы [min, max), не покрытые вырезами.

    Гэп «покрыт», если вырезы накрывают >= ``cover_frac`` его длины. Часть
    непокрытых — норма: word-safe клампинг и дедуп детектора заминок сознательно
    оставляют гэпы, чьё удаление рискует задеть слово.
    """
    spans = remove_spans(cuts)
    total = 0
    uncovered = 0
    for a, b in vad_gaps:
        raw = b - a
        if raw < min_duration or raw >= max_duration:
            continue
        total += 1
        if covered_length(a, b, spans) + _EPS < cover_frac * raw:
            uncovered += 1
    return {"gaps": total, "uncovered": uncovered}


def cut_summary(cuts: list[CutSegment], duration: float) -> dict:
    """СВОДКА: вырезы по типам (enabled), удалено секунд / %, итог."""
    dur = max(0.0, float(duration))
    by_type: dict[str, int] = {}
    for c in cuts:
        if c.enabled:
            by_type[c.type] = by_type.get(c.type, 0) + 1
    spans = merge_spans((max(0.0, c.start), min(dur, c.end) if dur else c.end)
                        for c in cuts
                        if c.enabled and c.action == ACTION_REMOVE)
    removed = sum(b - a for a, b in spans)
    pct = (100.0 * removed / dur) if dur > 0 else 0.0
    return {"cuts_total": sum(by_type.values()), "by_type": by_type,
            "removed_s": round(removed, 2), "removed_pct": round(pct, 2),
            "final_s": round(max(0.0, dur - removed), 2)}


def aggregate_results(analyzed: list[dict]) -> Optional[dict]:
    """Итоговая строка по проанализированным клипам.

    Счётчики суммируются; проценты считаются ЧЕСТНО по суммам (взвешенно),
    а не как среднее средних.
    """
    if not analyzed:
        return None
    dur = sum(r["duration_s"] for r in analyzed)
    removed = sum(r["summary"]["removed_s"] for r in analyzed)
    auto = sum(r["safety"]["auto_cuts"] for r in analyzed)
    bad = sum(r["safety"]["violating_cuts"] for r in analyzed)
    f_total = sum(r["fillers"]["total"] for r in analyzed)
    f_cov = sum(r["fillers"]["covered"] for r in analyzed)
    gaps = sum(r["pauses"]["long_gaps"] for r in analyzed)
    gaps_left = sum(r["pauses"]["uncovered"] for r in analyzed)
    hes = [r["hesitations"] for r in analyzed if r.get("hesitations")]
    by_type: dict[str, int] = {}
    for r in analyzed:
        for t, n in r["summary"]["by_type"].items():
            by_type[t] = by_type.get(t, 0) + n
    return {
        "clips": len(analyzed),
        "duration_s": round(dur, 2),
        "cuts_total": sum(r["summary"]["cuts_total"] for r in analyzed),
        "by_type": by_type,
        "removed_s": round(removed, 2),
        "removed_pct": round(100.0 * removed / dur, 2) if dur > 0 else 0.0,
        "final_s": round(max(0.0, dur - removed), 2),
        "safety": {"auto_cuts": auto, "violating_cuts": bad,
                   "violation_pct": round(100.0 * bad / auto, 2) if auto else 0.0},
        "fillers": {"total": f_total, "covered": f_cov,
                    "coverage_pct": round(100.0 * f_cov / f_total, 2)
                    if f_total else 100.0},
        "pauses": {"long_gaps": gaps, "uncovered": gaps_left},
        "hesitations": ({"gaps": sum(h["gaps"] for h in hes),
                         "uncovered": sum(h["uncovered"] for h in hes)}
                        if hes else None),
    }


# =============================================================================
# I/O: загрузка кэша, извлечение wav, прогон детекции
# =============================================================================

def _ensure_wav(clip: Path, cfg: Config, duration: float,
                audio_hash: str, notes: list[str]) -> Optional[Path]:
    """Вернуть путь к audio16k.wav клипа; извлечь через ffmpeg при отсутствии.

    Кладёт wav туда же, куда его кладёт редактор (``work/{stem}-{hash8}/``),
    чтобы бенчмарк и боевая сессия делили один кэш. ``None`` при любой ошибке
    (ffmpeg недоступен и т.п.) — VAD-метрики тогда просто пропускаются.
    """
    wav = Path(cfg.paths.work_dir) / f"{clip.stem}-{audio_hash[:8]}" / "audio16k.wav"
    if wav.exists():
        return wav
    try:
        from vpipe.ffmpeg_utils import FFmpeg
        from vpipe.probe import extract_audio
        ff = FFmpeg(cfg.ffmpeg)
        extract_audio(ff, clip, wav, total=duration or None)
        return wav
    except Exception as e:  # noqa: BLE001 — бенчмарк не должен падать без ffmpeg
        notes.append("audio16k.wav недоступен "
                     f"({e.__class__.__name__}) — VAD-метрики пропущены")
        return None


def benchmark_clip(clip: str | Path, cfg: Config, fillers: FillerLists,
                   profanity: ProfanityLists, *, vad: bool = True,
                   log=lambda *_a, **_k: None) -> dict:
    """Полный бенчмарк одного клипа. Транскрипт — ТОЛЬКО из кэша (без GPU)."""
    clip = Path(clip)
    res: dict = {"clip": str(clip), "name": clip.name,
                 "skipped": False, "reason": "", "notes": []}
    if not clip.exists():
        res.update(skipped=True, reason="файл не найден")
        return res

    audio_hash = hash_input(clip)
    res["audio_hash"] = audio_hash
    cache_path = Path(cfg.paths.cache_dir) / f"{audio_hash}.transcript.json"
    if not cache_path.exists():
        res.update(skipped=True,
                   reason="нет кэшированного транскрипта — пропущен "
                          "(GPU не запускаем; откройте клип в редакторе один раз)")
        return res
    try:
        tr = Transcript.load(cache_path)
    except Exception as e:  # noqa: BLE001 — битый кэш не должен ронять бенчмарк
        res.update(skipped=True, reason=f"кэш транскрипта не читается: {e}")
        return res

    duration = max(0.0, float(tr.duration))
    res["model"] = tr.model
    res["duration_s"] = round(duration, 2)

    notes: list[str] = res["notes"]
    wav: Optional[Path] = None
    if vad:
        wav = _ensure_wav(clip, cfg, duration, audio_hash, notes)
    else:
        notes.append("VAD отключён (--no-vad) — метрика заминок пропущена")

    cuts = run_detection(tr, cfg, fillers, profanity, source=str(clip),
                         llm=None, log=log, audio_path=wav).segments
    words = tr.all_words()

    vad_gaps: Optional[list[tuple[float, float]]] = None
    if wav is not None:
        try:
            from vpipe.detect.hesitations import _get_non_speech_gaps
            vad_gaps = _get_non_speech_gaps(wav, cfg.hesitations)
        except Exception as e:  # noqa: BLE001 — onnx/wav сбой -> метрика «н/д»
            notes.append(f"VAD не отработал ({e.__class__.__name__}) — "
                         "метрика заминок пропущена")

    res["summary"] = cut_summary(cuts, duration)
    res["safety"] = safety_metrics(words, cuts)
    res["fillers"] = filler_coverage(words, cuts, fillers)
    res["pauses"] = long_pause_residuals(words, cuts)
    res["hesitations"] = (hesitation_residuals(vad_gaps, cuts)
                          if vad_gaps is not None else None)
    return res


# =============================================================================
# Отчёты: markdown, JSON, ASCII-консоль
# =============================================================================

def _fmt(x, nd: int = 1) -> str:
    return f"{x:.{nd}f}" if isinstance(x, float) else str(x)


def _na(v, suffix: str = "") -> str:
    return "н/д" if v is None else f"{v}{suffix}"


def render_markdown(results: list[dict], generated: str) -> str:
    """Markdown-отчёт: сводка, safety, чистота, типы, нарушения, пропуски."""
    analyzed = [r for r in results if not r["skipped"]]
    skipped = [r for r in results if r["skipped"]]
    agg = aggregate_results(analyzed)

    L: list[str] = []
    L.append("# Бенчмарк качества резов — FastVideoEdit")
    L.append("")
    L.append(f"Сформирован: {generated} · клипов проанализировано: "
             f"{len(analyzed)} · пропущено: {len(skipped)}")
    L.append("")
    L.append("Методика: транскрипт берётся из кэша (GPU не используется), "
             "детекция — штатный `run_detection` (без LLM). "
             f"Допуск клиппинга слова: **{WORD_CLIP_TOLERANCE * 1000:.0f} мс**; "
             f"«длинная пауза»: гэп > **{LONG_GAP_THRESHOLD} с**; "
             f"«заминка»: VAD-гэп **{HES_GAP_MIN}–{HES_GAP_MAX} с**; "
             f"покрытие: ≥ **{COVER_FRACTION * 100:.0f} %** длительности.")
    L.append("")

    # --- Сводка ---------------------------------------------------------------
    L.append("## Сводка")
    L.append("")
    L.append("| Клип | Модель | Длительность, с | Вырезов | Удалено, с | "
             "Удалено, % | Итог, с |")
    L.append("|---|---|---:|---:|---:|---:|---:|")
    for r in analyzed:
        s = r["summary"]
        L.append(f"| {r['name']} | {r.get('model', '?')} | "
                 f"{_fmt(r['duration_s'])} | {s['cuts_total']} | "
                 f"{_fmt(s['removed_s'])} | {_fmt(s['removed_pct'])} | "
                 f"{_fmt(s['final_s'])} |")
    if agg:
        L.append(f"| **Итого / среднее** | — | {_fmt(agg['duration_s'])} | "
                 f"{agg['cuts_total']} | {_fmt(agg['removed_s'])} | "
                 f"{_fmt(agg['removed_pct'])} | {_fmt(agg['final_s'])} |")
    L.append("")

    # --- Safety -----------------------------------------------------------------
    L.append("## Safety — целостность слов (цель: 0 %)")
    L.append("")
    L.append("| Клип | Авто-вырезов | С клиппингом слова (>12 мс) | Нарушений, % |")
    L.append("|---|---:|---:|---:|")
    for r in analyzed:
        sf = r["safety"]
        L.append(f"| {r['name']} | {sf['auto_cuts']} | {sf['violating_cuts']} | "
                 f"{_fmt(sf['violation_pct'])} |")
    if agg:
        a = agg["safety"]
        L.append(f"| **Итого / среднее** | {a['auto_cuts']} | "
                 f"{a['violating_cuts']} | {_fmt(a['violation_pct'])} |")
    L.append("")

    # --- Чистота ---------------------------------------------------------------
    L.append("## Чистота (прокси-метрики)")
    L.append("")
    L.append("| Клип | Филлеров в речи | Покрыто | Покрытие, % | "
             "Пауз >0.9 с | Осталось длинных | VAD-гэпов 0.2–0.55 с | Не покрыто |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in analyzed:
        f, p, h = r["fillers"], r["pauses"], r["hesitations"]
        L.append(f"| {r['name']} | {f['total']} | {f['covered']} | "
                 f"{_fmt(f['coverage_pct'])} | {p['long_gaps']} | "
                 f"{p['uncovered']} | "
                 f"{_na(h['gaps'] if h else None)} | "
                 f"{_na(h['uncovered'] if h else None)} |")
    if agg:
        f, p, h = agg["fillers"], agg["pauses"], agg["hesitations"]
        L.append(f"| **Итого / среднее** | {f['total']} | {f['covered']} | "
                 f"{_fmt(f['coverage_pct'])} | {p['long_gaps']} | "
                 f"{p['uncovered']} | "
                 f"{_na(h['gaps'] if h else None)} | "
                 f"{_na(h['uncovered'] if h else None)} |")
    L.append("")
    L.append("Примечание: непокрытые VAD-гэпы — отчасти норма: word-safe "
             "клампинг и дедуп детектора заминок сознательно не режут гэпы, "
             "чьё удаление рискует задеть соседнее слово.")
    L.append("")

    # --- Вырезы по типам ---------------------------------------------------------
    types = sorted({t for r in analyzed for t in r["summary"]["by_type"]})
    if types:
        L.append("## Вырезы по типам")
        L.append("")
        L.append("| Клип | " + " | ".join(types) + " |")
        L.append("|---|" + "---:|" * len(types))
        for r in analyzed:
            bt = r["summary"]["by_type"]
            L.append(f"| {r['name']} | "
                     + " | ".join(str(bt.get(t, 0)) for t in types) + " |")
        if agg:
            bt = agg["by_type"]
            L.append("| **Итого** | "
                     + " | ".join(str(bt.get(t, 0)) for t in types) + " |")
        L.append("")

    # --- Нарушения safety ----------------------------------------------------------
    any_violations = any(r["safety"]["violations"] for r in analyzed)
    L.append("## Нарушения safety")
    L.append("")
    if not any_violations:
        L.append("Нарушений не найдено — ни один авто-вырез не клиппит слово "
                 f"больше чем на {WORD_CLIP_TOLERANCE * 1000:.0f} мс. ✅")
        L.append("")
    else:
        for r in analyzed:
            viol = r["safety"]["violations"]
            if not viol:
                continue
            L.append(f"### {r['name']}")
            L.append("")
            L.append("| Тип выреза | Вырез, с | Слово | Слово, с | Пересечение, мс |")
            L.append("|---|---|---|---|---:|")
            shown = viol[:20]
            for v in shown:
                L.append(f"| {v['cut_type']} | "
                         f"{_fmt(v['cut_start'], 3)}–{_fmt(v['cut_end'], 3)} | "
                         f"«{v['word']}» | "
                         f"{_fmt(v['word_start'], 3)}–{_fmt(v['word_end'], 3)} | "
                         f"{_fmt(v['overlap_ms'])} |")
            if len(viol) > len(shown):
                L.append(f"| … | ещё {len(viol) - len(shown)} | | | |")
            L.append("")

    # --- Пропущенные клипы -----------------------------------------------------------
    if skipped:
        L.append("## Пропущенные клипы")
        L.append("")
        for r in skipped:
            L.append(f"- **{r['name']}** — {r['reason']}")
        L.append("")

    # --- Заметки -----------------------------------------------------------------------
    notes = [(r["name"], n) for r in analyzed for n in r.get("notes", [])]
    if notes:
        L.append("## Заметки")
        L.append("")
        for name, n in notes:
            L.append(f"- {name}: {n}")
        L.append("")

    return "\n".join(L)


def build_json(results: list[dict], generated: str) -> dict:
    """Машиночитаемый отчёт (зеркало markdown + полные детали)."""
    analyzed = [r for r in results if not r["skipped"]]
    return {
        "generated": generated,
        "params": {
            "word_clip_tolerance_ms": WORD_CLIP_TOLERANCE * 1000.0,
            "long_gap_threshold_s": LONG_GAP_THRESHOLD,
            "hesitation_gap_s": [HES_GAP_MIN, HES_GAP_MAX],
            "cover_fraction": COVER_FRACTION,
        },
        "clips": results,
        "aggregate": aggregate_results(analyzed),
    }


def _ascii(s: str) -> str:
    """Консоль Windows может быть cp1251/cp866 — печатаем только ASCII."""
    return s.encode("ascii", "replace").decode("ascii")


def print_console_summary(results: list[dict], md_path: Path,
                          json_path: Path) -> None:
    """Краткий ASCII-итог в консоль (безопасно для cp1251)."""
    print("Benchmark: cut quality (FastVideoEdit)")
    for r in results:
        name = _ascii(r["name"])
        if r["skipped"]:
            print(f"[SKIP] {name} | no cached transcript or unreadable input "
                  "(open the clip in the editor once)")
            continue
        s, sf, f, p, h = (r["summary"], r["safety"], r["fillers"],
                          r["pauses"], r["hesitations"])
        hes_txt = (f"vad gaps left {h['uncovered']}/{h['gaps']}"
                   if h else "vad gaps n/a")
        print(f"[ OK ] {name} | safety violations "
              f"{sf['violating_cuts']}/{sf['auto_cuts']} "
              f"({sf['violation_pct']:.1f}%) | fillers covered "
              f"{f['covered']}/{f['total']} ({f['coverage_pct']:.1f}%) | "
              f"long pauses left {p['uncovered']}/{p['long_gaps']} | "
              f"{hes_txt} | removed {s['removed_s']:.1f}s "
              f"({s['removed_pct']:.1f}%) of {r['duration_s']:.1f}s")
    agg = aggregate_results([r for r in results if not r["skipped"]])
    if agg:
        print(f"TOTAL {agg['clips']} clip(s) | safety "
              f"{agg['safety']['violation_pct']:.1f}% | filler coverage "
              f"{agg['fillers']['coverage_pct']:.1f}% | long pauses left "
              f"{agg['pauses']['uncovered']}/{agg['pauses']['long_gaps']} | "
              f"removed {agg['removed_s']:.1f}s ({agg['removed_pct']:.1f}%)")
    else:
        print("TOTAL: no clips analyzed (no cached transcripts found)")
    print(f"Saved: {_ascii(str(md_path))} ; {_ascii(str(json_path))}")


# =============================================================================
# CLI
# =============================================================================

def _abs_paths(cfg: Config) -> Config:
    """Относительные пути конфига (./cache и т.п.) — от корня репозитория."""
    for fld in ("cache_dir", "work_dir", "out_dir"):
        v = Path(getattr(cfg.paths, fld))
        if not v.is_absolute():
            setattr(cfg.paths, fld, str(ROOT / v))
    return cfg


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="benchmark_cuts.py",
        description="Benchmark of automatic cut quality (cache-only, no GPU). "
                    "Writes benchmark_results.md / .json to the repo root.")
    ap.add_argument("clips", nargs="*",
                    help="clip paths (default: 3.mp4 2.mp4 1.mp4 in repo root)")
    ap.add_argument("--clip", action="append", default=[],
                    help="add one more clip (repeatable)")
    ap.add_argument("--config", default=str(ROOT / "config.yaml"),
                    help="config.yaml path (default: repo root)")
    ap.add_argument("--md", default=str(ROOT / "benchmark_results.md"),
                    help="markdown report path")
    ap.add_argument("--json", dest="json_path",
                    default=str(ROOT / "benchmark_results.json"),
                    help="JSON report path")
    ap.add_argument("--no-vad", action="store_true",
                    help="skip VAD (no wav extraction / onnx; "
                         "hesitation metric becomes n/a)")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="print detector logs per clip")
    args = ap.parse_args(argv)

    clips = [Path(c) for c in args.clips] if args.clips else \
            [ROOT / c for c in DEFAULT_CLIPS]
    clips += [Path(c) for c in args.clip]

    cfg = _abs_paths(load_config(args.config))
    fillers = load_fillers(ROOT / "fillers_ru.yaml")
    profanity = load_profanity(ROOT / "profanity_ru.yaml")

    log = (lambda m: print(_ascii(str(m)))) if args.verbose else (lambda *_: None)

    results = [benchmark_clip(c, cfg, fillers, profanity,
                              vad=not args.no_vad, log=log)
               for c in clips]

    generated = datetime.now().strftime("%Y-%m-%d %H:%M")
    md_path, json_path = Path(args.md), Path(args.json_path)
    md_path.write_text(render_markdown(results, generated) + "\n",
                       encoding="utf-8", newline="\n")
    json_path.write_text(json.dumps(build_json(results, generated),
                                    ensure_ascii=False, indent=2),
                         encoding="utf-8", newline="\n")

    print_console_summary(results, md_path, json_path)
    return 0 if any(not r["skipped"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
