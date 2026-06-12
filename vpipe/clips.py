"""Clip Maker — YouTube Shorts candidates from the ORIGINAL transcript.

The local LLM (qwen3:8b) proposes 20-60 s self-contained fragments of a long
talking-head video. Proven engineering principle (18 real runs, llm.md): the
model is reliable on SEMANTICS (what hooks, where a thought completes) and
unreliable on arithmetic and self-checks — so the prompt owns the meaning and
the CODE owns everything numeric/verifiable: duration trim, first-letter case,
hook quoting, overlap dedup. The model returns segment INDICES (never float
timestamps); we map them back to exact times.

Candidates live in ORIGINAL timeline coordinates (badtakes-style, NOT
chapters-style — the UI/cutlist work in original time). The cutlist is used
only to compute the EFFECTIVE duration (raw minus enabled internal cuts) and
to snap boundaries away from dead air.

The prompt + schema below are the verified v3 set (D:/tmp/clipmaker/llm.md,
JSON validity 100%, 0 salvage). Do NOT reword them — reshuffled wording
reshuffles the top.
"""
from __future__ import annotations

import re
from bisect import bisect_left
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Optional

from .config import ClipsCfg, LlmCfg
from .cutlist import resolve
from .llm import OllamaClient, segment_windows
from .models import ACTION_REMOVE, TYPE_PAUSE, CutList, Segment, Transcript
from .subtitles import _SENT_END
from .timeline import Timeline

# --- verified v3 prompt + schema (llm.md §1) — verbatim, do not "improve" ----
_SYSTEM = (
    "Ты — продюсер коротких вертикальных роликов (YouTube Shorts). Тебе дают "
    "фрагмент расшифровки длинного видео (человек говорит в камеру) в виде "
    "нумерованных сегментов; у каждого указано время начала (мин:сек). Найди "
    "фрагменты, из которых получится самостоятельный короткий ролик, "
    "цепляющий зрителя с первой фразы.\n\n"
    "Жёсткие требования к фрагменту:\n"
    "- от 4 до 8 ПОДРЯД идущих сегментов (это короткий ролик до минуты);\n"
    "- текст ПЕРВОГО сегмента начинается с ЗАГЛАВНОЙ буквы — это начало "
    "нового предложения. НЕ выбирай первым сегмент, начинающийся с маленькой "
    "буквы или с продолжения чужой фразы;\n"
    "- первая фраза — сам по себе хук: интрига, спорное заявление, вопрос, "
    "цифра или сильное обещание. Запрещены первые фразы-отсылки: «И в конце…», "
    "«То есть…», «Дальше…», «Поэтому…», «А ещё…», «Над ним…», «Он/Это…» без "
    "названия предмета;\n"
    "- ПОСЛЕДНИЙ сегмент завершает мысль: вывод, панчлайн или итог. Если "
    "мысль продолжается в следующем сегменте — выбери другую границу;\n"
    "- фрагмент самодостаточен: понятен зрителю, который не видел остальное "
    "видео;\n"
    "- фрагменты НЕ пересекаются друг с другом.\n\n"
    "Что цепляет: спорное мнение, неожиданный факт или вывод, личная история, "
    "конкретный совет/лайфхак, эмоция, сравнение «до/после», цифры.\n"
    "Что НЕ годится: приветствия и анонсы («сегодня мы поговорим…»), просьбы "
    "подписаться/лайкнуть, перечисления без вывода, пошаговые инструкции с "
    "экрана без контекста.\n\n"
    "Верни строго JSON вида {\"clips\":[{\"start_index\":N, \"end_index\":N, "
    "\"score\":0-100, \"hook_phrase\":\"...\", \"reason\":\"...\"}]}.\n"
    "start_index/end_index — номера ПЕРВОГО и ПОСЛЕДНЕГО сегмента "
    "(включительно). score: 80-100 — точно зацепит, 60-79 — хороший, 40-59 — "
    "средний; ниже 40 не предлагай. hook_phrase — дословные первые слова "
    "первого сегмента (5-12 слов). reason — коротко по-русски, почему фрагмент "
    "сработает. Не больше 3 фрагментов. Если достойных нет — верни пустой "
    "список."
)

# Flat schema, no min/max/enum (like badtakes/chapters) — numeric bounds are
# validated by code, not by the model.
_SCHEMA = {
    "type": "object",
    "properties": {
        "clips": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "start_index": {"type": "integer"},
                    "end_index": {"type": "integer"},
                    "score": {"type": "integer"},
                    "hook_phrase": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["start_index", "end_index", "score",
                             "hook_phrase", "reason"],
            },
        }
    },
    "required": ["clips"],
}

# --- one-call re-rank (§3.5/F6) ------------------------------------------------
# Window scores are compressed into 80–95 and NOT comparable across windows, so
# a raw top-K across windows is effectively random. One final LLM call compares
# ALL survivors side by side. Ids are STRICTLY 1-based: with 0-based ids the
# model provably loses element 0 (llm.md). System prompt is the working one
# from the live run — do not reword.
_RERANK_SYSTEM = (
    "Ты — продюсер YouTube Shorts. Тебе дают список кандидатов в короткие "
    "ролики, вырезанных из одного длинного видео: у каждого номер, "
    "длительность и текст. Сравни их МЕЖДУ СОБОЙ и оцени каждый по шкале "
    "0-100: насколько ролик зацепит случайного зрителя в ленте Shorts (хук с "
    "первой фразы, самодостаточность, польза/эмоция, желание досмотреть). "
    "Оценки должны различаться — это рейтинг, а не школьные пятёрки. Верни "
    "строго JSON: каждый номер из списка ровно один раз."
)

_RERANK_SCHEMA = {
    "type": "object",
    "properties": {
        "ranking": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"id": {"type": "integer"},
                               "score": {"type": "integer"}},
                "required": ["id", "score"],
            },
        }
    },
    "required": ["ranking"],
}

_RERANK_MAX = 15        # >15 candidates overflow num_ctx 16384 (~4k tokens, §3.5)
_RERANK_TEXT_MAX = 700  # per-candidate text budget in the re-rank prompt;
_RERANK_HEAD = 480      # longer texts are clipped to head 480 + " … " +
_RERANK_TAIL = 180      # tail 180 (§3.5)

# --- calibrated post-processing constants (llm.md §4, plan §3.6) -------------
_IOU_DUP = 0.5               # time dedup: IoU of [t0,t1] ranges
_CONTAINMENT_DUP = 0.7       # time dedup: inter / min_len (small clip inside big)
_TEXT_DUP_RATIO = 0.7        # retake dedup (real Prod9 retake = 0.733)
_TEXT_DUP_CHARS = 600        # SequenceMatcher over the first N normalized chars
_HOOK_QUOTE_CHARS = 25       # hook quoting: first ~25 normalized chars
_ZONE_RADIUS = 10            # lowercase-guard zone: ±10 neighbour segments
_ZONE_CAPITAL_FRAC = 0.40    # >=40% capitals -> punctuated zone (hard drop)
_REMOVED_OVERLAP_DROP = 0.5  # >50% of the range inside removed -> drop
_MIN_TRIM_SEGMENTS = 4       # the trim never goes below 4 segments
_LEAD_IN = 0.15              # start pad before the snapped word (s)
_TAIL_OUT = 0.25             # end pad after the last word (s)
# Soft demote (polish, never a drop): anaphoric clip openers the prompt ban
# does not fully stop ("То есть никакой магии" slipped through on Prod9).
_ANAPHORIC_OPENERS = ("то есть", "и вот", "ну,", "дальше", "поэтому")


@dataclass
class ClipCandidate:
    id: str                  # "c01"…
    seg_start: int           # GLOBAL indices into the ORIGINAL transcript
    seg_end: int             # inclusive
    start: float             # seconds, ORIGINAL coordinates (UI/cutlist live there)
    end: float
    dur_raw: float
    dur_eff: float           # raw − Timeline.removed_overlap(start, end)
    score: int               # final: re-rank score when rank_source=="llm",
                             # else the window score
    score_window: int        # raw window score (threshold/tie-break; NOT
                             # comparable across windows)
    hook_phrase: str
    reason: str
    fuzzy_boundary: bool = False   # lowercase start in an unpunctuated zone /
                                   # trim without a sentence boundary -> UI badge
    source_window: int = 0
    short: bool = False            # dur_eff in [hard_min, min_duration) — the
                                   # «коротковат» mark (plan §3.6.1: 15–20 s)
    rank_source: str = "round_robin"   # "llm" — порядок задал одновызовный
                                       # re-rank; "round_robin" — фолбэк (F6)


@dataclass
class _Raw:
    """Working candidate while the §3.6 rules run (pre-ClipCandidate)."""
    seg_start: int
    seg_end: int
    t0: float
    t1: float
    score_window: int
    hook_phrase: str
    reason: str
    source_window: int
    fuzzy_boundary: bool = False
    short: bool = False
    demote: int = 0          # 0 normal | 1 anaphoric opener (soft) | 2 made-up hook
    score_final: Optional[int] = None  # re-rank score (§3.5); None → window score


# --- small helpers ------------------------------------------------------------
def _fmt_mmss(t: float) -> str:
    t = int(t)
    return f"{t // 60}:{t % 60:02d}"


def _build_user(window: list[Segment]) -> str:
    lines = [f"{i} | {_fmt_mmss(s.start)} | {s.text.strip()}"
             for i, s in enumerate(window)]
    return ("Сегменты расшифровки (номер | время начала | текст):\n"
            + "\n".join(lines)
            + "\n\nВерни JSON: {\"clips\": [{\"start_index\": N, \"end_index\": N, "
              "\"score\": N, \"hook_phrase\": \"...\", \"reason\": \"...\"}]}")


_TOKEN_NORM = re.compile(r"[^0-9a-zа-яе]+")


def _norm_token(s: str) -> str:
    """Lowercase, fold ё→е, strip everything but letters/digits.

    Unlike textnorm.normalize this keeps latin letters and digits — hooks quote
    things like «96%» and «linux» verbatim.
    """
    return _TOKEN_NORM.sub("", s.lower().replace("ё", "е"))


def _norm_text(s: str) -> str:
    toks = (_norm_token(t) for t in s.split())
    return " ".join(t for t in toks if t)


def _first_alpha(text: str) -> str:
    for ch in text.strip():
        if ch.isalpha():
            return ch
    return ""


def _valid_int(x) -> bool:
    # bool is an int subclass — the model (or a mock) may return true/false.
    return isinstance(x, int) and not isinstance(x, bool)


# --- §3.6 rules ----------------------------------------------------------------
def _zone_is_punctuated(segments: list[Segment], idx: int) -> bool:
    """Zone test for the lowercase-guard (§3.6.2).

    >=40% of the ±10 neighbour segments start with a capital letter → this is a
    punctuated zone and a lowercase first segment means mid-sentence (drop). In
    Whisper's unpunctuated zones (Prod9 segments 200–279) a hard drop would
    kill the BEST candidates, so the caller keeps those with fuzzy_boundary.
    """
    lo = max(0, idx - _ZONE_RADIUS)
    hi = min(len(segments), idx + _ZONE_RADIUS + 1)
    neigh = [s for i, s in enumerate(segments[lo:hi], lo) if i != idx]
    if not neigh:
        return True
    caps = sum(1 for s in neigh if _first_alpha(s.text).isupper())
    return caps / len(neigh) >= _ZONE_CAPITAL_FRAC


def _hook_is_quoted(hook: str, segments: list[Segment], gs: int) -> bool:
    """§3.6.3 — the first ~25 normalized chars of hook_phrase must literally
    occur in seg[start..start+2]; otherwise the hook is made up → demote."""
    head = _norm_text(hook)[:_HOOK_QUOTE_CHARS]
    if not head:
        return True                      # nothing to verify — don't punish
    zone = " ".join(s.text for s in segments[gs:gs + 3])
    return head in _norm_text(zone)


def _is_anaphoric(text: str) -> bool:
    return text.strip().lower().startswith(_ANAPHORIC_OPENERS)


def _trim_to_max(segments: list[Segment], gs: int, ge: int,
                 max_dur: float) -> tuple[int, bool, bool]:
    """§3.6.1 — the model ignores numeric limits (proven on v1/v2), so code
    trims: walk BACK from the end to the farthest segment that (a) keeps the
    clip <= max_dur and (b) ends a sentence (_SENT_END); never below 4
    segments. No sentence boundary found → just <= max_dur + fuzzy flag.

    Returns (new_seg_end, fuzzy, ok). ok=False → even 4 segments overflow the
    limit and the candidate cannot be saved.
    """
    t0 = segments[gs].start
    if segments[ge].end - t0 <= max_dur:
        return ge, False, True
    lo = gs + _MIN_TRIM_SEGMENTS - 1
    best_any: Optional[int] = None
    for j in range(ge - 1, lo - 1, -1):
        if segments[j].end - t0 > max_dur:
            continue
        if best_any is None:
            best_any = j                 # farthest segment satisfying (a)
        if segments[j].text.rstrip().endswith(_SENT_END):
            return j, False, True        # …and (b): sentence boundary
    if best_any is not None:
        return best_any, True, True
    return ge, False, False


def _clip_text(segments: list[Segment], c: _Raw) -> str:
    return " ".join(s.text.strip() for s in segments[c.seg_start:c.seg_end + 1])


def _is_time_dup(a: _Raw, b: _Raw) -> bool:
    inter = min(a.t1, b.t1) - max(a.t0, b.t0)
    if inter <= 0:
        return False
    union = max(a.t1, b.t1) - min(a.t0, b.t0)
    iou = inter / union if union > 0 else 0.0
    containment = inter / max(1e-9, min(a.t1 - a.t0, b.t1 - b.t0))
    return iou >= _IOU_DUP or containment >= _CONTAINMENT_DUP


def _is_text_dup(text_a: str, text_b: str) -> bool:
    a = _norm_text(text_a)[:_TEXT_DUP_CHARS]
    b = _norm_text(text_b)[:_TEXT_DUP_CHARS]
    if not a or not b:
        return False
    return SequenceMatcher(None, a, b).ratio() >= _TEXT_DUP_RATIO


def _enabled_cuts_within(cutlist: CutList, t0: float, t1: float) -> int:
    """Number of enabled REMOVE cuts intersecting [t0, t1] — retake-dedup
    preference (founder decision №5: fewer cuts inside = the cleaner take)."""
    n = 0
    for s in cutlist.segments:
        if s.enabled and s.action == ACTION_REMOVE \
                and min(s.end, t1) - max(s.start, t0) > 0:
            n += 1
    return n


def _dedup(cands: list[_Raw], segments: list[Segment], cutlist: CutList) -> list[_Raw]:
    """§3.6.5 — sort by window score desc, then a greedy pass; each newcomer is
    compared against the already-kept ones.

    (а) time overlap (IoU>=0.5 OR containment>=0.7): keep the bigger window
        score, on a tie — the shorter one (closer to 20–60 s).
    (б) retake duplicates (normalized text ratio>=0.7): founder decision №5 —
        prefer the instance with FEWER enabled cuts inside its range (fewer
        stumbles = the cleaner take); on a tie — the bigger score.
    (в) intra-window overlaps are caught by rule (а).
    """
    order = sorted(cands, key=lambda c: (-c.score_window, c.t1 - c.t0))
    texts = {id(c): _clip_text(segments, c) for c in order}
    kept: list[_Raw] = []
    for c in order:
        dup = False
        for i, k in enumerate(kept):
            if _is_time_dup(c, k):
                # order is score-desc, so kept wins unless tied and c is shorter
                if c.score_window == k.score_window \
                        and (c.t1 - c.t0) < (k.t1 - k.t0):
                    kept[i] = c
                dup = True
                break
            if _is_text_dup(texts[id(c)], texts[id(k)]):
                cc = _enabled_cuts_within(cutlist, c.t0, c.t1)
                kc = _enabled_cuts_within(cutlist, k.t0, k.t1)
                if cc < kc or (cc == kc and c.score_window > k.score_window):
                    kept[i] = c
                dup = True
                break
        if not dup:
            kept.append(c)
    return kept


# Residual overlap above this fraction of the SHORTER clip's length is a
# product conflict: two Shorts must not share material. Pairs below the dedup
# thresholds (IoU<0.5, containment<0.7) can still share a boundary segment —
# seen live on Prod9 (c04/c08 shared ~10 s of seg 279, containment 0.25).
_OVERLAP_CONFLICT = 0.2


def _drop_residual_overlaps(cands: list[_Raw]) -> list[_Raw]:
    """Greedy non-overlap pass AFTER dedup: walk score-desc, drop any clip
    whose time overlap with an already-kept one exceeds ``_OVERLAP_CONFLICT``
    of the shorter clip. Tiny boundary kisses (<20%) survive — the product
    promise is «фрагменты не пересекаются», not «не соприкасаются»."""
    order = sorted(cands, key=lambda c: (-c.score_window, c.t1 - c.t0))
    kept: list[_Raw] = []
    for c in order:
        conflict = False
        for k in kept:
            inter = min(c.t1, k.t1) - max(c.t0, k.t0)
            if inter <= 0:
                continue
            shorter = max(1e-9, min(c.t1 - c.t0, k.t1 - k.t0))
            if inter / shorter > _OVERLAP_CONFLICT:
                conflict = True
                break
        if not conflict:
            kept.append(c)
    return kept


def _sort_mvp(cands: list[_Raw]) -> list[_Raw]:
    """Base ordering — window score desc with round-robin across windows
    (window scores are NOT comparable across windows, and one «generous»
    window must not flood the whole top). This is both the re-rank FALLBACK
    (LLM down / garbage reply / cfg.rerank=False) and the «original order» the
    re-rank repairs lean on (missing ids are appended in this order). Demoted
    candidates go to the tail: soft anaphoric first, made-up hooks last.
    """
    out: list[_Raw] = []
    for tier in (0, 1, 2):
        group = [c for c in cands if c.demote == tier]
        by_win: dict[int, list[_Raw]] = {}
        for c in group:
            by_win.setdefault(c.source_window, []).append(c)
        for lst in by_win.values():
            lst.sort(key=lambda c: (-c.score_window, c.t1 - c.t0))
        rank = 0
        while True:
            row = [lst[rank] for lst in by_win.values() if rank < len(lst)]
            if not row:
                break
            row.sort(key=lambda c: (-c.score_window, c.t1 - c.t0))
            out.extend(row)
            rank += 1
    return out


# --- one-call re-rank (§3.5/F6) ---------------------------------------------------
def _rerank_text(segments: list[Segment], c: _Raw) -> str:
    """Candidate text for the re-rank prompt: full when <=700 chars, otherwise
    head 480 + « … » + tail 180 (§3.5 — the tail shows whether the thought
    completes, which the system prompt scores)."""
    text = _clip_text(segments, c)
    if len(text) <= _RERANK_TEXT_MAX:
        return text
    return text[:_RERANK_HEAD].rstrip() + " … " + text[-_RERANK_TAIL:].lstrip()


def _build_rerank_user(pool: list[_Raw], segments: list[Segment],
                       tl: Timeline) -> str:
    """Compact prompt with ALL candidates: id, duration, hook, short text.

    Ids are STRICTLY 1-based — with 0-based ids the model provably drops
    element 0 (§3.5). The id→candidate mapping stays on our side (pool[id-1]).
    """
    lines = []
    for i, c in enumerate(pool, 1):
        dur = (c.t1 - c.t0) - tl.removed_overlap(c.t0, c.t1)
        lines.append(f"{i} | {max(1, round(dur))}с | хук: {c.hook_phrase or '—'} "
                     f"| {_rerank_text(segments, c)}")
    return ("Кандидаты (номер | длительность | хук | текст):\n"
            + "\n".join(lines)
            + "\n\nУпорядочь от лучшего к худшему (лучшие первыми). Верни "
              "JSON: {\"ranking\": [{\"id\": N, \"score\": N}]} — каждый номер "
              "из списка ровно один раз.")


def _rerank(ordered: list[_Raw], segments: list[Segment], tl: Timeline,
            llm: OllamaClient, log) -> Optional[list[_Raw]]:
    """One final LLM call over all surviving candidates (§3.5/F6).

    Input is the round-robin order; only the first ``_RERANK_MAX`` go into the
    prompt (token budget §3.5), the rest keep their order at the tail. Returns
    the new order, or ``None`` — the caller keeps the round-robin order
    (rank_source="round_robin").

    Response repairs (never a hard failure): unknown ids ignored, duplicate
    ids collapsed to the first occurrence, missing ids appended at the tail in
    the original round-robin order. Valid re-rank scores land in
    ``score_final`` (clamped 0–100) and replace the window score in the
    output; repaired tail entries keep their window score.

    keep_alive=0 — this is the last LLM call of the pass (frees VRAM).
    """
    pool = ordered[:_RERANK_MAX]
    user = _build_rerank_user(pool, segments, tl)
    try:
        data = llm.chat_json(_RERANK_SYSTEM, user, _RERANK_SCHEMA, keep_alive=0)
    except Exception as e:  # noqa: BLE001 — re-rank must never lose the pass
        log(f"  clips: re-rank failed ({e}); keeping round-robin order.")
        return None
    items = data.get("ranking") if isinstance(data, dict) else None
    if not isinstance(items, list):
        log("  clips: re-rank returned no usable ranking; keeping round-robin order.")
        return None
    seen: set[int] = set()
    picked: list[_Raw] = []
    for r in items:
        if not isinstance(r, dict):
            continue
        rid = r.get("id")
        if not _valid_int(rid) or not (1 <= rid <= len(pool)) or rid in seen:
            continue                     # unknown/duplicate id — ignore
        seen.add(rid)
        c = pool[rid - 1]
        sc = r.get("score")
        if _valid_int(sc):
            c.score_final = max(0, min(100, sc))
        picked.append(c)
    if not picked:
        log("  clips: re-rank returned no usable ranking; keeping round-robin order.")
        return None
    picked.extend(c for i, c in enumerate(pool, 1) if i not in seen)
    picked.extend(ordered[len(pool):])
    return picked


# --- word-snap (§3.6.4) ---------------------------------------------------------
def _snap_start_word(c: _Raw, segments: list[Segment], words, norm_words,
                     starts: list[float]) -> Optional[int]:
    """Find the word the clip should start on.

    First try the hook: search the first 3–5 normalized hook words in the word
    stream around seg[start] (±1 segment) — proven on Prod9, where the hook
    «почему…» lay in the TAIL of the previous segment. Fall back to the first
    word of the start segment.
    """
    hook_toks = [t for t in (_norm_token(x) for x in c.hook_phrase.split()) if t]
    if hook_toks:
        lo_t = segments[max(0, c.seg_start - 1)].start - 1e-6
        hi_t = segments[min(len(segments) - 1, c.seg_start + 1)].end + 1e-6
        idxs = [i for i in range(len(words))
                if norm_words[i] and lo_t <= words[i].start <= hi_t]
        k_max = min(5, len(hook_toks))
        k_min = min(3, len(hook_toks))
        for k in range(k_max, k_min - 1, -1):
            target = hook_toks[:k]
            for pos in range(len(idxs) - k + 1):
                if [norm_words[idxs[pos + j]] for j in range(k)] == target:
                    return idxs[pos]
    # fallback: the first word at/after the start segment's boundary
    pos = bisect_left(starts, segments[c.seg_start].start - 1e-3)
    if pos < len(words) and words[pos].start < segments[c.seg_end].end:
        return pos
    return None


def _apply_word_bounds(c: _Raw, segments: list[Segment], words, norm_words,
                       starts: list[float], duration: float,
                       max_dur: float, cutlist: CutList) -> tuple[float, float]:
    """§2.1 step 9 — word-snap of the start by hook_phrase, lead-in −0.15 s /
    tail +0.25 s pads (never past the neighbouring word — pauses.py pattern),
    then snap a boundary that landed inside a cutlist pause segment to its edge
    (a Short must not start with dead air)."""
    s_seg, e_seg = segments[c.seg_start], segments[c.seg_end]
    # --- start
    wi = _snap_start_word(c, segments, words, norm_words, starts)
    if wi is not None:
        w = words[wi]
        floor = words[wi - 1].end if wi > 0 else 0.0
        start = max(w.start - _LEAD_IN, min(floor, w.start), 0.0)
    else:
        start = max(0.0, s_seg.start - _LEAD_IN)
    # --- end: the last word starting before the end segment's boundary
    pos = bisect_left(starts, e_seg.end - 1e-3) - 1
    if 0 <= pos < len(words) and words[pos].end > start:
        w = words[pos]
        ceil_ = words[pos + 1].start if pos + 1 < len(words) else duration
        end = min(w.end + _TAIL_OUT, max(ceil_, w.end), duration)
    else:
        end = min(duration, e_seg.end + _TAIL_OUT)
    # --- snap off cutlist pause segments (dead air at a clip edge)
    snapped_s, snapped_e = start, end
    for s in cutlist.segments:
        if s.type != TYPE_PAUSE:
            continue
        if s.start < snapped_s < s.end:
            snapped_s = min(s.end, snapped_e)      # speech resumes at the right edge
        if s.start < snapped_e < s.end:
            snapped_e = max(s.start, snapped_s)    # speech ended at the left edge
    if snapped_e - snapped_s >= 0.05:
        start, end = snapped_s, snapped_e
    # --- invariant: never longer than max_dur raw (pads could add ±0.4 s)
    if end - start > max_dur:
        end = start + max_dur
    start = max(0.0, min(start, duration))
    end = max(start, min(end, duration))
    return start, end


# --- main entry -----------------------------------------------------------------
def suggest(transcript: Transcript, cutlist: CutList, cfg: ClipsCfg,
            llm_cfg: LlmCfg, llm: OllamaClient, *, log=print,
            on_progress=None, on_stage=None) -> list[ClipCandidate]:
    """Full pass: windows → validation → trim → dedup → re-rank (one final
    LLM call, §3.5/F6; fallback round-robin) → word-snap. Candidates come back
    in ORIGINAL coordinates; ``rank_source`` says who ordered them."""
    segments = transcript.segments
    if llm is None or not segments:
        return []

    # 1. The cutlist resolves to removed intervals — ONLY for dur_eff/snapping.
    removed, _ = resolve(cutlist)
    duration = float(cutlist.duration or transcript.duration)
    tl = Timeline(removed, duration)

    # 2. Windows with overlap=12 (not the default 5): a <=60 s clip straddling a
    #    boundary must fit whole into at least one window.
    wcfg = llm_cfg.model_copy(update={"segment_overlap": int(cfg.window_overlap)})
    windows = segment_windows(len(segments), wcfg)
    n_win = len(windows)

    raw: list[_Raw] = []
    for wi, (win_start, win_end) in enumerate(windows):
        if on_stage:
            on_stage(f"Клипы… {wi + 1}/{n_win}")
        if on_progress:
            on_progress(wi / max(1, n_win))
        window = segments[win_start:win_end]
        user = _build_user(window)
        # Keep qwen3 warm BETWEEN windows; unload after the last one so it
        # frees VRAM for the next stage (chapters.py pattern).
        ka = 0 if wi == n_win - 1 else cfg.keep_alive_between
        try:
            data = llm.chat_json(_SYSTEM, user, _SCHEMA, keep_alive=ka)
        except Exception as e:  # noqa: BLE001 — one bad window must not lose the pass
            log(f"  clips: window [{win_start}:{win_end}] failed ({e}); skipped.")
            continue
        items = data.get("clips", [])
        if not isinstance(items, list):
            continue
        win_raw: list[_Raw] = []
        for r in items:
            if not isinstance(r, dict):
                continue
            a, b, sc = r.get("start_index"), r.get("end_index"), r.get("score")
            # bool/str/negative/out-of-range — silently dropped (badtakes guard)
            if not (_valid_int(a) and _valid_int(b) and _valid_int(sc)):
                continue
            if not (0 <= a <= b < len(window)):
                continue
            gs, ge = win_start + a, win_start + b      # local -> GLOBAL indices
            win_raw.append(_Raw(
                seg_start=gs, seg_end=ge,
                t0=segments[gs].start, t1=segments[ge].end,
                score_window=max(0, min(100, sc)),
                hook_phrase=str(r.get("hook_phrase", "")).strip(),
                reason=str(r.get("reason", "")).strip(),
                source_window=wi))
        # «не больше 3» is baked into the prompt, but code enforces the number.
        win_raw.sort(key=lambda c: -c.score_window)
        raw.extend(win_raw[:max(1, int(cfg.max_per_window))])

    # 5–6. Per-candidate validation + trim + duration drop (§3.6.1/2/3/6/7).
    processed: list[_Raw] = []
    for c in raw:
        raw_dur = c.t1 - c.t0
        if raw_dur <= 0:
            continue
        # §3.6.6 — the candidate lives inside a user-disabled retake.
        if tl.removed_overlap(c.t0, c.t1) / raw_dur > _REMOVED_OVERLAP_DROP:
            continue
        # §3.6.2 — zone-dependent lowercase-guard (a hard drop everywhere would
        # kill the BEST Prod9 candidates in the unpunctuated zone).
        first = _first_alpha(segments[c.seg_start].text)
        if first and first.islower():
            if _zone_is_punctuated(segments, c.seg_start):
                continue
            c.fuzzy_boundary = True
        # §3.6.3 — a made-up hook demotes to the tail; §3.6.7 — soft demote of
        # anaphoric openers (polish, never a drop).
        if not _hook_is_quoted(c.hook_phrase, segments, c.seg_start):
            c.demote = 2
        elif _is_anaphoric(segments[c.seg_start].text):
            c.demote = 1
        # §3.6.1 — trim >max_duration back to a sentence boundary.
        new_ge, fuzzy_trim, ok = _trim_to_max(segments, c.seg_start, c.seg_end,
                                              cfg.max_duration)
        if not ok:
            continue
        if new_ge != c.seg_end:
            c.seg_end = new_ge
            c.t1 = segments[new_ge].end
            c.fuzzy_boundary = c.fuzzy_boundary or fuzzy_trim
        # effective duration = raw − enabled cuts inside the range
        dur_eff = (c.t1 - c.t0) - tl.removed_overlap(c.t0, c.t1)
        if dur_eff < cfg.hard_min:
            continue
        c.short = dur_eff < cfg.min_duration
        processed.append(c)

    # 7. Dedup (greedy, score-desc; incl. founder decision №5 for retakes).
    deduped = _dedup(processed, segments, cutlist)
    deduped = _drop_residual_overlaps(deduped)

    # 8. Base order: window score desc + round-robin across windows — the
    #    honest fallback AND the «original order» for re-rank repairs.
    ordered = _sort_mvp(deduped)
    # 8b. One-call re-rank (§3.5/F6): window scores are NOT comparable across
    #     windows, so one final LLM call orders ALL survivors. Any failure
    #     keeps the round-robin order. Skipped for <2 candidates — nothing to
    #     compare, no extra LLM call.
    rank_source = "round_robin"
    if cfg.rerank and len(ordered) >= 2:
        if on_stage:
            on_stage("Клипы… ранжирование")
        reranked = _rerank(ordered, segments, tl, llm, log)
        if reranked is not None:
            ordered = reranked
            rank_source = "llm"

    # 9. Word-snap of the start by hook_phrase + pads + pause-edge snapping.
    words = transcript.all_words()
    norm_words = [_norm_token(w.word) for w in words]
    starts = [w.start for w in words]
    out: list[ClipCandidate] = []
    # 10. Cap to max_candidates, assign ids c01…
    for i, c in enumerate(ordered[:max(0, int(cfg.max_candidates))]):
        start, end = _apply_word_bounds(c, segments, words, norm_words, starts,
                                        duration, cfg.max_duration, cutlist)
        dur_raw = end - start
        dur_eff = max(0.0, dur_raw - tl.removed_overlap(start, end))
        out.append(ClipCandidate(
            id=f"c{i + 1:02d}",
            seg_start=c.seg_start, seg_end=c.seg_end,
            start=round(start, 3), end=round(end, 3),
            dur_raw=round(dur_raw, 3), dur_eff=round(dur_eff, 3),
            # re-rank score replaces the window score for the UI bar; repaired
            # tail entries (missing from the LLM reply) keep the window score.
            score=(c.score_final if c.score_final is not None
                   else c.score_window),
            score_window=c.score_window,
            hook_phrase=c.hook_phrase, reason=c.reason,
            fuzzy_boundary=c.fuzzy_boundary,
            source_window=c.source_window,
            short=c.short,
            rank_source=rank_source))
    if on_progress:
        on_progress(1.0)
    return out
