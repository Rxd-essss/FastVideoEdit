"""Stage 7 — subtitles under the final (cut) timeline.

Remaps word timestamps onto the shortened timeline, packs words into cues that
respect Russian reading-speed/line conventions, masks profanity in the text
(е.g. б***ь) and writes .srt / .vtt (+ optional transcript.txt).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .config import AssStyleCfg, MaskingCfg, ProfanityLists, SubsCfg
from .detect.profanity import ProfanityMatcher
from .models import Transcript, Word
from .timeline import Timeline, remap_words

_SENT_END = ("...", ".", "!", "?", "…", ".", "?!")


def mask_word(word: str, cfg: MaskingCfg) -> str:
    """б***ь — keep first/last letters, replace the middle with mask chars.

    Always masks at least one real character: keep_first + keep_last is clamped
    to < length so a short word (or a generous keep config) can't leak whole.
    """
    n = len(word)
    if n == 0:
        return word
    kf = min(cfg.keep_first, max(0, n - 1))
    kl = min(cfg.keep_last, max(0, n - kf - 1))
    middle = n - kf - kl                       # guaranteed >= 1
    stars = max(cfg.min_stars, middle)
    tail = word[n - kl:] if kl else ""
    return word[:kf] + (cfg.mask_char * stars) + tail


def mask_text(text: str, matcher: ProfanityMatcher, cfg: MaskingCfg) -> str:
    import re
    return re.sub(r"[А-Яа-яЁё]+",
                  lambda m: mask_word(m.group(0), cfg) if matcher.is_profane(m.group(0))
                  else m.group(0),
                  text)


@dataclass
class Cue:
    start: float
    end: float
    text: str


def _display(w: Word, matcher: ProfanityMatcher, cfg: MaskingCfg) -> str:
    raw = w.word.strip()
    return mask_word(raw, cfg) if matcher.is_profane(raw) else raw


# Short Russian function words that read badly at the END of a wrapped line —
# prepositions, conjunctions and a couple of common particles. We avoid breaking
# right after one of these so the next line doesn't start orphaned. Editable.
_BAD_LINE_TAIL = {
    "в", "и", "а", "на", "по", "с", "к", "от", "до", "не", "что", "как",
    "о", "об", "у", "за", "из", "под", "над", "при", "для", "но", "да",
    "то", "же", "бы", "ли", "со", "во", "ко", "из-за", "из-под",
}


def _wrap(words_disp: list[str], max_chars: int, max_lines: int) -> str:
    """Wrap displayed words into <= max_lines.

    For the common 2-line case we pick the split point that best BALANCES the
    two line lengths (minimises their difference) while avoiding a break right
    after a short preposition/conjunction (so the next line isn't orphaned).
    Falls back to greedy filling for >2 lines or when no split fits.
    """
    if not words_disp:
        return ""

    joined = " ".join(words_disp)
    # Single line: fits, or we have no room to wrap.
    if len(joined) <= max_chars or max_lines <= 1 or len(words_disp) < 2:
        return joined

    if max_lines == 2:
        best: tuple | None = None   # (penalty, diff, split_index)
        for split in range(1, len(words_disp)):
            l1 = " ".join(words_disp[:split])
            l2 = " ".join(words_disp[split:])
            # Strongly prefer splits where both lines fit the width.
            over = max(0, len(l1) - max_chars) + max(0, len(l2) - max_chars)
            diff = abs(len(l1) - len(l2))
            tail = words_disp[split - 1].strip(".,!?…:;\"'»«()").lower()
            bad_tail = 1 if tail in _BAD_LINE_TAIL else 0
            # penalty tiers: overflow dominates, then bad tail, then imbalance.
            key = (over, bad_tail, diff)
            if best is None or key < best[0]:
                best = (key, split)
        if best is not None:
            split = best[1]
            return " ".join(words_disp[:split]) + "\n" + " ".join(words_disp[split:])

    # >2 lines (or fallback): greedy fill, also avoiding short-word tails.
    lines: list[str] = []
    cur = ""
    prev_w = ""
    for w in words_disp:
        cand = (cur + " " + w).strip()
        prev_tail = prev_w.strip(".,!?…:;\"'»«()").lower()
        force_keep = prev_tail in _BAD_LINE_TAIL
        if cur and len(cand) > max_chars and len(lines) < max_lines - 1 and not force_keep:
            lines.append(cur)
            cur = w
        else:
            cur = cand
        prev_w = w
    if cur:
        lines.append(cur)
    return "\n".join(lines[:max_lines]) if lines else joined


def build_cues(words: list[Word], matcher: ProfanityMatcher,
               subs: SubsCfg, mask: MaskingCfg, total: float) -> list[Cue]:
    disp = [_display(w, matcher, mask) for w in words]
    groups: list[list[int]] = []
    cur: list[int] = []
    cur_len = 0
    budget = subs.max_line_chars * subs.max_lines

    for i, w in enumerate(words):
        if cur:
            gap = w.start - words[cur[-1]].end
            dur = w.end - words[cur[0]].start
            if (gap > subs.new_cue_gap or dur > subs.max_dur
                    or cur_len + len(disp[i]) + 1 > budget):
                groups.append(cur)
                cur, cur_len = [], 0
        cur.append(i)
        cur_len += len(disp[i]) + 1
        if disp[i].endswith(_SENT_END) and (w.end - words[cur[0]].start) >= subs.min_dur:
            groups.append(cur)
            cur, cur_len = [], 0
    if cur:
        groups.append(cur)

    cues: list[Cue] = []
    cue_next_start: list[float] = []   # original start of the following group
    for gi, g in enumerate(groups):
        start = min(words[g[0]].start, total)
        end = min(max(words[g[-1]].end, start + subs.min_dur), total)
        if end <= start:                       # word at/after timeline end
            continue
        text = _wrap([disp[i] for i in g], subs.max_line_chars, subs.max_lines)
        cues.append(Cue(start, end, text))
        nxt = words[groups[gi + 1][0]].start if gi + 1 < len(groups) else total
        cue_next_start.append(min(nxt, total))

    # Enforce max reading speed (chars-per-second). A cue that is too short for
    # its text is extended on its END up to the next cue's start minus min_gap.
    # If that still isn't enough room it's left best-effort (we never overlap).
    max_cps = subs.max_cps if subs.max_cps and subs.max_cps > 0 else 0.0
    if max_cps > 0:
        for i, c in enumerate(cues):
            chars = len(c.text.replace("\n", ""))
            required = chars / max_cps
            if (c.end - c.start) < required:
                limit = cue_next_start[i] - subs.min_gap
                c.end = min(max(c.end, c.start + required), max(c.start, limit), total)

    # Enforce no overlap (keep order, leave a small gap).
    for i in range(len(cues) - 1):
        if cues[i].end > cues[i + 1].start - subs.min_gap:
            cues[i].end = max(cues[i].start + 0.2, cues[i + 1].start - subs.min_gap)
    return cues


def _ts(t: float, sep: str) -> str:
    if t < 0:
        t = 0.0
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def write_srt(cues: list[Cue], path: str | Path) -> None:
    out = []
    for i, c in enumerate(cues, 1):
        out.append(str(i))
        out.append(f"{_ts(c.start, ',')} --> {_ts(c.end, ',')}")
        out.append(c.text)
        out.append("")
    Path(path).write_text("\n".join(out), encoding="utf-8")


def write_vtt(cues: list[Cue], path: str | Path) -> None:
    out = ["WEBVTT", ""]
    for c in cues:
        out.append(f"{_ts(c.start, '.')} --> {_ts(c.end, '.')}")
        out.append(c.text)
        out.append("")
    Path(path).write_text("\n".join(out), encoding="utf-8")


def write_transcript(words: list[Word], matcher: ProfanityMatcher,
                     mask: MaskingCfg, path: str | Path) -> None:
    parts: list[str] = []
    line: list[str] = []
    for w in words:
        d = _display(w, matcher, mask)
        line.append(d)
        if d.endswith(_SENT_END):
            parts.append(" ".join(line))
            line = []
    if line:
        parts.append(" ".join(line))
    Path(path).write_text("\n".join(parts) + "\n", encoding="utf-8")


def generate(transcript: Transcript, removed: list[tuple[float, float]],
             subs: SubsCfg, mask: MaskingCfg, matcher: ProfanityMatcher,
             out_base: str | Path, log=print) -> dict:
    """Write subtitles under the cut timeline. Returns counts + paths."""
    tl = Timeline(removed, transcript.duration)
    words = remap_words(transcript.all_words(), tl)
    cues = build_cues(words, matcher, subs, mask, tl.new_duration())

    base = Path(out_base)
    srt = str(base.with_suffix(".srt"))
    write_srt(cues, srt)
    result = {"cues": len(cues), "srt": srt}
    if subs.write_vtt:
        vtt = str(base.with_suffix(".vtt"))
        write_vtt(cues, vtt)
        result["vtt"] = vtt
    if subs.write_transcript:
        txt = str(base.parent / "transcript.txt")
        write_transcript(words, matcher, mask, txt)
        result["transcript"] = txt
    log(f"  {len(cues)} cues -> {Path(srt).name}"
        + (f" + .vtt" if subs.write_vtt else ""))
    return result


# --- Feature A: burn-in (ASS/libass) subtitles with karaoke ------------------
_ASS_ALIGN = {"bottom": 2, "top": 8, "center": 5}


def _ass_text_escape(text: str) -> str:
    r"""Escape a cue's plain text for the ASS ``Text`` field.

    ASS treats ``{`` as the start of an override block and ``\`` as an escape
    introducer, so both must be neutralised; literal line breaks become the ASS
    hard break ``\N``. We do NOT touch ``}`` (harmless without an opening brace)
    beyond the brace escape above.
    """
    text = text.replace("\\", "\\\\")        # backslash first
    text = text.replace("{", "\\{").replace("}", "\\}")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\n", "\\N")
    return text


# --- Feature B (V11 §4b): kinetic keyword pop inside karaoke -------------------
# Stop-list of NON-content words that must never "pop": the line-tail function
# words (prepositions/conjunctions/particles — exactly the §4b «не-ключевые»
# set) plus a few common pronouns/forms that carry no emphasis. Anything NOT in
# here AND long enough is a content-word candidate (noun/verb «носитель смысла»).
_KINETIC_STOP = _BAD_LINE_TAIL | {
    "это", "этот", "эта", "эти", "вот", "так", "там", "тут", "уже", "ещё",
    "его", "её", "их", "мы", "вы", "он", "она", "они", "я", "ты", "оно",
    "был", "была", "было", "были", "есть", "над", "без", "про", "там",
    "тоже", "только", "очень", "когда", "если", "чтобы", "потому",
}
KINETIC_MIN_LEN = 4           # слово короче 4 букв не «носитель смысла» -> не поп
KINETIC_MAX_PER_CUE = 2       # 1–2 слова на реплику (§4b), НЕ каждую реплику
KINETIC_SCALE = 120           # вспухание до 120% (≤1.2×, §4b анти-кринж)
KINETIC_POP_IN_MS = 120       # рост за 120 мс (R3)
KINETIC_POP_HOLD_MS = 260     # держит 260 мс
KINETIC_POP_OUT_MS = 160      # возврат за 160 мс
_KINETIC_DEFAULT_ACCENT = "&H000B9EF5"   # #f59e0b BGR — акцент проекта (не кислота)


def _is_content_word(disp: str) -> bool:
    r"""Слово — «носитель смысла» (кандидат на кинетический поп, §4b)?

    Чистим пунктуацию/кавычки, нижний регистр; отсекаем стоп-лист (предлоги/
    союзы/частицы/местоимения) и короткие слова. Длинное содержательное слово
    (существительное/глагол), число с цифрами — годится.
    """
    core = disp.strip(".,!?…:;\"'»«()-—").lower()
    if not core:
        return False
    if any(ch.isdigit() for ch in core):     # числа/названия — всегда содержательны
        return True
    return len(core) >= KINETIC_MIN_LEN and core not in _KINETIC_STOP


def _kinetic_keywords(disp_words: list[str], *,
                      max_n: int = KINETIC_MAX_PER_CUE) -> set[int]:
    r"""Индексы 1–2 ключевых слов реплики для кинетического попа (§4b).

    Эвристика «носитель смысла, не стоп-лист»: из content-слов берём самые
    длинные (длина ≈ значимость), не больше ``max_n``. НЕ каждую реплику — если
    content-слов нет, set пуст (поп не ставится). Стабильно: при равной длине
    выигрывает более ранний индекс."""
    cands = [(len(disp_words[i]), -i, i)
             for i in range(len(disp_words))
             if _is_content_word(disp_words[i])]
    if not cands:
        return set()
    cands.sort(reverse=True)
    return {i for _l, _ni, i in cands[:max(0, max_n)]}


def _kinetic_pop_tag(start_ms: int, accent: str) -> str:
    r"""``\t``-поп ключевого слова: вспухание ``KINETIC_SCALE``% + акцент за
    ``POP_IN``, держит ``POP_HOLD``, возврат к 100% за ``POP_OUT`` (§4b, R3).

    Времена line-relative (libass ``\t`` от начала события): ``start_ms`` =
    смещение до произнесения слова (=Σ предыдущих ``\k`` ×10). Караоке-``\k``
    остального текста не трогаем — поп лишь добавляет ``\t`` в тот же блок.
    """
    t0 = max(0, int(start_ms))
    t1 = t0 + KINETIC_POP_IN_MS
    t2 = t1 + KINETIC_POP_HOLD_MS
    t3 = t2 + KINETIC_POP_OUT_MS
    return (f"\\t({t0},{t1},\\fscx{KINETIC_SCALE}\\fscy{KINETIC_SCALE}"
            f"\\1c{accent}\\3c{accent})"
            f"\\t({t2},{t3},\\fscx100\\fscy100)")


def _karaoke_text(cue: Cue, words: list[Word], matcher: ProfanityMatcher,
                  mask: MaskingCfg, eps: float = 0.02, *,
                  kinetic: bool = True,
                  accent: str = _KINETIC_DEFAULT_ACCENT) -> str:
    r"""Render one cue's text as a karaoke line: ``{\kNN}word`` per word.

    ``NN`` is the word's duration in centiseconds (ASS ``\k`` unit). Words are
    matched to the cue by their (already FINAL-coordinate) timings. A lead-in
    gap before the first word and inter-word gaps are folded into the following
    word's ``\k`` so the highlight stays in sync and the sum of all ``\k`` for
    the cue equals the cue's duration in centiseconds.

    Line breaks in ``cue.text`` are preserved: we keep the wrapped plain text as
    the layout reference and only attach ``\k`` tags to whitespace-separated
    tokens, mapping them positionally onto the in-cue words.

    V11 §4b — kinetic keyword pop: when ``kinetic`` is on, 1–2 content words of
    the cue additionally get a ``\t`` transform that swells them to
    ``KINETIC_SCALE``% + accent colour exactly when spoken, then returns to
    100%. The pop only ADDS a ``\t`` inside the word's own ``{...}`` block — the
    per-word ``\kNN`` karaoke fill (count and centisecond sum) is untouched
    (proven byte-compatible with the existing karaoke contract). Profane
    (masked) words never pop (no extra attention on a bleeped word).
    """
    in_cue = [w for w in words
              if w.start >= cue.start - eps and w.end <= cue.end + eps]
    in_cue.sort(key=lambda w: w.start)

    # If we somehow can't line words up with the cue, fall back to plain text.
    if not in_cue:
        return _ass_text_escape(cue.text)

    total_cs = max(len(in_cue), int(round((cue.end - cue.start) * 100)))
    # Per-word highlight window = the span from the previous word's end (or the
    # cue start) to this word's end, folding any preceding gap into the word so
    # the karaoke fill never stalls. Each window is at least 1 cs.
    spans: list[int] = []
    prev_end = cue.start
    for w in in_cue:
        start = max(prev_end, cue.start)
        cs = max(1, int(round((w.end - start) * 100)))
        spans.append(cs)
        prev_end = w.end

    # Make the \k sum exactly equal the cue duration (centiseconds): push the
    # residual onto the last word (keeping every \k >= 1). libass tolerates a
    # mismatch, but an exact sum keeps the highlight finishing on the cue's end.
    drift = total_cs - sum(spans)
    spans[-1] = max(1, spans[-1] + drift)

    disps = [_display(w, matcher, mask) for w in in_cue]
    # Kinetic keywords: only NON-profane content words are eligible to pop.
    key_idx: set[int] = set()
    if kinetic:
        eligible = [d if not matcher.is_profane(w.word.strip()) else ""
                    for w, d in zip(in_cue, disps)]
        key_idx = _kinetic_keywords(eligible)

    out_parts: list[str] = []
    elapsed_cs = 0                          # Σ предыдущих \k -> офсет слова в мс
    for i, (disp, cs) in enumerate(zip(disps, spans)):
        esc = _ass_text_escape(disp)
        if i in key_idx:
            pop = _kinetic_pop_tag(elapsed_cs * 10, accent)
            out_parts.append(f"{{\\k{cs}{pop}}}{esc}")
        else:
            out_parts.append(f"{{\\k{cs}}}{esc}")
        elapsed_cs += cs
    return " ".join(out_parts)


def write_ass(cues: list[Cue], path: str | Path, style: "AssStyleCfg", *,
              karaoke: bool = True, words: list[Word] | None = None,
              matcher: ProfanityMatcher | None = None,
              mask: MaskingCfg | None = None,
              play_res: tuple[int, int] = (1920, 1080)) -> None:
    r"""Write an ASS subtitle file for burn-in via the libass ``subtitles`` filter.

    ``cues`` and ``words`` must already be in FINAL (post-cut) coordinates so the
    burned timings line up with the rendered video. When ``karaoke`` is on and
    ``words`` is supplied, each cue becomes a per-word ``{\kNN}`` karaoke line;
    otherwise the plain wrapped text is used (with ``\N`` for line breaks).

    The file is written UTF-8 with BOM — libass on Windows reads BOM-prefixed
    Cyrillic most reliably.
    """
    pr_x, pr_y = int(play_res[0] or 1920), int(play_res[1] or 1080)
    align = _ASS_ALIGN.get(style.position, 2)

    header = [
        "[Script Info]",
        "ScriptType: v4.00+",
        "Collisions: Normal",
        "WrapStyle: 0",
        "ScaledBorderAndShadow: yes",
        f"PlayResX: {pr_x}",
        f"PlayResY: {pr_y}",
        "",
        "[V4+ Styles]",
        ("Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
         "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
         "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
         "Alignment, MarginL, MarginR, MarginV, Encoding"),
        # SecondaryColour is the karaoke "not yet sung" colour; PrimaryColour is
        # the highlighted (sung) colour. We want words to light up as spoken, so
        # Secondary = base text colour, Primary = karaoke highlight colour.
        ("Style: Default,{font},{size},{primary},{secondary},{outline_c},"
         "&H64000000,0,0,0,0,100,100,0,0,1,{outline},{shadow},{align},"
         "40,40,{margin_v},1").format(
            font=style.font, size=int(style.size),
            primary=(style.karaoke_color if karaoke else style.primary_color),
            secondary=style.primary_color,
            outline_c=style.outline_color,
            outline=_num(style.outline), shadow=_num(style.shadow),
            align=align, margin_v=int(style.margin_v)),
        "",
        "[Events]",
        ("Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, "
         "Effect, Text"),
    ]

    use_kara = bool(karaoke and words is not None)
    # The cues' plain text is already masked by build_cues(); when re-deriving
    # karaoke text from raw words we need a matcher to re-mask. Fall back to a
    # no-op matcher so a missing one never leaks an unmasked word.
    kmatcher = matcher if matcher is not None else ProfanityMatcher(ProfanityLists())
    kmask = mask or MaskingCfg()
    lines = list(header)
    for c in cues:
        if use_kara:
            # V11 §4b: kinetic keyword pop uses the project accent (distinct from
            # the karaoke fill so the swelling word reads as an emphasis, not
            # just the sung colour). Karaoke \k fill stays untouched.
            text = _karaoke_text(c, words or [], kmatcher, kmask,
                                 accent=_KINETIC_DEFAULT_ACCENT)
        else:
            text = _ass_text_escape(c.text)
        lines.append(
            f"Dialogue: 0,{_ass_ts(c.start)},{_ass_ts(c.end)},Default,,0,0,0,,{text}")

    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8-sig")


def _num(x: float) -> str:
    """Format a style number without a trailing '.0' for whole values."""
    f = float(x)
    return str(int(f)) if f == int(f) else f"{f:g}"


def _ass_ts(t: float) -> str:
    """ASS timestamp ``H:MM:SS.cc`` (centiseconds, one-digit hours)."""
    if t < 0:
        t = 0.0
    cs = int(round(t * 100))
    h, cs = divmod(cs, 360_000)
    m, cs = divmod(cs, 6_000)
    s, cs = divmod(cs, 100)
    return f"{h:d}:{m:02d}:{s:02d}.{cs:02d}"
