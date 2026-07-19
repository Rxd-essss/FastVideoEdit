import re

from vpipe.config import MaskingCfg, ProfanityLists, SubsCfg
from vpipe.detect.profanity import ProfanityMatcher
from vpipe.models import Word
from vpipe.subtitles import (KINETIC_MAX_PER_CUE, KINETIC_SCALE, Cue,
                             _is_content_word, _karaoke_text, _kinetic_keywords,
                             _ts, _wrap, build_cues, mask_text, mask_word)


def test_mask_word():
    cfg = MaskingCfg()  # keep_first=1, keep_last=1, min_stars=2
    assert mask_word("блядь", cfg) == "б***ь"
    assert mask_word("сука", cfg) == "с**а"
    # very short word still gets at least min_stars
    assert mask_word("ху", MaskingCfg(keep_last=0)) == "х**"
    # default config must NOT leak a 2-letter word (regression)
    assert mask_word("ху", cfg) == "х**"
    # generous keep config must still mask at least one real char (no leak)
    m = mask_word("сука", MaskingCfg(keep_first=2, keep_last=2))
    assert "к" not in m and m.startswith("су") and m.endswith("а")
    assert mask_word("a", cfg) == "**"   # 1-letter fully masked


def test_build_cues_skips_word_past_end():
    m = ProfanityMatcher(ProfanityLists(roots=[], allow=[]))
    cues = build_cues([Word("поздно", 10.0, 11.0)], m, SubsCfg(), MaskingCfg(), total=5.0)
    assert cues == []   # word entirely past timeline end -> no degenerate cue


def test_mask_text():
    m = ProfanityMatcher(ProfanityLists(roots=["бля"], allow=[]))
    assert mask_text("ну ты блядь даёшь", m, MaskingCfg()) == "ну ты б***ь даёшь"


def test_ts():
    assert _ts(0, ",") == "00:00:00,000"
    assert _ts(3661.5, ",") == "01:01:01,500"
    assert _ts(1.5, ".") == "00:00:01.500"


def test_build_cues_splits_on_gap():
    m = ProfanityMatcher(ProfanityLists(roots=[], allow=[]))
    words = [Word("раз", 0.0, 0.4), Word("два", 0.5, 0.9),
             Word("три", 5.0, 5.4), Word("четыре", 5.5, 6.0)]
    cues = build_cues(words, m, SubsCfg(new_cue_gap=0.7), MaskingCfg(), total=7.0)
    assert len(cues) == 2          # big gap 0.9 -> 5.0 forces a split
    assert cues[0].start == 0.0
    assert cues[1].start == 5.0
    # no overlap
    assert cues[0].end <= cues[1].start


# === V11 §4b: кинетичная подсветка ключевого слова в караоке ====================
_NOPROF = ProfanityMatcher(ProfanityLists(roots=[], allow=[]))


def _ks(dia):
    return [int(x) for x in re.findall(r"\\k(\d+)", dia)]


def test_is_content_word_heuristic():
    # содержательные слова — носители смысла (длинные сущ/глаг), числа
    assert _is_content_word("локально") and _is_content_word("бесплатно")
    assert _is_content_word("2024") and _is_content_word("RTX3080")
    # стоп-лист (предлоги/союзы/частицы/местоимения) и короткие — НЕ ключевые
    for w in ("и", "в", "на", "что", "это", "так", "уже", "мы", "его", "да"):
        assert not _is_content_word(w)
    # пунктуация/кавычки чистятся перед проверкой
    assert _is_content_word("«Linux».")
    assert not _is_content_word("—")


def test_kinetic_keywords_picks_1_2_longest_content():
    words = ["Всё", "работает", "полностью", "локально", "и", "бесплатно"]
    keys = _kinetic_keywords(words)
    assert len(keys) <= KINETIC_MAX_PER_CUE
    # самые длинные content-слова: «полностью»(9)/«бесплатно»(9) — НЕ предлог «и»
    chosen = {words[i] for i in keys}
    assert "и" not in chosen
    assert all(w not in ("Всё",) for w in chosen)       # короткие не берём


def test_kinetic_keywords_empty_when_no_content():
    # реплика из одних стоп-слов -> ничего не вспухает (НЕ каждую реплику, §4b)
    assert _kinetic_keywords(["и", "в", "на", "да", "же"]) == set()


def _cue_words(pairs, t0=0.5):
    """pairs: [(word, dur_cs)] -> (Cue, [Word]) с непрерывным таймингом."""
    words, t = [], t0
    for w, cs in pairs:
        words.append(Word(w, t, t + cs / 100.0))
        t += cs / 100.0
    return Cue(t0, t, " ".join(w for w, _ in pairs)), words


def test_kinetic_pop_keeps_karaoke_fill_intact():
    # тот же \k-тайминг с попом и без — караоке-заполнение ЦЕЛО (R3).
    cue, words = _cue_words([("Всё", 28), ("работает", 46), ("полностью", 50),
                             ("локально", 64), ("и", 18), ("бесплатно", 72)])
    plain = _karaoke_text(cue, words, _NOPROF, MaskingCfg(), kinetic=False)
    kin = _karaoke_text(cue, words, _NOPROF, MaskingCfg(), kinetic=True)
    # одинаковое число \k и одинаковая сумма (= длительность реплики в cs)
    assert _ks(plain) == _ks(kin)
    assert sum(_ks(kin)) == sum(_ks(plain)) == 278
    # все слова на месте в обоих
    for w, _ in [("Всё", 0), ("работает", 0), ("локально", 0), ("бесплатно", 0)]:
        assert w in plain and w in kin


def test_kinetic_pop_adds_t_scale_and_accent():
    cue, words = _cue_words([("Всё", 28), ("работает", 46), ("полностью", 50),
                             ("локально", 64), ("и", 18), ("бесплатно", 72)])
    kin = _karaoke_text(cue, words, _NOPROF, MaskingCfg(), kinetic=True,
                        accent="&H000B9EF5")
    # 1–2 попа \t(...\fscx120\fscy120\1c...) — вспухание ≤1.2× + акцент
    pops = re.findall(r"\\t\(\d+,\d+,\\fscx" + str(KINETIC_SCALE), kin)
    assert 1 <= len(pops) <= KINETIC_MAX_PER_CUE
    assert "\\1c&H000B9EF5" in kin and "\\3c&H000B9EF5" in kin
    # возврат к 100% вторым \t
    assert "\\fscx100\\fscy100" in kin


def test_kinetic_pop_offset_is_sum_of_prior_k():
    # офсет \t = Σ предыдущих \k ×10 мс (line-relative): поп срабатывает ровно
    # когда слово произносится.
    cue, words = _cue_words([("Всё", 28), ("работает", 46), ("полностью", 50),
                             ("локально", 64)])
    kin = _karaoke_text(cue, words, _NOPROF, MaskingCfg(), kinetic=True)
    # накопленная сумма \k (×10 мс) для каждого слова — допустимые офсеты попа
    cum_ms, acc = [], 0
    for c in _ks(kin):
        cum_ms.append(acc * 10)
        acc += c
    # каждый поп = 2 \t; первый \t каждого попа стартует на офсете слова
    pop_starts = [int(x) for x in re.findall(r"\\t\((\d+),\d+,\\fscx120", kin)]
    assert pop_starts                                   # хотя бы один поп
    for off in pop_starts:
        assert off in cum_ms


def test_kinetic_pop_skips_profanity():
    # мат не вспухает (без лишнего внимания на запиканном слове)
    m = ProfanityMatcher(ProfanityLists(roots=["бля"], allow=[]))
    cue, words = _cue_words([("отвратительно", 60), ("блядь", 50)])
    kin = _karaoke_text(cue, words, m, MaskingCfg(), kinetic=True)
    assert "б***ь" in kin and "блядь" not in kin
    # запиканное слово не получает \t-поп; вспухает только «отвратительно»
    masked_seg = [seg for seg in kin.split(" ") if "***" in seg][0]
    assert "\\t(" not in masked_seg


# --- regression: karaoke must not drop the last word of a trimmed cue ----------
def test_karaoke_keeps_last_word_when_cue_end_trimmed():
    # build_cues trims cue.end below the last word's end for back-to-back speech;
    # containment selection used to silently drop that tail word from the burn.
    words = [Word("раз", 0.0, 1.0), Word("два", 1.0, 2.0)]
    cue = Cue(0.0, 1.90, "раз два")            # cue.end 1.90 < last word end 2.00
    txt = _karaoke_text(cue, words, _NOPROF, MaskingCfg())
    assert "два" in txt
    assert len(_ks(txt)) == 2                  # both words carry a \k tag


def test_karaoke_excludes_neighbour_words_at_boundaries():
    # the previous cue's last word (ends at cue.start) and the next cue's first
    # word (starts at cue.end) must stay OUT — overlap selection uses a strict eps.
    words = [Word("до", -1.0, 0.0), Word("тут", 0.5, 1.5), Word("после", 2.0, 3.0)]
    cue = Cue(0.0, 2.0, "тут")
    txt = _karaoke_text(cue, words, _NOPROF, MaskingCfg())
    assert "тут" in txt and "до" not in txt and "после" not in txt
    assert len(_ks(txt)) == 1


# --- regression: max_cps must never shrink a cue below its own last word -------
def test_max_cps_never_shrinks_cue_end_below_last_word():
    m = ProfanityMatcher(ProfanityLists(roots=[], allow=[]))
    words = [Word("информация", 3.00, 3.50), Word("передаётся", 3.50, 4.00),
             Word("полностью", 4.00, 4.50), Word("локально", 4.50, 5.00),
             Word("быстро", 5.00, 5.50), Word("надёжно", 5.50, 5.98)]
    # One dense cue: required (≈54/17≈3.2s) > duration (2.98s); limit = total-min_gap
    # = 5.95 < last word end 5.98. Pre-fix the reading-speed pass shrank end to 5.95.
    cues = build_cues(words, m, SubsCfg(), MaskingCfg(), total=6.0)
    assert cues[-1].end >= 5.98 - 1e-6         # never pulled below the last word


# === _wrap: line-wrapping heuristic (direct coverage) ==========================
# _wrap decides how every cue looks. These tests lock its two documented paths:
#   * the common 2-line balance branch (subtitles.py:83-99)
#   * the >2-line / fallback greedy fill (subtitles.py:101-117)
# NOTE: the greedy fill's `len(lines) < max_lines - 1` guard caps the line count
# at max_lines, so the trailing `lines[:max_lines]` slice never drops a word;
# and the 2-line key `(over, bad_tail, diff)` puts overflow FIRST, so a split
# that fits max_chars always wins when one exists. Both are asserted below.
def _wrap_tokens(s):
    return [t for t in s.replace("\n", " ").split(" ") if t]


def test_wrap_single_line_when_it_fits():
    # short enough for one line -> no break inserted
    out = _wrap(["раз", "два"], max_chars=42, max_lines=2)
    assert out == "раз два" and "\n" not in out


def test_wrap_two_line_balances():
    # No 2-line split fits max_chars=8 here (раз два три = 11 > 8), so _wrap picks
    # the split that best BALANCES the two line lengths and lets libass re-wrap —
    # it does NOT guarantee each line <= max_chars when nothing fits. The balanced
    # split of "раз два три четыре" is "раз два" (7) / "три четыре" (10), imbalance 3.
    out = _wrap(["раз", "два", "три", "четыре"], max_chars=8, max_lines=2)
    assert out.count("\n") == 1
    assert _wrap_tokens(out) == ["раз", "два", "три", "четыре"]   # nothing dropped
    assert out.split("\n") == ["раз два", "три четыре"]           # minimal-imbalance split


def test_wrap_avoids_bad_line_tail():
    # never end line 1 right after the conjunction «и» when a fitting split exists
    out = _wrap(["я", "и", "ты", "здесь"], max_chars=6, max_lines=2)
    first = out.split("\n")[0]
    assert not first.rstrip().endswith(" и")
    assert first.strip() != "я и"
    assert _wrap_tokens(out) == ["я", "и", "ты", "здесь"]


def test_wrap_two_line_respects_max_chars_when_split_fits():
    # 4x4-char words: a 2/2 split fits max_chars=9. `over` is the leading key
    # term, so the fitting split always beats an overflowing one.
    out = _wrap(["аааа", "бббб", "вввв", "гггг"], max_chars=9, max_lines=2)
    for line in out.split("\n"):
        assert len(line) <= 9 or len(line.split(" ")) == 1


def test_wrap_multiline_preserves_all_words():
    # >2-line greedy path (max_lines=3): every input word must survive, in order.
    # The `len(lines) < max_lines - 1` guard caps the line count at max_lines, so
    # the trailing `lines[:max_lines]` slice never silently deletes a word.
    words = ["один", "два", "три", "четыре", "пять", "шесть", "семь"]
    out = _wrap(words, max_chars=8, max_lines=3)
    assert out.count("\n") <= 2                 # at most max_lines lines
    assert _wrap_tokens(out) == words           # order preserved, nothing dropped


# --- P2: karaoke carries _wrap's \N so libass stops re-wrapping -----------------
def test_karaoke_wraps_to_two_lines():
    # cue.text carries _wrap's balanced 2-line layout; the karaoke line must carry
    # the ASS hard break \N so max_lines / Russian break rules survive the burn.
    words = [Word("слово", 0.5, 0.9), Word("раз", 0.9, 1.3),
             Word("два", 1.3, 1.7), Word("три", 1.7, 2.1),
             Word("четыре", 2.1, 2.5), Word("пять", 2.5, 2.9)]
    cue = Cue(0.5, 2.9, "слово раз два\nтри четыре пять")
    out = _karaoke_text(cue, words, _NOPROF, MaskingCfg(), kinetic=False)
    assert out.count("\\N") == 1               # exactly one hard break inserted
    first, second = out.split("\\N")
    assert len(_ks(first)) == 3 and len(_ks(second)) == 3


def test_karaoke_single_line_no_newline():
    # single-line cue (cue.text has no '\n') keeps the flat join -- regression
    # guard for the fallback so existing single-line karaoke tests are untouched.
    cue, words = _cue_words([("привет", 40), ("мир", 40)])
    out = _karaoke_text(cue, words, _NOPROF, MaskingCfg(), kinetic=False)
    assert "\\N" not in out


# --- P3: kinetic pop restores fill+outline colour at pop end -------------------
def test_kinetic_pop_restores_colours():
    # the pop's second \t must animate colour back to the karaoke fill + outline,
    # else the popped word AND the rest of the cue stay accent-amber.
    cue, words = _cue_words([("отвратительно", 60),
                             ("замечательно", 60)])
    kin = _karaoke_text(cue, words, _NOPROF, MaskingCfg(), kinetic=True,
                        accent="&H000B9EF5", karaoke_color="&H00AABBCC",
                        outline_color="&H00112233")
    assert "\\fscx100\\fscy100\\1c&H00AABBCC\\3c&H00112233" in kin


# --- W6 #3: pop is SCOPED to the keyword — static reset right after it ---------
def test_kinetic_pop_scoped_with_static_reset_after_keyword():
    # ASS-теги действуют от вставки до КОНЦА строки события: без статического
    # restore-блока сразу за ключевым словом каждое СЛЕДУЮЩЕЕ слово тоже
    # вспухало до 120% и получало акцент-обводку на время окна попа.
    cue, words = _cue_words([("отвратительно", 60), ("и", 18), ("тут", 20)])
    kin = _karaoke_text(cue, words, _NOPROF, MaskingCfg(), kinetic=True,
                        accent="&H000B9EF5", karaoke_color="&H00AABBCC",
                        outline_color="&H00112233")
    # статический сброс СРАЗУ после ключевого слова (fail-before: отсутствовал)
    assert ("отвратительно{\\fscx100\\fscy100"
            "\\1c&H00AABBCC\\3c&H00112233}") in kin
    # хвост строки после ключевого слова — чистые {\kNN}-блоки без анимации
    tail = kin.split("отвратительно", 1)[1]
    assert "\\t(" not in tail


# --- P4: continuous cues chain end-to-start (no boundary blink) -----------------
def test_continuous_cues_chain_without_gap():
    # back-to-back speech (cue0 ends exactly where cue1 starts) must chain, not
    # leave the forced min_gap that blinks the subtitle off/on at the boundary.
    words = [Word("Привет.", 0.5, 2.0), Word("мир", 2.0, 3.5)]
    cues = build_cues(words, _NOPROF, SubsCfg(), MaskingCfg(), total=4.0)
    assert len(cues) == 2
    assert abs(cues[0].end - cues[1].start) < 1e-9      # touching, no blank frame


def test_real_pause_keeps_gap():
    # a genuine pause between cues keeps the min_gap (no chaining).
    words = [Word("Привет.", 0.5, 2.0), Word("мир", 2.5, 4.0)]
    cues = build_cues(words, _NOPROF, SubsCfg(), MaskingCfg(), total=5.0)
    assert len(cues) == 2
    assert cues[0].end <= cues[1].start - SubsCfg().min_gap

