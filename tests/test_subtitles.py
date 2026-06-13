import re

from vpipe.config import MaskingCfg, ProfanityLists, SubsCfg
from vpipe.detect.profanity import ProfanityMatcher
from vpipe.models import Word
from vpipe.subtitles import (KINETIC_MAX_PER_CUE, KINETIC_SCALE, Cue,
                             _is_content_word, _karaoke_text, _kinetic_keywords,
                             _ts, build_cues, mask_text, mask_word)


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
