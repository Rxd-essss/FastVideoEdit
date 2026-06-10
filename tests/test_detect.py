from vpipe.config import (FillerLists, FillersCfg, PausesCfg, ProfanityCfg,
                          ProfanityLists)
from vpipe.detect import fillers as fdet
from vpipe.detect import pauses as pdet
from vpipe.detect import profanity as prdet
from vpipe.models import Word


def W(t, txt, d=0.4):
    return Word(txt, t, t + d)


def test_pauses():
    words = [W(0.0, "a"), W(0.5, "b"), W(5.0, "c")]  # big gap 0.9->5.0
    out = pdet.detect(words, duration=8.0, cfg=PausesCfg())
    # one mid pause + trailing silence (c ends 5.4, dur 8)
    types = [s.type for s in out]
    assert all(t == "pause" for t in types)
    assert any(s.start > 0.9 and s.end < 5.0 for s in out)   # padded mid pause
    assert any(s.end == 8.0 for s in out)                    # trailing


def test_fillers_word_and_phrase():
    words = [W(0, "Привет"), W(1, "ну"), W(2, "как"), W(2.5, "бы"), W(3, "всё")]
    lists = FillerLists(mumbles=["ну+"], words=[], phrases=[["как", "бы"]])
    out = fdet.detect(words, FillersCfg(), lists)
    texts = sorted(s.text for s in out)
    assert "ну" in texts
    assert "как бы" in texts
    assert len(out) == 2


def test_fillers_dont_eat_neighbours():
    words = [W(0.0, "раз", 0.4), W(0.45, "ну", 0.2), W(0.7, "два", 0.4)]
    out = fdet.detect(words, FillersCfg(pad=0.5), FillerLists(mumbles=["ну+"]))
    seg = out[0]
    assert seg.start >= 0.4    # clamped to previous word end
    assert seg.end <= 0.7      # clamped to next word start


def test_profanity_yo_forms_caught():
    # ё in roots must work even though inputs are folded ё->е (regression)
    lists = ProfanityLists(roots=["ёб", "еб[аёуниыл]", "бля"], allow=[])
    m = prdet.ProfanityMatcher(lists)
    assert m.is_profane("заёб")
    assert m.is_profane("подъёб")
    assert m.is_profane("ёбаный")


def test_profanity_no_overmatch():
    lists = ProfanityLists(roots=["мандав", "бля"], allow=[])
    m = prdet.ProfanityMatcher(lists)
    assert not m.is_profane("команда")
    assert not m.is_profane("мандарин")
    assert m.is_profane("мандавошка")


def test_fillers_inverted_span_skipped():
    # overlapping ASR timestamps: prev ends (1.0) after next starts (0.6)
    words = [Word("раз", 0.0, 1.0), Word("ну", 0.5, 0.7), Word("два", 0.6, 1.5)]
    out = fdet.detect(words, FillersCfg(pad=0.0), FillerLists(mumbles=["ну+"]))
    assert all(s.end > s.start for s in out)        # no inverted interval emitted
    assert not any(s.text == "ну" for s in out)     # clamped-to-inverted -> skipped


def test_profanity_roots_and_allow():
    lists = ProfanityLists(roots=["бля", "еб[аёуниыл]"], allow=["хлеб", "ребенок"])
    m = prdet.ProfanityMatcher(lists)
    assert m.is_profane("блядь")
    assert m.is_profane("Бляяя!")
    assert not m.is_profane("хлеб")
    assert not m.is_profane("ребёнок")   # ё folded, in allow
    words = [W(0, "это"), W(1, "блядь"), W(2, "хлеб")]
    out = prdet.detect(words, ProfanityCfg(action="censor"), lists)
    assert len(out) == 1
    assert out[0].action == "censor"
    assert out[0].word == "блядь"
