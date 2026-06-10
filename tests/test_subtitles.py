from vpipe.config import MaskingCfg, ProfanityLists, SubsCfg
from vpipe.detect.profanity import ProfanityMatcher
from vpipe.models import Word
from vpipe.subtitles import _ts, build_cues, mask_text, mask_word


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
