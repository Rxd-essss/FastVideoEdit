from vpipe.chapters import Chapter, enforce_rules, _fmt
from vpipe.config import ChaptersCfg


def test_fmt():
    assert _fmt(0, False) == "00:00"
    assert _fmt(84, False) == "01:24"
    assert _fmt(3725, True) == "1:02:05"


def test_enforce_forces_zero_and_spacing():
    cfg = ChaptersCfg(min_length=10, min_chapters=3, max_chapters=30)
    chs = [Chapter(5, "A"), Chapter(8, "B"), Chapter(40, "C"), Chapter(80, "D")]
    out = enforce_rules(chs, new_duration=120, cfg=cfg)
    assert out[0].time == 0.0                       # first forced to 0
    times = [c.time for c in out]
    # consecutive chapters at least min_length apart
    assert all(b - a >= 10 for a, b in zip(times, times[1:]))


def test_enforce_caps_max():
    cfg = ChaptersCfg(min_length=1, max_chapters=2)
    chs = [Chapter(0, "A"), Chapter(10, "B"), Chapter(20, "C")]
    out = enforce_rules(chs, new_duration=100, cfg=cfg)
    assert len(out) == 2
