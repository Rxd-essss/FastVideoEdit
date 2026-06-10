from vpipe.config import CensorCfg
from vpipe.censor import build_censor_graph
from vpipe.models import ACTION_CENSOR, TYPE_PROFANITY, CutSegment


def C(a, b):
    return CutSegment(id="x", start=a, end=b, type=TYPE_PROFANITY,
                      action=ACTION_CENSOR, enabled=True)


def test_partial_graph():
    g = build_censor_graph([C(1.0, 1.5)], CensorCfg(method="partial"),
                           duration=10, sample_rate=48000, has_rubberband=True)
    assert g.startswith("[0:a]") and g.endswith("[cen]")
    assert "volume=" in g and "between(t," in g


def test_lowpass_graph():
    g = build_censor_graph([C(1.0, 1.5)], CensorCfg(method="lowpass"),
                           duration=10, sample_rate=48000, has_rubberband=True)
    assert "lowpass=f=500" in g and "enable='between(t," in g


def test_pitch_segmented_rubberband():
    g = build_censor_graph([C(2.0, 2.5)], CensorCfg(method="pitch"),
                           duration=10, sample_rate=48000, has_rubberband=True)
    assert "asplit=" in g and "concat=n=" in g and "rubberband=pitch=" in g


def test_pitch_segmented_fallback():
    g = build_censor_graph([C(2.0, 2.5)], CensorCfg(method="pitch"),
                           duration=10, sample_rate=44100, has_rubberband=False)
    assert "asetrate=" in g and "aresample=44100" in g and "atempo=" in g


def test_reverse_segmented():
    g = build_censor_graph([C(2.0, 2.5)], CensorCfg(method="reverse"),
                           duration=10, sample_rate=48000, has_rubberband=True)
    assert "areverse" in g and "concat=n=" in g


def test_pitch_big_shift_atempo_chain_in_range():
    import re
    cfg = CensorCfg(method="pitch", pitch={"semitones": 24, "use_rubberband": "false"})
    g = build_censor_graph([C(2.0, 2.5)], cfg, duration=10,
                           sample_rate=48000, has_rubberband=False)
    tempos = [float(x) for x in re.findall(r"atempo=([0-9.]+)", g)]
    assert tempos, "expected an atempo chain"
    assert all(0.5 - 1e-6 <= t <= 2.0 + 1e-6 for t in tempos)  # each stage valid
    prod = 1.0
    for t in tempos:
        prod *= t
    ratio = 2.0 ** (24 / 12)
    assert abs(prod - 1.0 / ratio) < 1e-3   # chain restores duration


def test_no_censors_returns_none():
    assert build_censor_graph([], CensorCfg(), 10, 48000, True) is None
