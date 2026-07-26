"""Opt-in end-to-end ffmpeg integration tier (audit #81).

Every other render test pins ffmpeg's argv against FakeFF and never runs the
binary, so an ffmpeg-illegal graph (bad subtitles quoting, a concat A/V
stream-count mismatch, a renamed 8.1.1 filter option) sails through green. This
module renders a tiny synthetic clip through the REAL ffmpeg and ffprobe-asserts
OUTPUT INVARIANTS only -- duration within tolerance, >=2 streams, non-empty --
not exact bytes/codecs, so ffmpeg version drift can't false-fail it.

SKIPPED by default (tests/conftest.requires_ffmpeg): CI installs no ffmpeg and
this tier must NEVER run there. The 12s testsrc+sine fixture is synthesized via
ffmpeg lavfi INSIDE a gated fixture, so no binary media is committed and nothing
executes unless you opt in:

    FVE_FFMPEG_TESTS=1 python -m pytest tests/test_integration_render.py -v
"""
from pathlib import Path

import pytest

from conftest import requires_ffmpeg
from vpipe.config import load_config
from vpipe.ffmpeg_utils import FFmpeg
from vpipe.probe import probe_media
from vpipe.render import render
from vpipe.models import CutList, CutSegment, TYPE_MANUAL, ACTION_REMOVE

_ROOT = Path(__file__).resolve().parent.parent
_CLIP_SECONDS = 12
_REMOVED = 2.0   # cuts below drop [1,2] and [4,5] -> 2.0s excised


@pytest.fixture(scope="module")
def ff():
    cfg = load_config(str(_ROOT / "config.yaml"))
    return FFmpeg(cfg.ffmpeg)


@pytest.fixture(scope="module")
def clip(ff, tmp_path_factory):
    """Synthesize a real 12s 320x240 testsrc + 300Hz sine clip via lavfi.

    Built INSIDE this gated fixture (never at import/collection time) so the
    default hermetic suite never invokes ffmpeg.
    """
    d = tmp_path_factory.mktemp("fve_media")
    src = d / "synth.mp4"
    ff.run(["-f", "lavfi", "-i",
            f"testsrc=size=320x240:rate=25:duration={_CLIP_SECONDS}",
            "-f", "lavfi", "-i", f"sine=frequency=300:duration={_CLIP_SECONDS}",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
            "-shortest", str(src)],
           total=float(_CLIP_SECONDS), desc="synthesize test clip")
    return src


def _probe_dur_streams(ff, path):
    d = ff.probe(path)
    return float(d["format"]["duration"]), len(d["streams"])


def _cutlist(src, dur):
    # Drop [1,2] and [4,5]; clamp cut points to the fixture's real duration so a
    # shorter clip could never over-cut (read dur at runtime, never hard-coded).
    def seg(sid, a, b):
        a = max(0.0, min(a, dur))
        b = max(0.0, min(b, dur))
        return CutSegment(id=sid, start=a, end=b, type=TYPE_MANUAL,
                          action=ACTION_REMOVE, enabled=True)
    return CutList(source=str(src), duration=dur,
                   segments=[seg("a", 1.0, 2.0), seg("b", 4.0, 5.0)])


@requires_ffmpeg
@pytest.mark.ffmpeg
@pytest.mark.parametrize("opts", [
    {},                                   # plain cuts + concat + single encode
    {"scale_h": 720},                     # cuts + rescale
    {"loudnorm": True, "deess": True},    # cuts + audio mastering chain
], ids=["plain", "scale720", "loudnorm_deess"])
def test_real_render_duration_and_streams(ff, clip, tmp_path, opts):
    # Fresh cfg per case so a mutated denoise flag can't leak across params.
    cfg = load_config(str(_ROOT / "config.yaml"))
    media = probe_media(ff, clip)
    if opts.get("loudnorm"):
        cfg.render.denoise.loudnorm = True
    if opts.get("deess"):
        cfg.render.denoise.deess = True
    out = tmp_path / "out.mp4"
    render(ff, media, _cutlist(clip, media.duration), cfg, str(out), str(tmp_path),
           log=lambda *a, **k: None, scale_h=opts.get("scale_h"))
    assert out.exists() and out.stat().st_size > 0
    got_dur, nstreams = _probe_dur_streams(ff, out)
    assert abs(got_dur - (media.duration - _REMOVED)) < 0.5   # no truncation/overrun
    assert nstreams >= 2                                      # video + audio survived concat
