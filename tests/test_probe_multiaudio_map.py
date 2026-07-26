# -*- coding: utf-8 -*-
"""Wave 4 (#58): pin transcription (extract_audio) and the no-cut copy fast-path
to the SAME first audio track (0:a:0) so a multi-track OBS source (mic+desktop)
is analyzed and rendered on the same stream. Single-audio sources unaffected.
Fakes only — no real ffmpeg."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vpipe.config import Config                                       # noqa: E402
from vpipe.models import CutList                                      # noqa: E402
from vpipe.probe import MediaInfo, extract_audio                      # noqa: E402
from vpipe.render import render                                       # noqa: E402


class _RecFF:
    """Records every run's args; x264 forced for determinism."""

    def __init__(self):
        self.runs: list[list[str]] = []

    def has_filter(self, name):
        return True

    def has_encoder(self, name):
        return False

    def run(self, args, total=None, on_progress=None, desc="ffmpeg"):
        self.runs.append(list(args))
        try:
            Path(args[-1]).write_bytes(b"")
        except OSError:
            pass
        return ""


def _map_targets(args):
    return [args[i + 1] for i, a in enumerate(args) if a == "-map"]


def _media(has_audio=True):
    return MediaInfo(path="in.mp4", duration=10.0, fps=30.0, width=1920,
                     height=1080, vcodec="h264", acodec="aac",
                     has_audio=has_audio, sample_rate=48000)


def test_extract_audio_maps_first_track(tmp_path):
    # FAILS before the fix: extract_audio has NO -map, so ffmpeg's default
    # selection picks the most-channels track (not necessarily the first).
    ff = _RecFF()
    out = tmp_path / "a.wav"
    extract_audio(ff, "in.mkv", str(out))
    args = ff.runs[-1]
    i = args.index("-map")
    assert args[i + 1] == "0:a:0?"        # first audio stream, tolerant


def test_nocut_copy_maps_single_audio(tmp_path):
    # FAILS before the fix: the no-cut copy fast-path used bare `-map 0:a`,
    # copying ALL audio tracks; must keep exactly one (0:a:0) like every path.
    ff = _RecFF()
    cl = CutList(source="in.mp4", duration=10.0, segments=[])
    out = str(tmp_path / "out.mp4")
    render(ff, _media(), cl, Config(), out, str(tmp_path),
           log=lambda *a, **k: None)
    args = ff.runs[-1]
    assert "-filter_complex" not in args              # lossless copy fast-path
    targets = _map_targets(args)
    assert targets == ["0:v", "0:a:0"]
    assert "0:a" not in targets                       # not the bare multi-track map
