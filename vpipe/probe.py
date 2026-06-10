"""Stage 1 — ingest & probe: read media info and extract 16 kHz mono audio."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .ffmpeg_utils import FFmpeg, parse_fps


@dataclass
class MediaInfo:
    path: str
    duration: float
    fps: float
    width: int
    height: int
    vcodec: str
    acodec: str
    has_audio: bool
    sample_rate: int

    def describe(self) -> str:
        return (f"{self.width}x{self.height} @ {self.fps:.3f} fps, "
                f"{self.duration:.1f}s, v={self.vcodec}, "
                f"a={self.acodec or 'none'}@{self.sample_rate or 0}Hz")


def probe_media(ff: FFmpeg, path: str | Path) -> MediaInfo:
    info = ff.probe(path)
    fmt = info.get("format", {})
    streams = info.get("streams", [])
    v = next((s for s in streams if s.get("codec_type") == "video"), None)
    a = next((s for s in streams if s.get("codec_type") == "audio"), None)

    duration = float(fmt.get("duration", 0.0) or 0.0)
    if duration <= 0 and v and v.get("duration"):
        duration = float(v["duration"])

    return MediaInfo(
        path=str(path),
        duration=duration,
        fps=parse_fps(v.get("avg_frame_rate") or v.get("r_frame_rate") or "0/0") if v else 0.0,
        width=int(v.get("width", 0)) if v else 0,
        height=int(v.get("height", 0)) if v else 0,
        vcodec=v.get("codec_name", "") if v else "",
        acodec=a.get("codec_name", "") if a else "",
        has_audio=a is not None,
        sample_rate=int(a.get("sample_rate", 0)) if a else 0,
    )


def hash_input(path: str | Path, chunk: int = 1 << 20) -> str:
    """Content hash of the input file (size + sampled bytes) for cache keys.

    Samples size + first 1 MiB + last 1 MiB + several evenly-spaced interior
    256 KiB windows rather than the whole file, so the key stays cheap for
    multi-GB videos while still changing if the content changes anywhere. The
    extra interior sampling makes collisions on edited files far less likely
    than a head/tail-only digest. Returns the full sha1 hex digest.
    """
    interior_win = 1 << 18  # 256 KiB per interior window
    interior_count = 8      # number of evenly-spaced interior windows
    p = Path(path)
    size = p.stat().st_size
    h = hashlib.sha1()
    h.update(str(size).encode())
    with open(p, "rb") as f:
        # Head.
        h.update(f.read(chunk))
        # Tail (only if it does not overlap the head).
        if size > chunk * 2:
            f.seek(-chunk, 2)
            h.update(f.read(chunk))
        # Evenly-spaced interior windows between head and tail.
        lo = chunk
        hi = size - chunk
        if hi - lo > interior_win:
            span = hi - lo
            for i in range(1, interior_count + 1):
                off = lo + (span * i) // (interior_count + 1)
                f.seek(off)
                h.update(f.read(interior_win))
    return h.hexdigest()


def extract_audio(ff: FFmpeg, src: str | Path, out_wav: str | Path,
                  total: Optional[float] = None, on_progress=None) -> str:
    """Extract 16 kHz mono PCM WAV (what faster-whisper wants)."""
    out_wav = str(out_wav)
    Path(out_wav).parent.mkdir(parents=True, exist_ok=True)
    ff.run(["-i", str(src), "-vn", "-ac", "1", "-ar", "16000",
            "-c:a", "pcm_s16le", out_wav],
           total=total, on_progress=on_progress, desc="audio extraction")
    return out_wav
