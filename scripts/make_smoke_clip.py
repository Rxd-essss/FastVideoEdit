"""Create a synthetic test clip + a crafted transcript in the cache.

This lets us exercise the full pipeline (detect -> censor -> cut -> subtitles ->
chapters -> summary) end-to-end without running real transcription. The crafted
transcript deliberately contains a long pause, filler words, a filler phrase and
one profane word.

    python scripts/make_smoke_clip.py
    python pipeline.py tests/_media/test.mp4 --out ./out --no-llm --apply
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vpipe.config import load_config
from vpipe.ffmpeg_utils import FFmpeg
from vpipe.models import Segment, Transcript, Word
from vpipe.probe import hash_input, probe_media

ROOT = Path(__file__).resolve().parent.parent
MEDIA = ROOT / "tests" / "_media"
CLIP = MEDIA / "test.mp4"


def make_clip(ff: FFmpeg) -> None:
    MEDIA.mkdir(parents=True, exist_ok=True)
    ff.run(["-f", "lavfi", "-i", "testsrc=size=320x240:rate=25:duration=12",
            "-f", "lavfi", "-i", "sine=frequency=300:duration=12",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
            "-shortest", str(CLIP)],
           total=12, desc="make test clip")


def craft_transcript(duration: float, audio_hash: str) -> Transcript:
    def w(txt, s, e):
        return Word(txt, s, e, 0.9)

    seg0 = Segment(0.0, 2.6, "привет это тест ну как бы", [
        w("привет", 0.0, 0.5), w("это", 0.6, 1.0), w("тест", 1.1, 1.7),
        w("ну", 1.8, 2.0), w("как", 2.05, 2.3), w("бы", 2.35, 2.6),
    ])
    # 3.4 s pause here (2.6 -> 6.0)
    seg1 = Segment(6.0, 9.6, "вот блядь всё отлично работает ребята", [
        w("вот", 6.0, 6.3), w("блядь", 6.4, 6.9), w("всё", 7.0, 7.4),
        w("отлично", 7.5, 8.2), w("работает", 8.3, 9.0), w("ребята", 9.1, 9.6),
    ])
    # 2.4 s trailing silence (9.6 -> 12)
    return Transcript(language="ru", duration=duration, model="(crafted)",
                      audio_hash=audio_hash, segments=[seg0, seg1])


def main() -> None:
    cfg = load_config(str(ROOT / "config.yaml"))
    ff = FFmpeg(cfg.ffmpeg)
    make_clip(ff)
    media = probe_media(ff, CLIP)
    h = hash_input(CLIP)
    cache_dir = ROOT / cfg.paths.cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    tr = craft_transcript(media.duration, h)
    out = cache_dir / f"{h}.transcript.json"
    tr.save(out)
    print(f"clip:       {CLIP}  ({media.describe()})")
    print(f"transcript: {out}")
    print("now run:  python pipeline.py tests/_media/test.mp4 --out ./out --no-llm --apply")


if __name__ == "__main__":
    main()
