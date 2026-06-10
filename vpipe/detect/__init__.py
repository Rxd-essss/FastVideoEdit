"""Stage 3 — build the editable cut list from a transcript."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from ..config import (Config, FillerLists, ProfanityLists)
from ..models import CutList, CutSegment, Transcript
from ..llm import OllamaClient
from . import pauses as _pauses
from . import fillers as _fillers
from . import profanity as _profanity
from . import badtakes as _badtakes
from . import hesitations as _hesitations


def run_detection(transcript: Transcript, cfg: Config,
                  fillers: FillerLists, profanity: ProfanityLists,
                  source: str, llm: Optional[OllamaClient] = None,
                  log=print,
                  audio_path: Optional[str | Path] = None) -> CutList:
    words = transcript.all_words()
    segs: list[CutSegment] = []

    if cfg.pauses.enabled:
        p = _pauses.detect(words, transcript.duration, cfg.pauses)
        log(f"  pauses: {len(p)}")
        segs += p
    if cfg.fillers.enabled:
        f = _fillers.detect(words, cfg.fillers, fillers)
        log(f"  fillers: {len(f)}")
        segs += f
    if cfg.profanity.enabled:
        pr = _profanity.detect(words, cfg.profanity, profanity)
        log(f"  profanity: {len(pr)}")
        segs += pr
    if cfg.bad_takes.enabled and llm is not None:
        try:
            bt = _badtakes.detect(transcript, cfg, llm)
            log(f"  bad takes (LLM): {len(bt)}")
            segs += bt
        except Exception as e:  # noqa: BLE001
            log(f"  bad takes skipped (LLM error: {e})")
    elif cfg.bad_takes.enabled:
        log("  bad takes skipped (LLM unavailable)")

    # Acoustic hesitations run LAST among auto-detectors so they can dedup
    # against everything already flagged above (pauses/fillers/profanity/takes).
    # Needs the 16 kHz wav for VAD; without it we simply skip (graceful).
    if cfg.hesitations.enabled:
        if audio_path is not None and Path(audio_path).exists():
            h = _hesitations.detect(audio_path, transcript.duration,
                                    cfg.hesitations, segs, words=words)
            log(f"  hesitations: {len(h)}")
            segs += h
        else:
            log("  hesitations skipped (no audio16k.wav)")

    # stable sort by start, then assign ids
    segs.sort(key=lambda s: (s.start, s.end))
    counters: dict[str, int] = {}
    for s in segs:
        n = counters.get(s.type, 0)
        counters[s.type] = n + 1
        s.id = f"{s.type[:2]}{n:03d}"

    return CutList(source=source, duration=transcript.duration, segments=segs)
