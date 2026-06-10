"""Acoustic «hesitation» detection via Silero VAD over the 16 kHz mono wav.

This catches speech stumbles that the *text* detectors (pauses / fillers /
profanity) structurally cannot see, because Whisper either does not transcribe
them at all or folds them into a neighbouring word:

  * stretched ``э-э-э`` / ``м-м`` that the ASR drops or shortens,
  * micro-cutoffs and mumbles between words,
  * short *non-speech* dead-air whose length is BELOW the pause detector's
    ``min_silence`` (so the pause detector never flags it).

Approach: run Silero VAD (``faster_whisper.vad`` — ONNX, **no torch**) over the
already-extracted ``audio16k.wav``, invert the speech timestamps into the
non-speech gaps between consecutive speech chunks, keep only gaps in
``[min_duration, max_duration)``, pad them inward, clamp to ``[0, duration]``
and drop any that meaningfully overlap an existing cut-list segment. Leading /
trailing silence is intentionally NOT flagged here — it is the pause detector's
job (with full word context), and the VAD edges there are the least reliable.

The detector is intentionally *additive*: it never mutates ``existing_segs``,
and on any failure (missing/unreadable wav, VAD import/runtime error) it returns
an empty list so the rest of detection is unaffected.
"""
from __future__ import annotations

import wave
from pathlib import Path

from ..config import HesitationsCfg
from ..models import ACTION_REMOVE, TYPE_HESITATION, CutSegment

_SAMPLE_RATE = 16000


def _read_wav_mono_f32(audio_path: str | Path):
    """Read a 16 kHz mono PCM_S16LE wav into a float32 array in [-1, 1].

    Uses only the ``wave`` + ``numpy`` stdlib/runtime deps (no soundfile /
    librosa / torch). If the file is multi-channel it is down-mixed to mono by
    averaging; a non-16 kHz rate is returned as-is and the caller scales by the
    real rate. Returns ``(samples, sample_rate)``.
    """
    import numpy as np

    with wave.open(str(audio_path), "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        rate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())

    if sampwidth != 2:
        # extract_audio() always writes pcm_s16le; anything else we don't trust.
        raise ValueError(f"unexpected wav sample width {sampwidth} (expected 2)")

    data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if n_channels > 1:
        data = data.reshape(-1, n_channels).mean(axis=1)
    return data, rate


def _get_non_speech_gaps(audio_path: str | Path,
                         cfg: HesitationsCfg) -> list[tuple[float, float]]:
    """Run Silero VAD and return the *interior* non-speech gaps in seconds.

    Each gap is ``(start, end)`` between the end of one speech chunk and the
    start of the next (in seconds). Leading silence (before the first chunk) and
    trailing silence (after the last) are deliberately excluded — pauses own
    those. The gaps are returned RAW (no length filtering); ``detect`` applies
    the thresholds and padding.
    """
    # Imported lazily so importing this module never forces onnxruntime to load
    # (and so a missing dep degrades to "no hesitations" rather than a crash).
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    audio, rate = _read_wav_mono_f32(audio_path)
    if audio.size == 0:
        return []

    options = VadOptions(
        threshold=cfg.vad_threshold,
        min_speech_duration_ms=cfg.vad_min_speech_ms,
        min_silence_duration_ms=cfg.vad_min_silence_ms,
        # We compute our OWN padding; Silero's speech_pad_ms would inflate every
        # speech chunk and shrink the very gaps we are trying to measure.
        speech_pad_ms=0,
    )
    speech = get_speech_timestamps(audio, vad_options=options, sampling_rate=rate)

    # Timestamps come back in samples; convert to seconds and sort defensively.
    spans = sorted(
        (ts["start"] / float(rate), ts["end"] / float(rate)) for ts in speech)

    gaps: list[tuple[float, float]] = []
    for (a_start, a_end), (b_start, b_end) in zip(spans, spans[1:]):
        g_start, g_end = a_end, b_start
        if g_end > g_start:
            gaps.append((g_start, g_end))
    return gaps


def _overlaps_existing(a: float, b: float,
                       existing: list[CutSegment],
                       threshold: float) -> bool:
    """True if ``[a, b]`` overlaps any existing segment by >= ``threshold``.

    The fraction is measured against the *candidate's* own length, so a
    candidate that is mostly covered by an already-flagged pause/filler/etc. is
    dropped as a duplicate even if it is much smaller than that segment.
    """
    span = b - a
    if span <= 0:
        return True
    for seg in existing:
        overlap = min(b, seg.end) - max(a, seg.start)
        if overlap > 0 and (overlap / span) >= threshold:
            return True
    return False


def detect(audio_path: str | Path, duration: float, cfg: HesitationsCfg,
           existing_segs: list[CutSegment], words: list | None = None) -> list[CutSegment]:
    """Detect hesitation cuts. Pure w.r.t. ``existing_segs`` (never mutated).

    Returns ``CutSegment``s of type :data:`TYPE_HESITATION` with ``id=""`` (the
    id is assigned later in ``run_detection``). Returns ``[]`` on any failure
    so detection as a whole is never broken by a bad/missing wav.

    ``words`` (Whisper word timestamps, optional) makes the cut WORD-SAFE: a VAD
    "non-speech" edge often lands a few ms inside a real word's tail/onset, so the
    raw gap can clip a word. We clamp each cut to lie strictly between the word
    *before* and the word *after* the gap — only inter-word non-speech is removed.
    (Partial-word cutting is reserved for the profanity censor, by design.)
    """
    dur = max(0.0, float(duration))
    try:
        gaps = _get_non_speech_gaps(audio_path, cfg)
    except Exception:  # noqa: BLE001 — VAD/wav failure must not break detection
        return []

    words = words or []
    out: list[CutSegment] = []
    for g_start, g_end in gaps:
        raw = g_end - g_start
        # Below min -> VAD artefact / inter-word breath. At/above max -> that is
        # a pause, the pause detector owns it (and would double-flag here).
        if raw < cfg.min_duration or raw >= cfg.max_duration:
            continue

        a = g_start + cfg.pad_start
        b = g_end - cfg.pad_end
        # Word-safe clamp: keep the cut inside the inter-word interval so it can
        # never bite into the surrounding words (the "огрызки слов" the user heard).
        if words:
            mid = 0.5 * (g_start + g_end)
            prev_end = max((w.end for w in words if w.start < mid), default=None)
            next_start = min((w.start for w in words if w.start >= mid), default=None)
            if prev_end is not None:
                a = max(a, prev_end)
            if next_start is not None:
                b = min(b, next_start)
        a = min(max(0.0, a), dur)
        b = min(max(0.0, b), dur)
        # Enforce the minimum on the FINAL (post-padding) length, not just the
        # raw gap — otherwise a gap barely above min_duration shrinks to a
        # 0.02–0.05 s cut (≈1 frame): imperceptible, and clutters the cut list.
        if (b - a) < cfg.min_duration:
            continue
        if _overlaps_existing(a, b, existing_segs, cfg.overlap_threshold):
            continue

        out.append(CutSegment(
            id="", start=round(a, 3), end=round(b, 3),
            type=TYPE_HESITATION, action=ACTION_REMOVE, enabled=True,
            text=f"заминка {raw:.2f}с"))
    return out
