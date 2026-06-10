"""Stage 4 — build the audio filtergraph that censors profanity in place.

The graph processes audio ONLY (video is kept bit-exact elsewhere) and is run
on the FULL original timeline to produce a lossless censored track; cutting
happens afterwards (see render.py). Four methods:

  partial  — keep an audible onset, mute the vocalic middle (click-free volume
             notch with short linear ramps). Default.
  lowpass  — muffle the word ("underwater") via a timeline-gated lowpass.
  pitch    — shift the word up N semitones (rubberband, or asetrate chain).
  reverse  — reverse the word's audio.

partial/lowpass are timeline-gated (one chain). pitch/reverse cannot be gated,
so the stream is split at every word boundary and only the profane pieces are
processed, then concatenated back.
"""
from __future__ import annotations

from typing import Optional

from .config import CensorCfg
from .models import CutSegment
from .timeline import merge_intervals


def _segs(censors: list[CutSegment], duration: float) -> list[tuple[float, float]]:
    iv = [(max(0.0, c.start), min(duration, c.end)) for c in censors]
    return merge_intervals([(a, b) for a, b in iv if b > a])


def _f(x: float) -> str:
    return f"{x:.4f}"


def _partial_chain(segs, cfg, in_label, out_label) -> str:
    p = cfg.partial
    filters = []
    for (t0, t1) in segs:
        dur = t1 - t0
        m0 = t0 + p.onset
        m1 = (t0 + p.onset + p.mute_fraction * dur) if p.keep_tail else t1
        m0 = min(m0, t1)
        m1 = min(m1, t1)
        if m1 <= m0:                      # word too short for an onset -> mute all
            m0, m1 = t0, t1
        f = max(0.001, p.fade)
        # value: 0 inside [m0,m1]; linear ramp down in [m0-f,m0]; up in [m1,m1+f]; else 1
        expr = (f"if(between(t,{_f(m0)},{_f(m1)}),0,"
                f"if(between(t,{_f(m0 - f)},{_f(m0)}),({_f(m0)}-t)/{_f(f)},"
                f"if(between(t,{_f(m1)},{_f(m1 + f)}),(t-{_f(m1)})/{_f(f)},1)))")
        filters.append(f"volume='{expr}':eval=frame")
    return in_label + ",".join(filters) + out_label


def _lowpass_chain(segs, cfg, in_label, out_label) -> str:
    lp = cfg.lowpass
    filters = [f"lowpass=f={lp.cutoff}:p={lp.poles}:enable='between(t,{_f(t0)},{_f(t1)})'"
               for (t0, t1) in segs]
    return in_label + ",".join(filters) + out_label


def _pieces(segs, duration):
    """Ordered contiguous (a, b, is_censor) pieces covering [0, duration]."""
    out = []
    cursor = 0.0
    for (t0, t1) in segs:
        if t0 > cursor:
            out.append((cursor, t0, False))
        out.append((t0, t1, True))
        cursor = t1
    if cursor < duration:
        out.append((cursor, duration, False))
    return [(a, b, c) for (a, b, c) in out if b - a > 1e-4]


def _segmented(segs, duration, effect_fn, in_label, out_label) -> str:
    """Split the stream at word boundaries, process the profane pieces, concat.

    Each effect (pitch/reverse) MUST emit a piece exactly ``(b - a)`` seconds
    long: the concatenated FLAC is later atrimmed on the ORIGINAL timeline by
    render.py, so any drift accumulates and de-syncs the whole track. asetrate
    + atempo and areverse are only approximately length-preserving, so after the
    effect we force the piece length with ``apad`` (pad short) then ``atrim``
    (clip long). The sum of all pieces therefore stays == ``duration``.
    """
    pieces = _pieces(segs, duration)
    n = len(pieces)
    parts = [f"{in_label}asplit={n}" + "".join(f"[p{i}]" for i in range(n)) + ";"]
    for i, (a, b, cen) in enumerate(pieces):
        chain = f"[p{i}]atrim=start={_f(a)}:end={_f(b)},asetpts=N/SR/TB"
        if cen:
            dur = b - a
            chain += "," + effect_fn(dur)
            # Enforce exact piece length so concat stays sample-aligned to the
            # original timeline (pad-then-trim is idempotent for rubberband,
            # which is already length-preserving, but a cheap safety net).
            chain += f",apad=whole_dur={_f(dur)},atrim=end={_f(dur)},asetpts=N/SR/TB"
        parts.append(chain + f"[q{i}];")
    parts.append("".join(f"[q{i}]" for i in range(n)) + f"concat=n={n}:v=0:a=1{out_label}")
    return "".join(parts)


def _atempo_chain(factor: float) -> str:
    """Express ``factor`` as a chain of atempo filters each within [0.5, 2.0].

    A single atempo only accepts 0.5..100.0, so a large pitch shift (whose
    duration-restoring tempo would be < 0.5) must be split into stages whose
    product equals ``factor``.
    """
    steps: list[float] = []
    f = factor
    while f < 0.5 - 1e-9:
        steps.append(0.5)
        f /= 0.5
    while f > 2.0 + 1e-9:
        steps.append(2.0)
        f /= 2.0
    steps.append(f)
    return ",".join(f"atempo={s:.6f}" for s in steps)


def _pitch_effect(cfg, sample_rate, has_rubberband):
    ratio = 2.0 ** (cfg.pitch.semitones / 12.0)
    mode = cfg.pitch.use_rubberband
    use_rb = has_rubberband if mode == "auto" else (mode == "true")
    if use_rb and has_rubberband:
        return lambda dur: f"rubberband=pitch={ratio:.6f}"
    sr = sample_rate or 48000
    new = int(round(sr * ratio))
    tempo_chain = _atempo_chain(1.0 / ratio)   # restore duration after asetrate
    return lambda dur: f"asetrate={new},aresample={sr},{tempo_chain}"


def _reverse_effect(cfg):
    f = max(0.001, cfg.reverse.fade)

    def fx(dur: float) -> str:
        s = "areverse"
        if dur > 2.2 * f:
            s += f",afade=t=in:st=0:d={_f(f)},afade=t=out:st={_f(dur - f)}:d={_f(f)}"
        return s
    return fx


def build_censor_graph(censors: list[CutSegment], cfg: CensorCfg, duration: float,
                       sample_rate: int, has_rubberband: bool,
                       in_label: str = "[0:a]", out_label: str = "[cen]"
                       ) -> Optional[str]:
    """Return a filter_complex string, or None if there is nothing to censor."""
    segs = _segs(censors, duration)
    if not segs:
        return None
    method = cfg.method
    if method == "partial":
        return _partial_chain(segs, cfg, in_label, out_label)
    if method == "lowpass":
        return _lowpass_chain(segs, cfg, in_label, out_label)
    if method == "pitch":
        return _segmented(segs, duration, _pitch_effect(cfg, sample_rate, has_rubberband),
                          in_label, out_label)
    if method == "reverse":
        return _segmented(segs, duration, _reverse_effect(cfg), in_label, out_label)
    raise ValueError(f"Unknown censor method: {method!r}")
