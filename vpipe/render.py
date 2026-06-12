"""Stage 6 — render the final video.

Two stages keep the VIDEO encoded exactly once at high quality:
  1. produce a LOSSLESS censored audio track on the original timeline (video
     untouched);
  2. one pass that frame-accurately drops the removed intervals from both the
     video and the censored audio, NVENC-encoding the video a single time.
"""
from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Optional

from .censor import build_censor_graph
from .config import Config
from .cutlist import resolve
from .ffmpeg_utils import FFmpeg
from .models import CutList
from .probe import MediaInfo
from .timeline import Timeline


def _f(x: float) -> str:
    return f"{x:.3f}"


def _audio_seg_filter(asrc: str, a: float, b: float, i: int, n_kept: int,
                      cut_fade: float) -> str:
    """Per-kept-segment audio filter: trim + (optional) seam de-click fades.

    Builds ``<asrc>atrim=...,asetpts=PTS-STARTPTS`` and, when ``cut_fade>0``,
    appends a fade-in / fade-out that ramps the cut edges to zero so the concat
    seam has no waveform jump (the «рвано» click). Fades apply only to INTERNAL
    seams — never the program's true start (segment 0) or end (segment n-1) — and
    are equal-length & length-preserving, so audio/video stay in sync."""
    af = f"{asrc}atrim=start={_f(a)}:end={_f(b)},asetpts=PTS-STARTPTS"
    seg = b - a
    if cut_fade > 0 and seg > 0:
        fade_in = i > 0
        fade_out = i < n_kept - 1
        # Split the budget when a short segment is faded on BOTH sides.
        d = min(cut_fade, seg / 2 if (fade_in and fade_out) else seg)
        if fade_in and d > 0:
            af += f",afade=t=in:st=0:d={_f(d)}"
        if fade_out and d > 0:
            af += f",afade=t=out:st={_f(seg - d)}:d={_f(d)}"
    return af


def _edge_fade_filters(edge_fade: float, out_dur: float) -> list[str]:
    """De-click fades for the program's TRUE edges (Clip Maker F8).

    A Shorts clip cut from mid-phrase / a noise bed starts and ends with a full-
    amplitude waveform step — a click. :func:`_audio_seg_filter` deliberately
    fades only INTERNAL seams, so the clip's own start/end stay hard. This
    returns an ``afade=in`` at t=0 and an ``afade=out`` ending exactly at
    ``out_dur`` (the FINAL audio duration, after all cuts), to be appended to
    the final audio chain. ``edge_fade`` is clamped to a sane 0..0.2 s and to
    half the output duration (a 30 ms clip must not fade past its middle);
    <=0 (or an empty program) disables the fades entirely.

    Placement contract (why these run AFTER the whole apost chain, incl.
    loudnorm): loudnorm is an adaptive gain stage — applied AFTER a fade it
    would pump the fading edges back up, partially undoing the de-click ramp,
    whereas afade as the LAST gain stage guarantees the output ramps from/to
    literal zero. The loudness target is not perturbed: 2×~25 ms of faded edge
    is far below loudnorm's 400 ms gating blocks and negligible over a 20-60 s
    integration window. And the 2-pass measurement chain
    (:func:`build_loudnorm_measure_chain`) intentionally knows nothing about
    edge fades — with the fade after loudnorm the measured stats still describe
    exactly the audio that enters loudnorm, bit-for-bit.
    """
    edge_fade = min(0.2, max(0.0, float(edge_fade or 0.0)))
    d = min(edge_fade, out_dur / 2)
    if d <= 0:
        return []
    return [f"afade=t=in:st=0:d={_f(d)}",
            f"afade=t=out:st={_f(out_dur - d)}:d={_f(d)}"]


# Force constant frame rate on every re-encode (cut/rescale) pass. VFR sources
# (phone/OBS screen recordings) otherwise drift or stutter at concat seams.
_CFR = ["-fps_mode", "cfr"]


_CONTAINER_FMT = {".mp4": "mp4", ".mov": "mov", ".mkv": "matroska",
                  ".webm": "webm", ".m4v": "mp4"}


def _run_atomic(ff: FFmpeg, args: list[str], out_path: str, *,
                total=None, on_progress=None, desc="ffmpeg") -> None:
    """Run ffmpeg writing to ``out_path + '.part'`` then atomically replace.

    The final file appears only after ffmpeg exits 0, so a crashed/cancelled
    render never leaves a truncated, playable-looking file behind. ``args`` must
    end with the output path placeholder == ``out_path``; we substitute the temp
    path for the trailing occurrence. Because the temp name's extension is
    ``.part`` (which ffmpeg cannot map to a muxer), we inject an explicit
    ``-f <format>`` derived from the real output extension before that arg.
    """
    tmp = out_path + ".part"
    fmt = _CONTAINER_FMT.get(Path(out_path).suffix.lower(), "mp4")
    # The output path is always the last positional arg in our call sites.
    run_args = list(args)
    if run_args and run_args[-1] == out_path:
        run_args[-1:] = ["-f", fmt, tmp]
    else:                                  # defensive: replace any exact match
        out: list[str] = []
        for a in run_args:
            if a == out_path:
                out += ["-f", fmt, tmp]
            else:
                out.append(a)
        run_args = out
    try:
        ff.run(run_args, total=total, on_progress=on_progress, desc=desc)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise
    os.replace(tmp, out_path)


def video_encoder_args(cfg: Config, has_nvenc: bool, log=print) -> list[str]:
    r = cfg.render
    use_nvenc = (r.encoder == "nvenc") and has_nvenc
    if r.encoder == "nvenc" and not has_nvenc:
        log("  NVENC not available — falling back to libx264.")
    if use_nvenc:
        n = r.nvenc
        args = ["-c:v", "h264_nvenc", "-preset", n.preset, "-tune", n.tune]
        if n.rc == "vbr":
            args += ["-rc", "vbr", "-cq", str(n.cq), "-b:v", "0"]
        else:
            args += ["-rc", "constqp", "-qp", str(n.qp)]
        args += ["-rc-lookahead", "32", "-spatial-aq", "1", "-temporal-aq", "1",
                 "-b_ref_mode", "middle", "-bf", "3",
                 "-profile:v", "high", "-pix_fmt", "yuv420p", *_CFR]
        return args
    x = r.x264
    return ["-c:v", "libx264", "-crf", str(x.crf), "-preset", x.preset,
            "-profile:v", "high", "-pix_fmt", "yuv420p", *_CFR]


def _audio_args(cfg: Config) -> list[str]:
    return ["-c:a", "aac", "-b:a", cfg.render.audio_bitrate]


# YouTube speech-loudness target. _LOUDNORM_DYNAMIC is the historical one-pass
# filter string (kept byte-for-byte); _LOUDNORM_MEASURE is the same target with
# JSON stats printing for the 2-pass measurement run.
_LOUDNORM_DYNAMIC = "loudnorm=I=-14:TP=-1.5:LRA=11"
_LOUDNORM_MEASURE = _LOUDNORM_DYNAMIC + ":print_format=json"

# Keys we need from ffmpeg's loudnorm JSON block to build the linear pass 2.
_LOUDNORM_KEYS = ("input_i", "input_tp", "input_lra", "input_thresh",
                  "target_offset")

# A flat (non-nested) {...} block that mentions "input_i" — loudnorm's stats
# JSON. [^{}] deliberately matches newlines, so the multi-line block is caught.
_LOUDNORM_JSON_RE = re.compile(r'\{[^{}]*"input_i"[^{}]*\}')


def parse_loudnorm_stats(stderr: str) -> Optional[dict]:
    """Extract loudnorm measurement stats from an ffmpeg stderr dump.

    ffmpeg prints the ``print_format=json`` block at the very end of the run
    (after the usual banner/progress noise), so we scan for every flat ``{...}``
    block containing ``"input_i"`` and take the LAST parseable one. Returns a
    dict of floats keyed by :data:`_LOUDNORM_KEYS`, or ``None`` when no usable
    block exists (missing keys, broken JSON, non-finite values like ``-inf`` on
    pure silence) — the caller then falls back to one-pass dynamic mode.
    """
    if not stderr:
        return None
    for block in reversed(_LOUDNORM_JSON_RE.findall(stderr)):
        try:
            data = json.loads(block)
            stats = {k: float(data[k]) for k in _LOUDNORM_KEYS}
        except (ValueError, KeyError, TypeError):
            continue
        if all(math.isfinite(v) for v in stats.values()):
            return stats
    return None


def _loudnorm_linear(measured: dict) -> Optional[str]:
    """The pass-2 LINEAR loudnorm filter built from measured stats.

    Returns ``None`` when the dict is unusable (missing/non-numeric/non-finite
    values) so :func:`build_apost` degrades to the dynamic one-pass string
    instead of emitting a broken filter.
    """
    try:
        vals = {k: float(measured[k]) for k in _LOUDNORM_KEYS}
    except (ValueError, KeyError, TypeError):
        return None
    if not all(math.isfinite(v) for v in vals.values()):
        return None
    return (_LOUDNORM_DYNAMIC
            + f":measured_I={vals['input_i']:.2f}"
            f":measured_TP={vals['input_tp']:.2f}"
            f":measured_LRA={vals['input_lra']:.2f}"
            f":measured_thresh={vals['input_thresh']:.2f}"
            f":offset={vals['target_offset']:.2f}"
            ":linear=true")


def build_apost(cfg: Config, loudnorm_measured: Optional[dict] = None) -> list[str]:
    """Audio post-filters: denoise + mastering ([] when everything is off).

    The list is meant to be ``",".join``-ed and applied to the *final* audio
    (the concat/mux output) so it always runs AFTER profanity censoring —
    censoring happens in Stage 1 and is consumed here as a lossless FLAC input.

    Order matters: highpass (kill low hum) -> afftdn (FFT noise reduction) ->
    optional dynaudnorm (gentle level normalisation) -> optional deesser (soft,
    i=0.4 — the filter's own default intensity is 0, i.e. a no-op) -> optional
    loudnorm -> aresample. All are native ffmpeg filters; no external .rnn
    model is used.

    The denoise trio (highpass/afftdn/dynaudnorm) is gated by ``denoise.enabled``;
    ``deess`` and ``loudnorm`` are INDEPENDENT mastering switches that work with
    the denoiser off. Only when all three toggles are off does this return ``[]``
    so the byte-for-byte audio copy fast-path is preserved.

    Loudness: without ``loudnorm_measured`` the loudnorm is the historical
    single-pass DYNAMIC mode (``loudnorm=I=-14:TP=-1.5:LRA=11``) — acceptable
    for speech. When ``loudnorm_measured`` (the dict from
    :func:`parse_loudnorm_stats`, produced by the 2-pass measurement run) is
    given AND usable, the accurate two-pass LINEAR variant with ``measured_*``
    values and ``linear=true`` is emitted instead; an unusable dict silently
    degrades to dynamic. loudnorm internally resamples to 192 kHz, so
    ``aresample=48000`` directly after it restores a sane rate in both modes.

    Engine: with ``denoise.engine == "deepfilter"`` the denoising already
    happened EXTERNALLY (DeepFilterNet CLI, see :func:`enhance_audio`), so the
    highpass/afftdn/dynaudnorm trio is skipped here — only the independent
    mastering add-ons (deesser/loudnorm) remain. :func:`render` resets the
    engine to ``"afftdn"`` on its working config when the CLI is unavailable,
    so the fallback path gets the trio back automatically.
    """
    d = cfg.render.denoise
    deess = bool(getattr(d, "deess", False))
    loudnorm = bool(getattr(d, "loudnorm", False))
    engine = str(getattr(d, "engine", "afftdn") or "").strip().lower()
    ffmpeg_denoise = bool(d.enabled) and engine != "deepfilter"
    if not (ffmpeg_denoise or deess or loudnorm):
        return []
    filters: list[str] = []
    if ffmpeg_denoise:
        if d.highpass_hz and d.highpass_hz > 0:
            filters.append(f"highpass=f={int(d.highpass_hz)}")
        filters.append(f"afftdn=nf={d.nf:.1f}")
        if d.normalize:
            filters.append("dynaudnorm=p=0.95:m=100")
    if deess:
        filters.append("deesser=i=0.4")
    if loudnorm:
        linear = _loudnorm_linear(loudnorm_measured) if loudnorm_measured else None
        filters.append(linear or _LOUDNORM_DYNAMIC)
        filters.append("aresample=48000")
    return filters


def _wants_loudnorm_2pass(cfg: Config) -> bool:
    """True when the user opted into the accurate 2-pass loudnorm.

    Gated on BOTH switches so the default config ("dynamic", or loudnorm off)
    keeps the legacy single-pass behaviour with zero extra ffmpeg runs.
    """
    d = cfg.render.denoise
    return (bool(getattr(d, "loudnorm", False))
            and str(getattr(d, "loudnorm_mode", "dynamic") or "").strip().lower()
            == "2pass")


def build_loudnorm_measure_chain(cfg: Config) -> str:
    """Audio chain for the loudnorm measurement pass (pass 1 of 2).

    EXACTLY the final pre-loudnorm chain (denoise + deesser as configured —
    reuses :func:`build_apost` on a copy with loudnorm forced off, so the two
    passes can never drift apart) followed by a measurement-only
    ``loudnorm=...:print_format=json``.
    """
    mcfg = cfg.model_copy(deep=True)
    mcfg.render.denoise.loudnorm = False
    return ",".join(build_apost(mcfg) + [_LOUDNORM_MEASURE])


def measure_loudness(ff: FFmpeg, cfg: Config, inputs: list[str], asrc_idx: str,
                     kept: Optional[list[tuple[float, float]]], *,
                     cut_fade: float = 0.0, total: Optional[float] = None,
                     log=print) -> Optional[dict]:
    """Loudnorm measurement pass: replay the FINAL audio into ``-f null``.

    Rebuilds the exact audio the encode pass will master — per-segment trims +
    seam de-click fades + concat when ``kept`` is given (cuts present), or the
    full censored/original track otherwise — runs it through
    :func:`build_loudnorm_measure_chain` and parses the JSON stats ffmpeg prints
    on stderr. Only the audio stream is mapped, so the video is never decoded
    and the pass takes seconds.

    Graceful degradation: ANY failure (ffmpeg error, no/broken JSON) logs an
    honest message and returns ``None`` — the caller falls back to the one-pass
    dynamic loudnorm and the render NEVER fails because of the measurement.
    """
    log("Измеряю громкость…")
    chain = build_loudnorm_measure_chain(cfg)
    asrc = f"[{asrc_idx}:a]"
    if kept:
        n_kept = len(kept)
        aparts = [_audio_seg_filter(asrc, a, b, i, n_kept, cut_fade) + f"[a{i}]"
                  for i, (a, b) in enumerate(kept)]
        labels = "".join(f"[a{i}]" for i in range(n_kept))
        graph = (";".join(aparts) + ";" + labels
                 + f"concat=n={n_kept}:v=0:a=1[mraw];[mraw]{chain}[mout]")
    else:
        graph = f"{asrc}{chain}[mout]"
    args = [*inputs, "-filter_complex", graph, "-map", "[mout]",
            "-f", "null", "-"]
    try:
        stderr = ff.run(args, total=total, desc="loudnorm measure")
    except Exception as e:  # noqa: BLE001 — measurement is strictly best-effort
        log(f"  loudnorm 2pass: измерение не удалось ({e}) — "
            "включаю обычный однопроходный режим.")
        return None
    stats = parse_loudnorm_stats(stderr or "")
    if stats is None:
        log("  loudnorm 2pass: ffmpeg не вернул JSON с измерениями — "
            "включаю обычный однопроходный режим.")
    else:
        log(f"  loudnorm 2pass: измерено I={stats['input_i']:.1f} LUFS, "
            f"TP={stats['input_tp']:.1f} dBTP — применяю linear-нормализацию.")
    return stats


def _faststart(cfg: Config) -> list[str]:
    return ["-movflags", "+faststart"] if cfg.render.faststart else []


def censor_audio(ff: FFmpeg, media: MediaInfo, cl: CutList, cfg: Config,
                 work_dir: str | Path, on_progress=None, log=print) -> Optional[str]:
    """Stage 1: write a lossless FLAC with profanity censored. None if no-op."""
    _, censors = resolve(cl)
    if not censors:
        return None
    has_rb = ff.has_filter("rubberband")
    graph = build_censor_graph(censors, cfg.censor, media.duration,
                               media.sample_rate, has_rb)
    if graph is None:
        return None
    out = str(Path(work_dir) / "censored.flac")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    log(f"  censoring {len(censors)} segment(s) via '{cfg.censor.method}'"
        + ("" if (cfg.censor.method != 'pitch' or has_rb) else " (asetrate fallback: no librubberband)"))
    ff.run(["-i", media.path, "-vn", "-filter_complex", graph,
            "-map", "[cen]", "-c:a", "flac", out],
           total=media.duration, on_progress=on_progress, desc="censor audio")
    return out


# --- DeepFilterNet 3 neural denoise (external CLI, opt-in engine) ------------
# Repo root (parent of vpipe/) — `tools/deep-filter.exe` ships relative to it.
_REPO_ROOT = Path(__file__).resolve().parents[1]


def _resolve_deepfilter_bin(configured: str) -> Optional[str]:
    """Resolve the ``deep-filter`` CLI binary to a runnable path (or ``None``).

    Mirrors the spirit of :func:`vpipe.ffmpeg_utils.resolve_bin` but NEVER
    raises — a missing neural denoiser must degrade to the afftdn chain, not
    kill the render. Lookup order: absolute path -> repo-root-relative (the
    vendored ``tools/deep-filter.exe``) -> cwd-relative -> PATH (the full
    configured name, then the bare stem, so ``deep-filter`` on PATH works even
    when the config says ``tools/deep-filter.exe``).
    """
    configured = str(configured or "").strip()
    if not configured:
        return None
    p = Path(configured)
    if p.is_absolute():
        return str(p) if p.exists() else None
    cand = _REPO_ROOT / configured
    if cand.exists():
        return str(cand)
    if p.exists():
        return str(p)
    return shutil.which(configured) or shutil.which(p.name) or shutil.which(p.stem)


def _cleanup_dfn_temp(work_dir: str | Path) -> None:
    """Best-effort removal of the DFN intermediate wavs after a SUCCESSFUL render.

    The 48 kHz pcm wavs are big (~220 MB per 20 min); the censored FLAC is kept
    (as before), but these are pure scratch. Failures are swallowed — cleanup
    must never fail a render that already produced its output.
    """
    wd = Path(work_dir)
    for p in (wd / "dfn_out" / "dfn_in.wav", wd / "dfn_in.wav"):
        try:
            p.unlink()
        except OSError:
            pass
    try:
        (wd / "dfn_out").rmdir()               # only if now empty
    except OSError:
        pass


def enhance_audio(ff: FFmpeg, src: str, cfg: Config, work_dir: str | Path,
                  log=print, on_progress=None,
                  total: Optional[float] = None) -> Optional[str]:
    """Stage 1.5: neural speech denoise via the DeepFilterNet 3 CLI.

    Modelled on :func:`censor_audio`: consumes ``src`` (the censored FLAC when
    profanity was censored, else the original media file), produces an enhanced
    wav on the ORIGINAL timeline and returns its path so :func:`render` can
    substitute it as the graph's audio input. Steps:

      1. ffmpeg extracts ``work_dir/dfn_in.wav`` — 48 kHz MONO pcm_s16le.
         HONEST LIMITATION: ``-ac 1`` downmixes stereo to mono. Fine for
         talking-head speech (this project's domain), wrong for stereo music
         beds — that is why the engine is strictly opt-in.
      2. ``deep-filter.exe [--pf] -D -o work_dir/dfn_out work_dir/dfn_in.wav``.
         The CLI writes ``<out-dir>/<input-basename>`` and creates the out dir
         itself (verified against v0.5.6). ``-D`` compensates the STFT/model
         lookahead delay so the output stays time-aligned with the source
         (measured: ~30 ms trimmed at the tail) — required because the cut /
         censor timestamps live on the original timeline.
      3. Returns ``work_dir/dfn_out/dfn_in.wav``.

    Graceful degradation: ANY failure (binary not found, extraction error,
    non-zero exit, missing/empty output) logs an honest message and returns
    ``None`` — the caller then falls back to the afftdn chain and the render
    NEVER fails because of the neural engine.
    """
    d = cfg.render.denoise
    configured = str(getattr(d, "deepfilter_bin", "deep-filter") or "")
    binp = _resolve_deepfilter_bin(configured)
    if not binp:
        log(f"  DeepFilterNet недоступен (бинарь '{configured}' не найден) — "
            "использую afftdn.")
        return None
    wd = Path(work_dir)
    wd.mkdir(parents=True, exist_ok=True)
    in_wav = wd / "dfn_in.wav"
    out_dir = wd / "dfn_out"
    out_wav = out_dir / in_wav.name            # CLI keeps the input basename
    try:
        ff.run(["-i", str(src), "-vn", "-ac", "1", "-ar", "48000",
                "-c:a", "pcm_s16le", str(in_wav)],
               total=total, on_progress=on_progress, desc="extract wav (DFN)")
    except Exception as e:  # noqa: BLE001 — enhancement is strictly best-effort
        log(f"  DeepFilterNet: извлечение WAV не удалось ({e}) — использую afftdn.")
        _cleanup_dfn_temp(wd)   # a partial dfn_in.wav must not pile up (audit C-1)
        return None
    post_filter = bool(getattr(d, "post_filter", True))
    cmd = [binp] + (["--pf"] if post_filter else []) + \
        ["-D", "-o", str(out_dir), str(in_wav)]
    log("  DeepFilterNet: нейроденойз (CPU"
        + (", post-filter" if post_filter else "") + ")…")
    try:
        r = subprocess.run(cmd, capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
    except OSError as e:
        log(f"  DeepFilterNet: запуск не удался ({e}) — использую afftdn.")
        _cleanup_dfn_temp(wd)
        return None
    if r.returncode != 0:
        tail = " | ".join((r.stderr or r.stdout or "").strip().splitlines()[-3:])
        log(f"  DeepFilterNet: exe завершился с ошибкой (exit {r.returncode})"
            + (f": {tail}" if tail else "") + " — использую afftdn.")
        _cleanup_dfn_temp(wd)
        return None
    try:
        ok = out_wav.exists() and out_wav.stat().st_size > 0
    except OSError:
        ok = False
    if not ok:
        log("  DeepFilterNet: выходной файл не появился — использую afftdn.")
        _cleanup_dfn_temp(wd)
        return None
    return str(out_wav)


def _ass_path_for_filter(p: str) -> str:
    r"""Escape a filesystem path for ffmpeg's ``subtitles=`` filter option.

    The libass filter parses its argument through ffmpeg's filtergraph lexer, so
    on Windows the drive colon and backslashes must be neutralised. We use
    forward slashes, escape the drive-letter colon (``C:`` -> ``C\:``), and
    escape the few characters that are filtergraph option delimiters. The result
    is meant to be wrapped in single quotes by the caller:
    ``subtitles='C\:/work/burn.ass'``.
    """
    p = p.replace("\\", "/")
    # Only the colon directly after a drive letter needs escaping (e.g. C:/).
    if len(p) >= 2 and p[1] == ":":
        p = p[0] + "\\:" + p[2:]
    # Filtergraph option delimiters that could prematurely end the value.
    p = p.replace(",", "\\,").replace("[", "\\[").replace("]", "\\]")
    p = p.replace(";", "\\;")
    p = p.replace("'", "\\'")   # a quote in the path must not close the value
    return p


def render(ff: FFmpeg, media: MediaInfo, cl: CutList, cfg: Config,
           out_path: str | Path, work_dir: str | Path,
           on_progress=None, log=print,
           scale_h: Optional[int] = None, fps: Optional[float] = None,
           ass_path: Optional[str] = None,
           crop_filter: Optional[str] = None,
           edge_fade: float = 0.0) -> dict:
    """Run both stages and write the final mp4. Returns a small summary dict.

    ``scale_h``/``fps`` override the output resolution height / frame rate; None
    keeps the source value. ``ass_path`` (when set) burns the ASS subtitle file
    into the video as the LAST video filter (after scale/fps), forcing a single
    re-encode pass over NVENC/x264. ``crop_filter`` (when set) is a fully-formed
    ``crop=...,scale=WxH`` string for the vertical (9:16) Shorts clip; it runs
    FIRST in the chain (crop -> scale -> fps -> subtitles) so the face crop and
    target resize happen before any burn-in. For vertical renders the caller
    passes ``scale_h=None`` because the exact target scale is baked into
    ``crop_filter``.

    ``edge_fade`` (Clip Maker F8): seconds of de-click afade-in/out applied to
    the TRUE edges of the FINAL audio (after all cuts/censoring/apost — see
    :func:`_edge_fade_filters` for the loudnorm-ordering rationale). The caller
    sets it EXPLICITLY and only for Shorts-clip renders; the default 0.0 keeps
    every regular full-video render byte-for-byte unchanged. Clamped to 0..0.2.
    """
    removed, censors = resolve(cl)
    tl = Timeline(removed, media.duration)
    new_dur = tl.new_duration()
    # Drop kept slivers shorter than ~1 video frame (a zero-width trim breaks
    # concat) OR shorter than min_segment — a tiny breath/VAD-edge blip between
    # two near-adjacent cuts that would just stutter. min_segment stays below the
    # shortest real word, so this merges cuts without dropping speech.
    frame = (1.2 / media.fps) if media.fps > 0 else 0.04
    min_seg = max(frame, float(getattr(cfg.render, "min_segment", 0.0) or 0.0))
    all_kept = tl.kept_segments()
    kept = [(a, b) for (a, b) in all_kept if (b - a) >= min_seg]
    if len(kept) < len(all_kept):
        log(f"  dropped {len(all_kept) - len(kept)} tiny kept sliver(s) "
            f"(< {min_seg*1000:.0f} ms) — merged near-adjacent cuts.")
    if not kept:
        raise RuntimeError("Every part of the video is marked for removal — nothing to render.")

    # Identity-resize fast-path: a "resize" to the source height or "fps" to the
    # source rate is a no-op, so drop it to keep the lossless copy path.
    if scale_h and media.height and int(scale_h) == int(media.height):
        scale_h = None
    if fps and media.fps and abs(float(fps) - media.fps) < 0.01:
        fps = None

    has_audio = bool(media.has_audio)
    # Wrap progress so the bar moves during both passes: when there is anything
    # to censor, the censor pass owns 0..0.3 and the encode pass 0.3..1.0;
    # otherwise the encode pass owns the whole bar. With the opt-in DeepFilterNet
    # engine a thin 0.10 slice is carved out for the enhance pass (censor then
    # owns 0..0.25); without it the historical mapping is untouched.
    will_censor = bool(has_audio and censors)
    dn = cfg.render.denoise
    use_dfn = bool(has_audio and dn.enabled
                   and str(getattr(dn, "engine", "afftdn") or "")
                   .strip().lower() == "deepfilter")
    censor_share = 0.25 if use_dfn else 0.30

    def _censor_prog(p):                            # 0..censor_share
        if on_progress is not None:
            on_progress(censor_share * min(1.0, max(0.0, p)))

    censor_prog = _censor_prog if will_censor else None

    # Censoring needs an audio track; skip it entirely for video-only sources.
    censored = (censor_audio(ff, media, cl, cfg, work_dir,
                             on_progress=censor_prog, log=log)
                if has_audio else None)

    # Stage 1.5 (opt-in): neural denoise of the censored/original audio via the
    # DeepFilterNet CLI. On ANY failure enhance_audio returns None — then the
    # render behaves exactly as if the user had picked the afftdn engine (the
    # trio returns to build_apost and to the loudnorm measurement chain). The
    # engine reset happens on a deep copy so the caller's config is untouched.
    enhanced: Optional[str] = None
    if use_dfn:
        enh_prog = None
        if on_progress is not None:
            e_start = censor_share if censored else 0.0

            def enh_prog(p, _op=on_progress, _s=e_start):
                _op(_s + 0.10 * min(1.0, max(0.0, p)))
        enhanced = enhance_audio(ff, censored or media.path, cfg, work_dir,
                                 log=log, on_progress=enh_prog,
                                 total=media.duration)
        if enhanced is None:
            log("  DeepFilterNet недоступен — использую afftdn.")
            cfg = cfg.model_copy(deep=True)
            cfg.render.denoise.engine = "afftdn"

    enc_start = ((censor_share if censored else 0.0)
                 + (0.10 if enhanced else 0.0))
    if enc_start > 0 and on_progress is not None:
        def encode_prog(p, _op=on_progress, _s=enc_start):   # enc_start..1.0
            _op(_s + (1.0 - _s) * min(1.0, max(0.0, p)))
    else:
        encode_prog = on_progress
    has_nvenc = ff.has_encoder("h264_nvenc")
    venc = video_encoder_args(cfg, has_nvenc, log=log)
    enc_name = "nvenc" if (cfg.render.encoder == "nvenc" and has_nvenc) else "x264"
    out_path = str(out_path)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    vpost = []
    # Vertical 9:16 crop+scale must run FIRST: crop -> scale -> fps -> subtitles.
    # crop_filter already bakes in the exact target scale (e.g. ...,scale=1080:1920),
    # so the caller passes scale_h=None and the generic scale step below is skipped.
    if crop_filter:
        vpost.append(crop_filter)
    if scale_h:
        vpost.append(f"scale=-2:{int(scale_h)}")
    if fps:
        vpost.append(f"fps={fps}")
    # Burn-in subtitles MUST be the last filter so karaoke timings line up with
    # the final (scaled/retimed) frames. A non-empty vpost forces re-encode, so
    # the no-cuts/no-scale copy fast-path is correctly bypassed when burning.
    if ass_path:
        vpost.append(f"subtitles='{_ass_path_for_filter(ass_path)}'")
    vpost_s = ",".join(vpost)
    # The graph's audio source: the DFN-enhanced wav (already censored, when
    # there was anything to censor) wins over the censored FLAC, which wins
    # over the original track. Either external file enters as input #1.
    audio_src = enhanced or censored
    asrc_idx = "1" if audio_src else "0"
    inputs = ["-i", media.path] + (["-i", audio_src] if audio_src else [])
    # Two-pass loudnorm (opt-in via denoise.loudnorm_mode="2pass"): measure the
    # FINAL audio (cuts + censor + denoise + deesser, WITHOUT loudnorm) first,
    # then hand the stats to build_apost for the accurate linear pass. The
    # measurement runs AFTER censoring (it consumes the censored FLAC) and is
    # audio-only, so it costs seconds. None (failure) -> dynamic fallback.
    measured: Optional[dict] = None
    if has_audio and _wants_loudnorm_2pass(cfg):
        m_fade = max(0.0, float(getattr(cfg.render, "cut_fade", 0.0) or 0.0))
        measured = measure_loudness(ff, cfg, inputs, asrc_idx,
                                    kept if removed else None,
                                    cut_fade=m_fade, total=new_dur, log=log)
    # Audio post-chain (denoise). Empty list when disabled -> the audio path stays
    # byte-for-byte identical to before (copy / passthrough fast-paths preserved).
    apost = build_apost(cfg, loudnorm_measured=measured) if has_audio else []
    apost_s = ",".join(apost)
    # F8: de-click fades on the clip's TRUE edges, appended AFTER the whole
    # apost chain (incl. loudnorm — ordering rationale in _edge_fade_filters).
    # The fade-out anchors to the FINAL audio duration: with cuts that is the
    # exact concat length (kept already excludes sub-min_segment slivers),
    # otherwise the full source duration. [] when edge_fade<=0 (every regular
    # full-video render) -> all fast-paths/graphs stay byte-for-byte identical.
    out_adur = sum(b - a for a, b in kept)
    efades = _edge_fade_filters(edge_fade, out_adur) if has_audio else []

    if not removed and not vpost:
        # No cuts, no rescale: just mux (video copied — no quality loss). Denoise,
        # when on, forces the audio through filter_complex (video still copied).
        if not has_audio:
            log("  no cuts — remuxing (video copied).")
            args = ["-i", media.path, "-map", "0:v",
                    "-c:v", "copy", "-an", *_faststart(cfg), out_path]
        elif apost or efades:
            achain = ",".join(apost + efades)   # == apost_s for regular renders
            log(f"  no cuts — remuxing (video copied); denoise audio ({achain}).")
            asrc_label = f"[{asrc_idx}:a]"  # DFN wav / FLAC if any, else original
            graph = f"{asrc_label}{achain}[outa]"
            args = [*inputs, "-filter_complex", graph,
                    "-map", "0:v", "-map", "[outa]",
                    "-c:v", "copy", *_audio_args(cfg), *_faststart(cfg), out_path]
        elif audio_src:
            log("  no cuts — remuxing (video copied).")
            args = ["-i", media.path, "-i", audio_src, "-map", "0:v", "-map", "1:a",
                    "-c:v", "copy", *_audio_args(cfg), *_faststart(cfg), out_path]
        else:
            log("  no cuts — remuxing (video copied).")
            args = ["-i", media.path, "-map", "0:v", "-map", "0:a",
                    "-c:v", "copy", "-c:a", "copy", *_faststart(cfg), out_path]
        # finally: the ~220 MB DFN temp wavs must not survive a failed/cancelled
        # encode either (audit C-1) — same in the other two _run_atomic branches.
        try:
            _run_atomic(ff, args, out_path, total=media.duration,
                        on_progress=encode_prog, desc="remux")
        finally:
            if enhanced:
                _cleanup_dfn_temp(work_dir)
        return {"out": out_path, "new_duration": media.duration,
                "removed": 0.0, "censored": len(censors),
                "encoder": "copy", "denoise": bool(apost) or bool(enhanced)}

    if not removed:
        # No cuts but rescale/fps requested -> re-encode video; pass audio through.
        if has_audio and (apost or efades):
            # Audio must enter the graph to apply the denoise/edge-fade filters.
            achain = ",".join(apost + efades)
            graph = (f"[0:v]{vpost_s}[outv];"
                     f"[{asrc_idx}:a]{achain}[outa]")
            args = [*inputs, "-filter_complex", graph, "-map", "[outv]",
                    "-map", "[outa]", *venc, *_audio_args(cfg),
                    *_faststart(cfg), out_path]
        elif has_audio:
            graph = f"[0:v]{vpost_s}[outv]"
            args = [*inputs, "-filter_complex", graph, "-map", "[outv]",
                    "-map", f"{asrc_idx}:a", *venc, *_audio_args(cfg),
                    *_faststart(cfg), out_path]
        else:
            graph = f"[0:v]{vpost_s}[outv]"
            args = [*inputs, "-filter_complex", graph, "-map", "[outv]",
                    "-an", *venc, *_faststart(cfg), out_path]
        log(f"  re-encoding (no cuts, {vpost_s or 'reencode'}) -> {enc_name}")
        try:
            _run_atomic(ff, args, out_path, total=media.duration,
                        on_progress=encode_prog, desc="render")
        finally:
            if enhanced:
                _cleanup_dfn_temp(work_dir)
        return {"out": out_path, "new_duration": media.duration,
                "removed": 0.0, "censored": len(censors),
                "encoder": enc_name, "denoise": bool(apost) or bool(enhanced)}

    # Cuts (optionally + rescale/fps): one frame-accurate pass.
    if has_audio:
        asrc = "[1:a]" if audio_src else "[0:a]"
        cut_fade = max(0.0, float(getattr(cfg.render, "cut_fade", 0.0) or 0.0))
        n_kept = len(kept)
        vparts, aparts, concat = [], [], []
        for i, (a, b) in enumerate(kept):
            vparts.append(f"[0:v]trim=start={_f(a)}:end={_f(b)},setpts=PTS-STARTPTS[v{i}]")
            aparts.append(_audio_seg_filter(asrc, a, b, i, n_kept, cut_fade) + f"[a{i}]")
            concat.append(f"[v{i}][a{i}]")
        base = ";".join(vparts + aparts) + ";" + "".join(concat)
        # When denoise/edge fades are on, the concat's audio leaves as
        # [outa_raw] and the post chain produces the final [outa] — so it runs
        # on the FULL retimed (already-censored) audio, never per-segment.
        # efades come strictly AFTER apost (i.e. after loudnorm) — see
        # _edge_fade_filters for why that order protects the de-click ramp.
        afinal = apost + efades
        a_lbl = "[outa_raw]" if afinal else "[outa]"
        if vpost:
            graph = base + f"concat=n={len(kept)}:v=1:a=1[vc]{a_lbl};[vc]{vpost_s}[outv]"
        else:
            graph = base + f"concat=n={len(kept)}:v=1:a=1[outv]{a_lbl}"
        if afinal:
            graph += f";[outa_raw]{','.join(afinal)}[outa]"
        args = [*inputs, "-filter_complex", graph, "-map", "[outv]", "-map", "[outa]",
                *venc, *_audio_args(cfg), *_faststart(cfg), out_path]
    else:
        # Video-only: concat with a=0 and map only the video stream.
        vparts, concat = [], []
        for i, (a, b) in enumerate(kept):
            vparts.append(f"[0:v]trim=start={_f(a)}:end={_f(b)},setpts=PTS-STARTPTS[v{i}]")
            concat.append(f"[v{i}]")
        base = ";".join(vparts) + ";" + "".join(concat)
        if vpost:
            graph = base + f"concat=n={len(kept)}:v=1:a=0[vc];[vc]{vpost_s}[outv]"
        else:
            graph = base + f"concat=n={len(kept)}:v=1:a=0[outv]"
        args = [*inputs, "-filter_complex", graph, "-map", "[outv]",
                "-an", *venc, *_faststart(cfg), out_path]
    log(f"  encoding {len(kept)} kept segment(s) -> {new_dur:.1f}s ({enc_name})")
    try:
        _run_atomic(ff, args, out_path, total=new_dur,
                    on_progress=encode_prog, desc="render")
    finally:
        if enhanced:
            _cleanup_dfn_temp(work_dir)
    return {"out": out_path, "new_duration": new_dur, "removed": tl.total_removed,
            "censored": len(censors), "encoder": enc_name,
            "denoise": bool(apost) or bool(enhanced)}
