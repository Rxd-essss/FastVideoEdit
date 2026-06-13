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
from .enrich import (MAX_ANIMS, MAX_STILLS, AnimOverlay, RenderEnrich,
                     StillOverlay, ZoomWindow)
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


# --- background music bed + auto-duck (C3, render.music) ---------------------
def music_source(cfg: Config, log=print) -> Optional[str]:
    """The validated background-music path, or ``None`` (disabled / missing).

    Render-side safety net: serve validates ``music.path`` with a 400 up
    front, but a config-file user can point it anywhere — a missing file logs
    an honest message and the render continues WITHOUT music (the bed must
    never fail a render)."""
    m = getattr(cfg.render, "music", None)
    if m is None or not getattr(m, "enabled", False):
        return None
    p = str(getattr(m, "path", "") or "").strip()
    if not p:
        log("  Фоновая музыка: путь к файлу не задан — рендер без музыки.")
        return None
    if not Path(p).is_file():
        log(f"  Фоновая музыка: файл не найден ({p}) — рендер без музыки.")
        return None
    return p


def build_music_mix(m, music_idx: int, speech_lbl: str, out_lbl: str,
                    out_dur: float) -> str:
    """Filtergraph block: looped bed -> gain -> sidechain duck -> sum w/ speech.

    ``[music_idx:a]`` is the looped (``-stream_loop -1``) music input: atrim
    caps it at the FINAL program duration (after cuts), ``volume`` drops it to
    ``gain_db``. The speech is split in two: one copy stays the on-air voice,
    the other is the sidechain KEY — ``sidechaincompress`` (main=music,
    key=speech) pushes the bed down while the host talks; ``duck_db`` maps to
    the filter's ``mix`` (see :class:`vpipe.config.MusicCfg`). ``amix`` sums
    voice + ducked bed with ``normalize=0`` (no -6 dB penalty on the speech)
    and ``duration=first`` (the SPEECH defines the program length)."""
    duck = min(0.0, float(getattr(m, "duck_db", -12.0) or 0.0))
    mix = min(1.0, max(0.0, 1.0 - 10.0 ** (duck / 20.0)))
    gain = float(getattr(m, "gain_db", -18.0) or 0.0)
    return (
        f"[{music_idx}:a]atrim=start=0:end={_f(out_dur)},asetpts=PTS-STARTPTS,"
        f"volume={gain:g}dB[bgm];"
        f"{speech_lbl}asplit=2[spd][spk];"
        f"[bgm][spk]sidechaincompress=threshold={float(m.threshold):g}"
        f":ratio={float(m.ratio):g}:attack={float(m.attack):g}"
        f":release={float(m.release):g}:mix={mix:.3f}[duck];"
        f"[spd][duck]amix=inputs=2:duration=first:dropout_transition=0"
        f":normalize=0{out_lbl}"
    )


def _split_apost_at_loudnorm(apost: list[str]) -> tuple[list[str], list[str]]:
    """Split the audio post chain into (speech-only, mastering) halves.

    Everything BEFORE the loudnorm filter (highpass/afftdn/dynaudnorm/deesser)
    treats the SPEECH and must run before the music is mixed in — denoising the
    bed would be wrong. loudnorm and its trailing aresample master the FINAL
    mix, so they run after. Without loudnorm the whole chain is speech-only."""
    for i, f in enumerate(apost):
        if f.startswith("loudnorm"):
            return apost[:i], apost[i:]
    return list(apost), []


def _music_audio_graph(cfg: Config, speech_lbl: str, music_idx: int,
                       out_dur: float, apost: list[str],
                       efades: list[str]) -> str:
    """Full audio sub-graph with the music bed, ending in ``[outa]``.

    Order contract (КРИТИЧНО): speech-only treatment (denoise trio + deesser)
    -> ducked-music mix -> loudnorm(+aresample) -> edge fades. loudnorm MUST
    master the FINAL mix — running it before amix would normalise the speech
    alone and the published loudness would be off by up to ``gain_db``."""
    pre, master = _split_apost_at_loudnorm(apost)
    parts: list[str] = []
    sp = speech_lbl
    if pre:
        parts.append(f"{sp}{','.join(pre)}[sp]")
        sp = "[sp]"
    tail = master + efades
    parts.append(build_music_mix(cfg.render.music, music_idx, sp,
                                 "[mix]" if tail else "[outa]", out_dur))
    if tail:
        parts.append(f"[mix]{','.join(tail)}[outa]")
    return ";".join(parts)


def measure_loudness(ff: FFmpeg, cfg: Config, inputs: list[str], asrc_idx: str,
                     kept: Optional[list[tuple[float, float]]], *,
                     cut_fade: float = 0.0, total: Optional[float] = None,
                     music_idx: Optional[int] = None, music_dur: float = 0.0,
                     log=print) -> Optional[dict]:
    """Loudnorm measurement pass: replay the FINAL audio into ``-f null``.

    Rebuilds the exact audio the encode pass will master — per-segment trims +
    seam de-click fades + concat when ``kept`` is given (cuts present), or the
    full censored/original track otherwise — runs it through
    :func:`build_loudnorm_measure_chain` and parses the JSON stats ffmpeg prints
    on stderr. Only the audio stream is mapped, so the video is never decoded
    and the pass takes seconds.

    ``music_idx`` (C3, the background-music bed): when set, the measurement
    includes the SAME ducked-music mix the encode pass will feed into loudnorm
    (speech treatment -> :func:`build_music_mix` -> measure) — otherwise the
    measured I/TP/LRA would describe the bare speech, not loudnorm's real
    input, and the linear pass 2 would land off-target by up to ``gain_db``.
    ``music_dur`` is the FINAL audio duration the bed is trimmed to.

    Graceful degradation: ANY failure (ffmpeg error, no/broken JSON) logs an
    honest message and returns ``None`` — the caller falls back to the one-pass
    dynamic loudnorm and the render NEVER fails because of the measurement.
    """
    log("Измеряю громкость…")
    asrc = f"[{asrc_idx}:a]"
    if kept:
        n_kept = len(kept)
        aparts = [_audio_seg_filter(asrc, a, b, i, n_kept, cut_fade) + f"[a{i}]"
                  for i, (a, b) in enumerate(kept)]
        labels = "".join(f"[a{i}]" for i in range(n_kept))
        head = (";".join(aparts) + ";" + labels
                + f"concat=n={n_kept}:v=0:a=1[mraw];")
        sp = "[mraw]"
    else:
        head, sp = "", asrc
    if music_idx is not None:
        # 2-pass symmetry with the bed: pre-loudnorm speech chain (build_apost
        # with loudnorm off — the build_loudnorm_measure_chain recipe), then
        # the IDENTICAL build_music_mix block, then the measurement loudnorm.
        mcfg = cfg.model_copy(deep=True)
        mcfg.render.denoise.loudnorm = False
        pre = build_apost(mcfg)
        if pre:
            head += f"{sp}{','.join(pre)}[mpre];"
            sp = "[mpre]"
        head += build_music_mix(cfg.render.music, music_idx, sp,
                                "[mmix]", music_dur) + ";"
        graph = head + f"[mmix]{_LOUDNORM_MEASURE}[mout]"
    else:
        graph = head + f"{sp}{build_loudnorm_measure_chain(cfg)}[mout]"
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


# --- enrich dynamics: punch-zoom + blur-backplate (V11 §3, §4a) ---------------
# Анти-кринж / перф-числа в КОДЕ (R3/R2 — qwen числа игнорирует). Окна (t0/t1) и
# z_max приходят из плана (FINAL-секунды); здесь — техника фильтра.
PUNCH_RAMP_S = 0.7            # рамп-ин/рамп-аут зума (smoothstep), §4a
# Даунскейл-blur (R2 §4): blur на 1/16 пикселей + bilinear-upscale ≡ boxblur=20,
# в 3.5× дешевле наивного полноэкранного boxblur. Числа доказаны кадрами спайка.
BLUR_DS_W = 480              # даунскейл-ширина ветки blur
BLUR_DS_H = 270              # даунскейл-высота
BLUR_BOX = "boxblur=4:2"     # boxblur на даунскейле (≡ sigma≈20 после upscale)
BLUR_EQ = "eq=brightness=-0.16:saturation=0.85"   # затемнение + десатурация фона


def _smoothstep_z_expr(zw: "ZoomWindow", ramp: float = PUNCH_RAMP_S) -> str:
    r"""Кусочная smoothstep-огибающая Z(t) для одного окна punch-zoom (§4a).

    Z(t): 1.0 вне окна; рамп-ин ``ramp`` c (smoothstep 1->z_max), hold на пике,
    рамп-аут ``ramp`` c (z_max->1). smoothstep ``p*p*(3-2p)`` — джиттер 0.082
    против 0.153 у zoompan (R3: zoompan ~1.9× дёрганей, отвергнут). Рампы
    клампятся, чтобы для короткого окна вход/выход не наложились.
    """
    t0, t1 = float(zw.t0), float(zw.t1)
    zmax = float(zw.z_max)
    rin = min(ramp, max(0.0, (t1 - t0) / 2.0))
    rout = rin
    ti0, ti1 = t0, t0 + rin              # рамп-ин
    to0, to1 = t1 - rout, t1            # рамп-аут
    one = "1"
    if rin <= 0:                        # вырожденное окно -> без зума
        return one
    # p_in = (t-ti0)/rin; ss_in = p*p*(3-2p); z_in = 1 + (zmax-1)*ss_in
    pin = f"((t-{_f(ti0)})/{_f(rin)})"
    ssin = f"({pin}*{pin}*(3-2*{pin}))"
    zin = f"(1+{_f(zmax - 1.0)}*{ssin})"
    pout = f"((t-{_f(to0)})/{_f(rout)})"
    ssout = f"({pout}*{pout}*(3-2*{pout}))"
    zout = f"({_f(zmax)}-{_f(zmax - 1.0)}*{ssout})"
    # if(in-window) { if(ramp-in) zin else if(ramp-out) zout else zmax } else 1
    inwin = f"between(t,{_f(t0)},{_f(t1)})"
    inramp = f"lt(t,{_f(ti1)})"
    outramp = f"gt(t,{_f(to0)})"
    body = (f"if({inramp},{zin},"
            f"if({outramp},{zout},{_f(zmax)}))")
    return f"if({inwin},{body},{one})"


def _punch_dims(media: MediaInfo, scale_h: Optional[int],
                crop_filter: Optional[str]) -> tuple[int, int]:
    r"""Финальные дименшены кадра (W, H) для punch-zoom scale-back.

    Зеркало ``serve._final_render_dims``: для вертикального рендера цель уже
    запечена в ``crop_filter`` (``...,scale=W:H``) — берём её; иначе высота =
    ``scale_h`` (или исходная), ширина пропорциональна источнику. Дименшены
    нужны ТОЛЬКО когда есть punch-окна; при их отсутствии не используются.
    """
    if crop_filter:
        m = re.search(r"scale=(\d+):(\d+)", crop_filter)
        if m:
            return int(m.group(1)), int(m.group(2))
    out_h = int(scale_h) if scale_h else int(media.height or 1080)
    src_w = int(media.width or 1920)
    src_h = int(media.height or 1080)
    out_w = int(round(src_w * out_h / src_h)) if src_h else 1920
    return max(1, out_w), max(1, out_h)


def _punch_filter(punches: list["ZoomWindow"], out_w: int, out_h: int) -> str:
    r"""crop+scale punch-zoom (§4a, ПОБЕДИЛ zoompan): кадр кропится по Z(t) и
    скейлится обратно в ``out_w``×``out_h`` — scale интерполирует субпиксельно
    (crop целочисленно снапит top-left, но финальный scale это прячет).

    Несколько окон складываются в ОДНУ Z(t) перемножением огибающих (окна
    планировщик разносит зазором ≥30-60 c, так что перемножение = max — активна
    максимум одна). Возвращает строку фильтра ``crop=...,scale=W:H`` (без
    меток) для вставки в comma-chain. Пустой список окон -> "" (no-op).
    """
    if not punches or out_w <= 0 or out_h <= 0:
        return ""
    exprs = [_smoothstep_z_expr(zw) for zw in punches]
    z = exprs[0] if len(exprs) == 1 else "*".join(f"({e})" for e in exprs)
    # crop по Z(t), центрированный; scale обратно в фикс. финальные дименшены
    # (НЕ iw:ih — после crop iw=cropped). \Z живёт в crop-выражениях.
    return (f"crop=w='iw/({z})':h='ih/({z})'"
            f":x='(iw-iw/({z}))/2':y='(ih-ih/({z}))/2',"
            f"scale={int(out_w)}:{int(out_h)}")


# --- enrich overlays (ENRICH_PLAN §2.1) ---------------------------------------
def _enrich_video_chain(src_lbl: str, vpre: list[str],
                        stills: list[StillOverlay], anims: list[AnimOverlay],
                        first_idx: int, vsubs: list[str],
                        *, card_windows: Optional[list[tuple[float, float]]] = None,
                        punches: Optional[list["ZoomWindow"]] = None,
                        out_w: int = 0, out_h: int = 0) -> str:
    r"""Video sub-graph from ``src_lbl`` to ``[outv]`` with enrich overlays.

    Statement order is the §2.1/§3/§4a contract: ``vpre`` (crop -> scale -> fps)
    first, then — V11 — punch-zoom (whole-frame crop+scale, §4a) and per-card
    blur-backplate (§3) BEFORE the overlay nodes (zoom the whole frame, then PiP
    on top; the frosted panel floats over its own blurred backdrop), then one
    overlay node per still/anim — the input indices follow the order the enrich
    ``-i`` entries were appended after the music input — and finally the
    subtitles filters from ``vsubs`` (enrich.ass with fontsdir FIRST, burn.ass
    LAST). All t0/t1 are FINAL (post-concat) seconds; the fractions are formatted
    by :func:`_f` (f-strings — always a dot, the filtergraph never sees a locale
    comma). Empty ``card_windows``/``punches`` keep the graph byte-for-byte the
    pre-V11 chain (purely additive).

    PNG stills get ``format=rgba`` + alpha fade in/out (R2 §1); WebM anims get
    ``setpts=PTS+t0/TB`` to shift their clock to the show window and — ONLY
    when the input is looped with ``-stream_loop -1`` — ``shortest=1`` on the
    overlay, otherwise framesync's default ``eof_action=repeat`` keeps
    repeating the bed's last frame FOREVER and the render never ends (R2
    trap #2). A finite (non-looped) anim must NOT get ``shortest=1`` — that
    would truncate the whole program at the anim's end.

    Blur-backplate (§3) is the R2 ``trimdscale`` technique: a ``split`` branch
    is down-scaled (``480:270``) → ``boxblur`` → darkened → up-scaled bilinear
    (≡ ``boxblur=20``, 3.5× cheaper, no blockiness) then ``trim``+``setpts``
    into the card window and overlaid with ``eof_action=pass``. The TRAP (R2 §4):
    ``enable=`` on the overlay does NOT gate the upstream blur — without ``trim``
    the blur is computed on EVERY frame of the whole clip. ``trim``+``setpts``
    confines the cost to the card seconds; ``eof_action=pass`` lets the main
    stream continue past the trimmed branch's end.
    """
    stmts: list[str] = []
    cur = src_lbl
    nb = 0
    if vpre:
        stmts.append(f"{cur}{','.join(vpre)}[vb{nb}]")
        cur = f"[vb{nb}]"
        nb += 1
    # V11 §4a — punch-zoom (whole-frame), BEFORE overlays/backplate.
    punch_f = _punch_filter(list(punches or []), out_w, out_h)
    if punch_f:
        out = f"[vb{nb}]"
        stmts.append(f"{cur}{punch_f}{out}")
        cur, nb = out, nb + 1
    # V11 §3 — blur-backplate per card window (R2 trimdscale), BEFORE overlays.
    for ci, (c0, c1) in enumerate(card_windows or []):
        c0 = max(0.0, float(c0))
        c1 = max(c0, float(c1))
        blur_lbl = f"[bbk{ci}]"
        base_lbl = f"[bbs{ci}]"
        trim_lbl = f"[bbb{ci}]"         # blur-ветка обрезана в окно карточки
        out = f"[vb{nb}]"
        # split: одна ветка — фон, вторая → blur-backplate в окно карточки.
        stmts.append(f"{cur}split=2{base_lbl}{blur_lbl}")
        # КРИТ-ПОРЯДОК (LAW §3/§4, R2 ловушка #1): trim ИДЁТ ПЕРВЫМ, ДО boxblur.
        # Раньше тут стоял scale2ref+trim-после — scale2ref тянет кадры из
        # ПОЛНОДЛИННОЙ reference-ветки в лок-степе, поэтому boxblur считался на
        # ВЕСЬ клип (+537% перф), а trim лишь выбрасывал готовые кадры. Перенос
        # trim перед даунскейл-blur ограничивает дорогой boxblur только секундами
        # карточки (оконно-пропорциональная стоимость, +60% как в спайке), а
        # плоский scale=W:H:flags=bilinear (вместо scale2ref) даёт тот же
        # bilinear-upscale ≡ boxblur=20 без полнодлинной reference-привязки.
        # setpts держит PTS в окне [c0,c1]; overlay eof_action=pass даёт main
        # пройти за конец обрезанной ветки.
        stmts.append(
            f"{blur_lbl}trim={_f(c0)}:{_f(c1)},setpts=PTS-STARTPTS+{_f(c0)}/TB,"
            f"scale={BLUR_DS_W}:{BLUR_DS_H},{BLUR_BOX},{BLUR_EQ},"
            f"scale={int(out_w)}:{int(out_h)}:flags=bilinear{trim_lbl}")
        stmts.append(
            f"{base_lbl}{trim_lbl}overlay=0:0"
            f":enable='between(t,{_f(c0)},{_f(c1)})':eof_action=pass{out}")
        cur, nb = out, nb + 1
    idx = first_idx
    for k, st in enumerate(stills):
        # TODO(kenburns v1.1): st.kenburns renders as a plain PNG overlay for
        # now; the zoompan + alphamerge branch (R2 §2) is deferred — the model
        # field exists so plans stay forward-compatible.
        fade = max(0.0, float(st.fade_s or 0.0))
        prep = f"[{idx}:v]format=rgba,scale={int(st.scale_w)}:-1"
        if fade > 0:
            prep += (f",fade=t=in:st={_f(st.t0)}:d={_f(fade)}:alpha=1"
                     f",fade=t=out:st={_f(max(st.t0, st.t1 - fade))}"
                     f":d={_f(fade)}:alpha=1")
        stmts.append(prep + f"[ov{k}]")
        out = f"[vb{nb}]"
        stmts.append(f"{cur}[ov{k}]overlay={st.x_expr}:{st.y_expr}"
                     f":enable='between(t,{_f(st.t0)},{_f(st.t1)})'{out}")
        cur, nb, idx = out, nb + 1, idx + 1
    for k, an in enumerate(anims):
        stmts.append(f"[{idx}:v]scale={int(an.scale_w)}:-1,"
                     f"setpts=PTS+{_f(an.t0)}/TB[an{k}]")
        out = f"[vb{nb}]"
        stmts.append(f"{cur}[an{k}]overlay={an.x_expr}:{an.y_expr}"
                     f":enable='between(t,{_f(an.t0)},{_f(an.t1)})'"
                     f"{':shortest=1' if an.loop else ''}{out}")
        cur, nb, idx = out, nb + 1, idx + 1
    if vsubs:
        stmts.append(f"{cur}{','.join(vsubs)}[outv]")
    else:
        # No subtitles: rename the last overlay's output label to [outv].
        stmts[-1] = stmts[-1][:-len(cur)] + "[outv]"
    return ";".join(stmts)


def render(ff: FFmpeg, media: MediaInfo, cl: CutList, cfg: Config,
           out_path: str | Path, work_dir: str | Path,
           on_progress=None, log=print,
           scale_h: Optional[int] = None, fps: Optional[float] = None,
           ass_path: Optional[str] = None,
           crop_filter: Optional[str] = None,
           edge_fade: float = 0.0,
           enrich: Optional[RenderEnrich] = None) -> dict:
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

    Background music (C3): when ``cfg.render.music`` is enabled with a valid
    file, the bed enters as the LAST (looped) input and is auto-ducked by the
    FINAL speech, mixed in BEFORE loudnorm (see :func:`_music_audio_graph`).
    Disabled (the default) keeps every graph byte-for-byte unchanged. Clip
    Maker renders never receive it — serve strips ``music`` from clip opts.

    ``enrich`` (ENRICH_PLAN §2.1): a render-ready :class:`vpipe.enrich
    .RenderEnrich` — already remapped to FINAL coordinates, validated and
    conflict-resolved by ``vpipe.enrich.plan_render``; render.py stays a dumb
    executor (the ``ass_path`` pattern). Its PNG stills / WebM-alpha anims
    enter as extra ``-i`` inputs AFTER the music bed, the overlay nodes sit
    between the crop/scale/fps prefix and the subtitles filters, and its
    ``cards_ass`` burns FIRST (with ``fontsdir``) so the card scrim never dims
    the karaoke subs — ``ass_path`` (burn.ass) stays the LAST video filter.
    A non-empty ``enrich`` forces a re-encode through the existing non-empty-
    vpost mechanism; ``None`` (or an all-empty plan) keeps every graph and
    fast-path byte-for-byte unchanged. The audio graph is NEVER touched.
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

    # An all-empty enrich plan behaves exactly like None (byte-for-byte legacy
    # graphs); a non-empty one rides the non-empty-vpost re-encode mechanism.
    # V11: punch-zoom (§4a) / card blur-backplate (§3) also make the plan
    # non-empty even with no stills/anims/cards_ass (they are pure video-chain
    # effects, no extra inputs).
    enr = (enrich if enrich is not None
           and (enrich.stills or enrich.anims or enrich.cards_ass
                or enrich.punches or enrich.card_windows) else None)
    enr_stills: list[StillOverlay] = list(enr.stills) if enr else []
    enr_anims: list[AnimOverlay] = list(enr.anims) if enr else []
    enr_punches: list[ZoomWindow] = list(enr.punches) if enr else []
    enr_card_windows: list[tuple[float, float]] = (
        list(enr.card_windows) if enr else [])
    # Engine safety net (§2.1 п.5, duplicates the planner's score-trim): the
    # planner already trimmed by score and sorted by t0 — here we only cap the
    # input count so a hand-written plan cannot explode the filtergraph.
    if len(enr_stills) > MAX_STILLS:
        log(f"  ВНИМАНИЕ: enrich даёт {len(enr_stills)} PNG-оверлеев — "
            f"лимит движка {MAX_STILLS}, лишние отброшены.")
        enr_stills = enr_stills[:MAX_STILLS]
    if len(enr_anims) > MAX_ANIMS:
        log(f"  ВНИМАНИЕ: enrich даёт {len(enr_anims)} WebM-оверлеев — "
            f"лимит движка {MAX_ANIMS}, лишние отброшены.")
        enr_anims = enr_anims[:MAX_ANIMS]
    has_overlays = bool(enr_stills or enr_anims)
    # V11: chain is needed when there are overlays OR video-chain dynamics.
    has_dynamics = bool(enr_punches or enr_card_windows)
    use_chain = has_overlays or has_dynamics
    # Финальные дименшены кадра — нужны punch-zoom (scale=W:H обратно после crop).
    # Та же логика, что serve._final_render_dims: vertical crop -> baked scale в
    # crop_filter; иначе высота = scale_h|исходная, ширина пропорциональна.
    pz_w, pz_h = _punch_dims(media, scale_h, crop_filter)

    # vpost = vpre (crop -> scale -> fps) + vsubs (subtitles last). Without
    # enrich overlays it is one comma chain — byte-for-byte the legacy vpost.
    vpre = []
    # Vertical 9:16 crop+scale must run FIRST: crop -> scale -> fps -> subtitles.
    # crop_filter already bakes in the exact target scale (e.g. ...,scale=1080:1920),
    # so the caller passes scale_h=None and the generic scale step below is skipped.
    if crop_filter:
        vpre.append(crop_filter)
    if scale_h:
        vpre.append(f"scale=-2:{int(scale_h)}")
    if fps:
        vpre.append(f"fps={fps}")
    vsubs = []
    # Card/CTA-text ASS goes FIRST with the vendored Inter fontsdir (§2.2: the
    # card scrim must not dim the karaoke subs below).
    if enr is not None and enr.cards_ass:
        vsubs.append(f"subtitles='{_ass_path_for_filter(enr.cards_ass)}'"
                     f":fontsdir='{_ass_path_for_filter(enr.fonts_dir)}'")
    # Burn-in subtitles MUST be the last filter so karaoke timings line up with
    # the final (scaled/retimed) frames. A non-empty vpost forces re-encode, so
    # the no-cuts/no-scale copy fast-path is correctly bypassed when burning.
    if ass_path:
        vsubs.append(f"subtitles='{_ass_path_for_filter(ass_path)}'")
    vpost = vpre + vsubs
    vpost_s = ",".join(vpost)
    # The graph's audio source: the DFN-enhanced wav (already censored, when
    # there was anything to censor) wins over the censored FLAC, which wins
    # over the original track. Either external file enters as input #1.
    audio_src = enhanced or censored
    asrc_idx = "1" if audio_src else "0"
    inputs = ["-i", media.path] + (["-i", audio_src] if audio_src else [])
    # The FINAL audio duration (after cuts): anchors the F8 edge fade-out and
    # caps the looped music bed (kept already excludes sub-min_segment slivers).
    out_adur = sum(b - a for a, b in kept)
    # C3: background music bed — a looped LAST input, auto-ducked by the speech
    # (sidechaincompress) and mixed in BEFORE loudnorm. Disabled / missing file
    # / video-only source -> music_idx stays None and every graph below is
    # byte-for-byte the legacy one.
    music_path = music_source(cfg, log=log) if has_audio else None
    if not has_audio and getattr(cfg.render, "music", None) is not None \
            and cfg.render.music.enabled:
        log("  Фоновая музыка: у источника нет звуковой дорожки — пропускаю.")
    music_idx: Optional[int] = None
    if music_path:
        music_idx = 2 if audio_src else 1
        # -stream_loop -1 loops the bed for the whole program; the graph's
        # atrim + amix duration=first cap it at the final duration.
        inputs = inputs + ["-stream_loop", "-1", "-i", music_path]
        log(f"  фоновая музыка: {Path(music_path).name} "
            f"(громкость {cfg.render.music.gain_db:g} дБ, "
            f"приглушение при речи {cfg.render.music.duck_db:g} дБ)")
    # Enrich overlay inputs (§2.1) — ALWAYS after the music bed, same dynamic
    # index pattern as music_idx. PNG stills need `-loop 1 -t <end of window>`
    # (a single frame otherwise — the alpha fades have nothing to run on, R2
    # §1); WebM-alpha anims MUST be decoded with `-c:v libvpx-vp9` BEFORE the
    # `-i` (ffmpeg's native vp9 decoder silently drops the alpha — R2 trap #1)
    # and loop via `-stream_loop -1` (their overlay then carries shortest=1).
    # measure_loudness later receives these inputs too: it maps only [mout]
    # audio, so the extra video inputs are never decoded (R1 §1.6) — verified
    # live against ffmpeg 8.1.1, incl. a looped webm (the -f null pass ends).
    enrich_in_idx = (2 if audio_src else 1) + (1 if music_path else 0)
    if has_overlays:
        for st in enr_stills:
            inputs = inputs + ["-loop", "1", "-t", _f(st.t1 + 0.5),
                               "-i", st.path]
        for an in enr_anims:
            inputs = (inputs + (["-stream_loop", "-1"] if an.loop else [])
                      + ["-c:v", "libvpx-vp9", "-i", an.path])
    if enr is not None:
        log(f"  обогащение: {len(enr_stills)} PNG + {len(enr_anims)} WebM"
            + (" + карточки/CTA-текст (ASS)" if enr.cards_ass else ""))
    # Two-pass loudnorm (opt-in via denoise.loudnorm_mode="2pass"): measure the
    # FINAL audio (cuts + censor + denoise + deesser + the ducked music bed,
    # WITHOUT loudnorm) first, then hand the stats to build_apost for the
    # accurate linear pass. The measurement runs AFTER censoring (it consumes
    # the censored FLAC) and is audio-only, so it costs seconds. None (failure)
    # -> dynamic fallback.
    measured: Optional[dict] = None
    if has_audio and _wants_loudnorm_2pass(cfg):
        m_fade = max(0.0, float(getattr(cfg.render, "cut_fade", 0.0) or 0.0))
        measured = measure_loudness(ff, cfg, inputs, asrc_idx,
                                    kept if removed else None,
                                    cut_fade=m_fade, total=new_dur,
                                    music_idx=music_idx, music_dur=out_adur,
                                    log=log)
    # Audio post-chain (denoise). Empty list when disabled -> the audio path stays
    # byte-for-byte identical to before (copy / passthrough fast-paths preserved).
    apost = build_apost(cfg, loudnorm_measured=measured) if has_audio else []
    apost_s = ",".join(apost)
    # F8: de-click fades on the clip's TRUE edges, appended AFTER the whole
    # apost chain (incl. loudnorm — ordering rationale in _edge_fade_filters).
    # The fade-out anchors to the FINAL audio duration: with cuts that is the
    # exact concat length, otherwise the full source duration. [] when
    # edge_fade<=0 (every regular full-video render) -> all fast-paths/graphs
    # stay byte-for-byte identical.
    efades = _edge_fade_filters(edge_fade, out_adur) if has_audio else []

    if not removed and not vpost and not use_chain:
        # No cuts, no rescale: just mux (video copied — no quality loss). Denoise,
        # when on, forces the audio through filter_complex (video still copied).
        # (use_chain joins the non-empty-vpost gate: overlay/punch/blur-only
        # enrich has an empty vpost string yet still demands a video re-encode.)
        if not has_audio:
            log("  no cuts — remuxing (video copied).")
            args = ["-i", media.path, "-map", "0:v",
                    "-c:v", "copy", "-an", *_faststart(cfg), out_path]
        elif apost or efades or music_path:
            if music_path:
                # C3: speech treatment -> ducked-bed mix -> loudnorm — the bed
                # forces the audio through filter_complex even with apost off.
                log("  no cuts — remuxing (video copied); музыка + авто-дакинг"
                    " (sidechaincompress).")
                graph = _music_audio_graph(cfg, f"[{asrc_idx}:a]", music_idx,
                                           out_adur, apost, efades)
            else:
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
                "encoder": "copy", "denoise": bool(apost) or bool(enhanced),
                "music": bool(music_path)}

    if not removed:
        # No cuts but rescale/fps/enrich requested -> re-encode video; pass
        # audio through. With overlays the video part becomes the §2.1 chain
        # ([0:v] -> vpre -> overlay nodes -> subtitles), byte-for-byte the
        # legacy single statement otherwise.
        vgraph = (_enrich_video_chain("[0:v]", vpre, enr_stills, enr_anims,
                                      enrich_in_idx, vsubs,
                                      card_windows=enr_card_windows,
                                      punches=enr_punches,
                                      out_w=pz_w, out_h=pz_h)
                  if use_chain else f"[0:v]{vpost_s}[outv]")
        if has_audio and (apost or efades or music_path):
            # Audio must enter the graph for the denoise/edge-fade/music filters.
            if music_path:
                graph = (vgraph + ";"
                         + _music_audio_graph(cfg, f"[{asrc_idx}:a]", music_idx,
                                              out_adur, apost, efades))
            else:
                achain = ",".join(apost + efades)
                graph = (vgraph + ";"
                         f"[{asrc_idx}:a]{achain}[outa]")
            args = [*inputs, "-filter_complex", graph, "-map", "[outv]",
                    "-map", "[outa]", *venc, *_audio_args(cfg),
                    *_faststart(cfg), out_path]
        elif has_audio:
            args = [*inputs, "-filter_complex", vgraph, "-map", "[outv]",
                    "-map", f"{asrc_idx}:a", *venc, *_audio_args(cfg),
                    *_faststart(cfg), out_path]
        else:
            args = [*inputs, "-filter_complex", vgraph, "-map", "[outv]",
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
                "encoder": enc_name, "denoise": bool(apost) or bool(enhanced),
                "music": bool(music_path)}

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
        # When denoise/edge fades/music are on, the concat's audio leaves as
        # [outa_raw] and the post chain produces the final [outa] — so it runs
        # on the FULL retimed (already-censored) audio, never per-segment.
        # efades come strictly AFTER apost (i.e. after loudnorm) — see
        # _edge_fade_filters for why that order protects the de-click ramp.
        afinal = apost + efades
        a_lbl = "[outa_raw]" if (afinal or music_path) else "[outa]"
        if use_chain:
            # §2.1/§3/§4a: overlays + punch-zoom + blur-backplate live AFTER
            # concat, on the FINAL timeline ([vc]).
            graph = (base + f"concat=n={len(kept)}:v=1:a=1[vc]{a_lbl};"
                     + _enrich_video_chain("[vc]", vpre, enr_stills, enr_anims,
                                           enrich_in_idx, vsubs,
                                           card_windows=enr_card_windows,
                                           punches=enr_punches,
                                           out_w=pz_w, out_h=pz_h))
        elif vpost:
            graph = base + f"concat=n={len(kept)}:v=1:a=1[vc]{a_lbl};[vc]{vpost_s}[outv]"
        else:
            graph = base + f"concat=n={len(kept)}:v=1:a=1[outv]{a_lbl}"
        if music_path:
            # C3: the retimed speech is the duck KEY; loudnorm (inside apost)
            # masters the final mix — see _music_audio_graph's order contract.
            graph += ";" + _music_audio_graph(cfg, "[outa_raw]", music_idx,
                                              out_adur, apost, efades)
        elif afinal:
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
        if use_chain:
            graph = (base + f"concat=n={len(kept)}:v=1:a=0[vc];"
                     + _enrich_video_chain("[vc]", vpre, enr_stills, enr_anims,
                                           enrich_in_idx, vsubs,
                                           card_windows=enr_card_windows,
                                           punches=enr_punches,
                                           out_w=pz_w, out_h=pz_h))
        elif vpost:
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
            "denoise": bool(apost) or bool(enhanced),
            "music": bool(music_path)}
