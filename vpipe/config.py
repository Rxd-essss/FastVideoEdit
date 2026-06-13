"""Typed configuration (pydantic) plus loaders for the editable word lists."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field


class _Base(BaseModel):
    model_config = ConfigDict(extra="ignore")


class PathsCfg(_Base):
    out_dir: str = "./out"
    cache_dir: str = "./cache"
    work_dir: str = "./work"


class FfmpegCfg(_Base):
    ffmpeg_bin: str = "ffmpeg"
    ffprobe_bin: str = "ffprobe"


class TranscribeCfg(_Base):
    model: str = "large-v3"
    compute_type: str = "int8_float16"
    device: str = "cuda"
    beam_size: int = 5
    language: str = "ru"
    vad_filter: bool = True
    vad_min_silence_ms: int = 500
    fallback_models: list[str] = Field(default_factory=lambda: ["medium", "small"])
    cache: bool = True


class PausesCfg(_Base):
    enabled: bool = True
    min_silence: float = 0.6
    pad_start: float = 0.15
    pad_end: float = 0.15
    min_keep: float = 0.05


class FillersCfg(_Base):
    enabled: bool = True
    pad: float = 0.04


class ProfanityCfg(_Base):
    enabled: bool = True
    action: str = "censor"


class PartialCfg(_Base):
    onset: float = 0.09
    mute_fraction: float = 0.6
    keep_tail: bool = True
    fade: float = 0.008


class PitchCfg(_Base):
    semitones: float = 6.0
    use_rubberband: str = "auto"   # auto | true | false


class LowpassCfg(_Base):
    cutoff: int = 500
    poles: int = 2


class ReverseCfg(_Base):
    fade: float = 0.005


class CensorCfg(_Base):
    method: str = "partial"
    partial: PartialCfg = Field(default_factory=PartialCfg)
    pitch: PitchCfg = Field(default_factory=PitchCfg)
    lowpass: LowpassCfg = Field(default_factory=LowpassCfg)
    reverse: ReverseCfg = Field(default_factory=ReverseCfg)


class BadTakesCfg(_Base):
    enabled: bool = True
    default_enabled: bool = False


class HesitationsCfg(_Base):
    """Acoustic «hesitation» detection (Silero VAD over the 16 kHz wav).

    Catches speech stumbles the TEXT detectors miss: stretched «э-э-э/м-м»,
    micro-cutoffs, mumbling and short *non-speech* dead-air BETWEEN words that
    is shorter than the pause detector's ``min_silence`` (so pauses never see
    it). Only gaps in ``[min_duration, max_duration)`` are flagged, then padded
    inward and deduped against the existing cut-list by overlap fraction.

    NOTE: keep ``max_duration`` <= ``pauses.min_silence`` so we don't restate a
    pause. The overlap dedup is a second safety net independent of that.
    """
    enabled: bool = True
    min_duration: float = 0.08    # min non-speech gap length to flag (s)
    max_duration: float = 0.55    # upper bound (above -> it's a pause, pdet owns it)
    pad_start: float = 0.04       # padding eaten from the gap's leading edge (s)
    pad_end: float = 0.04         # padding eaten from the gap's trailing edge (s)
    overlap_threshold: float = 0.5  # dedup: drop if this fraction overlaps an existing seg
    vad_threshold: float = 0.35   # Silero activation threshold (lower = more sensitive)
    vad_min_speech_ms: int = 100  # Silero min_speech_duration_ms
    vad_min_silence_ms: int = 30  # Silero min_silence_duration_ms (gap granularity)


class NvencCfg(_Base):
    preset: str = "p7"
    tune: str = "hq"
    rc: str = "constqp"
    qp: int = 19
    cq: int = 19


class X264Cfg(_Base):
    crf: int = 17
    preset: str = "slow"


class VerticalCfg(_Base):
    """Vertical (9:16) Shorts clip with auto face-crop.

    When ``enabled``, the render crops a ``target`` (default 1080x1920, i.e. 9:16)
    column out of the landscape source and scales it to that exact size. ``center``
    is either ``"auto"`` (detect the face X-center via :mod:`vpipe.facecrop`) or a
    float-as-string in ``[0, 1]`` giving the crop center fraction directly
    (``"0.5"`` = centred).
    """
    enabled: bool = False
    target: str = "1080x1920"          # WxH of the output frame
    center: str = "auto"               # "auto" | "0.0".."1.0"
    samples: int = 12                  # frames sampled for auto face-detection


class DenoiseCfg(_Base):
    """Audio denoise / speech-enhancement applied AFTER profanity censoring.

    ``enabled=False`` by default — audio is an *irreversible* render output and a
    mis-tuned ``afftdn`` "metallises" the voice (worse than the background hiss it
    removes), while too-high a ``highpass_hz`` thins low male voices. The user
    opts in consciously and verifies on their own material.

    The filter chain (when enabled) is, in order:
      highpass=f={highpass_hz}     low-frequency hum/rumble (AC, mains, room) —
                                   skipped when ``highpass_hz <= 0``.
      afftdn=nf={nf}               FFT noise-floor reduction; ``nf`` is in dB
                                   (negative; -25 = conservative, -30 = strong /
                                   risks artefacts on quiet speech).
      dynaudnorm=p=0.95:m=100      optional gentle loudness normalisation —
                                   only when ``normalize`` is True.

    Mastering add-ons (INDEPENDENT of ``enabled`` — each works on its own, so a
    user can master loudness without touching the denoiser):
      deess=True                   appends a soft ffmpeg ``deesser`` (i=0.4) to
                                   tame harsh sibilants ("s"/"sh" hiss). The
                                   filter's own default intensity is 0 (a
                                   no-op), hence the explicit gentle setting.
      loudnorm=True                appends ``loudnorm`` targeting YouTube
                                   speech loudness (I=-14 LUFS, TP=-1.5 dBTP,
                                   LRA=11). By default this is the DYNAMIC
                                   one-pass mode — fine for speech.
                                   loudnorm internally resamples to 192 kHz, so
                                   the chain always restores ``aresample=48000``
                                   right after it.
      loudnorm_mode="2pass"        upgrades the loudnorm above to the accurate
                                   two-pass LINEAR mode: a fast measurement pass
                                   runs the FINAL audio path (cuts + censor +
                                   denoise + deesser, no loudnorm) through
                                   ``loudnorm=...:print_format=json`` and feeds
                                   the measured I/TP/LRA/thresh/offset back as
                                   ``measured_*`` with ``linear=true``. The
                                   default ``"dynamic"`` keeps the legacy
                                   one-pass behaviour byte-for-byte. If the
                                   measurement pass fails (no JSON / ffmpeg
                                   error) the render logs it honestly and falls
                                   back to dynamic — it never fails the render.
                                   Only meaningful when ``loudnorm=True``.

    Engine choice (only meaningful when ``enabled=True``):
      engine="afftdn"              the legacy native-ffmpeg chain above —
                                   byte-for-byte default behaviour.
      engine="deepfilter"          neural denoise via the DeepFilterNet 3 CLI
                                   (``deep-filter.exe``, CPU-only, model baked
                                   in). The censored/original audio is exported
                                   to a 48 kHz MONO wav, enhanced externally and
                                   substituted as the render's audio input; the
                                   highpass/afftdn/dynaudnorm trio is then
                                   SKIPPED (deesser/loudnorm still apply). If
                                   the binary is missing or fails, the render
                                   logs it honestly and falls back to the
                                   afftdn chain — it never fails the render.
      deepfilter_bin               path to the CLI: absolute, relative to the
                                   repo root (``tools/deep-filter.exe`` ships
                                   there) or a bare name found on PATH.
      post_filter=True             adds ``--pf`` (slightly stronger attenuation
                                   of residual noise between words).
    """
    enabled: bool = False
    highpass_hz: int = 80              # cut hum/rumble below this (0 = skip)
    nf: float = -25.0                  # afftdn noise floor in dB (negative)
    normalize: bool = False            # add dynaudnorm after afftdn
    deess: bool = False                # soft de-esser (independent of enabled)
    loudnorm: bool = False             # one-pass -14 LUFS master (independent)
    loudnorm_mode: str = "dynamic"     # "dynamic" (one-pass, legacy) | "2pass"
                                       # (measure first, then linear loudnorm)
    engine: str = "afftdn"             # "afftdn" (ffmpeg, legacy) | "deepfilter"
                                       # (neural DFN3 via deep-filter.exe)
    deepfilter_bin: str = "tools/deep-filter.exe"  # abs / repo-root-rel / PATH
    post_filter: bool = True           # deep-filter --pf (post-filter)


class MusicCfg(_Base):
    """Background music bed with LOCAL auto-ducking (render.music).

    CapCut sells «Auto-Duck» as a Pro cloud feature (keyframes computed
    server-side); here the same effect is one ffmpeg ``sidechaincompress`` on
    this machine. The music file enters the render as a SECOND, looped input
    (``-stream_loop -1``), is trimmed to the final program duration, dropped to
    ``gain_db`` and compressed with the FINAL speech track (after cuts and
    censoring) as the sidechain KEY: while the host talks the bed dives, in
    pauses it breathes back up. The ducked bed and the speech are summed
    (``amix`` with ``normalize=0`` — no -6 dB penalty on the voice) BEFORE
    loudnorm, so the -14 LUFS master measures the actual published mix.

    ``duck_db`` is the ducking DEPTH — how far the bed is pushed down while
    speech is present. It maps to sidechaincompress's ``mix`` (dry/wet blend):
    ``mix = 1 - 10^(duck_db/20)``. With ratio 8 and the low threshold the wet
    path is squashed to near-silence during speech, so the output floor is
    ``dry·(1-mix)`` → an attenuation of ~``duck_db``. 0 = no ducking (plain
    bed), -30 ≈ the music all but vanishes under speech.

    Defaults follow the «подложка под голос» practice: bed at -18 dB under the
    speech, ducked a further ~12 dB while talking; threshold 0.02 (≈ -34 dBFS —
    even quiet speech triggers the duck), ratio 8 (firm), attack 20 ms (dives
    fast when a word starts), release 400 ms (returns smoothly, no pumping).

    Scope: ONLY the main full-video render. Clip Maker Shorts never get the
    bed — serve.py strips ``music`` from clip render_opts server-side.
    """
    enabled: bool = False
    path: Optional[str] = None    # audio file (or any video — ffmpeg takes its track)
    gain_db: float = -18.0        # bed level relative to speech, dB (<= 0)
    duck_db: float = -12.0        # extra attenuation while speech plays, dB (<= 0)
    threshold: float = 0.02       # sidechain level (linear) that starts the duck
    ratio: float = 8.0            # compression ratio above threshold
    attack: float = 20.0          # ms — how fast the bed dives
    release: float = 400.0        # ms — how fast it comes back in pauses


class EnrichRenderCfg(_Base):
    """Авто-обогащение при рендере (ENRICH_PLAN §5, render.enrich).

    Контракт как у music: без явного ``opts.enrich`` в /api/render обогащение
    ВЫКЛЮЧЕНО («нет ключа = выключено») — serve.py выставляет ``enabled`` из
    запроса на deep copy конфига. Клипы Clip Maker и автопак-клипы
    (cutlist_override-путь) обогащение не получают никогда (анти-скоуп §9).

    ``min_score`` — порог отсечки предложений ПОВЕРХ пер-предложенческого
    ``enabled``: Авто-пак шлёт 70 (консервативное применение без ревью-вкладки),
    обычный рендер живёт с 0 («берём всё, что включил юзер»).
    """
    enabled: bool = False
    min_score: int = 0                # 0..100; 0 = без отсечки по score


class ImageGenCfg(_Base):
    """Локальная SD-генерация контекстных картинок для авто-обогащения
    (PLAN_V11 §2, render.enrich.imagegen).

    Бэкенд — внешний бинарь stable-diffusion.cpp CUDA (паттерн
    ``denoise.deepfilter_bin``: ``sd-cli.exe`` + DLL рядом), модель SDXL-Turbo
    Q4_0 GGUF (~3.94 ГБ) скачивает ПОЛЬЗОВАТЕЛЬ — в репо не кладём. Полностью
    оффлайн (zero-upload), torch НЕ нужен.

    Дефолт ``imagegen_enabled=False`` — честный opt-in (тяжёлая модель). Если
    выключено / бинарь / модель не найдены → graceful degrade на эмодзи-фолбэк,
    задача НЕ падает (как ``engine="deepfilter"`` при отсутствии бинаря).

    ``imagegen_model`` — путь к .gguf, ОБЯЗАТЕЛЕН для работы (пусто = SD не
    настроен). ``imagegen_size`` — сторона кадра (768 — дефолт, VRAM-пик ~5.3 ГБ
    < 8 ГБ; 1024 только при выделенном GPU). ``imagegen_steps`` — шаги сэмплера
    (4 для Turbo). ``imagegen_vae_on_cpu`` — аварийный путь при дефиците VRAM
    (GPU-пик 2.7 ГБ, но ~10× медленнее) — НЕ дефолт."""
    imagegen_enabled: bool = False
    imagegen_bin: str = "tools/sd-cli.exe"   # abs / repo-root-rel / PATH
    imagegen_model: str = ""                 # путь к .gguf (обязателен для работы)
    imagegen_size: int = 768                 # сторона кадра, px
    imagegen_steps: int = 4                  # шаги сэмплера (Turbo = 4)
    imagegen_vae_on_cpu: bool = False        # аварийный VRAM-путь (медленно)


class RenderCfg(_Base):
    encoder: str = "nvenc"
    nvenc: NvencCfg = Field(default_factory=NvencCfg)
    x264: X264Cfg = Field(default_factory=X264Cfg)
    audio_bitrate: str = "320k"
    faststart: bool = True
    vertical: VerticalCfg = Field(default_factory=VerticalCfg)
    denoise: DenoiseCfg = Field(default_factory=DenoiseCfg)
    music: MusicCfg = Field(default_factory=MusicCfg)
    enrich: EnrichRenderCfg = Field(default_factory=EnrichRenderCfg)
    imagegen: ImageGenCfg = Field(default_factory=ImageGenCfg)
    # Smoothing at every cut seam. Without it the kept audio segments are
    # hard-concatenated and each join is a waveform discontinuity → an audible
    # click and an overall "choppy" feel. A short equal-length fade-out/fade-in
    # ramps both sides of every internal seam to zero so the join is click-free
    # ("soft", like the cloud editors). It is LENGTH-PRESERVING (no overlap), so
    # audio/video stay in sync. Applied only to INTERNAL seams (not the clip's
    # true start/end). 0 disables it (legacy hard cuts).
    cut_fade: float = 0.015           # seconds of fade on each side of a seam
                                      # (≈87% seam-click reduction, no audible dip)
    # Kept slivers shorter than this (seconds) are dropped so two near-adjacent
    # cuts merge instead of leaving a stuttery blip between them. Kept BELOW the
    # shortest real speech token (~60-80 ms) so it only removes breath/VAD-edge
    # remnants, never a word. 0 keeps the old behaviour (drop only sub-frame).
    min_segment: float = 0.04


class AssStyleCfg(_Base):
    """Style parameters for burn-in (ASS/libass) subtitles.

    Colours use the ASS &HAABBGGRR notation (alpha-blue-green-red, where AA=00
    is fully opaque). ``position`` maps to ASS Alignment (bottom=2, top=8,
    center=5). ``size``/``outline``/``shadow``/``margin_v`` are in the render's
    PlayRes pixels (we set PlayResX/Y to the output resolution).
    """
    enabled: bool = False
    font: str = "Arial"
    size: int = 52
    primary_color: str = "&H00FFFFFF"   # &HAABBGGRR — white, fully opaque
    outline_color: str = "&H00000000"   # black outline
    karaoke_color: str = "&H0000FFFF"   # yellow highlight for sung \k words
    outline: float = 2.0
    shadow: float = 1.0
    position: str = "bottom"           # "bottom" | "top" | "center"
    karaoke: bool = True
    margin_v: int = 40                 # vertical margin in PlayRes pixels


class SubsCfg(_Base):
    enabled: bool = True
    max_cps: float = 17.0
    max_line_chars: int = 42
    max_lines: int = 2
    min_dur: float = 1.0
    max_dur: float = 6.0
    min_gap: float = 0.05
    new_cue_gap: float = 0.7
    write_vtt: bool = True
    write_transcript: bool = True
    burn: AssStyleCfg = Field(default_factory=AssStyleCfg)


class MaskingCfg(_Base):
    keep_first: int = 1
    keep_last: int = 1
    mask_char: str = "*"
    min_stars: int = 2


class LlmCfg(_Base):
    enabled: bool = True
    backend: str = "ollama"
    model: str = "qwen3:8b"
    host: str = "http://localhost:11434"
    temperature: float = 0.0
    num_ctx: int = 16384
    timeout: int = 600
    # Long videos (26 min+) overflow a single prompt: split the transcript into
    # windows of this many segments and call the LLM per window (indices are
    # offset back to global). A small overlap keeps cross-boundary takes/chapters.
    max_segments_per_call: int = 80
    segment_overlap: int = 5
    # qwen3 emits <think> ... </think> that eats the budget on structured calls;
    # disable it for the JSON helpers.
    think: bool = False
    # Ollama keep_alive (seconds the model stays in VRAM after a call). 0 = unload
    # immediately so it doesn't share the 8 GB card with Whisper. -1/"5m" to keep.
    keep_alive: int = 0


class ChaptersCfg(_Base):
    enabled: bool = True
    min_chapters: int = 3
    min_length: float = 10.0
    max_chapters: int = 30


class MetadataCfg(_Base):
    enabled: bool = True
    max_title_chars: int = 100
    n_tags: int = 15
    max_hook_chars: int = 200


class ClipsCfg(_Base):
    enabled: bool = True
    max_per_window: int = 3        # прошито и в промпт («не больше 3»)
    min_duration: float = 20.0     # целевая нижняя граница, сек (eff)
    hard_min: float = 15.0         # ниже — дроп
    max_duration: float = 60.0     # трим кодом
    window_overlap: int = 12       # сегментов (≈60с при медиане ~5с/сегмент)
    keep_alive_between: int = 300  # сек, между окнами; 0 на последнем
    max_candidates: int = 15
    rerank: bool = True            # одновызовный финальный re-rank (F6/§3.5);
                                   # False → round-robin по окнам
    # F8: де-клик afade-in/out на ИСТИННЫХ краях клипа (его начале и конце),
    # сек на каждый край. Только для рендера клипов (/api/clips/render) —
    # обычный рендер полного ролика фейдов краёв не получает. 0 = выкл;
    # клампится к 0–0.2 в render().
    edge_fade: float = 0.025


class Config(_Base):
    language: str = "ru"
    paths: PathsCfg = Field(default_factory=PathsCfg)
    ffmpeg: FfmpegCfg = Field(default_factory=FfmpegCfg)
    transcribe: TranscribeCfg = Field(default_factory=TranscribeCfg)
    pauses: PausesCfg = Field(default_factory=PausesCfg)
    fillers: FillersCfg = Field(default_factory=FillersCfg)
    profanity: ProfanityCfg = Field(default_factory=ProfanityCfg)
    censor: CensorCfg = Field(default_factory=CensorCfg)
    bad_takes: BadTakesCfg = Field(default_factory=BadTakesCfg)
    hesitations: HesitationsCfg = Field(default_factory=HesitationsCfg)
    render: RenderCfg = Field(default_factory=RenderCfg)
    subtitles: SubsCfg = Field(default_factory=SubsCfg)
    masking: MaskingCfg = Field(default_factory=MaskingCfg)
    llm: LlmCfg = Field(default_factory=LlmCfg)
    chapters: ChaptersCfg = Field(default_factory=ChaptersCfg)
    metadata: MetadataCfg = Field(default_factory=MetadataCfg)
    clips: ClipsCfg = Field(default_factory=ClipsCfg)


def load_config(path: Optional[str | Path] = None) -> Config:
    data: dict = {}
    if path and Path(path).exists():
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return Config(**data)


# --- Editable word lists -----------------------------------------------------
@dataclass
class FillerLists:
    mumbles: list[str] = field(default_factory=list)
    words: list[str] = field(default_factory=list)
    phrases: list[list[str]] = field(default_factory=list)


@dataclass
class ProfanityLists:
    roots: list[str] = field(default_factory=list)
    allow: list[str] = field(default_factory=list)


def load_fillers(path: str | Path) -> FillerLists:
    p = Path(path)
    if not p.exists():
        return FillerLists()
    d = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return FillerLists(
        mumbles=list(d.get("mumbles", [])),
        words=list(d.get("words", [])),
        phrases=[list(x) for x in d.get("phrases", [])],
    )


def load_profanity(path: str | Path) -> ProfanityLists:
    p = Path(path)
    if not p.exists():
        return ProfanityLists()
    d = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return ProfanityLists(roots=list(d.get("roots", [])), allow=list(d.get("allow", [])))
