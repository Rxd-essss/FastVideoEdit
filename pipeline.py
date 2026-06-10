#!/usr/bin/env python
"""FastVideoEdit — local talking-head video pipeline (CLI).

Stages 1–5 then STOP for review:
    python pipeline.py input.mp4 --out ./out
Edit out/<name>.cutlist.json, then render (stages 6–9):
    python pipeline.py input.mp4 --out ./out --apply

--apply with no existing cut list runs detection then renders in one shot.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

for _s in (sys.stdout, sys.stderr):   # Windows consoles default to cp1251
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

try:
    import yaml  # noqa: F401  (canary for the project venv)
    import pydantic  # noqa: F401
except ModuleNotFoundError:
    sys.stderr.write(
        "\n[FastVideoEdit] Зависимости не установлены в этом Python (нет venv проекта).\n"
        "Запусти через лаунчер  run.bat / .\\run.ps1, или активируй venv:\n"
        "  .\\.venv\\Scripts\\Activate.ps1   затем   python pipeline.py ...\n\n")
    raise SystemExit(1)

from vpipe.config import (load_config, load_fillers, load_profanity)
from vpipe.cutlist import resolve, save_txt
from vpipe.detect import run_detection
from vpipe.detect.profanity import ProfanityMatcher
from vpipe.ffmpeg_utils import FFmpeg, FFmpegError
from vpipe.llm import get_client
from vpipe.models import CutList
from vpipe.probe import extract_audio, hash_input, probe_media
from vpipe.timeline import Timeline
from vpipe.transcribe import transcribe_audio
from vpipe import chapters as chapters_mod
from vpipe import render as render_mod
from vpipe import subtitles as subs_mod
from vpipe import summary as summary_mod

try:
    from tqdm import tqdm
except Exception:  # tqdm optional
    tqdm = None


def _progress(desc: str):
    if tqdm is None:
        def cb(frac: float) -> None:
            pass
        return None, cb
    bar = tqdm(total=100, desc=f"  {desc}", ncols=72, leave=False,
               file=sys.stdout, bar_format="{l_bar}{bar}| {n:.0f}%")
    state = {"n": 0}

    def cb(frac: float) -> None:
        v = min(100, int(frac * 100))
        if v > state["n"]:
            bar.update(v - state["n"])
            state["n"] = v
    return bar, cb


def stage(n: int, title: str) -> None:
    print(f"\n[{n}/9] {title}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Local YouTube video pipeline")
    ap.add_argument("input", help="input video file")
    ap.add_argument("--out", default=None, help="output directory (default from config)")
    ap.add_argument("--config", default="config.yaml", help="config file")
    ap.add_argument("--apply", action="store_true",
                    help="render using the (edited) cut list; auto-detect if none exists")
    ap.add_argument("--redetect", action="store_true",
                    help="re-run detection and OVERWRITE the cut list "
                         "(otherwise an existing cut list is preserved)")
    ap.add_argument("--no-llm", action="store_true", help="disable the local LLM")
    ap.add_argument("--censor-method", default=None,
                    choices=["partial", "pitch", "lowpass", "reverse"])
    ap.add_argument("--device", default=None, choices=["cuda", "cpu"])
    ap.add_argument("--model", default=None, help="whisper model override")
    args = ap.parse_args()

    inp = Path(args.input)
    if not inp.exists():
        print(f"Input not found: {inp}", file=sys.stderr)
        return 2

    cfg = load_config(args.config)
    here = Path(args.config).resolve().parent
    fillers = load_fillers(here / "fillers_ru.yaml")
    profanity = load_profanity(here / "profanity_ru.yaml")

    if args.out:
        cfg.paths.out_dir = args.out
    if args.no_llm:
        cfg.llm.enabled = False
    if args.censor_method:
        cfg.censor.method = args.censor_method
    if args.device:
        cfg.transcribe.device = args.device
    if args.model:
        cfg.transcribe.model = args.model

    out_dir = Path(cfg.paths.out_dir)
    cache_dir = Path(cfg.paths.cache_dir)
    work_dir = Path(cfg.paths.work_dir) / inp.stem
    for d in (out_dir, cache_dir, work_dir):
        d.mkdir(parents=True, exist_ok=True)
    base = out_dir / inp.stem
    cutlist_path = out_dir / f"{inp.stem}.cutlist.json"

    try:
        ff = FFmpeg(cfg.ffmpeg)
    except FFmpegError as e:
        print(e, file=sys.stderr)
        return 3

    # --- Stage 1: probe ------------------------------------------------------
    stage(1, "Ingest & probe")
    media = probe_media(ff, inp)
    print(f"  {media.describe()}")
    if not media.has_audio:
        print("  ERROR: input has no audio track.", file=sys.stderr)
        return 4

    # --- Stage 2: transcription ---------------------------------------------
    stage(2, "Transcription")
    audio_hash = hash_input(inp)
    cache_file = cache_dir / f"{audio_hash}.transcript.json"
    if cfg.transcribe.cache and cache_file.exists():
        from vpipe.models import Transcript
        transcript = Transcript.load(cache_file)
        print(f"  cache hit: {cache_file.name}")
    else:
        bar, cb = _progress("extract audio")
        wav = extract_audio(ff, inp, work_dir / "audio16k.wav",
                            total=media.duration, on_progress=cb)
        if bar:
            bar.close()
        transcript = transcribe_audio(wav, cfg.transcribe, media.duration,
                                      audio_hash, cache_dir=cache_dir, log=print)
    print(f"  {sum(len(s.words) for s in transcript.segments)} words, "
          f"{len(transcript.segments)} segments")

    # --- LLM availability ----------------------------------------------------
    llm = get_client(cfg.llm)
    if llm is not None:
        if not llm.available():
            print(f"  LLM: Ollama not reachable at {cfg.llm.host} — bad-takes/chapters use fallback.")
            llm = None
        elif not llm.has_model():
            print(f"  LLM: model '{cfg.llm.model}' not pulled (ollama pull {cfg.llm.model}) — fallback.")
            llm = None
        else:
            print(f"  LLM: {cfg.llm.model} ready.")
    else:
        print("  LLM: disabled.")

    matcher = ProfanityMatcher(profanity)

    # --- Stages 3–5: detect + review ----------------------------------------
    # Detect only when there is no cut list yet, or --redetect was asked for.
    # A plain re-run never clobbers a hand-edited cut list.
    if args.redetect or not cutlist_path.exists():
        stage(3, "Detect cut candidates")
        cutlist = run_detection(transcript, cfg, fillers, profanity,
                                source=str(inp), llm=llm, log=print)
        cutlist.save_json(cutlist_path)
        save_txt(cutlist, out_dir / f"{inp.stem}.cutlist.txt")
        print(f"  cut list -> {cutlist_path.name}  (+ .txt)")
        if not args.apply:
            stage(5, "Review")
            print(f"  Review {cutlist_path}")
            print(f"  Toggle \"enabled\" / change \"action\", then re-run with --apply.")
            n_en = sum(1 for s in cutlist.segments if s.enabled)
            print(f"  {n_en}/{len(cutlist.segments)} segments currently enabled.")
            return 0
    else:
        cutlist = CutList.load_json(cutlist_path)
        n_en = sum(1 for s in cutlist.segments if s.enabled)
        if not args.apply:
            stage(5, "Review")
            print(f"  Existing cut list kept (not regenerated): {cutlist_path.name}")
            print(f"  {n_en}/{len(cutlist.segments)} segments enabled.")
            print(f"  Re-run with --apply to render, or --redetect to rebuild from scratch.")
            return 0
        print(f"  using edited cut list: {cutlist_path.name} ({n_en} enabled)")

    removed, _ = resolve(cutlist)
    tl = Timeline(removed, media.duration)

    # --- Stage 6: render -----------------------------------------------------
    stage(6, "Render")
    bar, cb = _progress("render")
    render_res = render_mod.render(ff, media, cutlist, cfg, base.with_suffix(".mp4"),
                                   work_dir, on_progress=cb, log=print)
    if bar:
        bar.close()
    print(f"  -> {render_res['out']}")

    # --- Stage 7: subtitles --------------------------------------------------
    stage(7, "Subtitles")
    if cfg.subtitles.enabled:
        subs_res = subs_mod.generate(transcript, removed, cfg.subtitles,
                                     cfg.masking, matcher, base, log=print)
    else:
        subs_res = {"cues": 0}
        print("  disabled.")

    # --- Stage 8: chapters ---------------------------------------------------
    stage(8, "Chapters")
    if cfg.chapters.enabled:
        chapters_res = chapters_mod.generate(transcript, removed, cfg.chapters,
                                             out_dir / "chapters.txt", llm=llm,
                                             matcher=matcher, mask=cfg.masking, log=print)
    else:
        chapters_res = {"chapters": 0}
        print("  disabled.")

    # --- Stage 9: summary ----------------------------------------------------
    stage(9, "Summary")
    summary_mod.summarize(cutlist, tl.new_duration(), render_res,
                          subs_res, chapters_res, log=print)
    return 0


if __name__ == "__main__":
    sys.exit(main())
