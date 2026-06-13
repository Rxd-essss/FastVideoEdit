# -*- coding: utf-8 -*-
"""Draft CTA asset pack generator (ENRICH_PLAN §2.3) — P1 placeholder style.

Renders the four CTA overlay animations as RGBA frame sequences with Pillow
and encodes them to WebM VP9 yuva420p (the ONLY sane alpha format per R2 §3:
GIF kills gradient alpha and is banned, APNG is the documented fallback):

    vpipe/data/enrich/cta/subscribe_like.webm   rounded «ПОДПИСАТЬСЯ» badge +
                                                thumb-up, sine pulse 1.00–1.06
    vpipe/data/enrich/cta/comment.webm          speech bubble, pop_in
                                                0.95 -> 1.02 -> 1.0
    vpipe/data/enrich/cta/like.webm             thumb-up disc, sine pulse
    vpipe/data/enrich/cta/bell.webm             bell, damped wiggle (reserve)
    vpipe/data/enrich/cta/anim_presets.json     preset parameters (dur/easing/
                                                amplitude) consumed by P5/UI

HARD rules (R3/§2.3): the artwork is OUR OWN, drawn from scratch — no play
logo, no word «YouTube», no third-party packs. This is the DRAFT pack for the
G1 style gate; the final SVG-based set arrives in P5.

Determinism: no randomness, no timestamps. Frames are pure Pillow math; the
encode runs single-threaded (`-threads 1 -row-mt 0`) with `-bitexact` and
stripped metadata, so re-running the generator reproduces the .webm files
byte-for-byte on the same machine (across machines the PNG rasterization may
differ by a hair with another freetype — "близко" per plan).

Decode reminder for consumers (R2 traps, enforced in vpipe/render.py):
`-c:v libvpx-vp9` strictly BEFORE `-i` (the native decoder silently drops the
alpha) and `shortest=1` on the overlay whenever the input is looped with
`-stream_loop -1` (otherwise the render never ends).

Usage:
    python tools/make_enrich_assets.py [--ffmpeg PATH] [--out DIR] [--keep-frames]
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

REPO = Path(__file__).resolve().parents[1]
OUT_DIR_DEF = REPO / "vpipe" / "data" / "enrich" / "cta"
FONT_SEMIBOLD = REPO / "vpipe" / "data" / "enrich" / "fonts" / "Inter-SemiBold.ttf"

FPS = 25
N_FRAMES = 48                       # 1.92 s per loop (§2.3: 48 кадров / 25 fps)
SS = 2                              # supersampling factor for crisp edges
MAX_WEBM_BYTES = 200_000            # §7-P5 guard: every webm stays < 200 КБ

# Draft palette — простые черновые значки, чистовик в P5 (гейт G1/G4).
BADGE_BG = (224, 53, 75, 255)       # subscribe badge — red-ish, НЕ YouTube-лого
WHITE = (255, 255, 255, 255)
INK = (31, 41, 55, 255)             # dark slate (bubble dots / bell clapper)
ACCENT = (245, 158, 11, 255)        # #f59e0b — единый акцент проекта
BUBBLE_BG = (255, 255, 255, 242)
SHADOW = (0, 0, 0, 110)             # soft gradient-alpha shadow (VP9-лакмус)

# Animation presets (the json mirrors these numbers — single source below).
PULSE_AMP = 0.06                    # scale 1.00 -> 1.06 -> 1.00 по синусу
POP_FROM, POP_OVER, POP_TO = 0.95, 1.02, 1.0
POP_DUR_S = 0.48                    # pop-in lives in the first 12 frames
BELL_AMP_DEG = 12.0
BELL_CYCLES = 2.0                   # full wiggle periods per loop


# --- easing / per-frame transforms --------------------------------------------
def pulse_scale(i: int, n: int = N_FRAMES) -> float:
    """Seamless sine pulse: 1.0 at the loop seam, 1+amp mid-loop."""
    return 1.0 + (PULSE_AMP / 2.0) * (1.0 - math.cos(2.0 * math.pi * i / n))


def pop_in_scale(i: int) -> tuple[float, float]:
    """(scale, alpha) for the pop_in preset: 0.95 -> 1.02 -> 1.0, then hold."""
    pop_frames = int(round(POP_DUR_S * FPS))            # 12
    if i >= pop_frames:
        return POP_TO, 1.0
    p = i / max(1, pop_frames - 1)
    alpha = min(1.0, p / 0.45) if p < 0.45 else 1.0
    if p < 0.6:                                          # 0.95 -> 1.02 (ease-out)
        q = p / 0.6
        s = POP_FROM + (POP_OVER - POP_FROM) * (1.0 - (1.0 - q) ** 2)
    else:                                                # 1.02 -> 1.00 (ease-in-out)
        q = (p - 0.6) / 0.4
        s = POP_OVER + (POP_TO - POP_OVER) * (q * q * (3.0 - 2.0 * q))
    return s, alpha


def bell_angle(i: int, n: int = N_FRAMES) -> float:
    """Damped wiggle, 0 deg at both loop edges (the loop seam stays smooth)."""
    return BELL_AMP_DEG * math.sin(2.0 * math.pi * BELL_CYCLES * i / n) \
        * (1.0 - i / n)


# --- artwork (drawn once at SS resolution, transformed per frame) ---------------
def _shadow(size: tuple[int, int], box: tuple[int, int, int, int],
            radius: int) -> Image.Image:
    """Soft rounded-rect shadow — gradient alpha is the VP9-quality litmus."""
    img = Image.new("RGBA", size, (0, 0, 0, 0))
    ImageDraw.Draw(img).rounded_rectangle(box, radius=radius, fill=SHADOW)
    return img.filter(ImageFilter.GaussianBlur(10 * SS))


def _thumb_up(d: ImageDraw.ImageDraw, x: float, y: float, s: float,
              color=WHITE) -> None:
    """Simple draft thumb-up in a 100x100 box at (x, y) scaled by ``s``."""
    def b(*xy):
        return [x + v * s if k % 2 == 0 else y + v * s
                for k, v in enumerate(xy)]
    d.rounded_rectangle(b(10, 48, 28, 88), radius=6 * s, fill=color)
    d.rounded_rectangle(b(32, 46, 86, 88), radius=10 * s, fill=color)
    d.polygon(b(36, 50, 40, 22, 56, 26, 56, 50), fill=color)
    d.ellipse(b(40, 14, 58, 30), fill=color)


def draw_subscribe_like(size: tuple[int, int]) -> Image.Image:
    """Rounded badge «ПОДПИСАТЬСЯ» + thumb-up — one combined CTA (R5)."""
    w, h = size
    img = _shadow(size, (int(w * 0.035), int(h * 0.16),
                         int(w * 0.965), int(h * 0.86)), int(h * 0.18))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((int(w * 0.03), int(h * 0.12), int(w * 0.97),
                         int(h * 0.82)), radius=int(h * 0.18), fill=BADGE_BG)
    # thumb-up в круге слева
    cx, cy, r = int(w * 0.115), int(h * 0.47), int(h * 0.26)
    d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=WHITE)
    _thumb_up(d, cx - r * 0.66, cy - r * 0.70, r * 0.0132, color=BADGE_BG)
    # Текст вписывается в зону правее круга (детерминированный fit-цикл).
    text = "ПОДПИСАТЬСЯ"
    zone_x0, zone_x1 = cx + r + int(h * 0.10), int(w * 0.93)
    size = int(h * 0.30)
    while size > 8:
        font = ImageFont.truetype(str(FONT_SEMIBOLD), size)
        tx0, ty0, tx1, ty1 = d.textbbox((0, 0), text, font=font)
        if tx1 - tx0 <= zone_x1 - zone_x0:
            break
        size -= 2
    d.text(((zone_x0 + zone_x1) / 2 - (tx1 - tx0) / 2 - tx0,
            cy - (ty1 - ty0) / 2 - ty0), text, font=font, fill=WHITE)
    return img


def draw_like(size: tuple[int, int]) -> Image.Image:
    """Thumb-up on an accent disc (standalone like, ручные вставки)."""
    w, h = size
    img = _shadow(size, (int(w * 0.16), int(h * 0.20),
                         int(w * 0.84), int(h * 0.88)), int(w * 0.34))
    d = ImageDraw.Draw(img)
    d.ellipse((int(w * 0.12), int(h * 0.12), int(w * 0.88), int(h * 0.88)),
              fill=ACCENT)
    _thumb_up(d, w * 0.28, h * 0.26, w * 0.0046)
    return img


def draw_comment(size: tuple[int, int]) -> Image.Image:
    """Speech bubble with three dots (cta_comment icon)."""
    w, h = size
    img = _shadow(size, (int(w * 0.12), int(h * 0.22),
                         int(w * 0.88), int(h * 0.72)), int(w * 0.14))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((int(w * 0.10), int(h * 0.16), int(w * 0.90),
                         int(h * 0.66)), radius=int(w * 0.14), fill=BUBBLE_BG)
    d.polygon([(int(w * 0.26), int(h * 0.64)), (int(w * 0.44), int(h * 0.64)),
               (int(w * 0.24), int(h * 0.84))], fill=BUBBLE_BG)
    r = int(w * 0.045)
    for fx in (0.32, 0.50, 0.68):
        cx, cy = int(w * fx), int(h * 0.41)
        d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=INK)
    return img


def draw_bell(size: tuple[int, int]) -> Image.Image:
    """Notification bell (reserve asset)."""
    w, h = size
    img = _shadow(size, (int(w * 0.22), int(h * 0.24),
                         int(w * 0.78), int(h * 0.82)), int(w * 0.20))
    d = ImageDraw.Draw(img)
    d.ellipse((int(w * 0.45), int(h * 0.08), int(w * 0.55), int(h * 0.18)),
              fill=ACCENT)                                       # knob
    d.pieslice((int(w * 0.25), int(h * 0.13), int(w * 0.75), int(h * 0.73)),
               180, 360, fill=ACCENT)                            # dome
    d.rectangle((int(w * 0.25), int(h * 0.42), int(w * 0.75), int(h * 0.66)),
                fill=ACCENT)                                     # body
    d.rounded_rectangle((int(w * 0.17), int(h * 0.64), int(w * 0.83),
                         int(h * 0.74)), radius=int(w * 0.04), fill=ACCENT)
    d.ellipse((int(w * 0.44), int(h * 0.74), int(w * 0.56), int(h * 0.86)),
              fill=INK)                                          # clapper
    return img


# --- frame sequence -> webm -----------------------------------------------------
def render_frames(art: Image.Image, out_size: tuple[int, int], frames_dir: Path,
                  *, scale_fn=None, angle_fn=None, alpha_fn=None) -> None:
    """Write fr00..fr47 PNGs: rotate -> scale around the canvas centre."""
    frames_dir.mkdir(parents=True, exist_ok=True)
    big_w, big_h = art.size
    for i in range(N_FRAMES):
        frame = art
        if angle_fn is not None:
            frame = frame.rotate(angle_fn(i), resample=Image.BICUBIC,
                                 expand=False)
        s = scale_fn(i) if scale_fn is not None else 1.0
        canvas = Image.new("RGBA", (big_w, big_h), (0, 0, 0, 0))
        sw, sh = max(1, round(big_w * s)), max(1, round(big_h * s))
        scaled = frame.resize((sw, sh), Image.LANCZOS) if s != 1.0 else frame
        canvas.alpha_composite(scaled, ((big_w - sw) // 2, (big_h - sh) // 2))
        if alpha_fn is not None:
            a = alpha_fn(i)
            if a < 1.0:
                ch = canvas.getchannel("A").point(lambda v, _a=a: round(v * _a))
                canvas.putalpha(ch)
        canvas.resize(out_size, Image.LANCZOS).save(
            frames_dir / f"fr{i:02d}.png")


def encode_webm(ffmpeg: str, frames_dir: Path, out: Path) -> None:
    """PNG sequence -> WebM VP9 yuva420p (R2 §3 verdict), reproducible."""
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
           "-framerate", str(FPS), "-i", str(frames_dir / "fr%02d.png"),
           "-frames:v", str(N_FRAMES),
           "-c:v", "libvpx-vp9", "-pix_fmt", "yuva420p",
           "-crf", "28", "-b:v", "0", "-auto-alt-ref", "0",
           "-threads", "1", "-row-mt", "0",            # determinism
           "-map_metadata", "-1", "-bitexact",         # no timestamps/uids
           str(out)]
    subprocess.run(cmd, check=True)


def write_presets(out_dir: Path) -> Path:
    presets = {
        "version": 1,
        "fps": FPS,
        "frames": N_FRAMES,
        "presets": {
            "pulse": {"dur_s": round(N_FRAMES / FPS, 3), "easing": "sine",
                      "amplitude": PULSE_AMP, "loop": True},
            "pop_in": {"dur_s": POP_DUR_S, "easing": "overshoot",
                       "from": POP_FROM, "over": POP_OVER, "to": POP_TO,
                       "loop": False},
            "bell_wiggle": {"dur_s": round(N_FRAMES / FPS, 3),
                            "easing": "damped_sine",
                            "amplitude_deg": BELL_AMP_DEG,
                            "cycles": BELL_CYCLES, "loop": True},
        },
        "assets": {
            "subscribe_like.webm": "pulse",
            "comment.webm": "pop_in",
            "like.webm": "pulse",
            "bell.webm": "bell_wiggle",
        },
    }
    p = out_dir / "anim_presets.json"
    p.write_text(json.dumps(presets, ensure_ascii=False, indent=2) + "\n",
                 encoding="utf-8")
    return p


ASSETS = (
    # name, out_size, draw_fn, scale_fn, angle_fn, alpha_fn
    ("subscribe_like.webm", (640, 256), draw_subscribe_like,
     pulse_scale, None, None),
    ("comment.webm", (256, 256), draw_comment,
     lambda i: pop_in_scale(i)[0], None, lambda i: pop_in_scale(i)[1]),
    ("like.webm", (256, 256), draw_like, pulse_scale, None, None),
    ("bell.webm", (256, 256), draw_bell, None, bell_angle, None),
)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ffmpeg", default="ffmpeg",
                    help="ffmpeg binary (default: from PATH)")
    ap.add_argument("--out", default=str(OUT_DIR_DEF),
                    help=f"output dir (default: {OUT_DIR_DEF})")
    ap.add_argument("--keep-frames", action="store_true",
                    help="keep the intermediate PNG frames next to --out")
    args = ap.parse_args(argv)

    ffmpeg = shutil.which(args.ffmpeg) or args.ffmpeg
    if not Path(ffmpeg).is_file():
        print(f"ffmpeg не найден: {args.ffmpeg!r} — укажи --ffmpeg",
              file=sys.stderr)
        return 2
    if not FONT_SEMIBOLD.is_file():
        print(f"нет шрифта {FONT_SEMIBOLD} — сперва завендорь Inter (§2.4)",
              file=sys.stderr)
        return 2
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    tmp_root = (out_dir / "_frames" if args.keep_frames
                else Path(tempfile.mkdtemp(prefix="enrich_cta_")))
    try:
        for name, size, draw_fn, scale_fn, angle_fn, alpha_fn in ASSETS:
            art = draw_fn((size[0] * SS, size[1] * SS))
            frames = tmp_root / Path(name).stem
            render_frames(art, size, frames, scale_fn=scale_fn,
                          angle_fn=angle_fn, alpha_fn=alpha_fn)
            out = out_dir / name
            encode_webm(ffmpeg, frames, out)
            kb = out.stat().st_size / 1024.0
            flag = "" if out.stat().st_size < MAX_WEBM_BYTES else \
                "  !! БОЛЬШЕ 200 КБ — ужми (§7-P5 лимит)"
            print(f"  {name}: {kb:.1f} КБ{flag}")
    finally:
        if not args.keep_frames:
            shutil.rmtree(tmp_root, ignore_errors=True)
    print(f"  {write_presets(out_dir).name}: пресеты анимаций записаны")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
