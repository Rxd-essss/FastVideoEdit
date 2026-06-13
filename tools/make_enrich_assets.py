# -*- coding: utf-8 -*-
"""Chistovoy (final) CTA asset pack generator (ENRICH_PLAN §4 Tier 0, §7-P5).

Renders the four CTA overlay animations as RGBA frame sequences with Pillow and
encodes them to WebM VP9 yuva420p (the ONLY sane alpha format per R2 §3: GIF
kills gradient alpha and is banned, APNG is the documented fallback):

    vpipe/data/enrich/cta/subscribe_like.webm   dark pill «ПОДПИСАТЬСЯ» + thumb,
                                                sine pulse 1.00–1.06
    vpipe/data/enrich/cta/comment.webm          speech bubble + dots, pop_in
                                                0.95 -> 1.02 -> 1.0
    vpipe/data/enrich/cta/like.webm             thumb-up on accent disc, pulse
    vpipe/data/enrich/cta/bell.webm             bell glyph, damped wiggle
    vpipe/data/enrich/cta/anim_presets.json     preset parameters (dur/easing/
                                                amplitude) consumed by P5/UI

STYLE (§4, гейт G4) — собственный тёмно-премиум-стиль в духе UI проекта
(web/style.css): глубокие подложки surface (#18202e / #1c2433), сине-фиолетовый
акцент (#7c95ff → #5b76f7), мягкие скруглённые формы, аккуратная типографика
Inter SemiBold, верхний световой блик и нижняя тень с градиентной альфой
(градиент-альфа = лакмус качества VP9). Без кринжа, без «кислотных» цветов.

HARD rules (R3/§4): artwork — НАШ собственный, нарисован с нуля. НИКАКОГО
play-логотипа YouTube и слова «YouTube». Никаких чужих паков/Lottie.

Determinism: no randomness, no timestamps. Frames are pure Pillow math; the
encode runs single-threaded (`-threads 1 -row-mt 0`) with `-bitexact` and
stripped metadata, so re-running the generator reproduces the .webm files
byte-for-byte on the same machine (across machines the PNG rasterization may
differ by a hair with another freetype — "близко" per plan).

VP9 alpha note (verified 8.1.1): libvpx-vp9 stores alpha as a hidden secondary
stream + Matroska `alpha_mode=1` tag. `ffprobe` with its NATIVE decoder reports
`yuv420p` for such a file; only `ffprobe -c:v libvpx-vp9 …` (the decoder the
render graph forces) exposes the true `yuva420p`. The alpha IS there — decode a
frame back to PNG and the transparency round-trips. Verify with libvpx, not the
native probe.

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
SS = 3                              # supersampling factor for crisp edges
MAX_WEBM_BYTES = 200_000           # §7-P5 guard: every webm stays < 200 КБ

# --- тёмно-премиум-палитра (зеркало web/style.css) ----------------------------
PANEL = (24, 32, 46, 255)           # #18202e — surface2 (подложка значка)
PANEL_HI = (34, 44, 62, 255)        # верхний блик подложки (мягкий градиент)
LINE = (148, 163, 199, 46)          # #94a3c7 @ .18 — тонкая граница (--line-strong)
ACCENT = (124, 149, 255, 255)       # #7c95ff — акцент проекта (--accent)
ACCENT2 = (91, 118, 247, 255)       # #5b76f7 — низ акцентного градиента (--accent2)
WHITE = (236, 239, 246, 255)        # #e8ebf2 — текст (--text), не чисто-белый
INK = (200, 209, 230, 255)          # светлый «холодный» штрих на тёмной подложке
SHADOW = (0, 0, 0, 130)             # мягкая тень с градиент-альфой (VP9-лакмус)
GLOW = (124, 149, 255, 70)          # лёгкое сияние акцента (gradient-alpha)

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


# --- low-level draw helpers (all coords already at SS resolution) ---------------
def _lerp(a: tuple, b: tuple, t: float) -> tuple:
    return tuple(round(a[k] + (b[k] - a[k]) * t) for k in range(len(a)))


def _vgrad_rounded(size: tuple[int, int], box: tuple[float, float, float, float],
                   radius: float, top: tuple, bottom: tuple) -> Image.Image:
    """Vertical-gradient rounded rectangle (мягкий объём подложки/диска).

    Рисуем градиент построчно и обрезаем по rounded-rect маске — даёт чистый
    переход top->bottom с градиентной альфой по краю (лакмус VP9)."""
    w, h = size
    x0, y0, x1, y1 = box
    grad = Image.new("RGBA", size, (0, 0, 0, 0))
    px = grad.load()
    span = max(1.0, y1 - y0)
    ix0, ix1 = int(math.floor(x0)), int(math.ceil(x1))
    for y in range(int(math.floor(y0)), int(math.ceil(y1))):
        t = min(1.0, max(0.0, (y - y0) / span))
        col = _lerp(top, bottom, t)
        for x in range(ix0, ix1):
            if 0 <= x < w and 0 <= y < h:
                px[x, y] = col
    mask = Image.new("L", size, 0)
    ImageDraw.Draw(mask).rounded_rectangle(box, radius=radius, fill=255)
    out = Image.new("RGBA", size, (0, 0, 0, 0))
    out.paste(grad, (0, 0), mask)
    return out


def _drop_shadow(size: tuple[int, int], box: tuple[float, float, float, float],
                 radius: float, dy: float, blur: float,
                 color=SHADOW) -> Image.Image:
    """Soft rounded-rect drop shadow — gradient alpha is the VP9 litmus."""
    img = Image.new("RGBA", size, (0, 0, 0, 0))
    x0, y0, x1, y1 = box
    ImageDraw.Draw(img).rounded_rectangle(
        (x0, y0 + dy, x1, y1 + dy), radius=radius, fill=color)
    return img.filter(ImageFilter.GaussianBlur(blur))


def _soft_glow(size: tuple[int, int], box: tuple[float, float, float, float],
               radius: float, blur: float, color=GLOW) -> Image.Image:
    """Лёгкое акцентное сияние под значком (тонкая градиент-альфа)."""
    img = Image.new("RGBA", size, (0, 0, 0, 0))
    ImageDraw.Draw(img).rounded_rectangle(box, radius=radius, fill=color)
    return img.filter(ImageFilter.GaussianBlur(blur))


def _top_highlight(d: ImageDraw.ImageDraw,
                   box: tuple[float, float, float, float], radius: float,
                   width: int) -> None:
    """Тонкий световой контур по верху подложки (стеклянный объём)."""
    x0, y0, x1, y1 = box
    d.rounded_rectangle((x0, y0, x1, y1), radius=radius,
                        outline=LINE, width=width)
    # верхняя дуга чуть ярче — имитация падающего света сверху
    d.arc((x0 + width, y0 + width, x1 - width, y0 + 2 * radius),
          start=200, end=340, fill=(255, 255, 255, 38), width=width)


def _thumb_up(d: ImageDraw.ImageDraw, x: float, y: float, s: float,
              color=WHITE) -> None:
    """Чистый палец-лайк (rounded) в боксе 100x100 при (x, y), масштаб ``s``."""
    def b(*xy):
        return [x + v * s if k % 2 == 0 else y + v * s
                for k, v in enumerate(xy)]
    # манжета (кисть) + ладонь
    d.rounded_rectangle(b(8, 50, 30, 92), radius=7 * s, fill=color)
    d.rounded_rectangle(b(30, 48, 90, 92), radius=14 * s, fill=color)
    # большой палец: стебель + подушечка
    d.polygon(b(36, 52, 42, 20, 60, 24, 58, 52), fill=color)
    d.ellipse(b(40, 12, 62, 34), fill=color)


def _bell(d: ImageDraw.ImageDraw, w: int, h: int, color=WHITE) -> None:
    """Аккуратный колокольчик по центру (rounded dome + язычок)."""
    cx = w * 0.5
    # купол
    d.pieslice((w * 0.26, h * 0.16, w * 0.74, h * 0.66),
               180, 360, fill=color)
    d.rectangle((w * 0.26, h * 0.40, w * 0.74, h * 0.62), fill=color)
    # навершие (knob)
    r = w * 0.05
    d.ellipse((cx - r, h * 0.10, cx + r, h * 0.10 + 2 * r), fill=color)
    # основание (юбка) + язычок
    d.rounded_rectangle((w * 0.18, h * 0.60, w * 0.82, h * 0.71),
                        radius=w * 0.05, fill=color)
    rr = w * 0.07
    d.ellipse((cx - rr, h * 0.71, cx + rr, h * 0.71 + 2 * rr), fill=color)


# --- artwork (drawn once at SS resolution, transformed per frame) ---------------
def draw_subscribe_like(size: tuple[int, int]) -> Image.Image:
    """Тёмная скруглённая капсула «ПОДПИСАТЬСЯ» + лайк (один объединённый CTA, R5)."""
    w, h = size
    rad = int(h * 0.30)
    box = (int(w * 0.025), int(h * 0.12), int(w * 0.975), int(h * 0.88))
    img = _drop_shadow(size, box, rad, dy=int(h * 0.05), blur=11 * SS)
    img.alpha_composite(_vgrad_rounded(size, box, rad, PANEL_HI, PANEL))
    d = ImageDraw.Draw(img)
    _top_highlight(d, box, rad, max(1, SS))
    # акцентный диск слева с лайком
    cx, cy = int(w * 0.155), int(h * 0.50)
    r = int(h * 0.31)
    disc_box = (cx - r, cy - r, cx + r, cy + r)
    img.alpha_composite(_vgrad_rounded(size, disc_box, r, ACCENT, ACCENT2))
    d = ImageDraw.Draw(img)
    d.ellipse(disc_box, outline=(255, 255, 255, 40), width=max(1, SS))
    _thumb_up(d, cx - r * 0.62, cy - r * 0.66, r * 0.0128, color=WHITE)
    # «ПОДПИСАТЬСЯ» — детерминированный fit-цикл в зону правее диска
    text = "ПОДПИСАТЬСЯ"
    zone_x0, zone_x1 = cx + r + int(h * 0.10), int(w * 0.95)
    fsize = int(h * 0.32)
    font = ImageFont.truetype(str(FONT_SEMIBOLD), fsize)
    while fsize > 8:
        font = ImageFont.truetype(str(FONT_SEMIBOLD), fsize)
        tx0, ty0, tx1, ty1 = d.textbbox((0, 0), text, font=font)
        if tx1 - tx0 <= zone_x1 - zone_x0:
            break
        fsize -= 2
    d.text(((zone_x0 + zone_x1) / 2 - (tx1 - tx0) / 2 - tx0,
            cy - (ty1 - ty0) / 2 - ty0), text, font=font, fill=WHITE)
    return img


def draw_like(size: tuple[int, int]) -> Image.Image:
    """Палец-лайк на акцентном диске (standalone like, ручные вставки)."""
    w, h = size
    box = (int(w * 0.15), int(h * 0.15), int(w * 0.85), int(h * 0.85))
    r = (box[2] - box[0]) / 2
    img = _soft_glow(size, (box[0] - 6 * SS, box[1] - 4 * SS,
                            box[2] + 6 * SS, box[3] + 10 * SS), r, 14 * SS)
    img.alpha_composite(_drop_shadow(size, box, r, dy=int(h * 0.04), blur=10 * SS))
    img.alpha_composite(_vgrad_rounded(size, box, r, ACCENT, ACCENT2))
    d = ImageDraw.Draw(img)
    d.ellipse(box, outline=(255, 255, 255, 46), width=max(1, SS))
    _thumb_up(d, w * 0.30, h * 0.27, w * 0.0044, color=WHITE)
    return img


def draw_comment(size: tuple[int, int]) -> Image.Image:
    """Тёмное скруглённое облачко с тремя акцентными точками (cta_comment)."""
    w, h = size
    rad = int(w * 0.18)
    box = (int(w * 0.12), int(h * 0.16), int(w * 0.88), int(h * 0.64))
    img = _drop_shadow(size, box, rad, dy=int(h * 0.045), blur=10 * SS)
    img.alpha_composite(_vgrad_rounded(size, box, rad, PANEL_HI, PANEL))
    # хвостик облачка (треугольник в тон нижней части подложки)
    tail = Image.new("RGBA", size, (0, 0, 0, 0))
    ImageDraw.Draw(tail).polygon(
        [(int(w * 0.30), int(h * 0.60)), (int(w * 0.48), int(h * 0.60)),
         (int(w * 0.28), int(h * 0.82))], fill=PANEL)
    img.alpha_composite(tail)
    d = ImageDraw.Draw(img)
    _top_highlight(d, box, rad, max(1, SS))
    # три акцентные точки
    r = int(w * 0.052)
    for fx, col in ((0.32, ACCENT), (0.50, ACCENT), (0.68, ACCENT)):
        cx, cy = int(w * fx), int(h * 0.40)
        d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=col)
    return img


def draw_bell(size: tuple[int, int]) -> Image.Image:
    """Колокольчик-уведомление на акцентном диске (лёгкое покачивание)."""
    w, h = size
    box = (int(w * 0.15), int(h * 0.15), int(w * 0.85), int(h * 0.85))
    r = (box[2] - box[0]) / 2
    img = _soft_glow(size, (box[0] - 6 * SS, box[1] - 4 * SS,
                            box[2] + 6 * SS, box[3] + 10 * SS), r, 14 * SS)
    img.alpha_composite(_drop_shadow(size, box, r, dy=int(h * 0.04),
                                     blur=10 * SS))
    img.alpha_composite(_vgrad_rounded(size, box, r, ACCENT, ACCENT2))
    d = ImageDraw.Draw(img)
    d.ellipse(box, outline=(255, 255, 255, 46), width=max(1, SS))
    # колокол — белым на акцентном диске (зеркало стиля like-значка)
    bell_img = Image.new("RGBA", size, (0, 0, 0, 0))
    _bell(ImageDraw.Draw(bell_img), w, h, color=WHITE)
    img.alpha_composite(bell_img)
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
    """PNG sequence -> WebM VP9 yuva420p (R2 §3 verdict), reproducible.

    `-pix_fmt yuva420p` keeps the alpha plane; libvpx-vp9 stores it as a hidden
    secondary stream (probe with `-c:v libvpx-vp9` to see `yuva420p`)."""
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
    ok = True
    try:
        for name, size, draw_fn, scale_fn, angle_fn, alpha_fn in ASSETS:
            art = draw_fn((size[0] * SS, size[1] * SS))
            frames = tmp_root / Path(name).stem
            render_frames(art, size, frames, scale_fn=scale_fn,
                          angle_fn=angle_fn, alpha_fn=alpha_fn)
            out = out_dir / name
            encode_webm(ffmpeg, frames, out)
            kb = out.stat().st_size / 1024.0
            over = out.stat().st_size >= MAX_WEBM_BYTES
            ok = ok and not over
            flag = "" if not over else "  !! БОЛЬШЕ 200 КБ — ужми (§7-P5 лимит)"
            print(f"  {name}: {kb:.1f} КБ{flag}")
    finally:
        if not args.keep_frames:
            shutil.rmtree(tmp_root, ignore_errors=True)
    print(f"  {write_presets(out_dir).name}: пресеты анимаций записаны")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
