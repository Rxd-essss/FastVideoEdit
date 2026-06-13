# -*- coding: utf-8 -*-
"""Chistovoy (final) CTA asset pack generator (ENRICH_PLAN §4 Tier 0 / V11 §5).

Renders the CTA overlay animations as RGBA frame sequences with Pillow and
encodes them to WebM VP9 yuva420p (the ONLY sane alpha format per R2 §3: GIF
kills gradient alpha and is banned, APNG is the documented fallback):

    vpipe/data/enrich/cta/subscribe_like.webm    dark pill «ПОДПИСАТЬСЯ» + thumb;
                                                 REAL entrance — slide-in from the
                                                 left + ease-out-back bounce
                                                 (~+7% overshoot) + alpha fade-in
                                                 (first 40%) + accent glow-pulse,
                                                 then a sustained sine pulse
    vpipe/data/enrich/cta/subscribe_slide_avatar.webm
                                                 same slide-in family WITH a round
                                                 channel-avatar slot (placeholder)
                                                 on the right
    vpipe/data/enrich/cta/comment.webm           speech bubble + dots, pop_in
                                                 0.2 -> 1.12 -> 1.0 (overshoot)
    vpipe/data/enrich/cta/like.webm              thumb-up on accent disc, pop_in +
                                                 sustained pulse (same family)
    vpipe/data/enrich/cta/bell.webm              bell glyph, pop_in + damped wiggle
    vpipe/data/enrich/cta/anim_presets.json      preset parameters (dur/easing/
                                                 overshoot/glow) consumed by P5/UI

STYLE (§4, гейт G4) — собственный тёмно-премиум-стиль в духе UI проекта
(web/style.css): глубокие подложки surface (#18202e / #1c2433), сине-фиолетовый
акцент (#7c95ff → #5b76f7), мягкие скруглённые формы, аккуратная типографика
Inter SemiBold, верхний световой блик и нижняя тень с градиентной альфой
(градиент-альфа = лакмус качества VP9). Без кринжа, без «кислотных» цветов.

HARD rules (R3/§4): artwork — НАШ собственный, нарисован с нуля. НИКАКОГО
play-логотипа YouTube и слова «YouTube». Никаких чужих паков/Lottie.

ENTRANCE / loop (V11 §5): анимация ВЪЕЗДА требует НЕ-зацикленного
проигрывания — оверлей должен играть один раз от t0 (loop=False), иначе
въезд повторяется каждый цикл. Ассеты длятся ~1.4 с (въезд + отскок + покой) и
заканчиваются «успокоившимся» кадром, так что даже при ошибочном зацикливании
картинка просто повторяет уже-собранную плашку, а не дёргается. Путь
`AnimOverlay.loop=False` в `render._enrich_video_chain` поддержан (см. R3-отчёт
и отчёт этого трека) — финитный анимэ НЕ получает `shortest=1` (строка
`{':shortest=1' if an.loop else ''}` в render.py) и корректно сдвигается
`setpts=PTS+t0/TB`. Привязку CTA к loop=False правит трек cards-dyn (render.py),
не этот генератор.

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
                                       [--phase-frames DIR]
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
SS = 3                              # supersampling factor for crisp edges
MAX_WEBM_BYTES = 200_000           # §7-P5 guard: every webm stays < 200 КБ

# Frame budgets (entrance assets ~1.4 s, the small icons a touch shorter).
N_PILL = 36                        # 36 / 25 fps = 1.44 s («въезд + осёл + покой»)
N_ICON = 30                        # 30 / 25 fps = 1.20 s (pop-in + pulse/wiggle)

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
AVATAR_BG = (44, 56, 80, 255)       # тёмный диск-слот под аватар канала

# Animation presets (the json mirrors these numbers — single source below).
SLIDE_FRAC = 0.42                   # доля кадров на въезд (остальное — покой/пульс)
SLIDE_FROM_FRAC = -0.46            # старт за кадром слева (доля ширины ассета)
EASE_BACK_S = 1.70158              # overshoot-коэффициент => проскок ~+7%
FADE_FRAC = 0.40                    # альфа 0->1 за первые 40% въезда
GLOW_PULSE_AMP = 1.0               # пик силы glow-кольца на покое (0..1)
GLOW_PULSE_CYCLES = 1.6            # синус-циклов glow за фазу покоя

PULSE_AMP = 0.06                    # scale 1.00 -> 1.06 -> 1.00 по синусу (покой)
POP_FROM, POP_OVER, POP_TO = 0.20, 1.12, 1.0   # comment/icon pop-in overshoot
POP_DUR_S = 0.48                    # pop-in живёт в первых ~12 кадрах
POP_BACK_S = 2.2                    # сильнее overshoot для «выскока» из угла
BELL_AMP_DEG = 12.0
BELL_CYCLES = 2.0                   # full wiggle periods per loop


# --- easing -------------------------------------------------------------------
def clamp01(x: float) -> float:
    return min(1.0, max(0.0, x))


def ease_out_back(t: float, s: float = EASE_BACK_S) -> float:
    """Overshoot ease — «лёгкий отскок»: проскакивает 1.0 (~+7% при s≈1.7) и
    возвращается. Settles exactly at 1.0 for t==1."""
    t -= 1.0
    return 1.0 + (s + 1.0) * t * t * t + s * t * t


def pulse_scale(i: int, n: int, *, rest_from: int = 0) -> float:
    """Sustained sine pulse used on the rest phase / looping icons: 1.0 at the
    seam, 1+amp mid-cycle. ``rest_from`` shifts the phase origin so the pulse
    starts at 1.0 right where the entrance settles."""
    span = max(1, n - rest_from)
    j = max(0, i - rest_from)
    return 1.0 + (PULSE_AMP / 2.0) * (1.0 - math.cos(2.0 * math.pi * j / span))


def pop_in(i: int, n_frames: int) -> tuple[float, float]:
    """(scale, alpha) for the pop_in preset: POP_FROM -> POP_OVER -> POP_TO, then
    hold at 1.0. Alpha ramps in over the first ~35% of the pop."""
    pop_frames = min(n_frames, int(round(POP_DUR_S * FPS)))   # ~12
    if i >= pop_frames:
        return POP_TO, 1.0
    p = i / max(1, pop_frames - 1)
    e = ease_out_back(p, s=POP_BACK_S)            # 0 -> ~1.3 -> 1.0 overshoot path
    s = POP_FROM + (POP_TO - POP_FROM) * e
    # blend the explicit overshoot peak in the middle so it reads as «1.12»
    s = max(s, POP_OVER if 0.45 < p < 0.75 else s)
    alpha = clamp01(p / 0.35)
    return s, alpha


def bell_angle(i: int, n: int) -> float:
    """Damped wiggle, 0 deg at both edges (the loop seam stays smooth)."""
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


def _avatar(d: ImageDraw.ImageDraw, cx: float, cy: float, r: float,
            color=WHITE) -> None:
    """Плейсхолдер аватара канала: «голова + плечи» на тёмном диске-слоте.

    Слот заведомо пуст под подстановку реального круглого аватара (UI/render
    может маскировать картинку каналом в этот круг)."""
    d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=AVATAR_BG)
    d.ellipse((cx - r, cy - r, cx + r, cy + r),
              outline=(255, 255, 255, 50), width=max(1, SS))
    hr = r * 0.34                                  # голова
    d.ellipse((cx - hr, cy - r * 0.55, cx + hr, cy - r * 0.55 + 2 * hr),
              fill=color)
    sw = r * 0.62                                  # плечи (внутри диска)
    d.pieslice((cx - sw, cy + r * 0.05, cx + sw, cy + r * 1.25), 180, 360,
               fill=color)


# --- artwork (re-rendered per pulse phase so the accent glow actually breathes) -
def draw_subscribe_pill(size: tuple[int, int], *, with_avatar: bool = False,
                        glow: float = 0.0) -> Image.Image:
    """Тёмная скруглённая капсула «ПОДПИСАТЬСЯ» + лайк (один объединённый CTA, R5).

    ``glow`` (0..1) раздувает/яркит акцентное glow-кольцо за диском — рисуется
    реально, а не общим scale. ``with_avatar`` оставляет справа круглый
    слот-плейсхолдер под аватар канала."""
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
    if glow > 0:                                   # пульсирующее glow-кольцо
        gr = r + int(r * 0.30 * glow)
        img.alpha_composite(_soft_glow(
            size, (cx - gr, cy - gr, cx + gr, cy + gr), gr, 14 * SS,
            color=(124, 149, 255, int(120 * glow))))
    img.alpha_composite(_vgrad_rounded(size, disc_box, r, ACCENT, ACCENT2))
    d = ImageDraw.Draw(img)
    d.ellipse(disc_box, outline=(255, 255, 255, 40), width=max(1, SS))
    _thumb_up(d, cx - r * 0.62, cy - r * 0.66, r * 0.0128, color=WHITE)
    # «ПОДПИСАТЬСЯ» — детерминированный fit-цикл в зону правее диска
    text = "ПОДПИСАТЬСЯ"
    right = int(w * 0.86) if with_avatar else int(w * 0.95)
    zone_x0, zone_x1 = cx + r + int(h * 0.10), right
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
    if with_avatar:
        ar = int(h * 0.26)
        _avatar(d, int(w * 0.93), int(h * 0.50), ar, color=WHITE)
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
    for fx in (0.32, 0.50, 0.68):
        cx, cy = int(w * fx), int(h * 0.40)
        d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=ACCENT)
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
    bell_img = Image.new("RGBA", size, (0, 0, 0, 0))
    _bell(ImageDraw.Draw(bell_img), w, h, color=WHITE)
    img.alpha_composite(bell_img)
    return img


# --- frame composers ----------------------------------------------------------
def _apply_alpha(canvas: Image.Image, alpha: float) -> None:
    if alpha < 1.0:
        ch = canvas.getchannel("A").point(lambda v, _a=alpha: round(v * _a))
        canvas.putalpha(ch)


def render_slide_pill(draw_fn, out_size: tuple[int, int], n_frames: int,
                      frames_dir: Path) -> None:
    """ВЪЕЗД: slide-in слева + ease-out-back проскок + alpha fade-in, затем
    sustained accent glow-pulse. Артворк перерисовывается за фазу покоя, чтобы
    glow-кольцо реально дышало (а не общий scale).

    Phases (n=36 @25 = 1.44 s):
      0..15  slide from x=SLIDE_FROM_FRAC*W, ease-out-back, alpha 0->1  («въезд+осёл»)
      15..36 hold + sine glow pulse                                     («покой»)
    """
    frames_dir.mkdir(parents=True, exist_ok=True)
    ow, oh = out_size
    big = (ow * SS, oh * SS)
    slide_n = max(2, int(round(n_frames * SLIDE_FRAC)))      # ~15
    start_dx = int(big[0] * SLIDE_FROM_FRAC)                 # offscreen-left start
    for i in range(n_frames):
        canvas = Image.new("RGBA", big, (0, 0, 0, 0))
        if i < slide_n:
            t = i / max(1, slide_n - 1)
            e = ease_out_back(t)                             # overshoot ~+7% then back
            dx = round(start_dx * (1.0 - e))
            alpha = clamp01(t / FADE_FRAC)
            glow = 0.0
        else:
            j = i - slide_n
            rest_n = max(1, n_frames - slide_n)
            dx, alpha = 0, 1.0
            glow = (GLOW_PULSE_AMP
                    * (0.5 - 0.5 * math.cos(2.0 * math.pi * GLOW_PULSE_CYCLES
                                            * j / rest_n)))
        art = draw_fn(big, glow=glow)
        canvas.alpha_composite(art, (dx, 0))
        _apply_alpha(canvas, alpha)
        canvas.resize(out_size, Image.LANCZOS).save(frames_dir / f"fr{i:02d}.png")


def render_popin(draw_fn, out_size: tuple[int, int], n_frames: int,
                 frames_dir: Path, *, anchor=(0.28, 0.82)) -> None:
    """Pop-in: POP_FROM(α0) -> POP_OVER(overshoot) -> POP_TO, scale around an
    anchor (хвостик облака для comment) so it «выскакивает» из угла; затем
    держится 1.0."""
    frames_dir.mkdir(parents=True, exist_ok=True)
    ow, oh = out_size
    big = (ow * SS, oh * SS)
    art = draw_fn(big)
    ax, ay = int(big[0] * anchor[0]), int(big[1] * anchor[1])
    for i in range(n_frames):
        s, alpha = pop_in(i, n_frames)
        canvas = Image.new("RGBA", big, (0, 0, 0, 0))
        sw, sh = max(1, round(big[0] * s)), max(1, round(big[1] * s))
        scaled = art.resize((sw, sh), Image.LANCZOS) if s != 1.0 else art
        ox = round(ax - ax * (sw / big[0]))
        oy = round(ay - ay * (sh / big[1]))
        canvas.alpha_composite(scaled, (ox, oy))
        _apply_alpha(canvas, alpha)
        canvas.resize(out_size, Image.LANCZOS).save(frames_dir / f"fr{i:02d}.png")


def render_pop_pulse(draw_fn, out_size: tuple[int, int], n_frames: int,
                     frames_dir: Path) -> None:
    """Centre-anchored pop-in (icon выскакивает) затем sustained sine pulse —
    «то же семейство» для like.webm."""
    frames_dir.mkdir(parents=True, exist_ok=True)
    ow, oh = out_size
    big = (ow * SS, oh * SS)
    art = draw_fn(big)
    pop_frames = min(n_frames, int(round(POP_DUR_S * FPS)))
    for i in range(n_frames):
        if i < pop_frames:
            s, alpha = pop_in(i, n_frames)
        else:
            s, alpha = pulse_scale(i, n_frames, rest_from=pop_frames), 1.0
        canvas = Image.new("RGBA", big, (0, 0, 0, 0))
        sw, sh = max(1, round(big[0] * s)), max(1, round(big[1] * s))
        scaled = art.resize((sw, sh), Image.LANCZOS) if s != 1.0 else art
        canvas.alpha_composite(scaled, ((big[0] - sw) // 2, (big[1] - sh) // 2))
        _apply_alpha(canvas, alpha)
        canvas.resize(out_size, Image.LANCZOS).save(frames_dir / f"fr{i:02d}.png")


def render_pop_wiggle(draw_fn, out_size: tuple[int, int], n_frames: int,
                      frames_dir: Path) -> None:
    """Centre pop-in затем damped-sine wiggle — «то же семейство» для bell.webm."""
    frames_dir.mkdir(parents=True, exist_ok=True)
    ow, oh = out_size
    big = (ow * SS, oh * SS)
    art = draw_fn(big)
    pop_frames = min(n_frames, int(round(POP_DUR_S * FPS)))
    for i in range(n_frames):
        if i < pop_frames:
            s, alpha = pop_in(i, n_frames)
            ang = 0.0
        else:
            s, alpha = 1.0, 1.0
            ang = bell_angle(i - pop_frames, n_frames - pop_frames)
        frame = art.rotate(ang, resample=Image.BICUBIC, expand=False) \
            if ang else art
        canvas = Image.new("RGBA", big, (0, 0, 0, 0))
        sw, sh = max(1, round(big[0] * s)), max(1, round(big[1] * s))
        scaled = frame.resize((sw, sh), Image.LANCZOS) if s != 1.0 else frame
        canvas.alpha_composite(scaled, ((big[0] - sw) // 2, (big[1] - sh) // 2))
        _apply_alpha(canvas, alpha)
        canvas.resize(out_size, Image.LANCZOS).save(frames_dir / f"fr{i:02d}.png")


def encode_webm(ffmpeg: str, frames_dir: Path, out: Path, n_frames: int) -> None:
    """PNG sequence -> WebM VP9 yuva420p (R2 §3 verdict), reproducible.

    `-pix_fmt yuva420p` keeps the alpha plane; libvpx-vp9 stores it as a hidden
    secondary stream (probe with `-c:v libvpx-vp9` to see `yuva420p`)."""
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
           "-framerate", str(FPS), "-i", str(frames_dir / "fr%02d.png"),
           "-frames:v", str(n_frames),
           "-c:v", "libvpx-vp9", "-pix_fmt", "yuva420p",
           "-crf", "28", "-b:v", "0", "-auto-alt-ref", "0",
           "-threads", "1", "-row-mt", "0",            # determinism
           "-map_metadata", "-1", "-bitexact",         # no timestamps/uids
           str(out)]
    subprocess.run(cmd, check=True)


def write_presets(out_dir: Path) -> Path:
    presets = {
        "version": 2,
        "fps": FPS,
        "frames": {"pill": N_PILL, "icon": N_ICON},
        "presets": {
            "cta_slide_in": {
                "dur_s": round(N_PILL / FPS, 3),
                "easing": "ease_out_back",
                "overshoot": round(ease_out_back(SLIDE_FRAC), 3),
                "slide_frac": SLIDE_FRAC,
                "from_frac": SLIDE_FROM_FRAC,
                "fade_frac": FADE_FRAC,
                "loop": False,                  # въезд играть один раз от t0
            },
            "ease_out_back": {
                "easing": "ease_out_back",
                "s": EASE_BACK_S,
                "peak": round(max(ease_out_back(t / 100.0)
                                  for t in range(101)), 3),
            },
            "glow": {
                "easing": "sine",
                "amplitude": GLOW_PULSE_AMP,
                "cycles": GLOW_PULSE_CYCLES,
                "loop": True,
            },
            "pop_in": {
                "dur_s": POP_DUR_S,
                "easing": "ease_out_back",
                "from": POP_FROM, "over": POP_OVER, "to": POP_TO,
                "s": POP_BACK_S,
                "loop": False,
            },
            "pulse": {
                "dur_s": round(N_ICON / FPS, 3),
                "easing": "sine",
                "amplitude": PULSE_AMP,
                "loop": True,
            },
            "bell_wiggle": {
                "dur_s": round(N_ICON / FPS, 3),
                "easing": "damped_sine",
                "amplitude_deg": BELL_AMP_DEG,
                "cycles": BELL_CYCLES,
                "loop": True,
            },
        },
        "assets": {
            "subscribe_like.webm": "cta_slide_in",
            "subscribe_slide_avatar.webm": "cta_slide_in",
            "comment.webm": "pop_in",
            "like.webm": "pop_in",
            "bell.webm": "pop_in",
        },
    }
    p = out_dir / "anim_presets.json"
    p.write_text(json.dumps(presets, ensure_ascii=False, indent=2) + "\n",
                 encoding="utf-8")
    return p


# name, out_size, n_frames, composer, draw_fn
ASSETS = (
    ("subscribe_like.webm", (640, 200), N_PILL, "slide",
     lambda big, glow=0.0: draw_subscribe_pill(big, with_avatar=False, glow=glow)),
    ("subscribe_slide_avatar.webm", (640, 200), N_PILL, "slide",
     lambda big, glow=0.0: draw_subscribe_pill(big, with_avatar=True, glow=glow)),
    ("comment.webm", (256, 256), N_ICON, "popin", draw_comment),
    ("like.webm", (256, 256), N_ICON, "pop_pulse", draw_like),
    ("bell.webm", (256, 256), N_ICON, "pop_wiggle", draw_bell),
)


def _compose(kind: str, draw_fn, size, n_frames, frames_dir) -> None:
    if kind == "slide":
        render_slide_pill(draw_fn, size, n_frames, frames_dir)
    elif kind == "popin":
        render_popin(draw_fn, size, n_frames, frames_dir)
    elif kind == "pop_pulse":
        render_pop_pulse(draw_fn, size, n_frames, frames_dir)
    elif kind == "pop_wiggle":
        render_pop_wiggle(draw_fn, size, n_frames, frames_dir)
    else:                                                    # pragma: no cover
        raise ValueError(f"unknown composer {kind!r}")


def export_phase_frames(dst: Path, ffmpeg: str) -> None:
    """Render the subscribe-pill phases (въезд/отскок/покой) as standalone PNGs
    for the live gate (V11 §5 step 4)."""
    dst.mkdir(parents=True, exist_ok=True)
    size = (640, 200)
    big = (size[0] * SS, size[1] * SS)
    slide_n = max(2, int(round(N_PILL * SLIDE_FRAC)))
    start_dx = int(big[0] * SLIDE_FROM_FRAC)

    def _frame(i: int, with_avatar: bool) -> Image.Image:
        canvas = Image.new("RGBA", big, (0, 0, 0, 0))
        if i < slide_n:
            t = i / max(1, slide_n - 1)
            e = ease_out_back(t)
            dx = round(start_dx * (1.0 - e))
            alpha = clamp01(t / FADE_FRAC)
            glow = 0.0
        else:
            j, rest_n = i - slide_n, max(1, N_PILL - slide_n)
            dx, alpha = 0, 1.0
            glow = (GLOW_PULSE_AMP
                    * (0.5 - 0.5 * math.cos(2.0 * math.pi * GLOW_PULSE_CYCLES
                                            * j / rest_n)))
        art = draw_subscribe_pill(big, with_avatar=with_avatar, glow=glow)
        canvas.alpha_composite(art, (dx, 0))
        _apply_alpha(canvas, alpha)
        return canvas.resize(size, Image.LANCZOS)

    # фазовые кадры (на тёмной подложке, чтобы альфа читалась глазом)
    bg = Image.new("RGBA", size, (12, 16, 24, 255))
    phases = {
        "phase1_enter": int(slide_n * 0.45),               # въезд (полупрозрачно)
        "phase2_overshoot": slide_n - 1,                   # пик проскока
        "phase3_rest_glow": N_PILL - 1,                    # покой + glow-пик
    }
    for tag, i in phases.items():
        comp = bg.copy()
        comp.alpha_composite(_frame(i, with_avatar=False))
        comp.convert("RGB").save(dst / f"subscribe_{tag}.png")
        comp2 = bg.copy()
        comp2.alpha_composite(_frame(i, with_avatar=True))
        comp2.convert("RGB").save(dst / f"subscribe_avatar_{tag}.png")
    # фильмстрип всего пути
    strip_idx = [0, slide_n // 3, slide_n - 1, (slide_n + N_PILL) // 2, N_PILL - 1]
    strip = Image.new("RGB", (size[0], size[1] * len(strip_idx)), (12, 16, 24))
    for row, i in enumerate(strip_idx):
        comp = bg.copy()
        comp.alpha_composite(_frame(i, with_avatar=True))
        strip.paste(comp.convert("RGB"), (0, row * size[1]))
    strip.save(dst / "subscribe_filmstrip.png")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ffmpeg", default="ffmpeg",
                    help="ffmpeg binary (default: from PATH)")
    ap.add_argument("--out", default=str(OUT_DIR_DEF),
                    help=f"output dir (default: {OUT_DIR_DEF})")
    ap.add_argument("--keep-frames", action="store_true",
                    help="keep the intermediate PNG frames next to --out")
    ap.add_argument("--phase-frames", default=None,
                    help="also export subscribe phase PNGs (въезд/отскок/покой) "
                         "to this dir for the live gate")
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
        for name, size, n_frames, kind, draw_fn in ASSETS:
            frames = tmp_root / Path(name).stem
            _compose(kind, draw_fn, size, n_frames, frames)
            out = out_dir / name
            encode_webm(ffmpeg, frames, out, n_frames)
            kb = out.stat().st_size / 1024.0
            over = out.stat().st_size >= MAX_WEBM_BYTES
            ok = ok and not over
            flag = "" if not over else "  !! БОЛЬШЕ 200 КБ — ужми (§7-P5 лимит)"
            print(f"  {name}: {kb:.1f} КБ{flag}")
    finally:
        if not args.keep_frames:
            shutil.rmtree(tmp_root, ignore_errors=True)
    print(f"  {write_presets(out_dir).name}: пресеты анимаций записаны")
    if args.phase_frames:
        export_phase_frames(Path(args.phase_frames), ffmpeg)
        print(f"  фазовые кадры -> {args.phase_frames}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
