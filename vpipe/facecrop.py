"""Auto face-crop helper for the vertical (9:16) Shorts render.

Samples a handful of evenly-spaced frames from the source video, runs a face
detector on each, and returns the MEDIAN horizontal position of the detected
face(s) as a fraction in ``[0, 1]`` of the frame width. That value drives the X
offset of the 9:16 crop in :mod:`vpipe.render`, so the talking head stays in
frame instead of being centre-cropped blindly.

Detector cascade (plan 2.3), each step degrading gracefully to the next:

1. **YuNet** (``cv2.FaceDetectorYN`` + the vendored
   ``vpipe/data/face_detection_yunet_2023mar.onnx``) — robust to head turns,
   tilt and profile views where the Haar frontal cascade loses the face. The
   onnx is executed by OpenCV itself (no onnxruntime needed). Per frame the
   MOST CONFIDENT face is taken (YuNet returns a score per detection).
2. **Haar frontal cascade** — the original path, used verbatim whenever the
   YuNet model file is absent or ``FaceDetectorYN`` fails for any reason.
3. ``0.5`` (centre crop) — no faces found, or anything at all goes wrong.

``cv2`` (opencv-python-headless) is imported OPTIONALLY: if it is missing — or if
anything at all goes wrong, or no face is found — we fall back to ``0.5`` (centre
crop). The import is guarded exactly like the NVENC->x264 fallback in
``render.py``: try the preferred path, otherwise degrade gracefully without ever
raising. There is therefore no hard dependency on OpenCV for the feature to work.

Pure crop-rectangle math (independent of cv2) lives in :func:`crop_filter` so it
can be unit-tested without any media file or OpenCV install.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

try:                                   # optional — graceful fallback when absent
    import cv2 as _cv2
    _CV2_OK = True
except Exception:                      # noqa: BLE001 — ImportError or broken build
    _cv2 = None
    _CV2_OK = False


def cv2_available() -> bool:
    """True when OpenCV imported cleanly (so auto-detect can actually run)."""
    return _CV2_OK


# --- YuNet detector (plan 2.3) -----------------------------------------------
_YUNET_FILENAME = "face_detection_yunet_2023mar.onnx"
# Take the most confident face per frame; 0.6 keeps recall on turned/tilted
# heads (they score lower than frontal) while max-score selection keeps the
# occasional low-score background blob from winning.
_YUNET_SCORE_THRESHOLD = 0.6


def yunet_model_path() -> Path:
    """Vendored YuNet onnx location (``vpipe/data/``); existence NOT guaranteed."""
    return Path(__file__).resolve().parent / "data" / _YUNET_FILENAME


def _create_yunet(width: int, height: int):
    """``cv2.FaceDetectorYN`` instance, or ``None`` to signal the Haar fallback.

    ``None`` covers every degradation cause uniformly: cv2 missing, model file
    not vendored, OpenCV too old for the FaceDetectorYN API (< 4.5.4), or any
    error while loading the onnx. Never raises.
    """
    if not _CV2_OK:
        return None
    try:
        model = yunet_model_path()
        if not model.is_file():
            return None
        if getattr(_cv2, "FaceDetectorYN", None) is None:
            return None
        return _cv2.FaceDetectorYN.create(
            str(model), "", (int(width) or 320, int(height) or 320),
            score_threshold=_YUNET_SCORE_THRESHOLD,
            nms_threshold=0.3, top_k=50)
    except Exception:                  # noqa: BLE001 — degrade to Haar, never raise
        return None


def _yunet_center(detector, frame) -> Optional[float]:
    """X-center fraction of the MOST CONFIDENT YuNet face; ``None`` if no face.

    YuNet rows are ``[x, y, w, h, 10 landmark coords, score]``; we pick the row
    with the max score (spec: take the most confident face) rather than the
    per-frame median the Haar path uses. Any error counts as "no face in this
    frame" so a single bad frame cannot kill the whole sampling pass.
    """
    try:
        h_px, w_px = int(frame.shape[0]), int(frame.shape[1])
        if w_px <= 0 or h_px <= 0:
            return None
        detector.setInputSize((w_px, h_px))
        _rv, faces = detector.detect(frame)
        if faces is None or len(faces) == 0:
            return None
        best = max(faces, key=lambda f: float(f[14]) if len(f) > 14 else 0.0)
        cx = (float(best[0]) + float(best[2]) / 2.0) / float(w_px)
        return min(1.0, max(0.0, cx))
    except Exception:                  # noqa: BLE001 — treat as a no-face frame
        return None


def _haar_center(cascade, frame) -> Optional[float]:
    """Original Haar path: median face X-center fraction; ``None`` if no face."""
    gray = _cv2.cvtColor(frame, _cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4,
                                     minSize=(48, 48))
    if not len(faces):
        return None
    w_px = frame.shape[1] or 1
    xs = sorted((x + fw / 2.0) / w_px for (x, _y, fw, _fh) in faces)
    return xs[len(xs) // 2]            # median face in this frame


def detect_center(video_path: str | Path, ff=None, duration: float = 0.0,
                  *, samples: int = 12, start: float = 0.0,
                  end: Optional[float] = None, log=print) -> float:
    """Median face-center X fraction in ``[0, 1]``; ``0.5`` on any failure.

    Opens ``video_path`` with ``cv2.VideoCapture`` and seeks to ``samples`` evenly
    spaced timestamps. On each frame a face detector is run — YuNet when the
    vendored onnx model is available (most confident face per frame), otherwise
    the original frontal-face Haar cascade (median face per frame) — the X-center
    fraction is collected, and the median across all frames-with-a-face is
    returned. ``log`` reports which engine was used. ``ff`` is accepted for
    signature symmetry with the rest of the pipeline (reserved for a future
    ffmpeg-frame fallback) and is unused. Never raises.

    ``start``/``end`` (Clip Maker, план §2.3.3) restrict the sampled span to
    ``[start, end]`` — a Shorts clip crops by the face WITHIN the clip, where the
    speaker may sit elsewhere than the whole-video median. Samples are taken at
    ``t = start + (end - start) * (i + 0.5) / n``. The defaults (``start=0.0``,
    ``end=None`` -> ``duration``) reproduce the old whole-file sweep exactly; a
    nonsense range (inverted/out of bounds) degrades to the whole file.
    """
    if not _CV2_OK or duration is None or duration <= 0:
        return 0.5
    cap = None
    try:
        cap = _cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return 0.5
        # Engine cascade: YuNet (vendored onnx) -> Haar (original path) -> 0.5.
        frame_w = int(cap.get(_cv2.CAP_PROP_FRAME_WIDTH) or 0)
        frame_h = int(cap.get(_cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        yunet = _create_yunet(frame_w, frame_h)
        cascade = None
        if yunet is not None:
            log("  facecrop: YuNet face detector (vpipe/data/%s)" % _YUNET_FILENAME)
        else:
            cascade_path = _cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            cascade = _cv2.CascadeClassifier(cascade_path)
            if cascade.empty():        # cascade XML missing/corrupt -> center
                return 0.5
            log("  facecrop: Haar cascade face detector (YuNet model unavailable)")
        total_frames = cap.get(_cv2.CAP_PROP_FRAME_COUNT) or 0.0
        fps_src = cap.get(_cv2.CAP_PROP_FPS) or 0.0
        if fps_src <= 0:
            fps_src = 25.0
        n = max(1, int(samples))
        # Sampled span [t0, t1]: the clip range when given, the whole file by
        # default. Defensive clamping mirrors the rest of this function — any
        # garbage range falls back to the legacy whole-file sweep, never raises.
        try:
            t0 = max(0.0, float(start or 0.0))
        except (TypeError, ValueError):
            t0 = 0.0
        try:
            t1 = float(end) if end is not None else float(duration)
        except (TypeError, ValueError):
            t1 = float(duration)
        t1 = min(t1, float(duration))
        if t1 <= t0 or t0 >= duration:
            t0, t1 = 0.0, float(duration)
        centers: list[float] = []
        for i in range(n):
            t = t0 + (t1 - t0) * (i + 0.5) / n    # frame mid-point of each slice
            frame_idx = int(t * fps_src)
            if total_frames > 0:
                frame_idx = min(frame_idx, int(total_frames) - 1)
            frame_idx = max(0, frame_idx)
            cap.set(_cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            if yunet is not None:
                c = _yunet_center(yunet, frame)
            else:
                c = _haar_center(cascade, frame)
            if c is not None:
                centers.append(c)
        if not centers:
            return 0.5
        centers.sort()
        c = centers[len(centers) // 2]             # median across sampled frames
        return min(1.0, max(0.0, float(c)))
    except Exception:                              # noqa: BLE001 — never fail render
        return 0.5
    finally:
        if cap is not None:
            try:
                cap.release()
            except Exception:                      # noqa: BLE001
                pass


def parse_target(target: str | None, default: tuple[int, int] = (1080, 1920)
                 ) -> tuple[int, int]:
    """Parse a ``"WxH"`` target string into ``(w, h)``; fall back to ``default``."""
    if not target:
        return default
    try:
        w_s, h_s = str(target).lower().split("x", 1)
        w, h = int(w_s), int(h_s)
        if w > 0 and h > 0:
            return w, h
    except (ValueError, AttributeError):
        pass
    return default


def crop_filter(src_w: int, src_h: int, center_x: float,
                target: tuple[int, int] = (1080, 1920)) -> Optional[str]:
    """Build the ffmpeg ``crop,scale`` filter string for a 9:16 vertical clip.

    Crops a ``target`` aspect-ratio (e.g. 9:16) column out of the landscape
    source, horizontally positioned so ``center_x`` (face-center fraction in
    ``[0, 1]``) sits as near the middle of the crop as the frame allows, then
    scales to the exact ``target`` (e.g. ``1080x1920``).

    The crop width is ``ih * tw / th`` (height-driven so the full source height is
    kept). The X offset is clamped to ``[0, iw - crop_w]`` so the window never
    leaves the frame; ``center_x = 0.5`` yields a centre crop. Returns ``None``
    when the source is already at-or-narrower than the target aspect (portrait or
    square) — there the crop width would meet/exceed the source width, so we skip
    cropping and just scale to ``target`` (handled by the caller).

    The ``crop`` half uses ffmpeg ``iw``/``ih`` expressions so it stays correct
    per-segment even though the math here is evaluated in Python for the offset
    fraction only. Pure function — unit-tested without any media.
    """
    tw, th = int(target[0]), int(target[1])
    if tw <= 0 or th <= 0:
        return None
    target_ar = tw / th
    if src_w <= 0 or src_h <= 0:
        # Unknown source dims: trust target, do a center crop expression.
        cx = min(1.0, max(0.0, float(center_x)))
        return (f"crop=w='trunc(ih*{tw}/{th})':h=ih:"
                f"x='max(0,min(iw-trunc(ih*{tw}/{th}),(iw-trunc(ih*{tw}/{th}))*{cx:.4f}))':y=0,"
                f"scale={tw}:{th}")
    src_ar = src_w / src_h
    if src_ar <= target_ar:
        # Already vertical/square (or exactly target): nothing to crop.
        return None
    cx = min(1.0, max(0.0, float(center_x)))
    # crop=w='ih*tw/th':h=ih:x='clamp((iw-cropw)*cx, 0, iw-cropw)':y=0
    return (f"crop=w='trunc(ih*{tw}/{th})':h=ih:"
            f"x='max(0,min(iw-trunc(ih*{tw}/{th}),(iw-trunc(ih*{tw}/{th}))*{cx:.4f}))':y=0,"
            f"scale={tw}:{th}")


def vertical_filter(src_w: int, src_h: int, center_x: float,
                    target: tuple[int, int] = (1080, 1920)) -> str:
    """Always-usable ``crop,scale`` filter for a vertical render of ANY source.

    * Landscape (wider than target aspect): face-aware horizontal crop to the
      target column, then scale — delegates to :func:`crop_filter`.
    * Already-vertical/square source (where there is no horizontal room to crop):
      scale-to-cover the target then a centred crop to the exact size, so the
      output is exactly ``target`` with no distortion and no letterbox bars.

    Unlike :func:`crop_filter` this never returns ``None`` — the caller can always
    pass the result straight to ``render(crop_filter=...)`` and get an exact
    ``target`` frame. Pure function.
    """
    cf = crop_filter(src_w, src_h, center_x, target)
    if cf is not None:
        return cf
    tw, th = int(target[0] or 1080), int(target[1] or 1920)
    # Portrait/square: cover the frame (scale up the short side) then centre-crop.
    return (f"scale={tw}:{th}:force_original_aspect_ratio=increase,"
            f"crop={tw}:{th}")


def crop_rect(src_w: int, src_h: int, center_x: float,
              target: tuple[int, int] = (1080, 1920)) -> Optional[tuple[int, int, int, int]]:
    """Concrete integer crop rectangle ``(x, y, w, h)`` for the given source.

    Mirrors the clamped math of :func:`crop_filter` with numeric values (the
    ffmpeg expression evaluates the same way). Returns ``None`` when no crop is
    needed (source already at/under the target aspect). Used by unit tests to
    assert the clamping/boundary behaviour precisely.
    """
    tw, th = int(target[0]), int(target[1])
    if tw <= 0 or th <= 0 or src_w <= 0 or src_h <= 0:
        return None
    if src_w / src_h <= tw / th:
        return None
    crop_w = src_h * tw / th
    if crop_w >= src_w:                # degenerate: nothing wider to crop
        return None
    cx = min(1.0, max(0.0, float(center_x)))
    max_x = src_w - crop_w
    x = max(0.0, min(max_x, max_x * cx))
    return (int(round(x)), 0, int(round(crop_w)), int(src_h))
