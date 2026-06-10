"""Export edit decisions as an NLE timeline project (EDL / FCPXML).

The premium pitch: instead of (or in addition to) rendering an .mp4, hand the
cut decisions straight to Premiere Pro / DaVinci Resolve / Final Cut as a real
timeline so the editor can finish the work there — B-roll, colour, transitions.
No ffmpeg, no GPU, no re-encode: this is pure string/XML generation, so it is
instant.

Both formats describe the SAME timeline: one event/clip per *kept* segment, in
original-source order, laid end-to-end on the record/program timeline. The kept
segments come from ``Timeline.kept_segments()`` (original-source coordinates).

Generators here are deliberately free of pipeline I/O coupling — they take the
``kept`` segment list + a ``MediaInfo`` and write a single file — so they are
trivial to unit-test against synthetic inputs.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from math import gcd
from pathlib import Path
from urllib.parse import quote

from .probe import MediaInfo

# Fractional broadcast rates carry a 1001 timebase denominator (NTSC family).
# Anything else is treated as an integer rate (24/25/30/50/60…).
_KNOWN_NTSC: dict[float, tuple[int, int]] = {
    23.976: (24000, 1001),
    23.98: (24000, 1001),
    29.97: (30000, 1001),
    47.952: (48000, 1001),
    59.94: (60000, 1001),
    119.88: (120000, 1001),
}


def _fps_base(fps: float) -> tuple[int, int]:
    """Return the (numerator, denominator) timebase for ``fps``.

    NTSC-family fractional rates map to N*1000/1001 (e.g. 29.97 -> 30000/1001);
    everything else is treated as an integer rate -> (round(fps), 1). A
    non-positive/garbage fps falls back to 30/1 so we never divide by zero or
    emit a degenerate timeline.
    """
    try:
        f = float(fps)
    except (TypeError, ValueError):
        f = 0.0
    if f <= 0:
        return (30, 1)
    rounded = round(f, 3)
    if rounded in _KNOWN_NTSC:
        return _KNOWN_NTSC[rounded]
    return (max(1, round(f)), 1)


def _fps_int(fps: float) -> int:
    """Whole frames-per-second for timecode arithmetic (ceil of the rate).

    29.97 -> 30, 59.94 -> 60, 25 -> 25. EDL NON-DROP FRAME counts the nominal
    integer frame rate, so this is the rounded-up timebase numerator/denominator.
    """
    num, den = _fps_base(fps)
    # Ceil-division of num/den gives 30 for 30000/1001, 25 for 25/1, etc.
    return max(1, -(-num // den))


def _tc_from_frame(frame: int, fps_i: int) -> str:
    """Whole frame count -> CMX3600 non-drop timecode ``HH:MM:SS:FF``.

    The canonical integer-frame form: callers compute frame positions once and
    render both source and record timecodes through here so they stay consistent.
    Negatives clamp to zero; hours are not wrapped (clips are far under 24 h).
    """
    frame = max(0, int(frame))
    ff = frame % fps_i
    ss = (frame // fps_i) % 60
    mm = (frame // fps_i // 60) % 60
    hh = frame // fps_i // 3600
    return f"{hh:02d}:{mm:02d}:{ss:02d}:{ff:02d}"


def _secs_to_tc(secs: float, fps: float) -> str:
    """Float seconds -> CMX3600 timecode ``HH:MM:SS:FF`` (non-drop).

    Uses ``round`` (not truncation) on the total frame count so float noise like
    ``5.0 * 30 == 149.999…`` doesn't drop a frame. Hours wrap is not applied
    (talking-head clips are far under 24 h); negatives clamp to zero.
    """
    fps_i = _fps_int(fps)
    return _tc_from_frame(round(max(0.0, float(secs)) * fps_i), fps_i)


def _rational(secs: float, fps: float) -> str:
    """Float seconds -> FCPXML rational time string (``Ns`` or ``N/Ds``).

    FCPXML times are exact rationals quantised to the frame grid. For an integer
    rate the unit is 1/fps s; for an NTSC rate it is 1001/fps_num s. We snap the
    duration to the nearest whole frame, express it as ticks/denominator, then
    reduce the fraction (and collapse to ``Ns`` when it divides evenly) so the
    output is the canonical form Premiere/Resolve/FCP expect.
    """
    num, den = _fps_base(fps)
    frames = round(max(0.0, float(secs)) * num / den)
    # Duration in seconds = frames * (den / num). Express as ticks/denominator:
    ticks = frames * den
    denominator = num
    if ticks == 0:
        return "0s"
    g = gcd(ticks, denominator)
    ticks //= g
    denominator //= g
    if denominator == 1:
        return f"{ticks}s"
    return f"{ticks}/{denominator}s"


def _frames(secs: float, fps: float) -> int:
    """Float seconds -> nearest whole frame index on the exact ``fps`` grid.

    Integer rate: ``round(secs*fps)``. NTSC rate: ``round(secs*num/den)`` (e.g.
    ``num/den = 30000/1001`` for 29.97). One source of truth for every FCPXML
    frame position so offsets, starts, durations and the sequence length all
    reduce to integer-frame arithmetic (no float drift across the spine)."""
    num, den = _fps_base(fps)
    return round(max(0.0, float(secs)) * num / den)


def _rat_from_frames(frames: int, fps: float) -> str:
    """Whole frame count -> canonical FCPXML rational time string.

    ``frames`` frames last ``frames*(den/num)`` s -> ``frames*den / num`` s,
    reduced (and collapsed to ``Ns`` when it divides evenly). Pairing this with
    :func:`_frames` keeps the spine gapless: a clip's ``offset`` is the running
    sum of prior clip lengths *in frames*, never a re-rounded float cursor."""
    frames = max(0, int(frames))
    if frames == 0:
        return "0s"
    num, den = _fps_base(fps)
    ticks = frames * den
    denominator = num
    g = gcd(ticks, denominator)
    ticks //= g
    denominator //= g
    return f"{ticks}s" if denominator == 1 else f"{ticks}/{denominator}s"


def _file_url(path: str) -> str:
    """Canonical ``file://`` URL for an asset's media-rep src (FCPXML).

    Produces the three-slash form NLEs expect: ``file:///D:/dir/clip.mp4`` on
    Windows, ``file:///dir/clip.mp4`` on POSIX. We percent-encode the path body
    ourselves (keeping ``/`` and ``:`` intact) because ``pathname2url`` emits a
    non-portable ``////`` prefix on some Windows Python builds. Falls back to a
    best-effort encoding of the raw string if the path can't be resolved.
    """
    try:
        abs = Path(path).resolve()
        # Forward slashes; strip any leading slashes so we control the prefix.
        body = str(abs).replace("\\", "/").lstrip("/")
        # Keep path separators and the Windows drive colon unescaped.
        return "file:///" + quote(body, safe="/:")
    except Exception:  # noqa: BLE001
        return "file:///" + quote(str(path).replace("\\", "/").lstrip("/"), safe="/:")


def _fps_label(fps: float) -> str:
    """Human label for an FCPXML format name, e.g. ``2997`` or ``25``."""
    num, den = _fps_base(fps)
    if den == 1:
        return str(num)
    # 30000/1001 -> "2997", 23976 -> "2398" style two-decimal label.
    val = num / den
    return f"{round(val * 100):d}"


# --- EDL (CMX3600) -----------------------------------------------------------
def write_edl(
    path: str | Path,
    kept: list[tuple[float, float]],
    media: MediaInfo,
    title: str = "FastVideoEdit",
) -> None:
    """Write a CMX3600 EDL describing the kept segments.

    One ``C`` (cut) event per kept segment: source timecodes are the segment's
    in/out in the *original* media; record timecodes run continuously from
    00:00:00:00, each event butting against the previous one (the assembled
    program). A ``* FROM CLIP NAME`` comment names the source so the NLE can
    relink. ``FCM: NON-DROP FRAME`` is emitted for all rates (we count nominal
    integer frames; we never produce drop-frame ``;`` timecodes).
    """
    fps = media.fps
    fps_i = _fps_int(fps)
    clip_name = Path(media.path).name or "source"
    # EDL TITLE is ASCII-ish and historically <=70 chars; keep it on one line.
    safe_title = "".join(c for c in str(title) if c.isprintable() and c != "\n")[:70]

    lines = [f"TITLE: {safe_title}", "FCM: NON-DROP FRAME", ""]
    # Count in WHOLE FRAMES, not float seconds. Each event's source length (in
    # frames) and its record length MUST be identical, and the record must butt
    # exactly against the previous event — otherwise a strict importer (Resolve)
    # rejects the EDL and a lenient one silently shifts the assembly. Deriving
    # both source and record timecodes from one integer frame counter guarantees
    # source_frames == record_frames and a gapless record track on any boundary.
    rec_frame = 0
    evt = 0
    for src_in, src_out in kept:
        f_in = round(max(0.0, float(src_in)) * fps_i)
        f_out = round(max(0.0, float(src_out)) * fps_i)
        n = f_out - f_in
        if n <= 0:
            continue
        evt += 1
        lines.append(
            f"{evt:03d}  AX       V     C        "
            f"{_tc_from_frame(f_in, fps_i)} {_tc_from_frame(f_in + n, fps_i)} "
            f"{_tc_from_frame(rec_frame, fps_i)} {_tc_from_frame(rec_frame + n, fps_i)}")
        lines.append(f"* FROM CLIP NAME: {clip_name}")
        rec_frame += n

    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


# --- FCPXML 1.11 -------------------------------------------------------------
def write_fcpxml(
    path: str | Path,
    kept: list[tuple[float, float]],
    media: MediaInfo,
    title: str = "FastVideoEdit",
) -> None:
    """Write an FCPXML 1.11 project describing the kept segments.

    Structure: a single video ``format`` + one ``asset`` (the source) in
    ``resources``; a ``library/event/project/sequence/spine`` holding one
    ``asset-clip`` per kept segment. Each clip's ``offset`` is its position on
    the program timeline (segments butt end-to-end), ``start`` is the segment's
    in-point in the source, and ``duration`` is its length. All times are exact
    frame-quantised rationals via :func:`_rational`. Imports into Premiere Pro
    2023+, DaVinci Resolve 18+ and Final Cut Pro 10.6+.
    """
    fps = media.fps
    width = int(media.width or 1920)
    height = int(media.height or 1080)
    frame_dur = _rat_from_frames(1, fps)  # exactly one frame
    total_dur = _rat_from_frames(_frames(media.duration or 0.0, fps), fps)
    stem = Path(media.path).stem or "source"

    # Pre-compute every clip in INTEGER FRAMES so the spine is gapless: a clip's
    # offset is the running sum of prior clip lengths (not a re-rounded float
    # cursor), and the sequence duration is the exact sum of those lengths.
    # Without this, independently-rounded float offsets drift a frame apart and
    # a strict importer (Resolve) flags the timeline / leaves gaps on the spine.
    clip_specs: list[tuple[int, int, int]] = []   # (offset_frames, start_frames, len_frames)
    off_frames = 0
    for src_in, src_out in kept:
        f_in = _frames(src_in, fps)
        n = _frames(src_out, fps) - f_in
        if n <= 0:
            continue
        clip_specs.append((off_frames, f_in, n))
        off_frames += n
    prog_frames = off_frames   # assembled program length, in frames

    fcpxml = ET.Element("fcpxml", version="1.11")
    resources = ET.SubElement(fcpxml, "resources")
    fmt = ET.SubElement(resources, "format", {
        "id": "r1",
        "name": f"FFVideoFormat{height}p{_fps_label(fps)}",
        "frameDuration": frame_dur,
        "width": str(width),
        "height": str(height),
    })
    fmt.tail = ""
    asset = ET.SubElement(resources, "asset", {
        "id": "r2",
        "name": stem,
        "uid": stem,
        "start": "0s",
        "duration": total_dur,
        "hasVideo": "1",
        "hasAudio": "1" if getattr(media, "has_audio", False) else "0",
        "format": "r1",
        "videoSources": "1",
        "audioSources": "1" if getattr(media, "has_audio", False) else "0",
    })
    ET.SubElement(asset, "media-rep", {
        "kind": "original-media",
        "src": _file_url(media.path),
    })

    library = ET.SubElement(fcpxml, "library")
    event = ET.SubElement(library, "event", name=str(title))
    project = ET.SubElement(event, "project", name=str(title))
    # Sequence duration = sum of kept lengths in frames (the assembled program).
    sequence = ET.SubElement(project, "sequence", {
        "format": "r1",
        "duration": _rat_from_frames(prog_frames, fps),
        "tcStart": "0s",
        "tcFormat": "NDF",
    })
    spine = ET.SubElement(sequence, "spine")

    for off_f, start_f, len_f in clip_specs:
        ET.SubElement(spine, "asset-clip", {
            "ref": "r2",
            "name": stem,
            "lane": "0",
            "offset": _rat_from_frames(off_f, fps),
            "start": _rat_from_frames(start_f, fps),
            "duration": _rat_from_frames(len_f, fps),
            "format": "r1",
        })

    ET.indent(fcpxml, space="    ")
    xml_body = ET.tostring(fcpxml, encoding="unicode")
    out = ('<?xml version="1.0" encoding="UTF-8"?>\n'
           "<!DOCTYPE fcpxml>\n"
           f"{xml_body}\n")
    Path(path).write_text(out, encoding="utf-8")
