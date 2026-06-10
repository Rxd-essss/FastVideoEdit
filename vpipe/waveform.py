"""Compute downsampled audio peaks for the web waveform (wavesurfer)."""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np


def compute_peaks(ffmpeg_bin: str, src: str | Path, duration: float | None,
                  px_per_sec: int = 10, sr: int = 4000,
                  max_buckets: int = 24000) -> list[float]:
    """Return normalized [0,1] peak magnitudes for the audio of ``src``.

    The number of buckets scales with ``duration`` (``px_per_sec`` buckets per
    second), so a long clip keeps real detail instead of being smeared into a
    fixed bucket count. The result is capped at ``max_buckets`` to bound memory.

    The audio is streamed from ffmpeg as mono ``sr``-Hz ``s16le`` PCM and reduced
    chunk-by-chunk into a running per-bucket absolute-max array. Peak memory is
    therefore ``O(chunk + buckets)`` rather than the size of the whole signal --
    the full PCM is never buffered or copied to float32. Returns ``[]`` when there
    is no audio or ffmpeg produces no output.
    """
    buckets = max(1, min(max_buckets, int(round((duration or 0) * px_per_sec))))

    # Estimate the total number of samples so a running global sample index can
    # be mapped onto a bucket without knowing the real length ahead of time.
    total_samples = int(round((duration or 0) * sr))
    # Denominator for index -> bucket mapping. When duration is unknown we still
    # bound it (1s worth) so the math stays well-defined; with buckets==1 in that
    # case every sample collapses into the single bucket regardless.
    denom = max(1, total_samples)

    cmd = [ffmpeg_bin, "-nostdin", "-v", "error", "-i", str(src), "-vn",
           "-ac", "1", "-ar", str(sr), "-f", "s16le", "-"]

    peaks = np.zeros(buckets, dtype=np.int32)
    seen_any = False
    global_index = 0  # running count of int16 samples consumed so far
    # 1 MiB of int16 samples per read; carry an odd trailing byte across reads.
    chunk_bytes = 1 << 20
    leftover = b""

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    try:
        assert proc.stdout is not None
        while True:
            chunk = proc.stdout.read(chunk_bytes)
            if not chunk:
                break
            if leftover:
                chunk = leftover + chunk
                leftover = b""
            # int16 is 2 bytes; carry a trailing odd byte to the next read.
            if len(chunk) & 1:
                leftover = chunk[-1:]
                chunk = chunk[:-1]
            if not chunk:
                continue

            samples = np.frombuffer(chunk, dtype=np.int16)
            n = samples.size
            if n == 0:
                continue
            seen_any = True

            # abs() in int32 to avoid overflow on -32768; never copy to float32.
            mags = np.abs(samples.astype(np.int32))

            # Map each global sample index onto its bucket and reduce with an
            # unbuffered per-bucket maximum (running abs-max).
            idx = (np.arange(global_index, global_index + n, dtype=np.int64)
                   * buckets) // denom
            np.clip(idx, 0, buckets - 1, out=idx)
            np.maximum.at(peaks, idx, mags)

            global_index += n
    finally:
        if proc.stdout is not None:
            proc.stdout.close()
        proc.wait()

    # No audio stream / empty output / ffmpeg error => nothing to draw.
    if not seen_any:
        return []

    out = peaks.astype(np.float32) / 32768.0
    np.clip(out, 0.0, 1.0, out=out)
    return out.tolist()
