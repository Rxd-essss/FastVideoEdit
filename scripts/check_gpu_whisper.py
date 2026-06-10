"""Sanity-check that faster-whisper loads and runs on CUDA (cuDNN bootstrap)."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vpipe.transcribe import bootstrap_cuda_dlls

bootstrap_cuda_dlls()
from faster_whisper import WhisperModel

t = time.time()
m = WhisperModel("small", device="cuda", compute_type="int8_float16")
print(f"loaded 'small' on cuda in {time.time() - t:.1f}s")
segs, info = m.transcribe("tests/_media/test.mp4", language="ru", vad_filter=True)
n = len(list(segs))
print(f"transcribe ran ok; segments={n}; detected_lang={info.language}")
print("CUDA OK")
