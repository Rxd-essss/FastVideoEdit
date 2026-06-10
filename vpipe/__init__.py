"""FastVideoEdit — a fully local talking-head video pipeline for YouTube.

Stages: probe -> transcribe -> detect cuts -> review -> render -> subtitles ->
chapters -> summary. The CLI lives in ``pipeline.py``; the web review UI in
``serve.py`` reuses these same functions.
"""

__version__ = "0.1.0"
