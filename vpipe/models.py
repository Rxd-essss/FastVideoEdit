"""Core data structures shared across the pipeline (transcript + cut list)."""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

# --- Cut types / actions -----------------------------------------------------
TYPE_PAUSE = "pause"
TYPE_FILLER = "filler"
TYPE_PROFANITY = "profanity"
TYPE_BADTAKE = "bad_take"
TYPE_HESITATION = "hesitation"   # acoustic speech-stumble (VAD-detected micro dead-air)
TYPE_MANUAL = "manual"      # user-drawn cut from the web editor

ACTION_REMOVE = "remove"
ACTION_CENSOR = "censor"


# --- Transcript --------------------------------------------------------------
@dataclass
class Word:
    word: str
    start: float
    end: float
    prob: float = 1.0

    def to_dict(self) -> dict:
        return {"w": self.word, "s": round(self.start, 3),
                "e": round(self.end, 3), "p": round(self.prob, 3)}

    @staticmethod
    def from_dict(d: dict) -> "Word":
        return Word(d["w"], float(d["s"]), float(d["e"]), float(d.get("p", 1.0)))


@dataclass
class Segment:
    start: float
    end: float
    text: str
    words: list[Word] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"start": round(self.start, 3), "end": round(self.end, 3),
                "text": self.text, "words": [w.to_dict() for w in self.words]}

    @staticmethod
    def from_dict(d: dict) -> "Segment":
        return Segment(float(d["start"]), float(d["end"]), d.get("text", ""),
                       [Word.from_dict(w) for w in d.get("words", [])])


@dataclass
class Transcript:
    language: str
    duration: float
    model: str                      # the model that ACTUALLY produced this (may be
                                    # an OOM/error fallback, e.g. 'medium')
    audio_hash: str
    segments: list[Segment] = field(default_factory=list)
    version: int = 1
    # The model the user ASKED for. Lets the cache tell an involuntary OOM
    # fallback (requested large-v3, got medium → keep it, don't re-OOM next run)
    # apart from a deliberate model switch (requested changed → re-transcribe).
    requested_model: str = ""
    # A6: the device that ACTUALLY ran the transcription ("cuda" | "cpu") so the
    # UI can warn about a silent CPU fallback. None on caches written before
    # this field existed (old json has no key → from_dict yields None).
    device_used: Optional[str] = None

    def all_words(self) -> list[Word]:
        out: list[Word] = []
        for s in self.segments:
            out.extend(s.words)
        out.sort(key=lambda w: w.start)
        return out

    def to_dict(self) -> dict:
        return {"version": self.version, "language": self.language,
                "duration": self.duration, "model": self.model,
                "requested_model": self.requested_model,
                "device_used": self.device_used,
                "audio_hash": self.audio_hash,
                "segments": [s.to_dict() for s in self.segments]}

    @staticmethod
    def from_dict(d: dict) -> "Transcript":
        return Transcript(
            language=d.get("language", "ru"), duration=float(d.get("duration", 0.0)),
            model=d.get("model", ""), requested_model=d.get("requested_model", ""),
            device_used=d.get("device_used"),   # старые кэши без поля → None
            audio_hash=d.get("audio_hash", ""),
            segments=[Segment.from_dict(s) for s in d.get("segments", [])],
            version=int(d.get("version", 1)))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=1),
                              encoding="utf-8")

    @staticmethod
    def load(path: str | Path) -> "Transcript":
        return Transcript.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


# --- Cut list ----------------------------------------------------------------
@dataclass
class CutSegment:
    id: str
    start: float
    end: float
    type: str            # pause | filler | profanity | bad_take | hesitation | manual
    action: str          # remove | censor
    enabled: bool = True
    text: str = ""       # spoken text in this span (for review/UI)
    reason: str = ""     # why it was flagged (esp. bad_take)
    word: str = ""       # original profane word (for text masking)

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def to_dict(self) -> dict:
        return {"id": self.id, "start": round(self.start, 3), "end": round(self.end, 3),
                "type": self.type, "action": self.action, "enabled": self.enabled,
                "text": self.text, "reason": self.reason, "word": self.word}

    @staticmethod
    def from_dict(d: dict) -> "CutSegment":
        return CutSegment(
            id=d["id"], start=float(d["start"]), end=float(d["end"]),
            type=d["type"], action=d["action"], enabled=bool(d.get("enabled", True)),
            text=d.get("text", ""), reason=d.get("reason", ""), word=d.get("word", ""))


@dataclass
class CutList:
    source: str
    duration: float
    segments: list[CutSegment] = field(default_factory=list)
    version: int = 1

    def enabled_removes(self) -> list[tuple[float, float]]:
        return [(s.start, s.end) for s in self.segments
                if s.enabled and s.action == ACTION_REMOVE]

    def enabled_censors(self) -> list[CutSegment]:
        return [s for s in self.segments
                if s.enabled and s.action == ACTION_CENSOR]

    def to_dict(self) -> dict:
        return {"version": self.version, "source": self.source,
                "duration": self.duration,
                "segments": [s.to_dict() for s in self.segments]}

    @staticmethod
    def from_dict(d: dict) -> "CutList":
        return CutList(
            source=d.get("source", ""), duration=float(d.get("duration", 0.0)),
            segments=[CutSegment.from_dict(s) for s in d.get("segments", [])],
            version=int(d.get("version", 1)))

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
                              encoding="utf-8")

    @staticmethod
    def load_json(path: str | Path) -> "CutList":
        return CutList.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
