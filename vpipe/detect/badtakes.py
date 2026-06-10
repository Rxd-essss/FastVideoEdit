"""Bad-take suggestions via the local LLM (false starts, repeats, rambling).

The model returns SEGMENT INDICES (reliable) which we map back to exact
timestamps. Suggestions are added disabled-by-default — they are proposals the
user confirms in the review step, never auto-cuts.
"""
from __future__ import annotations

from ..config import Config
from ..llm import LLMUnavailable, OllamaClient, segment_windows
from ..models import ACTION_REMOVE, TYPE_BADTAKE, CutSegment, Transcript

_SYSTEM = (
    "Ты — ассистент видеомонтажёра. Тебе дают расшифровку речи (talking-head) "
    "по сегментам с номерами. Найди сегменты, которые стоит ВЫРЕЗАТЬ: фальстарты "
    "(оборванное начало фразы, которое затем повторяется), дубли/повторы одной и "
    "той же мысли, оговорки и явную «воду». НЕ удаляй содержательную речь. "
    "Верни строго JSON со списком номеров сегментов к удалению и краткой причиной "
    "на русском. Если удалять нечего — верни пустой список."
)

_SCHEMA = {
    "type": "object",
    "properties": {
        "removals": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "reason": {"type": "string"},
                },
                "required": ["index", "reason"],
            },
        }
    },
    "required": ["removals"],
}


def detect(transcript: Transcript, cfg: Config,
           llm: OllamaClient, log=print) -> list[CutSegment]:
    segments = transcript.segments
    if not segments:
        return []

    n = len(segments)
    # Long videos overflow a single prompt — the END of a 26 min+ video never
    # gets analysed. Walk fixed windows and offset per-window indices back to
    # GLOBAL segment indices, deduping the overlap.
    chosen: dict[int, str] = {}
    for win_start, win_end in segment_windows(n, cfg.llm):
        window = segments[win_start:win_end]
        lines = []
        for local_i, s in enumerate(window):
            lines.append(f"{local_i} | {s.start:.1f}-{s.end:.1f} | {s.text}")
        user = ("Сегменты расшифровки:\n" + "\n".join(lines) +
                "\n\nВерни JSON: {\"removals\": [{\"index\": <номер>, \"reason\": <причина>}]}")
        try:
            data = llm.chat_json(_SYSTEM, user, _SCHEMA)
        except (LLMUnavailable, Exception) as e:  # noqa: BLE001
            # Graceful: a single bad window must not lose the whole pass. Skip
            # this window and keep whatever the other windows produced.
            log(f"  bad takes: window [{win_start}:{win_end}] failed ({e}); skipped.")
            continue
        for r in data.get("removals", []):
            local_idx = r.get("index")
            if not isinstance(local_idx, bool) and isinstance(local_idx, int) \
                    and 0 <= local_idx < len(window):
                gidx = win_start + local_idx
                if gidx not in chosen:
                    chosen[gidx] = str(r.get("reason", "")).strip()

    out: list[CutSegment] = []
    for gidx in sorted(chosen):
        s = segments[gidx]
        out.append(CutSegment(
            id="", start=round(s.start, 3), end=round(s.end, 3),
            type=TYPE_BADTAKE, action=ACTION_REMOVE,
            enabled=cfg.bad_takes.default_enabled,
            text=s.text.strip(), reason=chosen[gidx]))
    return out
