"""Local LLM access (Ollama by default) with JSON-schema-enforced output.

Talks to the Ollama HTTP API over stdlib urllib so there is no extra dependency.
All calls force valid JSON via the top-level ``format`` field (a JSON Schema),
and ask the model for SEGMENT INDICES rather than float timestamps (which small
models hallucinate) — the caller maps indices back to exact times.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Optional

from .config import LlmCfg


class LLMUnavailable(RuntimeError):
    pass


class OllamaClient:
    def __init__(self, cfg: LlmCfg):
        self.cfg = cfg
        self.host = cfg.host.rstrip("/")

    # --- low level -----------------------------------------------------------
    def _get(self, path: str, timeout: float = 5.0) -> dict:
        req = urllib.request.Request(self.host + path, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))

    def _post(self, path: str, payload: dict) -> dict:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(self.host + path, data=data, method="POST",
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.cfg.timeout) as r:
            return json.loads(r.read().decode("utf-8"))

    # --- health --------------------------------------------------------------
    def available(self) -> bool:
        try:
            self._get("/api/version")
            return True
        except Exception:
            return False

    def has_model(self, model: Optional[str] = None) -> bool:
        model = model or self.cfg.model
        try:
            tags = self._get("/api/tags")
            names = {m.get("name", "") for m in tags.get("models", [])}
            if any(n == model for n in names):
                return True
            # A bare family name (no tag) may match any tag of that family, but a
            # tagged request (qwen3:8b) must match exactly — otherwise a different
            # size (qwen3:0.5b) would falsely report 'ready' and 404 at call time.
            if ":" not in model:
                return any(n.split(":")[0] == model for n in names)
            return False
        except Exception:
            return False

    def list_models(self) -> list[str]:
        """Return names of locally installed Ollama models (for the UI dropdown).

        Parses ``GET /api/tags``. Network failure / Ollama off -> ``[]`` (never
        raises), so the caller can degrade gracefully.
        """
        try:
            tags = self._get("/api/tags")
        except Exception:  # noqa: BLE001 — Ollama off / unreachable: empty list
            return []
        names = [m.get("name", "") for m in tags.get("models", [])]
        return [n for n in names if n]

    def _build_payload(self, system: str, user: str, schema: dict,
                       temperature: float, keep_alive=None) -> dict:
        # qwen3 (and other reasoning models) otherwise emit a <think>...</think>
        # block that consumes the whole context/output budget on structured
        # calls. Disable it both ways: the top-level "think" flag (newer Ollama)
        # and a ' /no_think' marker in the system prompt (model-level switch).
        if getattr(self.cfg, "think", False):
            sys_prompt = system
        else:
            sys_prompt = system.rstrip() + " /no_think"
        payload = {
            "model": self.cfg.model,
            "messages": [{"role": "system", "content": sys_prompt},
                         {"role": "user", "content": user}],
            "stream": False,
            "format": schema,
            "think": bool(getattr(self.cfg, "think", False)),
            # Unload the model right after the call so it does not keep holding
            # VRAM next to Whisper on an 8 GB card (transcription -> detection).
            # A caller may override (e.g. chapters keeps it warm across windows
            # so a long video doesn't reload qwen3 once per window).
            "keep_alive": (keep_alive if keep_alive is not None
                           else getattr(self.cfg, "keep_alive", 0)),
            "options": {"temperature": temperature,
                        "num_ctx": self.cfg.num_ctx},
        }
        return payload

    @staticmethod
    def _salvage_json(content: str) -> Optional[dict]:
        """Extract the first parseable balanced ``{...}`` object from a reply.

        Reasoning models can wrap the JSON in prose or a stray <think> block
        (which may itself contain unbalanced braces). For each ``{`` we scan a
        brace-balanced (string-aware) span and try to parse it; the first span
        that parses to an object wins. Returns None if nothing usable is found.
        """
        search_from = 0
        while True:
            start = content.find("{", search_from)
            if start < 0:
                return None
            depth = 0
            in_str = False
            esc = False
            closed_at = -1
            for i in range(start, len(content)):
                ch = content[i]
                if in_str:
                    if esc:
                        esc = False
                    elif ch == "\\":
                        esc = True
                    elif ch == '"':
                        in_str = False
                    continue
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        closed_at = i
                        break
            if closed_at >= 0:
                candidate = content[start:closed_at + 1]
                try:
                    parsed = json.loads(candidate)
                    if isinstance(parsed, dict):
                        return parsed
                except json.JSONDecodeError:
                    pass
            # this opener didn't yield a parseable object — try the next "{"
            search_from = start + 1

    def chat_json(self, system: str, user: str, schema: dict, keep_alive=None) -> dict:
        """One-shot chat that must return JSON validating against ``schema``.

        Retries once (at temperature 0) on a parse miss, and tries to salvage a
        balanced ``{...}`` substring before giving up. Transport failures raise
        ``LLMUnavailable`` so callers can fall back gracefully. ``keep_alive``
        overrides the config default for this call (e.g. keep the model warm
        between chapter windows, then unload on the last one).
        """
        last_content = ""
        for attempt in range(2):
            temperature = self.cfg.temperature if attempt == 0 else 0.0
            payload = self._build_payload(system, user, schema, temperature,
                                          keep_alive=keep_alive)
            try:
                resp = self._post("/api/chat", payload)
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                # URLError covers connection refused/DNS; TimeoutError/OSError
                # cover a slow inference read-timeout and dropped connections.
                raise LLMUnavailable(f"Ollama request failed: {e}") from e
            content = resp.get("message", {}).get("content", "")
            last_content = content
            if not content:
                continue  # retry once on an empty reply
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                salvaged = self._salvage_json(content)
                if salvaged is not None:
                    return salvaged
                # else: fall through to the retry / final raise
        if not last_content:
            raise LLMUnavailable("Empty response from LLM")
        raise LLMUnavailable(f"LLM returned non-JSON: {last_content[:200]}")


def segment_windows(n: int, cfg: LlmCfg) -> list[tuple[int, int]]:
    """Split ``n`` segments into overlapping ``[start, end)`` windows.

    Long videos overflow a single prompt, so the bad-takes / chapters callers
    walk these windows and offset the per-window indices back to GLOBAL segment
    indices. The overlap lets a take/chapter that straddles a window boundary be
    found in at least one window; callers dedupe the overlap.
    """
    size = max(1, int(getattr(cfg, "max_segments_per_call", 80)))
    overlap = max(0, int(getattr(cfg, "segment_overlap", 5)))
    overlap = min(overlap, size - 1)  # guarantee forward progress
    if n <= 0:
        return []
    if n <= size:
        return [(0, n)]
    windows: list[tuple[int, int]] = []
    start = 0
    while start < n:
        end = min(start + size, n)
        windows.append((start, end))
        if end >= n:
            break
        start = end - overlap
    return windows


def get_client(cfg: LlmCfg) -> Optional[OllamaClient]:
    """Return a ready client, or None (with reasons logged by the caller)."""
    if not cfg.enabled:
        return None
    if cfg.backend != "ollama":
        # llama_server path can be added later; for now only ollama is wired.
        return None
    client = OllamaClient(cfg)
    return client
