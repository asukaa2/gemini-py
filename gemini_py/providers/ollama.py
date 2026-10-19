"""Ollama provider — local LLM server via HTTP.

No external Python deps required. Talks to a local Ollama server at
http://localhost:11434 by default. Install Ollama from https://ollama.com.

Pull a model first:

    ollama pull llama3.2
    # or
    ollama pull qwen2.5-coder

Then use gemini-py with:

    gemini-py -p "Hello" --provider ollama --provider-model llama3.2
    # or
    gemini-py -p "Hello" --provider ollama --provider-model qwen2.5-coder \
      --provider-base-url http://localhost:11434
"""

from __future__ import annotations

import json
import urllib.request
from typing import Any, Dict, Iterator, List, Optional

from . import Provider
from ._converters import contents_to_ollama_messages
from .._response import _StreamedResponse, _UsageMetadata


DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "llama3.2"


class OllamaProvider(Provider):
    """Local Ollama server provider (HTTP, no extra deps)."""

    def __init__(
        self,
        *,
        base_url: Optional[str] = None,
        default_model: Optional[str] = None,
    ) -> None:
        self.base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        self.default_model = default_model or DEFAULT_MODEL
        # Quick connectivity check — warn (don't fail) if not reachable.
        try:
            self.list_models()
        except Exception:
            import sys
            sys.stderr.write(
                f"[gemini-py] Ollama at {self.base_url} not reachable. "
                f"Start it with `ollama serve` or install from https://ollama.com\n"
            )

    def generate_content(
        self,
        *,
        model: str,
        contents: List[Dict[str, Any]],
        config: Dict[str, Any],
    ) -> Any:
        messages = contents_to_ollama_messages(contents)
        payload = {
            "model": model or self.default_model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": config.get("temperature", 1.0),
                "num_predict": config.get("max_output_tokens", 8192),
            },
        }
        data = self._post("/api/chat", payload)
        return _ollama_to_internal(data)

    def generate_content_stream(
        self,
        *,
        model: str,
        contents: List[Dict[str, Any]],
        config: Dict[str, Any],
    ) -> Iterator[Any]:
        messages = contents_to_ollama_messages(contents)
        payload = {
            "model": model or self.default_model,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": config.get("temperature", 1.0),
                "num_predict": config.get("max_output_tokens", 8192),
            },
        }
        for line in self._post_stream("/api/chat", payload):
            line = line.strip()
            if not line:
                continue
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = (chunk.get("message") or {}).get("content", "")
            if text:
                yield _StreamedResponse(
                    parts=[],
                    text=text,
                    usage_metadata=None,
                    raw=chunk,
                )

    def list_models(self) -> List[str]:
        try:
            data = self._get("/api/tags")
            return [m.get("name", "") for m in data.get("models", [])]
        except Exception:
            return []

    def supports_tools(self) -> bool:
        # Ollama does support tool-calling, but the adapter isn't implemented
        # yet — text-only mode for now.
        return False

    @property
    def name(self) -> str:
        return "ollama"

    # --- HTTP helpers -----------------------------------------------------

    def _get(self, path: str) -> Dict[str, Any]:
        req = urllib.request.Request(f"{self.base_url}{path}")
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}{path}",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _post_stream(self, path: str, payload: Dict[str, Any]) -> Iterator[str]:
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}{path}",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=None) as resp:
            for line in resp:
                yield line.decode("utf-8", errors="replace")


def _ollama_to_internal(data: Dict[str, Any]) -> _StreamedResponse:
    """Convert an Ollama /api/chat response to a _StreamedResponse."""
    msg = data.get("message") or {}
    text = msg.get("content", "")
    finish = data.get("done_reason", "") or ("STOP" if data.get("done") else "")
    usage = None
    if "prompt_eval_count" in data or "eval_count" in data:
        prompt_count = data.get("prompt_eval_count", 0) or 0
        completion_count = data.get("eval_count", 0) or 0
        usage = _UsageMetadata(
            prompt_token_count=prompt_count,
            candidates_token_count=completion_count,
            total_token_count=prompt_count + completion_count,
        )
    return _StreamedResponse(
        parts=[],
        text=text,
        usage_metadata=usage,
        raw=data,
        finish_reason=finish,
    )
