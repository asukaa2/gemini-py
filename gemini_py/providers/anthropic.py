"""Anthropic provider — uses the `anthropic` Python package.

Supports text generation and streaming. Function-calling is not yet
adapted (text-only mode for now). Install with `pip install anthropic`.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Iterator, List, Optional

from . import Provider
from ._converters import contents_to_chat_messages
from .._response import _StreamedResponse, _UsageMetadata


DEFAULT_MODEL = "claude-3-5-sonnet-20241022"


class AnthropicProvider(Provider):
    """Provider backed by the `anthropic` Python SDK."""

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        default_model: Optional[str] = None,
    ) -> None:
        try:
            import anthropic  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "anthropic is not installed. Run `pip install anthropic`."
            ) from e

        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.base_url = base_url or os.environ.get("ANTHROPIC_BASE_URL")
        self.default_model = default_model or os.environ.get("ANTHROPIC_MODEL") or DEFAULT_MODEL

        if not self.api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Get one from https://console.anthropic.com/ "
                "or set --api-key."
            )

        kwargs = {"api_key": self.api_key}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        self._client = anthropic.Anthropic(**kwargs)

    def generate_content(
        self,
        *,
        model: str,
        contents: List[Dict[str, Any]],
        config: Dict[str, Any],
    ) -> Any:
        messages = contents_to_chat_messages(contents)
        # Strip out 'system' role — Anthropic wants it as a separate arg.
        system_text = ""
        non_system: List[Dict[str, Any]] = []
        for m in messages:
            if m["role"] == "system":
                system_text += m["content"] + "\n"
            else:
                non_system.append(m)

        kwargs: Dict[str, Any] = {
            "model": model or self.default_model,
            "messages": non_system,
            "max_tokens": config.get("max_output_tokens", 8192),
            "temperature": config.get("temperature", 1.0),
        }
        if system_text.strip():
            kwargs["system"] = system_text.strip()
        response = self._client.messages.create(**kwargs)
        return _anthropic_to_internal(response)

    def generate_content_stream(
        self,
        *,
        model: str,
        contents: List[Dict[str, Any]],
        config: Dict[str, Any],
    ) -> Iterator[Any]:
        messages = contents_to_chat_messages(contents)
        system_text = ""
        non_system: List[Dict[str, Any]] = []
        for m in messages:
            if m["role"] == "system":
                system_text += m["content"] + "\n"
            else:
                non_system.append(m)

        kwargs: Dict[str, Any] = {
            "model": model or self.default_model,
            "messages": non_system,
            "max_tokens": config.get("max_output_tokens", 8192),
            "temperature": config.get("temperature", 1.0),
        }
        if system_text.strip():
            kwargs["system"] = system_text.strip()
        with self._client.messages.stream(**kwargs) as stream:
            for text in stream.text_stream:
                if text:
                    yield _StreamedResponse(
                        parts=[],
                        text=text,
                        usage_metadata=None,
                        raw=None,
                    )

    def list_models(self) -> List[str]:
        # Anthropic doesn't expose a list-models API; return known models.
        return [
            "claude-3-5-sonnet-20241022",
            "claude-3-5-haiku-20241022",
            "claude-3-opus-20240229",
        ]

    def supports_tools(self) -> bool:
        return False

    @property
    def name(self) -> str:
        return "anthropic"


def _anthropic_to_internal(response: Any) -> _StreamedResponse:
    """Convert an Anthropic Message to a _StreamedResponse."""
    try:
        # Anthropic returns content blocks; concat text ones.
        text_parts: List[str] = []
        for block in response.content or []:
            t = getattr(block, "text", None)
            if t:
                text_parts.append(t)
        text = "".join(text_parts)
        finish = getattr(response, "stop_reason", "") or ""
    except Exception:
        text = ""
        finish = ""
    usage = None
    if getattr(response, "usage", None):
        u = response.usage
        usage = _UsageMetadata(
            prompt_token_count=getattr(u, "input_tokens", 0) or 0,
            candidates_token_count=getattr(u, "output_tokens", 0) or 0,
            total_token_count=(
                (getattr(u, "input_tokens", 0) or 0)
                + (getattr(u, "output_tokens", 0) or 0)
            ),
        )
    return _StreamedResponse(
        parts=[],
        text=text,
        usage_metadata=usage,
        raw=response,
        finish_reason=finish,
    )
