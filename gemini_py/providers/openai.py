"""OpenAI provider — uses the `openai` Python package.

Supports text generation and streaming. Function-calling is not yet
adapted (text-only mode for now). Install with `pip install openai`.

Compatible with OpenAI-compatible APIs (DeepSeek, Together, Groq,
OpenRouter, vLLM, LM Studio, etc.) by setting base_url.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Iterator, List, Optional

from . import Provider
from ._converters import contents_to_chat_messages
from .._response import _StreamedResponse, _UsageMetadata


DEFAULT_MODEL = "gpt-4o-mini"


class OpenAIProvider(Provider):
    """Provider backed by the `openai` Python SDK."""

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        default_model: Optional[str] = None,
    ) -> None:
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "openai is not installed. Run `pip install openai`."
            ) from e

        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        self.default_model = default_model or os.environ.get("OPENAI_MODEL") or DEFAULT_MODEL

        if not self.api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Get one from https://platform.openai.com/api-keys "
                "or set --api-key."
            )

        self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def generate_content(
        self,
        *,
        model: str,
        contents: List[Dict[str, Any]],
        config: Dict[str, Any],
    ) -> Any:
        messages = contents_to_chat_messages(contents)
        response = self._client.chat.completions.create(
            model=model or self.default_model,
            messages=messages,
            temperature=config.get("temperature", 1.0),
            max_tokens=config.get("max_output_tokens", 8192),
            stream=False,
        )
        return _openai_to_internal(response)

    def generate_content_stream(
        self,
        *,
        model: str,
        contents: List[Dict[str, Any]],
        config: Dict[str, Any],
    ) -> Iterator[Any]:
        messages = contents_to_chat_messages(contents)
        stream = self._client.chat.completions.create(
            model=model or self.default_model,
            messages=messages,
            temperature=config.get("temperature", 1.0),
            max_tokens=config.get("max_output_tokens", 8192),
            stream=True,
        )
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            text = getattr(delta, "content", "") or ""
            if text:
                # Yield a synthesized chunk with just the text.
                yield _StreamedResponse(
                    parts=[],
                    text=text,
                    usage_metadata=None,
                    raw=chunk,
                )

    def list_models(self) -> List[str]:
        try:
            return [m.id for m in self._client.models.list().data]
        except Exception:
            return []

    def supports_tools(self) -> bool:
        return False  # function-calling adapter not yet implemented

    @property
    def name(self) -> str:
        return "openai"


def _openai_to_internal(response: Any) -> _StreamedResponse:
    """Convert an OpenAI ChatCompletion to a _StreamedResponse."""
    try:
        choice = response.choices[0]
        text = choice.message.content or ""
        finish = choice.finish_reason or ""
    except Exception:
        text = ""
        finish = ""
    usage = None
    if getattr(response, "usage", None):
        u = response.usage
        usage = _UsageMetadata(
            prompt_token_count=getattr(u, "prompt_tokens", 0) or 0,
            candidates_token_count=getattr(u, "completion_tokens", 0) or 0,
            total_token_count=getattr(u, "total_tokens", 0) or 0,
        )
    return _StreamedResponse(
        parts=[],
        text=text,
        usage_metadata=usage,
        raw=response,
        finish_reason=finish,
    )
