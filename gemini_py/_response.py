"""Internal response-shape helper classes.

The agent loop in `client.py` accesses responses via this shape:

    response.candidates[].content.parts[].text      -> str | None
    response.candidates[].content.parts[].function_call -> FunctionCall | None
    response.usage_metadata                         -> UsageMetadata | None

The Gemini SDK returns objects of this shape directly. For non-Gemini
providers (OpenAI, Anthropic, Ollama, transformers, custom callables),
we synthesize objects of the same shape so the agent loop is unchanged.

Moved here from `client.py` so that provider adapters can import them
without circular dependencies.
"""

from __future__ import annotations

from typing import Any, List, Optional


class _TextPart:
    """A minimal Part-like object carrying just text (no function_call)."""

    __slots__ = ("text", "function_call", "thought_signature")

    def __init__(self, text: str) -> None:
        self.text = text
        self.function_call = None
        self.thought_signature = None


class _FunctionCallPart:
    """A Part-like object carrying a function_call (no text).

    `function_call` is a small object with `name`, `args`, `id`,
    `thought_signature` attributes — built via _FunctionCall.
    """

    __slots__ = ("text", "function_call", "thought_signature")

    def __init__(self, name: str, args: dict, call_id: Optional[str] = None) -> None:
        self.text = None
        self.function_call = _FunctionCall(name=name, args=args, id=call_id)
        self.thought_signature = None


class _FunctionCall:
    __slots__ = ("name", "args", "id", "thought_signature")

    def __init__(self, *, name: str, args: dict, id: Optional[str] = None) -> None:
        self.name = name
        self.args = args
        self.id = id
        self.thought_signature = None


class _Content:
    __slots__ = ("parts",)

    def __init__(self, parts: List[Any]) -> None:
        self.parts = parts


class _Candidate:
    __slots__ = ("content", "finish_reason")

    def __init__(self, content: _Content, finish_reason: str = "") -> None:
        self.content = content
        self.finish_reason = finish_reason


class _UsageMetadata:
    __slots__ = (
        "prompt_token_count",
        "candidates_token_count",
        "total_token_count",
        "cached_content_token_count",
        "thoughts_token_count",
    )

    def __init__(
        self,
        *,
        prompt_token_count: int = 0,
        candidates_token_count: int = 0,
        total_token_count: int = 0,
        cached_content_token_count: int = 0,
        thoughts_token_count: int = 0,
    ) -> None:
        self.prompt_token_count = prompt_token_count
        self.candidates_token_count = candidates_token_count
        self.total_token_count = total_token_count
        self.cached_content_token_count = cached_content_token_count
        self.thoughts_token_count = thoughts_token_count


class _StreamedResponse:
    """A response-shaped object that the agent loop can consume.

    Built by:
      - The streaming path in `GeminiClient._stream_and_collect` (from
        real Gemini streaming chunks).
      - Provider adapters for non-Gemini providers (OpenAI, Anthropic,
        Ollama, transformers, custom callables) — they synthesize one
        of these from their own response format.
    """

    def __init__(
        self,
        parts: List[Any],
        text: str,
        usage_metadata: Optional[Any] = None,
        raw: Any = None,
        finish_reason: str = "",
    ) -> None:
        # If we have no parts at all (e.g. model refused), synthesize an
        # empty text part so the loop's text extraction still works.
        if not parts and text:
            parts = [_TextPart(text)]
        candidate = _Candidate(_Content(parts), finish_reason=finish_reason)
        self.candidates = [candidate]
        self.text = text  # convenience: full text
        self.usage_metadata = usage_metadata
        self.raw = raw


# --- Adapter helpers -------------------------------------------------------


def text_response(text: str, *, usage: Optional[dict] = None, raw: Any = None) -> _StreamedResponse:
    """Build a response from a plain text string (for non-Gemini providers)."""
    parts = [_TextPart(text)]
    um = None
    if usage:
        um = _UsageMetadata(**usage)
    return _StreamedResponse(parts=parts, text=text, usage_metadata=um, raw=raw)


def text_chunk(text: str) -> _StreamedResponse:
    """Build a streaming chunk carrying incremental text."""
    return _StreamedResponse(parts=[_TextPart(text)], text=text, usage_metadata=None, raw=None)
