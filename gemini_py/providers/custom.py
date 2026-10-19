"""Custom provider — user-defined callable.

Lets you plug in any function that takes (prompt, model, **config) and
returns text. Useful for testing, mock backends, or wrapping exotic
APIs that don't have a dedicated provider yet.

Example:

    from gemini_py import GeminiClient
    from gemini_py.providers.custom import CustomProvider

    def my_generate(prompt: str, *, model: str, **config) -> str:
        # Replace with your real backend
        return f"You said: {prompt} (model={model})"

    provider = CustomProvider(
        generate_fn=my_generate,
        models=["my-model-1", "my-model-2"],
        name="my-backend",
    )
    client = GeminiClient(provider=provider)
    print(client.generate_text("Hello").text)
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterator, List, Optional

from . import Provider
from ._converters import extract_user_prompt
from .._response import _StreamedResponse


class CustomProvider(Provider):
    """Provider backed by user-supplied callables."""

    def __init__(
        self,
        *,
        generate_fn: Optional[Callable[..., str]] = None,
        stream_fn: Optional[Callable[..., Iterator[str]]] = None,
        models: Optional[List[str]] = None,
        name: str = "custom",
    ) -> None:
        if generate_fn is None and stream_fn is None:
            raise ValueError(
                "CustomProvider requires at least generate_fn or stream_fn."
            )
        self._generate_fn = generate_fn
        self._stream_fn = stream_fn
        self._models = models or []
        self._name = name

    def generate_content(
        self,
        *,
        model: str,
        contents: List[Dict[str, Any]],
        config: Dict[str, Any],
    ) -> Any:
        prompt = extract_user_prompt(contents)
        if self._generate_fn is None:
            # Fall back to streaming + join.
            chunks = list(self._stream_fn(prompt, model=model, **config))
            text = "".join(chunks)
        else:
            text = self._generate_fn(prompt, model=model, **config)
        return _StreamedResponse(
            parts=[],
            text=str(text),
            usage_metadata=None,
            raw=None,
            finish_reason="STOP",
        )

    def generate_content_stream(
        self,
        *,
        model: str,
        contents: List[Dict[str, Any]],
        config: Dict[str, Any],
    ) -> Iterator[Any]:
        prompt = extract_user_prompt(contents)
        if self._stream_fn is not None:
            for chunk in self._stream_fn(prompt, model=model, **config):
                yield _StreamedResponse(
                    parts=[],
                    text=str(chunk),
                    usage_metadata=None,
                    raw=None,
                )
        else:
            # Non-streaming fallback — emit the full response as one chunk.
            text = self._generate_fn(prompt, model=model, **config)
            yield _StreamedResponse(
                parts=[],
                text=str(text),
                usage_metadata=None,
                raw=None,
            )

    def list_models(self) -> List[str]:
        return list(self._models)

    def supports_tools(self) -> bool:
        return False

    @property
    def name(self) -> str:
        return self._name
