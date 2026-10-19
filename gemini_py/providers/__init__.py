"""Multi-provider support for gemini-py.

This package defines a `Provider` protocol that abstracts away the
underlying LLM backend. The agent loop in `GeminiClient` calls
`provider.generate_content(...)` / `provider.generate_content_stream(...)`
and consumes the response in a uniform shape (built via `_response.py`).

Built-in providers:

- `GeminiProvider`    — wraps `google.genai.Client` (default).
- `OpenAIProvider`    — uses the `openai` Python package.
- `AnthropicProvider` — uses the `anthropic` Python package.
- `OllamaProvider`    — HTTP to a local Ollama server (no extra deps).
- `TransformersProvider` — runs HuggingFace transformers locally.
- `CustomProvider`    — takes a callable (generate_fn / stream_fn).

Pick one via the factory `make_provider(name=..., **kwargs)` or by
constructing the class directly. The CLI's `--provider NAME` flag
routes through `make_provider`.

Note: non-Gemini providers currently support text generation only.
Function-calling / agentic flow (with tools) only works with the
Gemini provider. We're working on adapters for OpenAI/Anthropic native
tool-calling.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, List, Optional


class Provider(ABC):
    """Abstract provider protocol.

    Subclasses implement `generate_content` and `generate_content_stream`.
    Both take `model`, `contents`, `config` and return objects shaped
    like `gemini_py._response._StreamedResponse` (i.e. with
    `.candidates[].content.parts[]` and `.usage_metadata`).
    """

    @abstractmethod
    def generate_content(
        self,
        *,
        model: str,
        contents: List[Dict[str, Any]],
        config: Dict[str, Any],
    ) -> Any:
        """Return a response-shaped object (see _response.py)."""
        ...

    @abstractmethod
    def generate_content_stream(
        self,
        *,
        model: str,
        contents: List[Dict[str, Any]],
        config: Dict[str, Any],
    ) -> Iterator[Any]:
        """Yield response chunks (each with .text or candidates)."""
        ...

    def list_models(self) -> List[str]:
        """Return available model names. Default: empty list."""
        return []

    @property
    def name(self) -> str:
        """Provider name (defaults to class name)."""
        return type(self).__name__

    def supports_tools(self) -> bool:
        """Whether this provider supports function-calling / agentic flow.
        Non-Gemini providers return False — they only support text.
        """
        return False

    def close(self) -> None:
        """Release any resources. Default: no-op."""
        pass


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def make_provider(
    name: str,
    *,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    auth: Optional[Any] = None,
) -> Provider:
    """Construct a provider by name.

    Args:
        name: Provider name. One of:
            'gemini' (default), 'openai', 'anthropic', 'ollama',
            'transformers', 'custom'.
        api_key: API key (for openai/anthropic). Falls back to env vars
            OPENAI_API_KEY / ANTHROPIC_API_KEY.
        base_url: Override the base URL (for openai/anthropic/ollama).
        model: Default model name (for ollama/transformers).
        auth: AuthConfig (for gemini). Falls back to AuthConfig.from_env().

    Returns:
        A Provider instance.

    Raises:
        ValueError: if `name` is not a known provider.
        RuntimeError: if a required package isn't installed (e.g. openai).
    """
    name_lower = (name or "").lower().strip()
    if name_lower in ("gemini", "google", ""):
        from .gemini import GeminiProvider
        if auth is None:
            from ..auth import AuthConfig
            auth = AuthConfig.from_env()
        return GeminiProvider(auth=auth)
    if name_lower == "openai":
        from .openai import OpenAIProvider
        return OpenAIProvider(api_key=api_key, base_url=base_url, default_model=model)
    if name_lower == "anthropic":
        from .anthropic import AnthropicProvider
        return AnthropicProvider(api_key=api_key, base_url=base_url, default_model=model)
    if name_lower == "ollama":
        from .ollama import OllamaProvider
        return OllamaProvider(base_url=base_url, default_model=model)
    if name_lower in ("transformers", "hf", "huggingface"):
        from .transformers import TransformersProvider
        return TransformersProvider(model_name=model)
    if name_lower == "custom":
        # Custom needs a callable — can't construct from name alone.
        raise ValueError(
            "CustomProvider can't be constructed by name alone. Construct it "
            "directly: from gemini_py.providers.custom import CustomProvider; "
            "provider = CustomProvider(generate_fn=...)"
        )
    raise ValueError(
        f"unknown provider '{name}'. Pick one of: gemini, openai, anthropic, "
        f"ollama, transformers, custom."
    )


# Re-export the provider classes for convenient import:
from .gemini import GeminiProvider  # noqa: E402,F401
