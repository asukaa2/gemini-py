"""Gemini provider — wraps `google.genai.Client`.

This is the default provider and supports the full feature set:
function-calling, streaming, thought-signatures, tool execution.
"""

from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional

from . import Provider
from .._response import _StreamedResponse


class GeminiProvider(Provider):
    """Provider backed by `google.genai.Client`.

    Construction triggers `AuthConfig.from_env()` if `auth` is None.
    Supports all features: streaming, function-calling, thought-signatures.
    """

    def __init__(self, auth: Optional[Any] = None) -> None:
        if auth is None:
            from ..auth import AuthConfig
            auth = AuthConfig.from_env()
        self.auth = auth
        try:
            from google import genai  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "google-genai is not installed. Run `pip install google-genai`."
            ) from e

        kwargs = auth.to_client_kwargs()
        try:
            self._client = genai.Client(**kwargs)
        except Exception as e:
            from ..client import GeminiError
            raise GeminiError(f"Failed to initialize google.genai.Client: {e}") from e

    def generate_content(
        self,
        *,
        model: str,
        contents: List[Dict[str, Any]],
        config: Dict[str, Any],
    ) -> Any:
        """Call the Gemini API. Returns the raw google.genai response."""
        return self._client.models.generate_content(
            model=model,
            contents=contents,
            config=config,
        )

    def generate_content_stream(
        self,
        *,
        model: str,
        contents: List[Dict[str, Any]],
        config: Dict[str, Any],
    ) -> Iterator[Any]:
        """Stream chunks. Yields raw google.genai response chunks."""
        return self._client.models.generate_content_stream(
            model=model,
            contents=contents,
            config=config,
        )

    def list_models(self) -> List[str]:
        try:
            return [m.name for m in self._client.models.list()]
        except Exception:
            return []

    def supports_tools(self) -> bool:
        return True

    @property
    def name(self) -> str:
        return "gemini"
