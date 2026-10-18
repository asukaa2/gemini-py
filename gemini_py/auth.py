"""Authentication configuration for the Gemini Python SDK.

Mirrors the auth methods supported by the official `gemini-cli`:
    1. GEMINI_API_KEY       — Google AI Studio API key (recommended for headless)
    2. GOOGLE_API_KEY       — Vertex AI / Cloud API key
    3. Vertex AI ADC        — `GOOGLE_APPLICATION_CREDENTIALS` or `gcloud auth`
                              application-default login
    4. Google Cloud env     — `GOOGLE_CLOUD_PROJECT` + `GOOGLE_CLOUD_LOCATION`
                              for Vertex AI routing

For interactive OAuth ("Sign in with Google") we fall back to ADC if available;
we don't bundle a full OAuth flow in this Python port, but the `google-genai`
SDK will pick up ADC from the environment automatically.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


class AuthMethod(str, Enum):
    """Supported authentication methods, mirroring gemini-cli."""

    GEMINI_API_KEY = "gemini_api_key"   # Google AI Studio API key
    GOOGLE_API_KEY = "google_api_key"   # Google Cloud API key (Vertex)
    VERTEX_AI_ADC = "vertex_adc"        # Vertex AI w/ Application Default Creds
    AUTO = "auto"                       # Auto-detect from environment


def _env_search_paths() -> list:
    """Return the list of .env search paths, computed lazily at call time.

    Order: project-local first, then user-wide. This means a project's
    `.env` can override the user-wide `~/.gemini-py/.env`.
    """
    return [
        Path.cwd() / ".env",
        Path.cwd() / ".gemini" / ".env",
        Path.home() / ".gemini" / ".env",
        Path.home() / ".gemini-py" / ".env",
    ]


def _load_env_files() -> None:
    """Load variables from all .env files in general-to-specific order.

    Files are loaded general → specific, so a more specific file (project
    .env) overrides a less specific one (user .env). This differs slightly
    from the gemini-cli behavior ("load first found only") because we want
    `login()` to work reliably even when a project has its own `.env` that
    doesn't set GEMINI_API_KEY.
    """
    paths = _env_search_paths()
    # Reverse so general files load first, specific files last. Later loads
    # (with override=True) override earlier ones — specific wins.
    for path in reversed(paths):
        if path.is_file():
            load_dotenv(path, override=True)


@dataclass
class AuthConfig:
    """Authentication configuration for the Gemini SDK.

    Construct via `AuthConfig.from_env()` to read from environment variables
    and `.env` files automatically, or construct manually for full control.
    """

    method: AuthMethod = AuthMethod.AUTO
    api_key: Optional[str] = None
    # Vertex AI settings
    vertex_project: Optional[str] = None
    vertex_location: Optional[str] = None
    vertex_credentials_path: Optional[str] = None

    def __post_init__(self) -> None:
        # If method is AUTO, resolve it now.
        if self.method == AuthMethod.AUTO:
            self.method = self._resolve_auto()

    # ---- Construction ---------------------------------------------------

    @classmethod
    def from_env(cls, load_dotenv_files: bool = True) -> "AuthConfig":
        """Build an AuthConfig from environment variables.

        Reads (in priority order):
            GEMINI_API_KEY          — Google AI Studio key
            GOOGLE_API_KEY          — Google Cloud API key (Vertex)
            GOOGLE_APPLICATION_CREDENTIALS — Vertex AI service account JSON
            GOOGLE_CLOUD_PROJECT    — Vertex AI project ID
            GOOGLE_CLOUD_LOCATION   — Vertex AI region (default: us-central1)
        """
        if load_dotenv_files:
            _load_env_files()

        # Build initial config from env
        gemini_key = os.environ.get("GEMINI_API_KEY")
        google_key = os.environ.get("GOOGLE_API_KEY")
        adc_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        project = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get(
            "GOOGLE_CLOUD_PROJECT_ID"
        )
        location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")

        # If both GEMINI and GOOGLE keys exist, prefer GEMINI_API_KEY.
        method = AuthMethod.AUTO
        if gemini_key:
            method = AuthMethod.GEMINI_API_KEY
        elif google_key:
            method = AuthMethod.GOOGLE_API_KEY
        elif adc_path:
            method = AuthMethod.VERTEX_AI_ADC

        return cls(
            method=method,
            api_key=gemini_key or google_key,
            vertex_project=project,
            vertex_location=location,
            vertex_credentials_path=adc_path,
        )

    # ---- Resolution -----------------------------------------------------

    def _resolve_auto(self) -> AuthMethod:
        """Pick the best method based on which credentials are available."""
        if self.api_key and not self.vertex_credentials_path:
            # If we have GEMINI_API_KEY specifically, prefer that.
            if os.environ.get("GEMINI_API_KEY"):
                return AuthMethod.GEMINI_API_KEY
            return AuthMethod.GOOGLE_API_KEY
        if self.vertex_credentials_path or self.vertex_project:
            return AuthMethod.VERTEX_AI_ADC
        # Fall back to ADC (google-genai will pick up ADC if available).
        return AuthMethod.VERTEX_AI_ADC

    # ---- Client config --------------------------------------------------

    def to_client_kwargs(self) -> dict:
        """Convert to kwargs accepted by google.genai.Client.

        Returns:
            Dict with appropriate keys for the chosen auth method.
        """
        if self.method == AuthMethod.GEMINI_API_KEY:
            if not self.api_key:
                raise ValueError(
                    "GEMINI_API_KEY is not set. Get one from "
                    "https://aistudio.google.com/app/apikey"
                )
            return {"api_key": self.api_key}

        if self.method == AuthMethod.GOOGLE_API_KEY:
            if not self.api_key:
                raise ValueError("GOOGLE_API_KEY is not set.")
            if not self.vertex_project:
                raise ValueError(
                    "GOOGLE_CLOUD_PROJECT must be set when using GOOGLE_API_KEY."
                )
            return {
                "api_key": self.api_key,
                "vertexai": True,
                "project": self.vertex_project,
                "location": self.vertex_location,
            }

        if self.method == AuthMethod.VERTEX_AI_ADC:
            if not self.vertex_project:
                raise ValueError(
                    "GOOGLE_CLOUD_PROJECT must be set for Vertex AI."
                )
            kwargs: dict = {
                "vertexai": True,
                "project": self.vertex_project,
                "location": self.vertex_location,
            }
            # Let google-genai auto-discover ADC if no path is set.
            if self.vertex_credentials_path:
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = (
                    self.vertex_credentials_path
                )
            return kwargs

        raise ValueError(f"Unknown auth method: {self.method}")

    def describe(self) -> str:
        """Human-readable description of the chosen auth method."""
        if self.method == AuthMethod.GEMINI_API_KEY:
            return "Gemini API Key (Google AI Studio)"
        if self.method == AuthMethod.GOOGLE_API_KEY:
            return f"Google Cloud API Key (Vertex AI, project={self.vertex_project})"
        if self.method == AuthMethod.VERTEX_AI_ADC:
            return (
                f"Vertex AI ADC (project={self.vertex_project}, "
                f"location={self.vertex_location})"
            )
        return "Auto (no credentials detected)"
