"""Generation configuration and model selection.

Mirrors the latest AI Studio model lineup (as of 2026-10-13) plus the
gemini-cli alias convention. Aliases resolve as follows:

    auto          -> gemini-3.8-flash        (AI Studio recommended default)
    argon         -> gemini-4-argon          (most capable, GA via Dec 2026)
    pro           -> gemini-3.1-pro          (advanced reasoning)
    flash         -> gemini-3.8-flash        (frontier speed)
    flash-lite    -> gemini-3.5-flash-lite   (low-latency subagent)
    flash-latest  -> gemini-flash-latest     (Google's rolling "latest" alias)
    omni          -> gemini-omni-1.1-flash   (video generation/editing)
    transcribe    -> gemini-3.5-transcribe   (speech-to-text)
    tts           -> gemini-3.8-flash-tts     (text-to-speech)

Source: https://ai.google.dev/gemini-api/docs/changelog (Oct 13, 2026 entry)
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, Optional


class ModelAlias(str, Enum):
    """Shortcuts that map to specific Gemini model names.

    Resolutions are kept in sync with the AI Studio model lineup as of
    October 2026. See https://ai.google.dev/gemini-api/docs/changelog.
    """

    AUTO = "auto"
    ARGON = "argon"
    PRO = "pro"
    FLASH = "flash"
    FLASH_LITE = "flash-lite"
    FLASH_LATEST = "flash-latest"
    OMNI = "omni"
    TRANSCRIBE = "transcribe"
    TTS = "tts"

    def resolve(self) -> str:
        """Resolve an alias to a concrete model name."""
        mapping = {
            # Default — AI Studio's recommended model for new projects.
            ModelAlias.AUTO: "gemini-3.8-flash",
            # Most capable model (GA through Dec 31, 2026).
            ModelAlias.ARGON: "gemini-4-argon",
            # Advanced reasoning for coding and complex tasks.
            ModelAlias.PRO: "gemini-3.1-pro",
            # Frontier intelligence built for speed.
            ModelAlias.FLASH: "gemini-3.8-flash",
            # Low-latency, cost-effective subagent.
            ModelAlias.FLASH_LITE: "gemini-3.5-flash-lite",
            # Google's rolling "latest" alias — points to newest Flash.
            ModelAlias.FLASH_LATEST: "gemini-flash-latest",
            # Video generation and editing.
            ModelAlias.OMNI: "gemini-omni-1.1-flash",
            # Speech-to-text.
            ModelAlias.TRANSCRIBE: "gemini-3.5-transcribe",
            # Text-to-speech.
            ModelAlias.TTS: "gemini-3.8-flash-tts",
        }
        return mapping[self]

    @classmethod
    def is_alias(cls, value: str) -> bool:
        """Check if a string is a known alias."""
        try:
            cls(value)
            return True
        except ValueError:
            return False

    @classmethod
    def all_aliases(cls) -> Dict[str, str]:
        """Return a {alias: resolved_model} dict for help text / `--help`."""
        return {alias.value: alias.resolve() for alias in cls}


@dataclass
class GeminiConfig:
    """Configuration for a single Gemini generation request.

    Mirrors the most common gemini-cli flags. Maps directly to the
    `google.genai` SDK's `GenerateContentConfig`.
    """

    model: str = ModelAlias.AUTO.resolve()
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 64
    max_output_tokens: int = 8192
    system_instruction: Optional[str] = None
    # Tool control
    enable_shell_tool: bool = True
    enable_filesystem_tools: bool = True
    enable_web_search: bool = False
    enable_web_fetch: bool = False
    # Safety
    safety_settings: list = field(default_factory=list)

    def resolve_model(self) -> str:
        """Return the resolved model name (alias -> concrete)."""
        if ModelAlias.is_alias(self.model):
            return ModelAlias(self.model).resolve()
        return self.model

    def to_generate_content_config(self) -> Dict[str, Any]:
        """Convert to a dict usable by `google.genai.types.GenerateContentConfig`."""
        cfg: Dict[str, Any] = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "max_output_tokens": self.max_output_tokens,
        }
        if self.system_instruction:
            cfg["system_instruction"] = self.system_instruction
        if self.safety_settings:
            cfg["safety_settings"] = self.safety_settings
        return cfg

    def to_dict(self) -> Dict[str, Any]:
        """Serializable representation for session persistence."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GeminiConfig":
        """Reconstruct from a serialized dict."""
        # Only keep fields we know about.
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in known})
