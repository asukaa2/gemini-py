"""Generation configuration and model selection.

Mirrors the model aliases supported by the official `gemini-cli`:
    auto        -> gemini-2.5-pro
    pro         -> gemini-2.5-pro
    flash       -> gemini-2.5-flash
    flash-lite  -> gemini-2.5-flash-lite
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, Optional


class ModelAlias(str, Enum):
    """Shortcuts that map to specific Gemini model names."""

    AUTO = "auto"
    PRO = "pro"
    FLASH = "flash"
    FLASH_LITE = "flash-lite"

    def resolve(self) -> str:
        """Resolve an alias to a concrete model name."""
        mapping = {
            ModelAlias.AUTO: "gemini-2.5-pro",
            ModelAlias.PRO: "gemini-2.5-pro",
            ModelAlias.FLASH: "gemini-2.5-flash",
            ModelAlias.FLASH_LITE: "gemini-2.5-flash-lite",
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
