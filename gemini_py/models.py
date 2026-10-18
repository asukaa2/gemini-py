"""Data models for messages, responses, and usage stats."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union


class MessageRole(str, Enum):
    """Roles for chat messages."""

    USER = "user"
    MODEL = "model"
    SYSTEM = "system"
    TOOL = "tool"


@dataclass
class Message:
    """A single message in a conversation.

    `content` can be a string or a list of part dicts (for multimodal content).
    """

    role: MessageRole
    content: Union[str, List[Dict[str, Any]]]
    # For tool calls / results
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    tool_call_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"role": self.role.value, "parts": []}
        if isinstance(self.content, str):
            out["parts"].append({"text": self.content})
        else:
            out["parts"].extend(self.content)
        if self.tool_calls:
            out["tool_calls"] = self.tool_calls
        if self.tool_call_id:
            out["tool_call_id"] = self.tool_call_id
        return out

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        role = MessageRole(data.get("role", "user"))
        parts = data.get("parts", [])
        if parts and isinstance(parts[0], dict) and "text" in parts[0]:
            content = parts[0]["text"]
        else:
            content = data.get("content", "")
        return cls(
            role=role,
            content=content,
            tool_calls=data.get("tool_calls", []),
            tool_call_id=data.get("tool_call_id"),
        )


@dataclass
class UsageStats:
    """Token usage statistics for a generation."""

    prompt_token_count: int = 0
    candidates_token_count: int = 0
    total_token_count: int = 0
    cached_content_token_count: int = 0
    thoughts_token_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt_token_count": self.prompt_token_count,
            "candidates_token_count": self.candidates_token_count,
            "total_token_count": self.total_token_count,
            "cached_content_token_count": self.cached_content_token_count,
            "thoughts_token_count": self.thoughts_token_count,
        }

    @classmethod
    def from_metadata(cls, metadata: Any) -> "UsageStats":
        """Build from google-genai usage_metadata object."""
        if metadata is None:
            return cls()
        return cls(
            prompt_token_count=getattr(metadata, "prompt_token_count", 0) or 0,
            candidates_token_count=getattr(metadata, "candidates_token_count", 0) or 0,
            total_token_count=getattr(metadata, "total_token_count", 0) or 0,
            cached_content_token_count=getattr(
                metadata, "cached_content_token_count", 0
            )
            or 0,
            thoughts_token_count=getattr(metadata, "thoughts_token_count", 0) or 0,
        )


@dataclass
class GenerateResult:
    """Result of a single generate_text call."""

    text: str
    usage: UsageStats = field(default_factory=UsageStats)
    model: str = ""
    finish_reason: str = ""
    # For agentic flows
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    tool_results: List[Dict[str, Any]] = field(default_factory=list)
    # Raw response for advanced consumers
    raw: Any = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "response": self.text,
            "stats": self.usage.to_dict(),
            "model": self.model,
            "finish_reason": self.finish_reason,
            "tool_calls": self.tool_calls,
            "tool_results": self.tool_results,
        }
