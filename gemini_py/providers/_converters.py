"""Converters between Gemini's contents format and other providers' formats.

The agent loop stores conversation state in Gemini's contents format:

    [
        {"role": "user", "parts": [{"text": "..."}]},
        {"role": "model", "parts": [{"function_call": {...}}]},
        {"role": "user", "parts": [{"function_response": {...}}]},
        ...
    ]

For non-Gemini providers (OpenAI, Anthropic, Ollama), we flatten this
to their native chat-message format. Text-only mode — function calls
are skipped for non-Gemini providers (text-mode only).
"""

from __future__ import annotations

from typing import Any, Dict, List


def contents_to_text(contents: List[Dict[str, Any]]) -> str:
    """Flatten contents to a single text string.

    Used by providers that don't support multi-turn or function-calling
    (custom callables, simple transformers inference). Each message is
    prefixed with its role: `[user] ...`, `[model] ...`.
    """
    parts: List[str] = []
    for c in contents:
        if not isinstance(c, dict):
            continue
        role = c.get("role", "user")
        for p in c.get("parts", []) or []:
            if not isinstance(p, dict):
                continue
            text = p.get("text") or ""
            # Also extract text from TextPart-like objects.
            if not text and hasattr(p, "text"):
                text = getattr(p, "text", "") or ""
            if text:
                parts.append(f"[{role}] {text}")
    return "\n\n".join(parts)


def extract_user_prompt(contents: List[Dict[str, Any]]) -> str:
    """Extract just the LAST user message's text.

    Used by CustomProvider / simple backends that expect a single prompt
    string (no multi-turn conversation history, no role markers).
    """
    # Walk backwards to find the last 'user' message with text.
    for c in reversed(contents):
        if not isinstance(c, dict):
            continue
        if c.get("role") != "user":
            continue
        for p in c.get("parts", []) or []:
            if isinstance(p, dict) and p.get("text"):
                return p["text"]
            if hasattr(p, "text") and getattr(p, "text", None):
                return p.text
    # Fallback: any text part from any message.
    return contents_to_text(contents)


def contents_to_chat_messages(contents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert Gemini contents to OpenAI/Anthropic chat-message format.

    Maps: 'user' -> 'user', 'model' -> 'assistant', 'system' -> 'system'.
    Function-call and function-response parts are dropped (text-only mode).
    """
    out: List[Dict[str, Any]] = []
    for c in contents:
        if not isinstance(c, dict):
            continue
        role = c.get("role", "user")
        # Map role names.
        if role == "model":
            chat_role = "assistant"
        elif role in ("user", "system", "tool"):
            chat_role = role
        else:
            chat_role = "user"
        # Concatenate all text parts in this message.
        text_parts: List[str] = []
        for p in c.get("parts", []) or []:
            if not isinstance(p, dict):
                if hasattr(p, "text") and getattr(p, "text", None):
                    text_parts.append(p.text)
                continue
            text = p.get("text") or ""
            if text:
                text_parts.append(text)
        if text_parts:
            out.append({"role": chat_role, "content": "\n".join(text_parts)})
    return out


def contents_to_ollama_messages(contents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Same as contents_to_chat_messages (Ollama uses the OpenAI message shape)."""
    return contents_to_chat_messages(contents)
