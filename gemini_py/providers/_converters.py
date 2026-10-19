"""Converters between Gemini's contents format and other providers' formats.

The agent loop stores conversation state in Gemini's contents format:

    [
        {"role": "user", "parts": [{"text": "..."}]},
        {"role": "model", "parts": [{"function_call": {...}}]},
        {"role": "user", "parts": [{"function_response": {...}}]},
        ...
    ]

For non-Gemini providers (OpenAI, Anthropic, Ollama, transformers), we
flatten this to their native chat-message format. For OpenAI-compatible
backends and HuggingFace chat templates, function_call parts become
`tool_calls` on assistant messages, and function_response parts become
'tool' role messages with `tool_call_id` + content.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List


def contents_to_text(contents: List[Dict[str, Any]]) -> str:
    """Flatten contents to a single text string.

    Used by providers that don't support multi-turn or function-calling.
    Each message is prefixed with its role: `[user] ...`, `[model] ...`.
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
    return contents_to_text(contents)


# --- Part-extraction helpers (handle both dict-style and object-style) ----


def _extract_part_text(p: Any) -> str:
    if isinstance(p, dict):
        return p.get("text", "") or ""
    return getattr(p, "text", "") or ""


def _extract_part_function_call(p: Any) -> Dict[str, Any]:
    """Extract function_call from a Part (dict or object). Returns {} if none."""
    if isinstance(p, dict):
        fc = p.get("function_call")
        if not fc:
            return {}
        if isinstance(fc, dict):
            return {
                "name": fc.get("name", ""),
                "args": fc.get("args", {}) or {},
                "id": fc.get("id"),
            }
        # fc could be an object
    fc = getattr(p, "function_call", None) if not isinstance(p, dict) else None
    if fc is None and isinstance(p, dict):
        fc = p.get("function_call")
    if fc is None:
        return {}
    return {
        "name": getattr(fc, "name", "") or (fc.get("name", "") if isinstance(fc, dict) else ""),
        "args": dict(getattr(fc, "args", {}) or {}) if not isinstance(fc, dict) else (fc.get("args", {}) or {}),
        "id": getattr(fc, "id", None) if not isinstance(fc, dict) else fc.get("id"),
    }


def _extract_part_function_response(p: Any) -> Dict[str, Any]:
    """Extract function_response from a Part (dict or object). Returns {} if none."""
    if isinstance(p, dict):
        fr = p.get("function_response")
        if not fr:
            return {}
        return {
            "name": fr.get("name", ""),
            "result": (fr.get("response") or {}).get("result", ""),
            "id": fr.get("id"),
        }
    fr = getattr(p, "function_response", None)
    if fr is None:
        return {}
    response_obj = getattr(fr, "response", None)
    if response_obj is None:
        result = ""
    elif isinstance(response_obj, dict):
        result = response_obj.get("result", "")
    else:
        result = getattr(response_obj, "result", "") or ""
    return {
        "name": getattr(fr, "name", ""),
        "result": result,
        "id": getattr(fr, "id", None),
    }


def contents_to_chat_messages(contents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert Gemini contents to OpenAI/Anthropic/HF chat-message format.

    Handles text parts, function_call parts (becomes assistant message
    with `tool_calls`), and function_response parts (becomes 'tool' role
    message with `tool_call_id` + content).

    Maps roles: 'user' -> 'user', 'model' -> 'assistant', 'system' -> 'system'.
    """
    out: List[Dict[str, Any]] = []
    for c in contents:
        if not isinstance(c, dict):
            continue
        role = c.get("role", "user")
        if role == "model":
            chat_role = "assistant"
        elif role in ("user", "system", "tool"):
            chat_role = role
        else:
            chat_role = "user"
        parts = c.get("parts", []) or []

        text_parts: List[str] = []
        tool_calls: List[Dict[str, Any]] = []
        tool_results: List[Dict[str, Any]] = []

        for p in parts:
            text = _extract_part_text(p)
            if text:
                text_parts.append(text)
                continue
            fc = _extract_part_function_call(p)
            if fc.get("name"):
                tool_calls.append({
                    "id": fc.get("id") or f"call_{len(tool_calls)}",
                    "type": "function",
                    "function": {
                        "name": fc["name"],
                        "arguments": json.dumps(fc.get("args", {}), ensure_ascii=False),
                    },
                })
                continue
            fr = _extract_part_function_response(p)
            if fr.get("name"):
                tool_results.append(fr)

        # Emit tool results as separate 'tool' messages first.
        for tr in tool_results:
            out.append({
                "role": "tool",
                "tool_call_id": tr.get("id") or "",
                "content": tr.get("result", ""),
            })

        # Then the main message (assistant with tool_calls, or user/system).
        if text_parts or tool_calls:
            msg: Dict[str, Any] = {
                "role": chat_role,
                "content": "\n".join(text_parts) if text_parts else "",
            }
            if tool_calls:
                msg["tool_calls"] = tool_calls
            out.append(msg)

    return out


def contents_to_ollama_messages(contents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Same as contents_to_chat_messages (Ollama uses the OpenAI message shape)."""
    return contents_to_chat_messages(contents)
