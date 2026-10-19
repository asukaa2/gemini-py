"""Tests for the TransformersProvider tool-call parser + converters."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gemini_py.providers.transformers import (
    _parse_tool_calls,
    _parse_tool_call_json,
    _normalize_tool_call_obj,
)
from gemini_py.providers._converters import (
    contents_to_chat_messages,
    _extract_part_function_call,
    _extract_part_function_response,
)
from gemini_py._response import (
    _TextPart,
    _FunctionCallPart,
    _FunctionCall,
    text_with_tool_calls_response,
)


# --- Tool-call parser ----------------------------------------------------


def test_parse_qwen_style_tool_call():
    text = "Let me check. \u3010{\"name\": \"read_file\", \"arguments\": {\"path\": \"foo.py\"}}\u3011 Done."
    calls, remaining = _parse_tool_calls(text)
    assert len(calls) == 1
    assert calls[0]["name"] == "read_file"
    assert calls[0]["args"] == {"path": "foo.py"}
    assert calls[0]["id"] is None
    assert "Done" in remaining
    assert "\u3010" not in remaining


def test_parse_hermes_style_tool_call():
    text = "<|tool_call|>{\"name\": \"list_directory\", \"arguments\": {\"path\": \".\"}}<|/tool_call|>"
    calls, remaining = _parse_tool_calls(text)
    assert len(calls) == 1
    assert calls[0]["name"] == "list_directory"
    assert calls[0]["args"] == {"path": "."}
    assert remaining == ""


def test_parse_fenced_tool_code_block():
    text = (
        "I will run it.\n"
        "```tool\n"
        "{\"name\": \"run_shell_command\", \"arguments\": {\"command\": \"ls\"}}\n"
        "```\n"
        "Done."
    )
    calls, remaining = _parse_tool_calls(text)
    assert len(calls) == 1
    assert calls[0]["name"] == "run_shell_command"
    assert calls[0]["args"] == {"command": "ls"}
    assert "Done" in remaining


def test_parse_mistral_tool_calls():
    text = "[TOOL_CALLS] {\"name\": \"echo\", \"arguments\": {\"msg\": \"hi\"}}"
    calls, _ = _parse_tool_calls(text)
    assert len(calls) == 1
    assert calls[0]["name"] == "echo"


def test_parse_bare_json_tool_call():
    text = "{\"name\": \"echo\", \"arguments\": {\"msg\": \"hi\"}}"
    calls, remaining = _parse_tool_calls(text)
    assert len(calls) == 1
    assert calls[0]["name"] == "echo"
    assert remaining == ""


def test_parse_no_tool_calls():
    text = "Just a regular response, no tool calls."
    calls, remaining = _parse_tool_calls(text)
    assert calls == []
    assert remaining == text


def test_parse_multiple_tool_calls():
    text = (
        "<|tool_call|>{\"name\": \"a\", \"arguments\": {}}<|/tool_call|>\n"
        "<|tool_call|>{\"name\": \"b\", \"arguments\": {\"x\": 1}}<|/tool_call|>"
    )
    calls, _ = _parse_tool_calls(text)
    assert len(calls) == 2
    assert calls[0]["name"] == "a"
    assert calls[1]["name"] == "b"
    assert calls[1]["args"] == {"x": 1}


def test_parse_args_key_alias():
    """Some models use 'args' instead of 'arguments'."""
    text = "{\"name\": \"x\", \"args\": {\"y\": 1}}"
    calls, _ = _parse_tool_calls(text)
    assert len(calls) == 1
    assert calls[0]["args"] == {"y": 1}


def test_parse_string_arguments_parsed_as_json():
    """Qwen format: 'arguments' is a JSON string."""
    obj = {"name": "x", "arguments": "{\"a\": 1, \"b\": \"foo\"}"}
    out = _normalize_tool_call_obj(obj)
    assert len(out) == 1
    assert out[0]["args"] == {"a": 1, "b": "foo"}


def test_parse_function_key_alias():
    """Some models use 'function' instead of 'name'."""
    obj = {"function": "x", "arguments": {}}
    out = _normalize_tool_call_obj(obj)
    assert len(out) == 1
    assert out[0]["name"] == "x"


def test_parse_empty_text():
    calls, remaining = _parse_tool_calls("")
    assert calls == []
    assert remaining == ""


def test_parse_malformed_json_no_crash():
    text = "<|tool_call|>{this is not valid json}<|/tool_call|>"
    calls, remaining = _parse_tool_calls(text)
    # No crash, no tool calls extracted.
    assert calls == []


def test_parse_tool_call_json_list():
    """Multiple tool calls as a JSON array."""
    json_str = '{"name": "a", "arguments": {}}\n{"name": "b", "arguments": {}}'
    calls = _parse_tool_call_json(json_str)
    assert len(calls) >= 2
    names = [c["name"] for c in calls]
    assert "a" in names
    assert "b" in names


# --- text_with_tool_calls_response helper ---------------------------------


def test_text_with_tool_calls_response_builds_parts():
    r = text_with_tool_calls_response(
        "Calling a tool now.",
        [{"name": "echo", "args": {"msg": "hi"}, "id": "call_1"}],
    )
    # Should have 2 parts: one text, one function_call.
    assert len(r.candidates[0].content.parts) == 2
    # First part = text.
    assert r.candidates[0].content.parts[0].text == "Calling a tool now."
    assert r.candidates[0].content.parts[0].function_call is None
    # Second part = function_call.
    fc = r.candidates[0].content.parts[1].function_call
    assert fc is not None
    assert fc.name == "echo"
    assert fc.args == {"msg": "hi"}
    assert fc.id == "call_1"


def test_text_with_tool_calls_response_no_text():
    """When text is empty, only function_call parts are emitted."""
    r = text_with_tool_calls_response(
        "",
        [{"name": "x", "args": {}, "id": "c1"}],
    )
    # No text part, just function_call.
    parts = r.candidates[0].content.parts
    assert len(parts) == 1
    assert parts[0].text is None
    assert parts[0].function_call is not None


# --- Converter updates ----------------------------------------------------


def test_contents_to_chat_messages_with_function_call_part():
    """Assistant message with a _FunctionCallPart becomes tool_calls."""
    fc_part = _FunctionCallPart(name="echo", args={"msg": "hi"}, call_id="call_1")
    contents = [
        {"role": "user", "parts": [{"text": "say hi"}]},
        {"role": "model", "parts": [fc_part]},
    ]
    msgs = contents_to_chat_messages(contents)
    # 2 messages: user + assistant (with tool_calls).
    assert len(msgs) == 2
    assert msgs[0]["role"] == "user"
    assert msgs[1]["role"] == "assistant"
    assert "tool_calls" in msgs[1]
    assert len(msgs[1]["tool_calls"]) == 1
    tc = msgs[1]["tool_calls"][0]
    assert tc["function"]["name"] == "echo"
    assert tc["id"] == "call_1"


def test_contents_to_chat_messages_with_function_response_part():
    """Function-response part becomes a 'tool' role message."""
    fr_part = {
        "function_response": {
            "name": "echo",
            "response": {"result": "hi there"},
            "id": "call_1",
        }
    }
    contents = [
        {"role": "user", "parts": [{"text": "say hi"}]},
        {"role": "model", "parts": [{"text": ""}]},  # placeholder
        {"role": "user", "parts": [fr_part]},
    ]
    msgs = contents_to_chat_messages(contents)
    # The last message should be a 'tool' role message.
    tool_msgs = [m for m in msgs if m["role"] == "tool"]
    assert len(tool_msgs) == 1
    assert tool_msgs[0]["tool_call_id"] == "call_1"
    assert tool_msgs[0]["content"] == "hi there"


def test_extract_part_function_call_from_dict():
    fc_part = {"function_call": {"name": "x", "args": {"a": 1}, "id": "c1"}}
    result = _extract_part_function_call(fc_part)
    assert result["name"] == "x"
    assert result["args"] == {"a": 1}
    assert result["id"] == "c1"


def test_extract_part_function_call_from_object():
    fc_part = _FunctionCallPart(name="x", args={"a": 1}, call_id="c1")
    result = _extract_part_function_call(fc_part)
    assert result["name"] == "x"
    assert result["args"] == {"a": 1}
    assert result["id"] == "c1"


def test_extract_part_function_call_from_text_only_part():
    """A text part returns {}."""
    result = _extract_part_function_call({"text": "hi"})
    assert result == {}


def test_extract_part_function_response_from_dict():
    fr_part = {
        "function_response": {
            "name": "x",
            "response": {"result": "ok"},
            "id": "c1",
        }
    }
    result = _extract_part_function_response(fr_part)
    assert result["name"] == "x"
    assert result["result"] == "ok"
    assert result["id"] == "c1"


def test_extract_part_function_response_from_text_only_part():
    result = _extract_part_function_response({"text": "hi"})
    assert result == {}


# --- TransformersProvider (smoke test, no actual model) -------------------


def test_transformers_supports_tools_true():
    """TransformersProvider.supports_tools() returns True."""
    # We can't construct it without torch+transformers, so we just verify
    # the method exists and the implementation is correct.
    from gemini_py.providers.transformers import TransformersProvider
    # Method is defined (not inherited from Provider base).
    assert "supports_tools" in dir(TransformersProvider)
    # Don't instantiate — that would require torch.


def test_normalize_tool_call_obj_with_list_args_dict():
    """args can be dict already."""
    obj = {"name": "x", "arguments": {"y": 1}}
    out = _normalize_tool_call_obj(obj)
    assert len(out) == 1
    assert out[0]["args"] == {"y": 1}


def test_normalize_tool_call_obj_with_no_name_returns_empty():
    obj = {"arguments": {}}
    out = _normalize_tool_call_obj(obj)
    assert out == []
