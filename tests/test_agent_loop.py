"""Tests for the agent-loop fixes (id linkage + loop detection)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gemini_py.tools import (
    build_default_tools,
    execute_tool_call,
    find_tool,
)


def test_execute_tool_call_forwards_id_when_present():
    """The id field from the function_call must be forwarded to the
    function_response so the Gemini API can link request ↔ response.

    Missing linkage is a common cause of agent loops where the model
    re-issues the same call every turn (since it can't tell the call was
    already executed).
    """
    tools = build_default_tools(enable_shell=True, enable_filesystem=True)
    fc = {
        "name": "run_shell_command",
        "args": {"command": "echo hi"},
        "id": "call_abc123",
    }
    result = execute_tool_call(tools, fc)
    fr = result["function_response"]
    assert fr["name"] == "run_shell_command"
    assert fr["id"] == "call_abc123"
    assert "hi" in fr["response"]["result"]


def test_execute_tool_call_omits_id_when_absent():
    """If the call has no id (older models), the response should not
    include the field at all — a None id would be invalid."""
    tools = build_default_tools(enable_shell=True, enable_filesystem=True)
    fc = {"name": "run_shell_command", "args": {"command": "echo hi"}}
    result = execute_tool_call(tools, fc)
    fr = result["function_response"]
    assert "id" not in fr
    assert "hi" in fr["response"]["result"]


def test_execute_tool_call_forwards_id_on_error_too():
    """Even on tool-execution errors, forward the id so the model can
    recover instead of re-issuing."""
    tools = build_default_tools()
    fc = {"name": "nonexistent", "args": {}, "id": "call_xyz"}
    result = execute_tool_call(tools, fc)
    fr = result["function_response"]
    assert fr["id"] == "call_xyz"
    assert "unknown tool" in fr["response"]["result"]


def test_cli_default_max_iterations_is_25():
    """Bumped from 10 -> 25 to give the agent more room before falling
    back to the 'max iterations reached' diagnostic."""
    from gemini_py.cli import build_parser
    parser = build_parser()
    args = parser.parse_args(["-p", "hello"])
    assert args.max_iterations == 25


def test_cli_max_iterations_override():
    from gemini_py.cli import build_parser
    parser = build_parser()
    args = parser.parse_args(["-p", "hello", "--max-iterations", "50"])
    assert args.max_iterations == 50


# --- Loop detection (smoke test of the fingerprint logic) ------------------


def test_loop_fingerprint_logic():
    """Verify the fingerprinting approach used by the loop detector."""
    import json
    from collections import Counter

    def fingerprint(fc):
        return f"{fc['name']}:{json.dumps(fc.get('args') or {}, sort_keys=True)}"

    # Simulate 3 iterations where the model calls the same tool with
    # identical args every time.
    call = {"name": "run_shell_command", "args": {"command": "ls"}}
    fps = [fingerprint(call)] * 3

    counts = Counter(fps)
    looped = [fp for fp, n in counts.items() if n >= 3]
    assert len(looped) == 1
    assert looped[0].startswith("run_shell_command:")


def test_loop_fingerprint_distinguishes_different_args():
    """Same tool name but different args should NOT count as a loop."""
    import json
    from collections import Counter

    def fingerprint(fc):
        return f"{fc['name']}:{json.dumps(fc.get('args') or {}, sort_keys=True)}"

    fps = [
        fingerprint({"name": "run_shell_command", "args": {"command": "ls"}}),
        fingerprint({"name": "run_shell_command", "args": {"command": "pwd"}}),
        fingerprint({"name": "run_shell_command", "args": {"command": "ls"}}),
    ]
    counts = Counter(fps)
    looped = [fp for fp, n in counts.items() if n >= 3]
    assert looped == []
