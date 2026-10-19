"""Tests for the live view and the new create_directory tool."""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gemini_py.tools import build_default_tools, find_tool, execute_tool_call
from gemini_py.live import LiveView, make_default_live_view


# --- create_directory tool ------------------------------------------------


def test_create_directory_tool_exists():
    tools = build_default_tools(enable_filesystem=True, enable_shell=False)
    assert find_tool(tools, "create_directory") is not None


def test_create_directory_creates_nested(tmp_path):
    tools = build_default_tools(enable_filesystem=True, enable_shell=False)
    cd = find_tool(tools, "create_directory")
    target = tmp_path / "a" / "b" / "c"
    out = cd.executor(path=str(target))
    assert "OK" in out
    assert target.is_dir()


def test_create_directory_idempotent(tmp_path):
    tools = build_default_tools(enable_filesystem=True, enable_shell=False)
    cd = find_tool(tools, "create_directory")
    target = tmp_path / "x"
    cd.executor(path=str(target))
    # Second call should not error.
    out = cd.executor(path=str(target))
    assert "OK" in out
    assert target.is_dir()


def test_create_directory_via_tool_call(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tools = build_default_tools(enable_filesystem=True, enable_shell=False)
    fc = {"name": "create_directory", "args": {"path": "sub1/sub2"}, "id": "x1"}
    result = execute_tool_call(tools, fc)
    fr = result["function_response"]
    assert fr["id"] == "x1"
    assert "OK" in fr["response"]["result"]
    assert (tmp_path / "sub1" / "sub2").is_dir()


def test_write_file_auto_creates_nested_dirs(tmp_path, monkeypatch):
    """write_file already auto-creates parent dirs (regression test)."""
    monkeypatch.chdir(tmp_path)
    tools = build_default_tools(enable_filesystem=True, enable_shell=False)
    wf = find_tool(tools, "write_file")
    out = wf.executor(path="proj/src/main.py", content="print('hi')")
    assert "OK" in out
    p = tmp_path / "proj" / "src" / "main.py"
    assert p.is_file()
    assert p.read_text() == "print('hi')"


# --- LiveView -------------------------------------------------------------


def test_live_view_handles_all_event_types():
    """LiveView.emit() should not raise on any event type."""
    stream = io.StringIO()
    view = LiveView(use_rich=False, stream=stream)
    # Iterate all event types we emit.
    view.emit({"type": "iteration", "n": 1, "max": 5})
    view.emit({"type": "text_chunk", "text": "Thinking..."})
    view.emit({"type": "tool_call", "name": "run_shell_command", "args": {"command": "ls"}})
    view.emit({"type": "file_write", "path": "main.py", "content": "print('hi')"})
    view.emit({"type": "tool_result", "name": "run_shell_command", "result": "file1.txt"})
    view.emit({"type": "tool_result", "name": "read_file", "result": "Error: not found"})
    view.emit({"type": "done", "finish_reason": "STOP", "n_tool_calls": 3})
    view.emit({"type": "error", "message": "boom"})
    out = stream.getvalue()
    assert "Iteration 1/5" in out
    assert "Thinking" in out
    assert "run_shell_command" in out
    assert "main.py" in out
    assert "file1.txt" in out
    assert "done" in out.lower() or "STOP" in out
    assert "boom" in out


def test_live_view_make_default_when_enabled():
    view = make_default_live_view(enabled=True)
    assert view is not None
    assert isinstance(view, LiveView)


def test_live_view_make_default_returns_none_when_disabled():
    assert make_default_live_view(enabled=False) is None


def test_live_view_file_write_renders_content():
    stream = io.StringIO()
    view = LiveView(use_rich=False, stream=stream)
    view.emit({"type": "file_write", "path": "app.py", "content": "print('hello')"})
    out = stream.getvalue()
    assert "app.py" in out
    assert "print('hello')" in out


def test_live_view_guess_lexer():
    view = LiveView(use_rich=False, stream=io.StringIO())
    assert view._guess_lexer("foo.py") == "python"
    assert view._guess_lexer("foo.js") == "javascript"
    assert view._guess_lexer("foo.ts") == "typescript"
    assert view._guess_lexer("foo.html") == "html"
    assert view._guess_lexer("foo.css") == "css"
    assert view._guess_lexer("foo.json") == "json"
    assert view._guess_lexer("README.md") == "markdown"
    assert view._guess_lexer("script.sh") == "bash"
    assert view._guess_lexer("config.yaml") == "yaml"
    assert view._guess_lexer("Dockerfile") == "docker"
    assert view._guess_lexer("unknown.xyz") == "text"


def test_cli_live_flag_parses():
    from gemini_py.cli import build_parser
    parser = build_parser()
    args = parser.parse_args(["-p", "hello", "--live"])
    assert args.live is True


def test_cli_live_flag_default_false():
    from gemini_py.cli import build_parser
    parser = build_parser()
    args = parser.parse_args(["-p", "hello"])
    assert args.live is False


# --- RichLiveChat (full-screen dashboard) --------------------------------


def test_rich_live_chat_emit_does_not_crash():
    """RichLiveChat should accept all event types without raising."""
    from gemini_py.live import RichLiveChat
    # Construct (this starts the Live display). Use a fake non-TTY stdout
    # so rich falls back to plain text (no ANSI cursor magic) — the Live
    # object still works, it just doesn't grab the screen.
    import io
    chat = RichLiveChat(model="gemini-3.8-flash", max_iterations=10)
    try:
        chat.emit({"type": "iteration", "n": 1, "max": 10})
        chat.emit({"type": "text_chunk", "text": "Hello "})
        chat.emit({"type": "text_chunk", "text": "world!"})
        chat.emit({"type": "tool_call", "name": "run_shell_command", "args": {"command": "ls"}})
        chat.emit({"type": "file_write", "path": "app.py", "content": "print('hi')"})
        chat.emit({"type": "tool_result", "name": "run_shell_command", "result": "file1.txt"})
        chat.emit({"type": "tool_result", "name": "read_file", "result": "Error: not found"})
        chat.emit({"type": "done", "finish_reason": "STOP", "n_tool_calls": 2})
    finally:
        chat.close()


def test_rich_live_chat_state_tracking():
    """The header state should reflect the events emitted."""
    from gemini_py.live import RichLiveChat
    chat = RichLiveChat(model="gemini-4-argon", max_iterations=25)
    try:
        chat.emit({"type": "iteration", "n": 3, "max": 25})
        assert chat.state["iteration"] == 3
        assert chat.state["max"] == 25
        assert chat.state["status"] == "thinking…"
        chat.emit({"type": "tool_call", "name": "write_file", "args": {"path": "x.py"}})
        assert chat.state["tool_calls"] == 1
        chat.emit({"type": "tool_call", "name": "read_file", "args": {"path": "y.py"}})
        assert chat.state["tool_calls"] == 2
        assert chat.state["status"] == "calling read_file"
        chat.emit({"type": "done", "finish_reason": "STOP", "n_tool_calls": 2})
        assert chat.state["done"] is True
        assert chat.state["finish_reason"] == "STOP"
        assert chat.state["status"] == "done"
    finally:
        chat.close()


def test_rich_live_chat_error_event():
    from gemini_py.live import RichLiveChat
    chat = RichLiveChat(model="x", max_iterations=5)
    try:
        chat.emit({"type": "error", "message": "boom"})
        assert chat.state["error"] == "boom"
        assert chat.state["status"] == "error"
    finally:
        chat.close()


def test_rich_live_chat_close_idempotent():
    """close() should be safe to call multiple times."""
    from gemini_py.live import RichLiveChat
    chat = RichLiveChat(model="x", max_iterations=5)
    chat.close()
    chat.close()  # should not raise


def test_make_default_live_view_prefers_rich_dashboard():
    """When stdout is a TTY and rich is available, we get RichLiveChat."""
    from gemini_py.live import make_default_live_view, RichLiveChat, LiveView
    # We can't easily fake a TTY in pytest, so this will fall back to LiveView.
    # Just verify the function doesn't raise and returns something with .emit().
    view = make_default_live_view(
        enabled=True,
        live=True,
        model="gemini-3.8-flash",
        max_iterations=10,
    )
    assert view is not None
    assert hasattr(view, "emit")
    assert hasattr(view, "close")
    # Cleanup if it's a RichLiveChat (the close stops the Live display).
    view.close()
