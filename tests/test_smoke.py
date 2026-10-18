"""Smoke tests for gemini_py — runnable without an API key."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gemini_py.auth import AuthConfig, AuthMethod
from gemini_py.config import GeminiConfig, ModelAlias
from gemini_py.models import Message, MessageRole
from gemini_py.tools import build_default_tools, execute_tool_call, find_tool
from gemini_py.tools.shell import run_shell_command
from gemini_py.sessions import SessionStore
from gemini_py.output import OutputFormat, format_result
from gemini_py import __version__


def test_version():
    assert __version__ == "0.1.0"


def test_model_alias_resolves():
    assert ModelAlias.AUTO.resolve() == "gemini-2.5-pro"
    assert ModelAlias.PRO.resolve() == "gemini-2.5-pro"
    assert ModelAlias.FLASH.resolve() == "gemini-2.5-flash"
    assert ModelAlias.FLASH_LITE.resolve() == "gemini-2.5-flash-lite"
    assert ModelAlias.is_alias("flash") is True
    assert ModelAlias.is_alias("gemini-2.5-pro") is False


def test_config_round_trip():
    cfg = GeminiConfig(model="flash", temperature=0.5, max_output_tokens=1000)
    d = cfg.to_dict()
    assert d["model"] == "flash"
    assert d["temperature"] == 0.5
    cfg2 = GeminiConfig.from_dict(d)
    assert cfg2.model == "flash"
    assert cfg2.temperature == 0.5
    assert cfg2.resolve_model() == "gemini-2.5-flash"


def test_message_round_trip():
    m = Message(role=MessageRole.USER, content="Hello")
    d = m.to_dict()
    assert d["role"] == "user"
    assert d["parts"][0]["text"] == "Hello"
    m2 = Message.from_dict(d)
    assert m2.role == MessageRole.USER
    assert m2.content == "Hello"


def test_auth_resolution(monkeypatch):
    # Clear any leaked env vars
    for v in (
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "GOOGLE_CLOUD_PROJECT",
        "GOOGLE_CLOUD_LOCATION",
    ):
        monkeypatch.delenv(v, raising=False)

    # No creds -> auto resolves to vertex_adc (no project, would error on build)
    cfg = AuthConfig.from_env(load_dotenv_files=False)
    assert cfg.method == AuthMethod.VERTEX_AI_ADC

    # With GEMINI_API_KEY
    monkeypatch.setenv("GEMINI_API_KEY", "test_key_123")
    cfg = AuthConfig.from_env(load_dotenv_files=False)
    assert cfg.method == AuthMethod.GEMINI_API_KEY
    assert cfg.api_key == "test_key_123"
    kwargs = cfg.to_client_kwargs()
    assert kwargs == {"api_key": "test_key_123"}

    # With GOOGLE_API_KEY + project -> Vertex
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "google_key_456")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "my-proj")
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "us-central1")
    cfg = AuthConfig.from_env(load_dotenv_files=False)
    assert cfg.method == AuthMethod.GOOGLE_API_KEY
    kwargs = cfg.to_client_kwargs()
    assert kwargs["api_key"] == "google_key_456"
    assert kwargs["vertexai"] is True
    assert kwargs["project"] == "my-proj"
    assert kwargs["location"] == "us-central1"


def test_shell_tool_runs_simple_command():
    out = run_shell_command("echo hello_world")
    assert "hello_world" in out
    assert "[exit=0]" in out


def test_shell_tool_handles_failure():
    out = run_shell_command("exit 7")
    assert "[exit=7]" in out


def test_tools_registry_contains_expected():
    tools = build_default_tools(enable_shell=True, enable_filesystem=True, enable_web=True)
    names = [t.name for t in tools]
    assert "list_directory" in names
    assert "read_file" in names
    assert "write_file" in names
    assert "grep_search" in names
    assert "run_shell_command" in names
    assert "web_search" in names
    assert "web_fetch" in names


def test_tool_call_execution_for_shell():
    tools = build_default_tools(enable_shell=True, enable_filesystem=True)
    res = execute_tool_call(tools, {"name": "run_shell_command", "args": {"command": "echo abc"}})
    assert res["function_response"]["name"] == "run_shell_command"
    assert "abc" in res["function_response"]["response"]["result"]


def test_tool_call_unknown_tool():
    tools = build_default_tools()
    res = execute_tool_call(tools, {"name": "nonexistent", "args": {}})
    assert "unknown tool" in res["function_response"]["response"]["result"]


def test_filesystem_tools(tmp_path):
    tools = build_default_tools(enable_filesystem=True, enable_shell=False)
    # write_file
    write_tool = find_tool(tools, "write_file")
    out = write_tool.executor(path=str(tmp_path / "hello.txt"), content="hi there")
    assert "OK" in out
    # read_file
    read_tool = find_tool(tools, "read_file")
    out = read_tool.executor(path=str(tmp_path / "hello.txt"))
    assert out == "hi there"
    # list_directory
    list_tool = find_tool(tools, "list_directory")
    out = list_tool.executor(path=str(tmp_path))
    assert "hello.txt" in out
    # grep_search
    grep_tool = find_tool(tools, "grep_search")
    out = grep_tool.executor(pattern="hi", path=str(tmp_path))
    assert "hi there" in out


def test_session_store_crud(tmp_path, monkeypatch):
    # Use a temp dir for sessions
    monkeypatch.setenv("HOME", str(tmp_path))
    store = SessionStore(base_dir=tmp_path / "sessions")
    rec = store.create(model="gemini-2.5-pro")
    assert rec.session_id
    got = store.get(rec.session_id)
    assert got is not None
    assert got.session_id == rec.session_id
    # Append a message
    store.append_message(rec.session_id, Message(role=MessageRole.USER, content="hi"))
    got = store.get(rec.session_id)
    assert len(got.messages) == 1
    # List
    all_recs = store.list_all()
    assert len(all_recs) == 1
    # Latest
    latest = store.latest()
    assert latest.session_id == rec.session_id
    # Delete
    assert store.delete(rec.session_id) is True
    assert store.get(rec.session_id) is None


def test_format_result_text():
    from gemini_py.models import GenerateResult, UsageStats
    r = GenerateResult(text="hello", usage=UsageStats(prompt_token_count=1, candidates_token_count=2, total_token_count=3), model="gemini-2.5-pro")
    assert format_result(r, OutputFormat.TEXT) == "hello"


def test_format_result_json():
    import json
    from gemini_py.models import GenerateResult, UsageStats
    r = GenerateResult(text="hello", usage=UsageStats(total_token_count=3), model="gemini-2.5-pro")
    out = json.loads(format_result(r, OutputFormat.JSON))
    assert out["response"] == "hello"
    assert out["stats"]["total_token_count"] == 3


def test_format_result_stream_json():
    import json
    from gemini_py.models import GenerateResult, UsageStats
    r = GenerateResult(text="hello", usage=UsageStats(total_token_count=3), model="gemini-2.5-pro")
    out = format_result(r, OutputFormat.STREAM_JSON)
    lines = [json.loads(l) for l in out.strip().split("\n") if l]
    assert lines[0]["type"] == "init"
    assert lines[1]["type"] == "result"
    assert lines[1]["response"] == "hello"


def test_cli_parser_help_runs():
    from gemini_py.cli import build_parser
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--help"])


def test_cli_parser_flags_parse():
    from gemini_py.cli import build_parser
    parser = build_parser()
    args = parser.parse_args(["-p", "hello world", "--model", "flash", "--output-format", "json"])
    assert args.prompt == "hello world"
    assert args.model == "flash"
    assert args.output_format == "json"
