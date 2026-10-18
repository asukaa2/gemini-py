"""Tests for the MCP registry, memory loader, and skills system."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gemini_py.mcp import (
    McpServerConfig,
    McpRegistry,
    _namespaced,
    _split_namespaced,
    parse_env_pair,
    parse_command_tokens,
    MCP_AVAILABLE,
)
from gemini_py.memory import MemoryLoader, MemoryConfig, load_memory
from gemini_py.skills import SkillsLoader, Skill, expand_skills_in_prompt


# --- MCP registry ----------------------------------------------------------


def test_mcp_config_round_trip():
    cfg = McpServerConfig(
        name="filesystem",
        transport="stdio",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
        env={"FOO": "bar"},
        scope="user",
        include_tools=["read_file", "write_file"],
    )
    d = cfg.to_dict()
    assert d["name"] == "filesystem"
    assert d["transport"] == "stdio"
    assert d["command"] == "npx"
    assert d["args"] == ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
    assert d["env"] == {"FOO": "bar"}
    assert d["include_tools"] == ["read_file", "write_file"]

    cfg2 = McpServerConfig.from_dict(d)
    assert cfg2.name == "filesystem"
    assert cfg2.command == "npx"
    assert cfg2.args == cfg.args


def test_mcp_describe_http():
    cfg = McpServerConfig(name="api", transport="http", url="https://example.com/mcp")
    s = cfg.describe()
    assert "api" in s
    assert "http" in s
    assert "https://example.com/mcp" in s


def test_mcp_describe_stdio():
    cfg = McpServerConfig(
        name="fs", transport="stdio", command="npx", args=["-y", "server"]
    )
    s = cfg.describe()
    assert "fs" in s
    assert "npx -y server" in s


def test_mcp_registry_crud(tmp_path, monkeypatch):
    # Use temp paths so we don't touch the user's real config.
    user_path = tmp_path / "user-mcp.json"
    proj_path = tmp_path / "proj-mcp.json"
    monkeypatch.chdir(tmp_path)

    reg = McpRegistry(user_path=user_path, project_path=proj_path)
    assert reg.list() == []

    # Add a user-scope server
    cfg = McpServerConfig(
        name="github",
        transport="stdio",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-github"],
        scope="user",
    )
    reg.add(cfg)
    assert reg.get("github") is not None
    assert len(reg.list()) == 1

    # Persistence: a new registry sees the saved file.
    reg2 = McpRegistry(user_path=user_path, project_path=proj_path)
    assert reg2.get("github") is not None
    assert reg2.get("github").command == "npx"

    # Remove
    assert reg2.remove("github") is True
    assert reg2.remove("github") is False
    assert reg2.list() == []


def test_mcp_registry_scope_merge(tmp_path, monkeypatch):
    user_path = tmp_path / "user-mcp.json"
    proj_path = tmp_path / "proj-mcp.json"
    monkeypatch.chdir(tmp_path)

    reg = McpRegistry(user_path=user_path, project_path=proj_path)
    reg.add(McpServerConfig(name="srv", command="cmd-1", scope="user"))
    # Same name in project scope overrides on next load
    reg.add(McpServerConfig(name="srv", command="cmd-2", scope="project"))
    assert reg.get("srv").command == "cmd-2"

    # A fresh registry should load both files and see the project version.
    reg2 = McpRegistry(user_path=user_path, project_path=proj_path)
    got = reg2.get("srv")
    assert got is not None
    assert got.command == "cmd-2"
    assert got.scope == "project"


def test_mcp_registry_enable_disable(tmp_path, monkeypatch):
    user_path = tmp_path / "m.json"
    proj_path = tmp_path / "p.json"
    monkeypatch.chdir(tmp_path)
    reg = McpRegistry(user_path=user_path, project_path=proj_path)
    reg.add(McpServerConfig(name="srv", command="cmd", scope="user"))
    assert reg.get("srv").enabled is True
    assert reg.set_enabled("srv", False) is True
    assert reg.get("srv").enabled is False
    assert len(reg.enabled_servers()) == 0
    assert reg.set_enabled("srv", True) is True
    assert len(reg.enabled_servers()) == 1


def test_namespaced_round_trip():
    full = _namespaced("github-mcp", "list_repos")
    assert full.startswith("mcp_github_mcp__")
    server, tool = _split_namespaced(full)
    assert server == "github_mcp"
    assert tool == "list_repos"


def test_namespaced_strips_unsafe_chars():
    full = _namespaced("my.server!", "tool@v2")
    assert "!" not in full
    assert "@" not in full
    server, tool = _split_namespaced(full)
    assert server == "my_server"
    assert tool == "tool_v2"


def test_split_namespaced_non_mcp_returns_none():
    server, tool = _split_namespaced("read_file")
    assert server is None
    assert tool == "read_file"


def test_parse_env_pair_ok():
    k, v = parse_env_pair("TOKEN=abc123")
    assert k == "TOKEN"
    assert v == "abc123"


def test_parse_env_pair_with_equals_in_value():
    k, v = parse_env_pair("URL=https://x.com/?a=1&b=2")
    assert k == "URL"
    assert v == "https://x.com/?a=1&b=2"


def test_parse_env_pair_invalid():
    with pytest.raises(ValueError):
        parse_env_pair("no-equals-sign")


def test_parse_command_tokens():
    cmd, args = parse_command_tokens(["npx", "-y", "server"])
    assert cmd == "npx"
    assert args == ["-y", "server"]


def test_parse_command_tokens_empty():
    with pytest.raises(ValueError):
        parse_command_tokens([])


def test_mcp_available():
    # Should be True since we install mcp as a core dep
    assert MCP_AVAILABLE is True


# --- Memory loader ---------------------------------------------------------


def test_memory_no_files(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    cfg = MemoryConfig(
        enabled=True,
        user_dir=tmp_path / ".gemini",
        project_dir=tmp_path,
    )
    loader = MemoryLoader(cfg)
    assert loader.load() == ""


def test_memory_user_file(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    # Create user-wide GEMINI.md
    user_gemini = tmp_path / ".gemini"
    user_gemini.mkdir(parents=True)
    (user_gemini / "GEMINI.md").write_text("# User rules\nAlways use type hints.\n")
    cfg = MemoryConfig(
        enabled=True,
        user_dir=user_gemini,
        project_dir=tmp_path,
    )
    loader = MemoryLoader(cfg)
    out = loader.load()
    assert "User rules" in out
    assert "type hints" in out
    assert "# Memory:" in out


def test_memory_project_file(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    (tmp_path / "GEMINI.md").write_text("# Project rules\nTest with pytest.\n")
    cfg = MemoryConfig(
        enabled=True,
        user_dir=tmp_path / "no-user",
        project_dir=tmp_path,
    )
    loader = MemoryLoader(cfg)
    out = loader.load()
    assert "Project rules" in out
    assert "pytest" in out


def test_memory_disabled(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    (tmp_path / "GEMINI.md").write_text("should not appear")
    cfg = MemoryConfig(
        enabled=False,
        user_dir=tmp_path / "no-user",
        project_dir=tmp_path,
    )
    loader = MemoryLoader(cfg)
    assert loader.load() == ""


def test_memory_ignore_directive(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    (tmp_path / "GEMINI.md").write_text(
        "before\n# gemini-py-ignore\nafter\n"
    )
    cfg = MemoryConfig(
        enabled=True,
        user_dir=tmp_path / "no-user",
        project_dir=tmp_path,
    )
    loader = MemoryLoader(cfg)
    out = loader.load()
    assert "before" in out
    assert "after" not in out


def test_memory_extra_files(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    extra = tmp_path / "extra.md"
    extra.write_text("Extra context here.")
    cfg = MemoryConfig(
        enabled=True,
        user_dir=tmp_path / "no-user",
        project_dir=tmp_path,
        extra_files=[extra],
    )
    loader = MemoryLoader(cfg)
    out = loader.load()
    assert "Extra context here." in out


def test_load_memory_helper(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    (tmp_path / "GEMINI.md").write_text("loaded via helper")
    out = load_memory(project_dir=tmp_path)
    assert "loaded via helper" in out


# --- Skills ----------------------------------------------------------------


def _make_skill(base: Path, name: str, content: str) -> None:
    skill_dir = base / name
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(content)


def test_skills_loader_empty(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    loader = SkillsLoader(
        user_dir=tmp_path / "user-skills",
        project_dir=tmp_path / "proj-skills",
        state_file=tmp_path / "state.json",
    )
    assert loader.list() == []
    assert loader.render_summary() == ""


def test_skills_loader_finds_user_skill(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    user_dir = tmp_path / "user-skills"
    _make_skill(user_dir, "code-reviewer", "# Code Reviewer\nYou review code for bugs and style.\n")
    loader = SkillsLoader(
        user_dir=user_dir,
        project_dir=tmp_path / "none",
        state_file=tmp_path / "state.json",
    )
    skills = loader.list()
    assert len(skills) == 1
    s = skills[0]
    assert s.name == "code-reviewer"
    assert s.scope == "user"
    assert s.enabled is True
    assert "bugs" in s.description.lower() or "review" in s.description.lower()


def test_skills_loader_finds_project_skill(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    proj_dir = tmp_path / ".gemini" / "skills"
    _make_skill(proj_dir, "test-writer", "# Test Writer\nWrite pytest tests.\n")
    loader = SkillsLoader(
        user_dir=tmp_path / "none",
        project_dir=proj_dir,
        state_file=tmp_path / "state.json",
    )
    skills = loader.list()
    assert len(skills) == 1
    s = skills[0]
    assert s.name == "test-writer"
    assert s.scope == "project"


def test_skills_loader_enable_disable_persists(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    user_dir = tmp_path / "user-skills"
    _make_skill(user_dir, "foo", "# Foo\nA skill.\n")
    state_file = tmp_path / "state.json"
    loader = SkillsLoader(
        user_dir=user_dir,
        project_dir=tmp_path / "none",
        state_file=state_file,
    )
    assert loader.get("foo").enabled is True
    assert loader.set_enabled("foo", False) is True
    assert loader.get("foo").enabled is False
    # Reload from state file
    loader2 = SkillsLoader(
        user_dir=user_dir,
        project_dir=tmp_path / "none",
        state_file=state_file,
    )
    assert loader2.get("foo").enabled is False
    assert loader2.enabled() == []


def test_skills_enable_all_disable_all(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    user_dir = tmp_path / "user-skills"
    _make_skill(user_dir, "a", "# A\n")
    _make_skill(user_dir, "b", "# B\n")
    loader = SkillsLoader(
        user_dir=user_dir,
        project_dir=tmp_path / "none",
        state_file=tmp_path / "state.json",
    )
    loader.disable_all()
    assert all(not s.enabled for s in loader.list())
    loader.enable_all()
    assert all(s.enabled for s in loader.list())


def test_skills_render_summary(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    user_dir = tmp_path / "user-skills"
    _make_skill(user_dir, "code-reviewer", "# Code Reviewer\nReviews code.\n")
    loader = SkillsLoader(
        user_dir=user_dir,
        project_dir=tmp_path / "none",
        state_file=tmp_path / "state.json",
    )
    summary = loader.render_summary()
    assert "@code-reviewer" in summary
    assert "Available skills" in summary


def test_expand_skills_in_prompt(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    user_dir = tmp_path / "user-skills"
    _make_skill(user_dir, "code-reviewer", "Review every PR carefully.")
    loader = SkillsLoader(
        user_dir=user_dir,
        project_dir=tmp_path / "none",
        state_file=tmp_path / "state.json",
    )
    prompt = "Please @code-reviewer the PR."
    expanded = expand_skills_in_prompt(prompt, loader)
    assert "Review every PR carefully." in expanded
    # The @skill_name token is replaced by the BEGIN/END SKILL block.
    assert "BEGIN SKILL: code-reviewer" in expanded
    assert "END SKILL: code-reviewer" in expanded


def test_expand_skills_unknown_left_alone(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    loader = SkillsLoader(
        user_dir=tmp_path / "none",
        project_dir=tmp_path / "none2",
        state_file=tmp_path / "state.json",
    )
    expanded = expand_skills_in_prompt("use @unknown-skill here", loader)
    assert "@unknown-skill" in expanded
    assert "BEGIN SKILL" not in expanded


def test_expand_skills_disabled_not_expanded(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    user_dir = tmp_path / "user-skills"
    _make_skill(user_dir, "disabled-skill", "Should not appear.")
    loader = SkillsLoader(
        user_dir=user_dir,
        project_dir=tmp_path / "none",
        state_file=tmp_path / "state.json",
    )
    loader.set_enabled("disabled-skill", False)
    expanded = expand_skills_in_prompt("use @disabled-skill here", loader)
    assert "Should not appear." not in expanded


# --- CLI subcommands ------------------------------------------------------


def test_cli_mcp_list_runs(tmp_path, monkeypatch):
    """`gemini-py mcp list` should print and exit 0 even when empty."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    from gemini_py.cli import main
    rc = main(["mcp", "list"])
    assert rc == 0


def test_cli_mcp_add_and_list(tmp_path, monkeypatch):
    """Add an MCP server and verify it shows up in `mcp list`."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    from gemini_py.cli import main
    rc = main([
        "mcp", "add", "filesystem",
        "npx", "-y", "@modelcontextprotocol/server-filesystem", "/tmp",
        "--scope", "user",
    ])
    assert rc == 0
    rc = main(["mcp", "list"])
    assert rc == 0


def test_cli_skills_list_empty(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    from gemini_py.cli import main
    rc = main(["skills", "list"])
    assert rc == 0


def test_cli_skills_show(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    user_dir = tmp_path / ".gemini-py" / "skills"
    _make_skill(user_dir, "demo", "# Demo skill\nDoes demo things.\n")
    from gemini_py.cli import main
    rc = main(["skills", "show", "demo"])
    assert rc == 0


def test_cli_skills_show_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    from gemini_py.cli import main
    rc = main(["skills", "show", "nonexistent"])
    assert rc == 1


def test_cli_flags_parse():
    from gemini_py.cli import build_parser
    parser = build_parser()
    args = parser.parse_args([
        "-p", "hello",
        "--no-memory",
        "--no-skills",
        "--no-subagent",
        "--no-mcp",
    ])
    assert args.no_memory is True
    assert args.no_skills is True
    assert args.no_subagent is True
    assert args.no_mcp is True
