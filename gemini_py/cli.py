"""gemini-py command-line interface.

Mirrors the gemini-cli surface area using argparse (no GUI, no TUI).

Commands:
    gemini-py [-p PROMPT] [OPTIONS]
    gemini-py sessions list
    gemini-py sessions delete INDEX
    gemini-py extensions list
    gemini-py mcp list

Examples:
    # One-shot non-interactive prompt
    gemini-py -p "Explain how HTTP/3 works"

    # Pipe stdin into the prompt
    cat README.md | gemini-py -p "Summarize this file"

    # JSON output for scripting
    gemini-py -p "What time is it in Jakarta?" --output-format json

    # Stream-JSON events
    gemini-py -p "List 3 Python tips" --output-format stream-json

    # Resume the most recent session
    gemini-py -r latest -p "What did we discuss?"

    # Specify model alias
    gemini-py -p "Quick fact" --model flash
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional

from . import __version__
from .auth import AuthConfig
from .config import GeminiConfig, ModelAlias
from .client import GeminiClient, GeminiError
from .output import OutputFormat, format_result, print_stream
from .prompts import build_system_prompt
from .sessions import SessionStore
from .tools import build_default_tools


# --- Argument parser --------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser."""
    parser = argparse.ArgumentParser(
        prog="gemini-py",
        description=(
            "gemini-py — Python SDK and CLI for Google Gemini "
            "(headless port of google-gemini/gemini-cli)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  gemini-py -p 'Explain how HTTP/3 works'\n"
            "  cat README.md | gemini-py -p 'Summarize this file'\n"
            "  gemini-py -p 'Quick fact' --model flash\n"
            "  gemini-py -p 'Summarize' --output-format json\n"
            "  gemini-py sessions list\n"
        ),
    )
    parser.add_argument(
        "query",
        nargs="*",
        help="Positional prompt. If empty, reads from -p or stdin.",
    )
    parser.add_argument("-p", "--prompt", help="Prompt text (forces non-interactive).")
    parser.add_argument(
        "-i",
        "--prompt-interactive",
        help="Execute prompt then continue interactively (note: this Python port has no TUI; it just runs the prompt).",
    )
    parser.add_argument("-m", "--model", default="auto", help="Model alias or name (default: auto).")
    parser.add_argument(
        "-o",
        "--output-format",
        choices=["text", "json", "stream-json"],
        default="text",
        help="Output format (default: text).",
    )
    parser.add_argument(
        "-r",
        "--resume",
        help="Resume a previous session by ID, 'latest', or index.",
    )
    parser.add_argument("--list-sessions", action="store_true", help="List sessions and exit.")
    parser.add_argument("--delete-session", help="Delete session by index (use --list-sessions to see).")
    parser.add_argument("--include-directories", help="Additional workspace directories (comma-separated).")
    parser.add_argument(
        "--sandbox",
        action="store_true",
        help="Run with sandboxing (note: this Python port currently ignores this flag).",
    )
    parser.add_argument(
        "--approval-mode",
        choices=["default", "auto_edit", "yolo", "plan"],
        default="default",
        help="Tool approval mode (default: default — confirm before shell).",
    )
    parser.add_argument("-y", "--yolo", action="store_true", help="Auto-approve all tool calls (alias of --approval-mode=yolo).")
    parser.add_argument("--skip-trust", action="store_true", help="Skip folder trust check (always yes).")
    parser.add_argument("-d", "--debug", action="store_true", help="Verbose logging.")
    parser.add_argument("-v", "--version", action="version", version=f"gemini-py {__version__}")
    parser.add_argument("--no-tools", action="store_true", help="Disable built-in tools (text-only).")
    parser.add_argument("--no-shell", action="store_true", help="Disable the shell tool only.")
    parser.add_argument("--enable-web", action="store_true", help="Enable web_search and web_fetch tools.")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Live view: stream the model's thinking + tool calls + tool results to stderr in real-time.",
    )
    parser.add_argument("--no-memory", action="store_true", help="Disable hierarchical GEMINI.md auto-memory.")
    parser.add_argument("--no-skills", action="store_true", help="Disable @skill_name expansion and skill listing.")
    parser.add_argument("--no-subagent", action="store_true", help="Disable the delegate_subagent tool.")
    parser.add_argument("--no-mcp", action="store_true", help="Do not connect to configured MCP servers.")
    parser.add_argument(
        "--system-prompt",
        help="Override the system prompt. Use '-' to read from stdin.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=25,
        help="Max tool-call round-trips in agent mode (default: 25).",
    )
    parser.add_argument("--max-output-tokens", type=int, default=8192, help="Max output tokens.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p nucleus sampling.")

    # NOTE: Subcommands (sessions/extensions/mcp) are handled by a separate
    # parser in `_build_subcommand_parser()`, dispatched from `main()` before
    # this parser runs. This avoids a conflict with the positional `query`
    # argument (nargs='*') that would otherwise swallow the subcommand name.

    return parser


def _build_subcommand_parser() -> argparse.ArgumentParser:
    """A parser that only handles the subcommands (sessions/extensions/mcp/skills/login/logout/whoami)."""
    parser = argparse.ArgumentParser(
        prog="gemini-py",
        description="Subcommands for sessions/extensions/mcp/skills/login management.",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND", required=True)
    _add_sessions_subparser(subparsers)
    _add_extensions_subparser(subparsers)
    _add_mcp_subparser(subparsers)
    _add_skills_subparser(subparsers)
    _add_login_subparser(subparsers)
    _add_logout_subparser(subparsers)
    _add_whoami_subparser(subparsers)
    return parser


def _add_sessions_subparser(subparsers) -> None:
    p = subparsers.add_parser("sessions", help="Manage sessions.")
    sub = p.add_subparsers(dest="sessions_cmd", required=True)
    sub.add_parser("list", help="List sessions for the current project.")
    delete = sub.add_parser("delete", help="Delete a session by index.")
    delete.add_argument("index", type=int, help="Session index from `sessions list`.")
    resume = sub.add_parser("resume", help="Resume a session by index or 'latest'.")
    resume.add_argument("which", help="Session index or 'latest'.")


def _add_extensions_subparser(subparsers) -> None:
    p = subparsers.add_parser("extensions", help="Manage extensions (stub).")
    sub = p.add_subparsers(dest="extensions_cmd", required=True)
    sub.add_parser("list", help="List installed extensions (always empty in this Python port).")


def _add_mcp_subparser(subparsers) -> None:
    p = subparsers.add_parser("mcp", help="Manage MCP servers.")
    sub = p.add_subparsers(dest="mcp_cmd", required=True)
    sub.add_parser("list", help="List configured MCP servers.")

    add = sub.add_parser("add", help="Add an MCP server.")
    add.add_argument("name", help="Server name (used to namespace its tools).")
    add.add_argument("command_or_url", help="Command (stdio) or URL (http). For stdio, additional positional args are forwarded.")
    # Use REMAINDER so that flags like -y / --foo meant for the subprocess
    # are passed through verbatim instead of being eaten by argparse.
    add.add_argument("cmd_args", nargs=argparse.REMAINDER, help="Args for the stdio command (after the command, including flags).")
    add.add_argument("--transport", choices=["stdio", "http"], default="auto", help="Transport type (default: auto-detect from URL).")
    add.add_argument("--env", action="append", default=[], help="Env var KEY=value (repeatable).")
    add.add_argument("--cwd", help="Working directory for the stdio subprocess.")
    add.add_argument("--scope", choices=["user", "project"], default="user", help="Where to persist the config.")
    add.add_argument("--include-tools", help="Comma-separated tool names to expose (default: all).")
    add.add_argument("--disabled", action="store_true", help="Add but leave disabled.")

    remove = sub.add_parser("remove", help="Remove an MCP server by name.")
    remove.add_argument("name", help="Server name.")

    enable = sub.add_parser("enable", help="Enable an MCP server.")
    enable.add_argument("name", help="Server name.")

    disable = sub.add_parser("disable", help="Disable an MCP server.")
    disable.add_argument("name", help="Server name.")


def _add_skills_subparser(subparsers) -> None:
    p = subparsers.add_parser("skills", help="Manage agent skills.")
    sub = p.add_subparsers(dest="skills_cmd", required=True)
    sub.add_parser("list", help="List discovered skills.")
    show = sub.add_parser("show", help="Print a skill's full prompt content.")
    show.add_argument("name", help="Skill name.")
    enable = sub.add_parser("enable", help="Enable a skill.")
    enable.add_argument("name", help="Skill name.")
    disable = sub.add_parser("disable", help="Disable a skill.")
    disable.add_argument("name", help="Skill name.")
    sub.add_parser("enable-all", help="Enable all discovered skills.")
    sub.add_parser("disable-all", help="Disable all discovered skills.")


def _add_login_subparser(subparsers) -> None:
    p = subparsers.add_parser("login", help="Authenticate and persist credentials.")
    p.add_argument(
        "--api-key",
        help="API key. If omitted, read interactively from a hidden prompt.",
    )
    p.add_argument(
        "--method",
        choices=["gemini", "google"],
        default="gemini",
        help="Auth method (default: gemini). 'google' = Vertex AI API key.",
    )
    p.add_argument("--project", help="Google Cloud project ID (required for --method=google).")
    p.add_argument("--location", default="us-central1", help="Vertex AI region (default: us-central1).")
    p.add_argument("--scope", choices=["user", "project"], default="user", help="Where to persist (default: user).")


def _add_logout_subparser(subparsers) -> None:
    p = subparsers.add_parser("logout", help="Clear persisted credentials.")
    p.add_argument("--scope", choices=["user", "project"], default="user", help="Which file to clear (default: user).")


def _add_whoami_subparser(subparsers) -> None:
    subparsers.add_parser("whoami", help="Show currently-active credentials (masked).")


# --- Main entry -------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    """Main CLI entry point. Returns exit code."""
    raw_argv = list(sys.argv[1:]) if argv is None else list(argv)

    # Detect subcommands BEFORE argparse, so that the positional `query`
    # argument doesn't swallow them when no -p is given.
    SUBCOMMANDS = {"sessions", "extensions", "mcp", "skills", "login", "logout", "whoami"}
    if raw_argv and raw_argv[0] in SUBCOMMANDS:
        # Build a parser that only knows about subcommands + their options.
        parser = _build_subcommand_parser()
        args = parser.parse_args(raw_argv)
        if args.command == "sessions":
            return _cmd_sessions(args)
        if args.command == "extensions":
            return _cmd_extensions(args)
        if args.command == "mcp":
            return _cmd_mcp(args)
        if args.command == "skills":
            return _cmd_skills(args)
        if args.command == "login":
            return _cmd_login(args)
        if args.command == "logout":
            return _cmd_logout(args)
        if args.command == "whoami":
            return _cmd_whoami(args)
        return 1

    parser = build_parser()
    args = parser.parse_args(raw_argv)

    if args.list_sessions:
        return _cmd_list_sessions()

    if args.delete_session:
        return _cmd_delete_session(args.delete_session)

    # Resolve prompt: -p > -i > positional > stdin
    prompt = args.prompt or args.prompt_interactive or ""
    if not prompt and args.query:
        prompt = " ".join(args.query)
    if not prompt and not sys.stdin.isatty():
        try:
            stdin_text = sys.stdin.read()
        except KeyboardInterrupt:
            return 130
        if stdin_text:
            prompt = prompt + ("\n\n" + stdin_text if prompt else stdin_text)

    if not prompt:
        parser.print_help()
        return 0

    # Build config
    cfg = GeminiConfig(
        model=args.model,
        temperature=args.temperature,
        top_p=args.top_p,
        max_output_tokens=args.max_output_tokens,
    )
    if args.system_prompt:
        sp = args.system_prompt
        if sp == "-":
            try:
                sp = sys.stdin.read()
            except Exception:
                sp = ""
        cfg.system_instruction = sp

    # Build client
    try:
        client = GeminiClient(config=cfg)
    except GeminiError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    # Session resume
    store = SessionStore()
    session_rec = None
    messages = []
    if args.resume:
        session_rec = _resolve_resume(store, args.resume)
        if session_rec is None:
            print(f"Error: session not found for '{args.resume}'", file=sys.stderr)
            return 1
        messages = session_rec.messages
    else:
        session_rec = store.create(model=cfg.resolve_model(), config=cfg)

    # Tools
    if args.no_tools:
        tools = []
    else:
        tools = build_default_tools(
            enable_shell=not args.no_shell,
            enable_filesystem=True,
            enable_web=args.enable_web,
        )

    # Approval mode affects tool execution
    approval_mode = args.approval_mode
    if args.yolo and approval_mode != "yolo":
        approval_mode = "yolo"

    # Output format
    fmt = OutputFormat(args.output_format)

    # Run
    try:
        if fmt == OutputFormat.STREAM_JSON:
            # Stream events as JSONL
            for event in client._stream_events(
                prompt,
                cfg=cfg,
                tools=tools,
                on_event=None,
            ):
                print_stream(event)
            return 0

        # Connect to MCP servers (if enabled & any are configured).
        mcp_client = None
        if not args.no_mcp:
            try:
                from .mcp import McpRegistry, McpClient, MCP_AVAILABLE
                if MCP_AVAILABLE:
                    registry = McpRegistry()
                    if registry.enabled_servers():
                        mcp_client = McpClient(registry)
                        mcp_client.start()
            except Exception as e:
                sys.stderr.write(f"[gemini-py] MCP setup failed: {e}\n")
                mcp_client = None

        try:
            # Regular non-stream run
            live_view = None
            if args.live:
                try:
                    from .live import make_default_live_view
                    live_view = make_default_live_view(
                        enabled=True,
                        live=True,
                        model=cfg.resolve_model(),
                        max_iterations=args.max_iterations,
                    )
                except Exception:
                    live_view = None
            result = client.run_agent_turn(
                prompt,
                tools=tools,
                config=cfg,
                max_iterations=args.max_iterations,
                enable_memory=not args.no_memory,
                enable_skills=not args.no_skills,
                enable_subagent=not args.no_subagent,
                mcp_client=mcp_client,
                live=args.live,
                on_event=(live_view.emit if live_view else None),
            )
            if live_view:
                # Stop the Live dashboard BEFORE printing the final answer,
                # otherwise the dashboard would overwrite our stdout output.
                live_view.close()
        finally:
            if mcp_client is not None:
                try:
                    mcp_client.stop()
                except Exception:
                    pass
        print(format_result(result, fmt))
        return 0
    except GeminiError as e:
        if fmt == OutputFormat.JSON:
            import json as _json
            print(_json.dumps({"error": str(e)}, indent=2))
        else:
            print(f"Error: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130


# --- Subcommand handlers ----------------------------------------------------


def _cmd_sessions(args) -> int:
    """Handle `gemini-py sessions <list|delete|resume>`."""
    store = SessionStore()
    if args.sessions_cmd == "list":
        return _cmd_list_sessions()
    if args.sessions_cmd == "delete":
        return _cmd_delete_session(str(args.index))
    if args.sessions_cmd == "resume":
        # In subcommand form, we just print the session's last messages.
        rec = _resolve_resume(store, args.which)
        if rec is None:
            print(f"Error: session not found for '{args.which}'", file=sys.stderr)
            return 1
        for msg in rec.messages:
            role = msg.get("role", "?")
            parts = msg.get("parts", [])
            text = parts[0].get("text", "") if parts and isinstance(parts[0], dict) else ""
            print(f"[{role}] {text[:200]}")
        return 0
    return 1


def _cmd_list_sessions() -> int:
    """List sessions for the current project."""
    store = SessionStore()
    records = store.list_all(project_dir=str(Path.cwd()))
    if not records:
        print("(no sessions found for this project)")
        return 0
    # Sort by updated_at descending
    records.sort(key=lambda r: r.updated_at, reverse=True)
    print(f"Sessions for {Path.cwd()}:")
    for i, r in enumerate(records):
        from datetime import datetime

        ts = datetime.fromtimestamp(r.updated_at).strftime("%Y-%m-%d %H:%M:%S")
        n_msgs = len(r.messages)
        print(f"  [{i}] {r.session_id}  updated={ts}  messages={n_msgs}  model={r.model}")
    return 0


def _cmd_delete_session(which: str) -> int:
    """Delete a session by index."""
    store = SessionStore()
    records = store.list_all(project_dir=str(Path.cwd()))
    records.sort(key=lambda r: r.updated_at, reverse=True)
    try:
        idx = int(which)
        if idx < 0 or idx >= len(records):
            raise IndexError
    except (ValueError, IndexError):
        print(f"Error: invalid session index: {which}", file=sys.stderr)
        return 1
    rec = records[idx]
    if store.delete(rec.session_id):
        print(f"Deleted session [{idx}] {rec.session_id}")
        return 0
    print(f"Error: failed to delete session {rec.session_id}", file=sys.stderr)
    return 1


def _cmd_extensions(args) -> int:
    if args.extensions_cmd == "list":
        print("(no extensions installed — Python port does not support gemini-cli extensions)")
        return 0
    return 0


def _cmd_mcp(args) -> int:
    """Handle `gemini-py mcp <list|add|remove|enable|disable>`."""
    from .mcp import McpRegistry, McpServerConfig, MCP_AVAILABLE, parse_env_pair

    if not MCP_AVAILABLE:
        print(
            "Error: mcp package not installed. Run `pip install mcp`.",
            file=sys.stderr,
        )
        return 1

    registry = McpRegistry()

    if args.mcp_cmd == "list":
        servers = registry.list()
        if not servers:
            print("(no MCP servers configured)")
            print()
            print("Add one with: gemini-py mcp add <name> <command> [args...]")
            print("       or:    gemini-py mcp add <name> <url> --transport http")
            return 0
        print(f"MCP servers ({len(servers)}):")
        for s in servers:
            state = "enabled" if s.enabled else "disabled"
            print(f"  [{state:8s}] {s.describe()}")
            if s.include_tools:
                print(f"              include-tools: {', '.join(s.include_tools)}")
        return 0

    if args.mcp_cmd == "add":
        # Auto-detect transport: URL → http, else stdio.
        transport = args.transport
        if transport == "auto":
            transport = "http" if _looks_like_url(args.command_or_url) else "stdio"

        # Build env dict
        env: dict = {}
        for pair in args.env:
            try:
                k, v = parse_env_pair(pair)
                env[k] = v
            except ValueError as e:
                print(f"Error: {e}", file=sys.stderr)
                return 1

        include = (
            [t.strip() for t in args.include_tools.split(",") if t.strip()]
            if args.include_tools
            else []
        )

        if transport == "http":
            cfg = McpServerConfig(
                name=args.name,
                transport="http",
                url=args.command_or_url,
                scope=args.scope,
                include_tools=include,
                enabled=not args.disabled,
            )
        else:
            cfg = McpServerConfig(
                name=args.name,
                transport="stdio",
                command=args.command_or_url,
                args=list(args.cmd_args),
                env=env,
                cwd=args.cwd,
                scope=args.scope,
                include_tools=include,
                enabled=not args.disabled,
            )
        registry.add(cfg)
        print(f"Added MCP server '{cfg.name}' (scope={cfg.scope}, transport={cfg.transport})")
        return 0

    if args.mcp_cmd == "remove":
        if registry.remove(args.name):
            print(f"Removed MCP server '{args.name}'")
            return 0
        print(f"Error: no MCP server named '{args.name}'", file=sys.stderr)
        return 1

    if args.mcp_cmd == "enable":
        if registry.set_enabled(args.name, True):
            print(f"Enabled MCP server '{args.name}'")
            return 0
        print(f"Error: no MCP server named '{args.name}'", file=sys.stderr)
        return 1

    if args.mcp_cmd == "disable":
        if registry.set_enabled(args.name, False):
            print(f"Disabled MCP server '{args.name}'")
            return 0
        print(f"Error: no MCP server named '{args.name}'", file=sys.stderr)
        return 1

    return 1


def _cmd_skills(args) -> int:
    """Handle `gemini-py skills <list|show|enable|disable|enable-all|disable-all>`."""
    from .skills import SkillsLoader

    loader = SkillsLoader()

    if args.skills_cmd == "list":
        skills = loader.list()
        if not skills:
            print("(no skills discovered)")
            print()
            print("Skills live in: ~/.gemini-py/skills/<name>/SKILL.md")
            print("              or ./.gemini/skills/<name>/SKILL.md")
            return 0
        print(f"Skills ({len(skills)}):")
        for s in skills:
            state = "enabled" if s.enabled else "disabled"
            desc = s.description or "(no description)"
            print(f"  [{state:8s}] @{s.name}  ({s.scope})  {desc[:80]}")
        return 0

    if args.skills_cmd == "show":
        skill = loader.get(args.name)
        if skill is None:
            print(f"Error: no skill named '{args.name}'", file=sys.stderr)
            return 1
        print(f"# Skill: {skill.name}")
        print(f"# Source: {skill.file_path}")
        print(f"# Scope:  {skill.scope}")
        print(f"# State:  {'enabled' if skill.enabled else 'disabled'}")
        print()
        print(skill.full_prompt())
        return 0

    if args.skills_cmd == "enable":
        if loader.set_enabled(args.name, True):
            print(f"Enabled skill '{args.name}'")
            return 0
        print(f"Error: no skill named '{args.name}'", file=sys.stderr)
        return 1

    if args.skills_cmd == "disable":
        if loader.set_enabled(args.name, False):
            print(f"Disabled skill '{args.name}'")
            return 0
        print(f"Error: no skill named '{args.name}'", file=sys.stderr)
        return 1

    if args.skills_cmd == "enable-all":
        loader.enable_all()
        print(f"Enabled all {len(loader.list())} skills")
        return 0

    if args.skills_cmd == "disable-all":
        loader.disable_all()
        print(f"Disabled all {len(loader.list())} skills")
        return 0

    return 1


def _looks_like_url(s: str) -> bool:
    return s.startswith(("http://", "https://"))


# --- login / logout / whoami -----------------------------------------------


def _cmd_login(args) -> int:
    """Handle `gemini-py login`."""
    from .login import login as _login, LoginError

    api_key = args.api_key
    if not api_key:
        # Read interactively from stdin (hidden if getpass is available).
        if sys.stdin.isatty():
            try:
                import getpass
                api_key = getpass.getpass("Enter your API key (input hidden): ")
            except Exception:
                api_key = input("Enter your API key: ").strip()
        else:
            # Non-interactive: read first line of stdin.
            try:
                api_key = sys.stdin.readline().strip()
            except Exception:
                api_key = ""
        if not api_key:
            print("Error: no API key provided.", file=sys.stderr)
            return 1

    try:
        result = _login(
            api_key,
            method=args.method,
            project=args.project,
            location=args.location,
            scope=args.scope,
        )
    except LoginError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    print(result.describe())
    if args.method == "google":
        print()
        print("To use Vertex AI ADC instead, unset GOOGLE_API_KEY and set GOOGLE_APPLICATION_CREDENTIALS.")
    return 0


def _cmd_logout(args) -> int:
    """Handle `gemini-py logout`."""
    from .login import logout as _logout, LoginError

    try:
        deleted = _logout(scope=args.scope)
    except LoginError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    if deleted:
        print(f"Logged out (removed {args.scope} credentials).")
    else:
        print(f"No {args.scope} credentials to remove.")
    return 0


def _cmd_whoami(args) -> int:
    """Handle `gemini-py whoami`."""
    from .login import whoami
    info = whoami()
    if info is None:
        print("(no credentials configured)")
        print()
        print("Login with:  gemini-py login")
        print("           or  from gemini_py import login; login('YOUR_KEY')")
        return 0
    print(f"Method:  {info['method']}")
    print(f"API key: {info['api_key']}")
    if info["project"]:
        print(f"Project: {info['project']}")
    if info["location"]:
        print(f"Location: {info['location']}")
    return 0


def _resolve_resume(store: SessionStore, which: str):
    """Resolve a `--resume` argument to a SessionRecord."""
    if which == "latest":
        return store.latest(project_dir=str(Path.cwd()))
    # Try as index
    try:
        idx = int(which)
        records = store.list_all(project_dir=str(Path.cwd()))
        records.sort(key=lambda r: r.updated_at, reverse=True)
        if 0 <= idx < len(records):
            return records[idx]
    except ValueError:
        pass
    # Try as session_id
    return store.get(which)


if __name__ == "__main__":
    raise SystemExit(main())
