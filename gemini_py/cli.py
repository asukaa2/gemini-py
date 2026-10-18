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
        "--system-prompt",
        help="Override the system prompt. Use '-' to read from stdin.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=10,
        help="Max tool-call round-trips in agent mode (default: 10).",
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
    """A parser that only handles the subcommands (sessions/extensions/mcp)."""
    parser = argparse.ArgumentParser(
        prog="gemini-py",
        description="Subcommands for sessions/extensions/mcp management.",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND", required=True)
    _add_sessions_subparser(subparsers)
    _add_extensions_subparser(subparsers)
    _add_mcp_subparser(subparsers)
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
    p = subparsers.add_parser("mcp", help="Manage MCP servers (stub).")
    sub = p.add_subparsers(dest="mcp_cmd", required=True)
    sub.add_parser("list", help="List configured MCP servers (always empty in this Python port).")


# --- Main entry -------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    """Main CLI entry point. Returns exit code."""
    raw_argv = list(sys.argv[1:]) if argv is None else list(argv)

    # Detect subcommands BEFORE argparse, so that the positional `query`
    # argument doesn't swallow them when no -p is given.
    SUBCOMMANDS = {"sessions", "extensions", "mcp"}
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

        # Regular non-stream run
        result = client.run_agent_turn(
            prompt,
            tools=tools,
            config=cfg,
            max_iterations=args.max_iterations,
        )
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
    if args.mcp_cmd == "list":
        print("(no MCP servers configured — Python port does not include an MCP registry)")
        return 0
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
