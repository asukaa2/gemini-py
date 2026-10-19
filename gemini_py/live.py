"""Live view: stream agent events to stderr in real-time.

Mirrors gemini-cli's interactive TUI feedback: as the model thinks,
the user sees the text streaming in; when a tool is called, the call
name + args are rendered; when a tool returns, the result is shown
(truncated). For `write_file` calls, the content is rendered with
syntax highlighting so the user can watch the code being generated.

All output goes to **stderr** so stdout stays clean for piping /
scripting (use `--output-format json` or `stream-json` for that).

Events handled:
    text_chunk    — incremental text from the model (the "thinking")
    tool_call     — model is invoking a tool with given args
    tool_result   — tool returned a result
    iteration     — agent is starting iteration N of max
    done          — agent finished
"""

from __future__ import annotations

import sys
from typing import Any, Dict, Optional


class LiveView:
    """Print agent events to stderr with optional rich formatting.

    Falls back to plain text if `rich` is unavailable.
    """

    def __init__(self, use_rich: bool = True, stream=None) -> None:
        self._stream = stream or sys.stderr
        self._use_rich = False
        if use_rich:
            try:
                from rich.console import Console
                from rich.panel import Panel
                from rich.syntax import Syntax
                from rich.markdown import Markdown
                # type: ignore[attr-defined]
                self._console = Console(
                    file=self._stream, force_terminal=True, soft_wrap=True
                )
                self._Panel = Panel
                self._Syntax = Syntax
                self._Markdown = Markdown
                self._use_rich = True
            except Exception:
                self._use_rich = False
        # Track current line so we can prefix thinking text with a marker.
        self._in_thinking = False

    # --- Public API ------------------------------------------------------

    def emit(self, event: Dict[str, Any]) -> None:
        """Emit a single event. Safe to call from any thread."""
        etype = event.get("type")
        if etype == "text_chunk":
            self._on_text_chunk(event.get("text", ""))
        elif etype == "tool_call":
            self._on_tool_call(event)
        elif etype == "tool_result":
            self._on_tool_result(event)
        elif etype == "file_write":
            self._on_file_write(event)
        elif etype == "iteration":
            self._on_iteration(event)
        elif etype == "done":
            self._on_done(event)
        elif etype == "error":
            self._on_error(event)

    # --- Event handlers --------------------------------------------------

    def _on_iteration(self, event: Dict[str, Any]) -> None:
        n = event.get("n", "?")
        mx = event.get("max", "?")
        if self._use_rich:
            self._console.print(
                self._Panel(
                    f"[bold cyan]Iteration {n}/{mx}[/]",
                    border_style="cyan",
                    padding=(0, 1),
                )
            )
        else:
            self._write(f"\n--- Iteration {n}/{mx} ---\n")
        # Reset thinking state for the new iteration.
        self._in_thinking = False

    def _on_text_chunk(self, text: str) -> None:
        if not text:
            return
        if not self._in_thinking:
            # Start of a thinking block.
            if self._use_rich:
                self._console.print("[dim italic]thinking…[/]")
            else:
                self._write("--- thinking ---\n")
            self._in_thinking = True
        if self._use_rich:
            # Print without trailing newline so streaming looks live.
            self._console.print(text, end="", soft_wrap=True, highlight=False)
        else:
            self._write(text)

    def _on_tool_call(self, event: Dict[str, Any]) -> None:
        # End any in-progress thinking block.
        if self._in_thinking:
            if self._use_rich:
                self._console.print("")  # newline
            else:
                self._write("\n")
            self._in_thinking = False

        name = event.get("name", "?")
        args = event.get("args", {}) or {}
        if name == "write_file":
            # Special-case: show as a code panel instead of a one-liner.
            path = args.get("path", "?")
            content = args.get("content", "")
            self._render_file_write(path, content)
            return

        args_str = self._format_args(args)
        if self._use_rich:
            self._console.print(
                f"[bold yellow]→ call[/] [bold]{name}[/] [dim]{args_str}[/]"
            )
        else:
            self._write(f"→ call {name} {args_str}\n")

    def _on_tool_result(self, event: Dict[str, Any]) -> None:
        name = event.get("name", "?")
        result = event.get("result", "")
        # Truncate long results.
        is_error = isinstance(result, str) and result.lower().startswith("error")
        max_len = 2000
        truncated = len(result) > max_len
        preview = result[:max_len] + ("…[truncated]" if truncated else "")

        if self._use_rich:
            color = "red" if is_error else "green"
            self._console.print(
                f"[{color}]← {name}[/] " + (f"\n{preview}" if "\n" in preview else f"[dim]{preview}[/]")
            )
        else:
            tag = "ERROR" if is_error else "OK"
            self._write(f"← {name} [{tag}]: {preview}\n")

    def _on_file_write(self, event: Dict[str, Any]) -> None:
        self._render_file_write(event.get("path", "?"), event.get("content", ""))

    def _render_file_write(self, path: str, content: str) -> None:
        """Render a write_file call as a code panel with syntax highlighting."""
        n_lines = content.count("\n") + 1
        n_chars = len(content)
        header = f"write_file → {path}  ({n_lines} lines, {n_chars} chars)"
        # Pick lexer by file extension.
        lexer = self._guess_lexer(path)
        if self._use_rich:
            try:
                syntax = self._Syntax(
                    content, lexer, theme="monokai", line_numbers=True, word_wrap=True
                )
                self._console.print(
                    self._Panel(
                        syntax,
                        title=f"[bold magenta]{header}[/]",
                        border_style="magenta",
                        padding=(0, 1),
                    )
                )
                return
            except Exception:
                pass
        # Fallback: plain text.
        self._write(f"{header}\n{content}\n")

    def _on_done(self, event: Dict[str, Any]) -> None:
        if self._in_thinking:
            if self._use_rich:
                self._console.print("")
            else:
                self._write("\n")
            self._in_thinking = False
        finish = event.get("finish_reason", "")
        n_tools = event.get("n_tool_calls", 0)
        if self._use_rich:
            self._console.print(
                f"[bold green]✓ done[/] [dim](finish={finish or 'n/a'}, "
                f"tool calls={n_tools})[/]"
            )
        else:
            self._write(f"\n--- done (finish={finish or 'n/a'}, tool_calls={n_tools}) ---\n")

    def _on_error(self, event: Dict[str, Any]) -> None:
        msg = event.get("message", "unknown error")
        if self._use_rich:
            self._console.print(f"[bold red]✗ error:[/] {msg}")
        else:
            self._write(f"ERROR: {msg}\n")

    # --- Helpers ---------------------------------------------------------

    def _format_args(self, args: Dict[str, Any]) -> str:
        if not args:
            return "{}"
        # Short, single-line representation.
        parts = []
        for k, v in args.items():
            v_str = repr(v) if isinstance(v, str) else str(v)
            if len(v_str) > 80:
                v_str = v_str[:77] + "…"
            parts.append(f"{k}={v_str}")
        return "{" + ", ".join(parts) + "}"

    def _guess_lexer(self, path: str) -> str:
        """Map a file extension to a rich Syntax lexer name."""
        lower = path.lower()
        if lower.endswith((".py", ".pyw")):
            return "python"
        if lower.endswith((".js", ".mjs", ".cjs")):
            return "javascript"
        if lower.endswith((".ts", ".tsx")):
            return "typescript"
        if lower.endswith((".jsx",)):
            return "jsx"
        if lower.endswith((".html", ".htm")):
            return "html"
        if lower.endswith((".css",)):
            return "css"
        if lower.endswith((".json",)):
            return "json"
        if lower.endswith((".md", ".markdown")):
            return "markdown"
        if lower.endswith((".sh", ".bash")):
            return "bash"
        if lower.endswith((".yml", ".yaml")):
            return "yaml"
        if lower.endswith((".toml",)):
            return "toml"
        if lower.endswith((".sql",)):
            return "sql"
        if lower.endswith((".go",)):
            return "go"
        if lower.endswith((".rs",)):
            return "rust"
        if lower.endswith((".java",)):
            return "java"
        if lower.endswith((".c", ".h")):
            return "c"
        if lower.endswith((".cpp", ".cc", ".hpp")):
            return "cpp"
        if lower.endswith((".rb",)):
            return "ruby"
        if lower.endswith((".php",)):
            return "php"
        if lower.endswith((".xml", ".svg")):
            return "xml"
        if lower.endswith((".ini", ".cfg")):
            return "ini"
        if lower.endswith((".env",)):
            return "bash"
        if lower.endswith("dockerfile") or lower.endswith(".dockerfile"):
            return "docker"
        return "text"

    def _write(self, s: str) -> None:
        try:
            self._stream.write(s)
            self._stream.flush()
        except Exception:
            pass


def make_default_live_view(enabled: bool) -> Optional[LiveView]:
    """Construct a LiveView if enabled, else None (events get dropped)."""
    if not enabled:
        return None
    return LiveView(use_rich=True)
