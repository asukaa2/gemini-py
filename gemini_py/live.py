"""Live chat GUI using `rich.live.Live` + `rich.layout.Layout`.

Mirrors gemini-cli's interactive TUI: while the agent runs, a dashboard
renders to the terminal in real-time, split into:

    ┌── header: model · iteration N/M · tool count · status ──┐
    │ body:   streaming transcript                            │
    │           - user prompt                                 │
    │           - model thinking (dim italic)                 │
    │           - tool_call  (name + args)                    │
    │           - file_write (code panel with syntax hl)      │
    │           - tool_result (truncated, color-coded)        │
    │           - iteration separator                         │
    ├── footer: hint + progress bar ─────────────────────────┤
    └─────────────────────────────────────────────────────────┘

The dashboard re-renders up to 12x/second via `rich.live.Live`. When
the agent finishes, the Live display is stopped and the final answer
prints to stdout normally (so piping still works).

Two implementations:
    - `RichLiveChat`  : full-screen dashboard (requires rich + TTY)
    - `LiveView`      : simple stderr streaming fallback (no rich / non-TTY)

The factory `make_default_live_view(enabled, live=True)` picks the right
one based on availability.
"""

from __future__ import annotations

import sys
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Plain-text fallback (no rich required)
# ---------------------------------------------------------------------------


class LiveView:
    """Print agent events to stderr (plain text). Fallback when rich is
    unavailable or stdout is not a TTY.
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
                self._console = Console(
                    file=self._stream, force_terminal=True, soft_wrap=True
                )
                self._Panel = Panel
                self._Syntax = Syntax
                self._Markdown = Markdown
                self._use_rich = True
            except Exception:
                self._use_rich = False
        self._in_thinking = False

    def emit(self, event: Dict[str, Any]) -> None:
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

    def close(self) -> None:
        """No-op for the simple view (everything is already on stderr)."""
        if self._in_thinking:
            self._write("\n")
            self._in_thinking = False

    # --- Event handlers (kept short — see commit history for the long form) -

    def _on_iteration(self, event: Dict[str, Any]) -> None:
        n = event.get("n", "?")
        mx = event.get("max", "?")
        if self._use_rich:
            from rich.panel import Panel as _P
            self._console.print(
                _P(
                    f"[bold cyan]Iteration {n}/{mx}[/]",
                    border_style="cyan",
                    padding=(0, 1),
                )
            )
        else:
            self._write(f"\n--- Iteration {n}/{mx} ---\n")
        self._in_thinking = False

    def _on_text_chunk(self, text: str) -> None:
        if not text:
            return
        if not self._in_thinking:
            if self._use_rich:
                self._console.print("[dim italic]thinking…[/]")
            else:
                self._write("--- thinking ---\n")
            self._in_thinking = True
        if self._use_rich:
            self._console.print(text, end="", soft_wrap=True, highlight=False)
        else:
            self._write(text)

    def _on_tool_call(self, event: Dict[str, Any]) -> None:
        if self._in_thinking:
            if self._use_rich:
                self._console.print("")
            else:
                self._write("\n")
            self._in_thinking = False
        name = event.get("name", "?")
        args = event.get("args", {}) or {}
        if name == "write_file":
            self._render_file_write(args.get("path", "?"), args.get("content", ""))
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
        n_lines = content.count("\n") + 1
        n_chars = len(content)
        header = f"write_file → {path}  ({n_lines} lines, {n_chars} chars)"
        lexer = self._guess_lexer(path)
        if self._use_rich:
            try:
                from rich.syntax import Syntax as _S
                from rich.panel import Panel as _P
                syntax = _S(content, lexer, theme="monokai", line_numbers=True, word_wrap=True)
                self._console.print(
                    _P(
                        syntax,
                        title=f"[bold magenta]{header}[/]",
                        border_style="magenta",
                        padding=(0, 1),
                    )
                )
                return
            except Exception:
                pass
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

    # --- Helpers -------------------------------------------------------------

    def _format_args(self, args: Dict[str, Any]) -> str:
        if not args:
            return "{}"
        parts = []
        for k, v in args.items():
            v_str = repr(v) if isinstance(v, str) else str(v)
            if len(v_str) > 80:
                v_str = v_str[:77] + "…"
            parts.append(f"{k}={v_str}")
        return "{" + ", ".join(parts) + "}"

    def _guess_lexer(self, path: str) -> str:
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
        if lower.endswith(".env"):
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


# ---------------------------------------------------------------------------
# Full-screen dashboard using rich.live.Live
# ---------------------------------------------------------------------------


class RichLiveChat:
    """Full-screen chat-style dashboard using `rich.live.Live`.

    Renders a continuously-updating dashboard split into:
        - header panel (model + iteration + tool count + status)
        - body panel (streaming transcript)
        - footer panel (progress hint)

    The dashboard re-renders up to 12x/second. Call `close()` to stop
    the Live display when the agent finishes.
    """

    REFRESH_PER_SECOND = 12

    def __init__(self, model: str = "", max_iterations: int = 0) -> None:
        try:
            from rich.console import Console
            from rich.live import Live
            from rich.layout import Layout
            from rich.panel import Panel
            from rich.text import Text
            from rich.table import Table
            from rich.syntax import Syntax
            from rich.markdown import Markdown
            from rich.align import Align
            from rich.columns import Columns
            from rich.rule import Rule
        except ImportError as e:
            raise RuntimeError(
                "rich is required for RichLiveChat. pip install rich."
            ) from e

        self._Layout = Layout
        self._Panel = Panel
        self._Text = Text
        self._Table = Table
        self._Syntax = Syntax
        self._Markdown = Markdown
        self._Align = Align
        self._Columns = Columns
        self._Rule = Rule

        self.console = Console(force_terminal=True)
        # State rendered by the dashboard.
        self.state: Dict[str, Any] = {
            "model": model,
            "iteration": 0,
            "max": max_iterations,
            "tool_calls": 0,
            "status": "starting…",
            "finish_reason": "",
            "done": False,
            "error": None,
        }
        # Body lines (each is a renderable or string).
        self._body_renderables: List[Any] = []
        # Pending thinking text accumulator (so we don't create one Text per chunk).
        self._pending_thinking: Optional[Any] = None

        # The Live display.
        #
        # screen=True   — use the terminal's alt-screen buffer so the
        #                 dashboard fills the screen and updates IN
        #                 PLACE (like vim / less / gemini-cli's TUI).
        #                 No scrolling, no flicker, no "muncul mulu".
        #
        # transient=True — when close() is called, the dashboard is
        #                  cleared from the terminal entirely. The final
        #                  answer then prints to stdout cleanly, with no
        #                  leftover dashboard above it.
        #
        # We pass an INITIAL renderable (not a callable — some rich
        # versions don't auto-call bound methods) and call
        # live.update(self._render()) on each emit() to push fresh state.
        # refresh_per_second controls the auto-refresh rate of whatever
        # renderable is currently set.
        self.live = Live(
            self._render(),
            console=self.console,
            refresh_per_second=self.REFRESH_PER_SECOND,
            transient=True,
            screen=True,
        )
        self.live.start()

    # --- Public API --------------------------------------------------------

    def emit(self, event: Dict[str, Any]) -> None:
        """Update internal state from an event and push a fresh renderable
        to the Live display.
        """
        etype = event.get("type")
        if etype == "iteration":
            self._on_iteration(event)
        elif etype == "text_chunk":
            self._on_text_chunk(event.get("text", ""))
        elif etype == "tool_call":
            self._on_tool_call(event)
        elif etype == "file_write":
            self._on_file_write(event)
        elif etype == "tool_result":
            self._on_tool_result(event)
        elif etype == "done":
            self._on_done(event)
        elif etype == "error":
            self._on_error(event)
        # Push the freshly-rendered dashboard to Live. This is the fix for
        # the NotRenderableError — passing a bound method to Live doesn't
        # work in all rich versions, so we update() with the rendered object.
        try:
            self.live.update(self._render())
        except Exception:
            pass

    def close(self) -> None:
        """Stop the Live display. Final answer should be printed AFTER this."""
        try:
            self.live.stop()
        except Exception:
            pass

    # --- Event handlers ----------------------------------------------------

    def _on_iteration(self, event: Dict[str, Any]) -> None:
        n = event.get("n", 0)
        mx = event.get("max", 0)
        self.state["iteration"] = n
        self.state["max"] = mx
        self.state["status"] = "thinking…"
        self._flush_thinking()
        self._body_renderables.append(
            self._Rule(f"iteration {n}/{mx}", style="cyan dim")
        )

    def _on_text_chunk(self, text: str) -> None:
        if not text:
            return
        # Accumulate into the pending thinking Text to avoid thousands of
        # tiny Text renderables in the body.
        if self._pending_thinking is None:
            self._pending_thinking = self._Text(text, style="dim italic")
            self._body_renderables.append(self._pending_thinking)
        else:
            self._pending_thinking.append(text)

    def _on_tool_call(self, event: Dict[str, Any]) -> None:
        self._flush_thinking()
        name = event.get("name", "?")
        args = event.get("args", {}) or {}
        self.state["tool_calls"] += 1
        self.state["status"] = f"calling {name}"

        # Special-case write_file: the file_write event (sent before this
        # tool_call) already rendered the code panel, so just emit a short
        # call line here.
        if name == "write_file":
            path = args.get("path", "?")
            line = self._Text.assemble(
                ("→ ", "bold yellow"),
                ("call ", "yellow"),
                (name, "bold"),
                ("(", "dim"),
                (f"path={path!r}", "cyan"),
                (")", "dim"),
            )
            self._body_renderables.append(line)
            return

        args_str = self._format_args(args)
        line = self._Text.assemble(
            ("→ ", "bold yellow"),
            ("call ", "yellow"),
            (name, "bold"),
            (" ", "dim"),
            (args_str, "dim"),
        )
        self._body_renderables.append(line)

    def _on_file_write(self, event: Dict[str, Any]) -> None:
        self._flush_thinking()
        path = event.get("path", "?")
        content = event.get("content", "")
        n_lines = content.count("\n") + 1
        n_chars = len(content)
        lexer = self._guess_lexer(path)
        try:
            syntax = self._Syntax(
                content, lexer, theme="monokai", line_numbers=True, word_wrap=True
            )
            title = (
                f"[bold magenta]write_file → {path}  "
                f"({n_lines} lines, {n_chars} chars)[/]"
            )
            self._body_renderables.append(
                self._Panel(
                    syntax,
                    title=title,
                    border_style="magenta",
                    padding=(0, 1),
                )
            )
        except Exception:
            self._body_renderables.append(
                self._Text(
                    f"write_file → {path}  ({n_lines} lines, {n_chars} chars)\n{content}",
                    style="magenta",
                )
            )

    def _on_tool_result(self, event: Dict[str, Any]) -> None:
        self._flush_thinking()
        name = event.get("name", "?")
        result = event.get("result", "")
        is_error = isinstance(result, str) and result.lower().startswith("error")
        max_len = 1500
        truncated = len(result) > max_len
        preview = result[:max_len] + ("…[truncated]" if truncated else "")

        color = "red" if is_error else "green"
        # Multi-line results get their own panel.
        if "\n" in preview:
            try:
                self._body_renderables.append(
                    self._Panel(
                        self._Text(preview, style=color),
                        title=f"[{color}]← {name}[/]",
                        border_style=color,
                        padding=(0, 1),
                    )
                )
                return
            except Exception:
                pass
        # Single-line result.
        self._body_renderables.append(
            self._Text.assemble(
                (f"← {name}  ", f"bold {color}"),
                (preview, "dim"),
            )
        )

    def _on_done(self, event: Dict[str, Any]) -> None:
        self._flush_thinking()
        self.state["done"] = True
        self.state["finish_reason"] = event.get("finish_reason", "")
        self.state["status"] = "done"
        self._body_renderables.append(
            self._Text.assemble(
                ("✓ ", "bold green"),
                ("done  ", "bold green"),
                (
                    f"(finish={self.state['finish_reason'] or 'n/a'}, "
                    f"tool_calls={self.state['tool_calls']})",
                    "dim",
                ),
            )
        )

    def _on_error(self, event: Dict[str, Any]) -> None:
        self._flush_thinking()
        msg = event.get("message", "unknown error")
        self.state["error"] = msg
        self.state["status"] = "error"
        self._body_renderables.append(
            self._Text.assemble(
                ("✗ error: ", "bold red"),
                (msg, "red"),
            )
        )

    # --- Rendering --------------------------------------------------------

    def _render(self):
        """Build the full dashboard renderable (called by Live)."""
        layout = self._Layout()
        layout.split_column(
            self._Layout(name="header", size=3),
            self._Layout(name="body"),
            self._Layout(name="footer", size=3),
        )
        layout["header"].update(self._render_header())
        layout["body"].update(self._render_body())
        layout["footer"].update(self._render_footer())
        return layout

    def _render_header(self):
        s = self.state
        status_style = "bold green" if s["done"] else (
            "bold red" if s["error"] else "bold yellow"
        )
        status_text = s["status"] or "idle"
        header_content = self._Text.assemble(
            ("gemini-py  ", "bold magenta"),
            ("model=", "dim"),
            (s.get("model") or "?", "cyan"),
            ("  iter=", "dim"),
            (f"{s.get('iteration', 0)}/{s.get('max', 0)}", "bold"),
            ("  tools=", "dim"),
            (str(s.get("tool_calls", 0)), "yellow"),
            ("  status=", "dim"),
            (status_text, status_style),
        )
        return self._Panel(
            header_content,
            border_style="magenta",
            padding=(0, 1),
        )

    def _render_body(self):
        # If no events yet, show a placeholder.
        if not self._body_renderables:
            return self._Panel(
                self._Text("(waiting for the model to start…)", style="dim italic"),
                border_style="dim",
                padding=(1, 1),
            )
        # Use a Group to stack all body renderables.
        from rich.console import Group
        return self._Panel(
            Group(*self._body_renderables),
            border_style="blue",
            padding=(0, 1),
            title="[bold blue]transcript[/]",
        )

    def _render_footer(self):
        s = self.state
        if s["done"]:
            text = self._Text.assemble(
                ("● ", "bold green"),
                ("agent finished — printing final answer to stdout", "dim"),
            )
        elif s["error"]:
            text = self._Text.assemble(
                ("● ", "bold red"),
                (f"error: {s['error']}", "red"),
            )
        else:
            text = self._Text.assemble(
                ("● ", "bold yellow"),
                ("working…  (Ctrl+C to abort)", "dim"),
            )
        return self._Panel(text, border_style="dim", padding=(0, 1))

    # --- Helpers -----------------------------------------------------------

    def _flush_thinking(self) -> None:
        """End any pending thinking text so the next event starts fresh."""
        if self._pending_thinking is not None:
            self._pending_thinking = None

    def _format_args(self, args: Dict[str, Any]) -> str:
        if not args:
            return "{}"
        parts = []
        for k, v in args.items():
            v_str = repr(v) if isinstance(v, str) else str(v)
            if len(v_str) > 80:
                v_str = v_str[:77] + "…"
            parts.append(f"{k}={v_str}")
        return "{" + ", ".join(parts) + "}"

    def _guess_lexer(self, path: str) -> str:
        # Reuse the same lexer table as LiveView.
        return LiveView(use_rich=False, stream=sys.stderr)._guess_lexer(path)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def make_default_live_view(
    *,
    enabled: bool,
    live: bool = True,
    model: str = "",
    max_iterations: int = 0,
) -> Optional[Any]:
    """Pick the right live view implementation.

    We ALWAYS use `LiveView` (the simple stderr-streaming version with
    rich formatting). It:

      - prints each event exactly ONCE to stderr (no re-renders,
        no "muncul banyak" repeats)
      - uses rich for colors + syntax-highlighted code panels
      - keeps stdout clean for the final answer

    The previous `RichLiveChat` (alt-screen dashboard with `Live` +
    `Layout`) was abandoned because:
      - `screen=True` doesn't work reliably across terminals (some
        fall back to writing new lines per refresh, causing the
        dashboard to scroll instead of update in place)
      - `screen=False` + growing transcript body causes cursor
        movement to fail, also producing new lines per refresh
      - User reported "MUNCUL NYA BANYAK BANGET, HARUSNYA 1 AJA"
        (appears many times, should appear just once) — exactly
        the streaming-re-renders problem

    `LiveView` is simpler, more reliable, and matches the original
    gemini-cli's behavior (stream events as they happen, no fancy
    in-place dashboard).

    Args:
        enabled: Master switch. If False, returns None (events dropped).
        live: Kept for backward-compat (ignored — we always stream).
        model: Unused (kept for backward-compat).
        max_iterations: Unused (kept for backward-compat).

    Returns:
        LiveView | None
    """
    if not enabled:
        return None
    return LiveView(use_rich=True)
