"""Generate screenshots of the gemini-py live view dashboard for docs.

Outputs (in docs/assets/):
    live-view-thinking.svg   — mid-generation state (thinking…)
    live-view-tool-call.svg  — tool call in progress
    live-view-tool-result.svg — tool result received
    live-view-file-write.svg — write_file code panel with syntax highlight
    live-view-done.svg       — finished state
    live-view-ansi.txt       — raw ANSI text for terminal preview

Uses rich.console.Console.export_svg() to capture the rendered Layout.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make sure we use the package under dev.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console, Group
from rich.layout import Layout
from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax
from rich.text import Text


OUTPUT_DIR = Path(__file__).resolve().parent.parent / "docs" / "assets"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def header(state: dict) -> Panel:
    status_style = (
        "bold green" if state.get("done")
        else "bold red" if state.get("error")
        else "bold yellow"
    )
    return Panel(
        Text.assemble(
            ("gemini-py  ", "bold magenta"),
            ("model=", "dim"),
            (state.get("model", "gemini-3.8-flash"), "cyan"),
            ("  iter=", "dim"),
            (f"{state.get('iteration', 0)}/{state.get('max', 25)}", "bold"),
            ("  tools=", "dim"),
            (str(state.get("tool_calls", 0)), "yellow"),
            ("  status=", "dim"),
            (state.get("status", "idle"), status_style),
        ),
        border_style="magenta",
        padding=(0, 1),
    )


def footer(state: dict) -> Panel:
    if state.get("done"):
        text = Text.assemble(
            ("● ", "bold green"),
            ("agent finished — printing final answer to stdout", "dim"),
        )
    elif state.get("error"):
        text = Text.assemble(
            ("● ", "bold red"),
            (f"error: {state['error']}", "red"),
        )
    else:
        text = Text.assemble(
            ("● ", "bold yellow"),
            ("working…  (Ctrl+C to abort)", "dim"),
        )
    return Panel(text, border_style="dim", padding=(0, 1))


def placeholder_body() -> Panel:
    return Panel(
        Text("(waiting for the model to start…)", style="dim italic"),
        border_style="dim",
        padding=(1, 1),
    )


def build_dashboard(state: dict, body_renderables: list) -> Group:
    """Build a vertical Group of (header panel + body panel + footer panel).

    We use Group instead of Layout because Layout collapses to 0 height
    when rendered without a real terminal (e.g. in console.capture()).
    Group just stacks the panels one after another, which renders fine
    in the captured SVG.
    """
    if body_renderables:
        body_panel = Panel(
            Group(*body_renderables),
            border_style="blue",
            padding=(0, 1),
            title="[bold blue]transcript[/]",
        )
    else:
        body_panel = placeholder_body()
    return Group(header(state), body_panel, footer(state))


def render_to_svg(group: Group, title: str, width: int = 130) -> str:
    """Render a Group to an SVG string via rich.console.Console.export_svg."""
    console = Console(record=True, width=width, force_terminal=True, color_system="truecolor")
    # Don't use console.capture() — it clears the record buffer at exit
    # and only keeps the LAST printed segment, causing truncation.
    # Instead, just call console.print() directly — the record stays.
    console.print(group)
    return console.export_svg(title=title)


# Sample code for the file_write panel
FLASK_CODE = '''from flask import Flask, request, jsonify

app = Flask(__name__)
todos = []

@app.route("/todos", methods=["GET"])
def list_todos():
    return jsonify(todos)

@app.route("/todos", methods=["POST"])
def add_todo():
    todos.append(request.json)
    return jsonify({"ok": True}), 201

if __name__ == "__main__":
    app.run(debug=True)'''


def scenario_thinking() -> Layout:
    state = {
        "model": "gemini-3.8-flash",
        "iteration": 1, "max": 25, "tool_calls": 0,
        "status": "thinking…", "done": False,
    }
    thinking_text = Text(
        "Sure! I'll build a simple Flask todo API for you. Let me start by "
        "creating a project folder, then write the main app file, and "
        "finally add a small test file to make sure everything works.",
        style="dim italic",
    )
    body = [
        Rule("iteration 1/25", style="cyan dim"),
        thinking_text,
    ]
    return build_dashboard(state, body)


def scenario_tool_call() -> Layout:
    state = {
        "model": "gemini-3.8-flash",
        "iteration": 1, "max": 25, "tool_calls": 1,
        "status": "calling list_directory", "done": False,
    }
    body = [
        Rule("iteration 1/25", style="cyan dim"),
        Text("Sure! Let me check what's in the current directory first.",
             style="dim italic"),
        Text.assemble(
            ("→ call ", "bold yellow"),
            ("list_directory", "bold"),
            (" {path=\".\"}", "dim"),
        ),
    ]
    return build_dashboard(state, body)


def scenario_tool_result() -> Layout:
    state = {
        "model": "gemini-3.8-flash",
        "iteration": 1, "max": 25, "tool_calls": 1,
        "status": "thinking…", "done": False,
    }
    body = [
        Rule("iteration 1/25", style="cyan dim"),
        Text("Sure! Let me check what's in the current directory first.",
             style="dim italic"),
        Text.assemble(
            ("→ call ", "bold yellow"),
            ("list_directory", "bold"),
            (" {path=\".\"}", "dim"),
        ),
        Text.assemble(
            ("← list_directory  ", "bold green"),
            ("README.md  app.py  requirements.txt  tests/", "dim"),
        ),
    ]
    return build_dashboard(state, body)


def scenario_file_write() -> Layout:
    state = {
        "model": "gemini-3.8-flash",
        "iteration": 2, "max": 25, "tool_calls": 3,
        "status": "calling write_file", "done": False,
    }
    n_lines = FLASK_CODE.count("\n") + 1
    n_chars = len(FLASK_CODE)
    body = [
        Rule("iteration 1/25", style="cyan dim"),
        Text("Sure! Let me check what's in the current directory first.",
             style="dim italic"),
        Text.assemble(
            ("→ call ", "bold yellow"),
            ("list_directory", "bold"),
            (" {path=\".\"}", "dim"),
        ),
        Text.assemble(
            ("← list_directory  ", "bold green"),
            ("README.md  app.py  requirements.txt  tests/", "dim"),
        ),
        Rule("iteration 2/25", style="cyan dim"),
        Text("Now let me create the main Flask app file.",
             style="dim italic"),
        Text.assemble(
            ("→ call ", "bold yellow"),
            ("create_directory", "bold"),
            (" {path=\"todo-api/\"}", "dim"),
        ),
        Text.assemble(
            ("← create_directory  ", "bold green"),
            ("OK: directory ready at /home/user/todo-api", "dim"),
        ),
        Text.assemble(
            ("→ call ", "bold yellow"),
            ("write_file", "bold"),
            (" {path=\"todo-api/app.py\"}", "dim"),
        ),
        Panel(
            Syntax(
                FLASK_CODE,
                "python",
                theme="monokai",
                line_numbers=True,
                word_wrap=True,
            ),
            title=(
                f"[bold magenta]write_file → todo-api/app.py  "
                f"({n_lines} lines, {n_chars} chars)[/]"
            ),
            border_style="magenta",
            padding=(0, 1),
        ),
    ]
    return build_dashboard(state, body)


def scenario_done() -> Layout:
    state = {
        "model": "gemini-3.8-flash",
        "iteration": 3, "max": 25, "tool_calls": 4,
        "status": "done", "done": True,
    }
    body = [
        Rule("iteration 1/25", style="cyan dim"),
        Text("Sure! Let me check what's in the current directory first.",
             style="dim italic"),
        Text.assemble(
            ("→ call ", "bold yellow"),
            ("list_directory", "bold"),
            (" {path=\".\"}", "dim"),
        ),
        Text.assemble(
            ("← list_directory  ", "bold green"),
            ("README.md  app.py  requirements.txt  tests/", "dim"),
        ),
        Rule("iteration 2/25", style="cyan dim"),
        Text("Now let me create the main Flask app file.",
             style="dim italic"),
        Text.assemble(
            ("→ call ", "bold yellow"),
            ("create_directory", "bold"),
            (" {path=\"todo-api/\"}", "dim"),
        ),
        Text.assemble(
            ("← create_directory  ", "bold green"),
            ("OK: directory ready at /home/user/todo-api", "dim"),
        ),
        Text.assemble(
            ("→ call ", "bold yellow"),
            ("write_file", "bold"),
            (" {path=\"todo-api/app.py\"}", "dim"),
        ),
        Text.assemble(
            ("← write_file  ", "bold green"),
            ("OK: wrote 280 chars to todo-api/app.py", "dim"),
        ),
        Rule("iteration 3/25", style="cyan dim"),
        Text.assemble(
            ("✓ done  ", "bold green"),
            ("(finish=STOP, tool_calls=4)", "dim"),
        ),
    ]
    return build_dashboard(state, body)


def main() -> None:
    scenarios = [
        ("live-view-thinking.svg", scenario_thinking(),
         "gemini-py live view — thinking"),
        ("live-view-tool-call.svg", scenario_tool_call(),
         "gemini-py live view — tool call"),
        ("live-view-tool-result.svg", scenario_tool_result(),
         "gemini-py live view — tool result"),
        ("live-view-file-write.svg", scenario_file_write(),
         "gemini-py live view — write_file code panel"),
        ("live-view-done.svg", scenario_done(),
         "gemini-py live view — done"),
    ]
    for filename, layout, title in scenarios:
        svg = render_to_svg(layout, title=title, width=130)
        out_path = OUTPUT_DIR / filename
        out_path.write_text(svg, encoding="utf-8")
        size_kb = out_path.stat().st_size / 1024
        print(f"  + {filename} ({size_kb:.1f} KB)")

    # Also save the rendered ANSI text — terminal users can `cat` it to
    # see what the dashboard actually looks like in a real TTY.
    console = Console(record=True, width=130, force_terminal=True, color_system="truecolor")
    with console.capture() as cap:
        console.print(scenario_file_write())
    ansi_text = cap.get()
    (OUTPUT_DIR / "live-view-ansi.txt").write_text(ansi_text, encoding="utf-8")
    print(f"  + live-view-ansi.txt (raw ANSI for terminal preview)")

    print(f"\nAll screenshots saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
