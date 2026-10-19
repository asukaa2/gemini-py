# Live View

The `--live` flag renders a real-time dashboard so you can watch the
AI think, call tools, and write code as it happens — like the original
gemini-cli's interactive TUI, but using `rich.live.Live` + `Layout`.

## What it looks like

```
┌── header: gemini-py · model=gemini-3.8-flash · iter 1/25 · tools=0 · status=thinking… ──┐
│ body:    streaming transcript (re-rendered 12x/sec):                                   │
│           ── iteration 1/25 ──                                                          │
│           (model thinking in dim italic, accumulates as text streams in)                 │
│           → call run_shell_command {command='ls'}                                      │
│           ← run_shell_command  file1.txt file2.txt                                     │
│           ┌─ write_file → app.py  (5 lines, 120 chars) ──────────────────────────────┐ │
│           │ 1  from flask import Flask                                                 │ │
│           │ 2  app = Flask(__name__)                                                  │ │
│           │ ...                                                                       │ │
│           └───────────────────────────────────────────────────────────────────────────┘ │
│           ← write_file  OK: wrote 120 chars to app.py                                    │
│           ✓ done  (finish=STOP, tool_calls=2)                                           │
├── footer: ● working…  (Ctrl+C to abort) ───────────────────────────────────────────────┤
└──────────────────────────────────────────────────────────────────────────────────────────┘
```

## Quick start

```bash
# Live dashboard with default tools
gemini-py -p "Build a simple Flask todo app" --live

# Live + auto-approve all tool calls
gemini-py -p "Build X" --live --yolo

# Live + disable tools (just watch the model think)
gemini-py -p "Explain HTTP/3" --live --no-tools
```

## What you'll see

### Header panel

Top bar showing the current state:

- **model** — the resolved model name (e.g. `gemini-3.8-flash`)
- **iter N/M** — current iteration vs `--max-iterations`
- **tools** — cumulative tool-call count
- **status** — `thinking…` / `calling <tool>` / `done` / `error`

### Body panel (transcript)

The streaming transcript, top to bottom:

- **iteration separators** — `── iteration N/M ──` (cyan rule)
- **model thinking** — text chunks in dim italic, accumulates as the
  model streams them in
- **tool calls** — `→ call <name> {args}` (yellow)
- **file writes** — a syntax-highlighted code panel with the file
  path, line/char count, and the full content with line numbers
- **tool results** — `← <name> <result>` (green=ok, red=error).
  Multi-line results get their own panel.
- **done marker** — `✓ done (finish=STOP, tool_calls=N)` (green)
- **error marker** — `✗ error: <message>` (red)

### Footer panel

Bottom bar with status:

- `● working…  (Ctrl+C to abort)` (yellow) — while the agent is running
- `● agent finished — printing final answer to stdout` (green) — when done
- `● error: <msg>` (red) — on errors

## Syntax highlighting

`write_file` calls render the content with syntax highlighting. The
lexer is auto-detected from the file extension:

| Extension(s) | Lexer |
| --- | --- |
| `.py`, `.pyw` | python |
| `.js`, `.mjs`, `.cjs` | javascript |
| `.ts`, `.tsx` | typescript |
| `.jsx` | jsx |
| `.html`, `.htm` | html |
| `.css` | css |
| `.json` | json |
| `.md`, `.markdown` | markdown |
| `.sh`, `.bash` | bash |
| `.yml`, `.yaml` | yaml |
| `.toml` | toml |
| `.sql` | sql |
| `.go` | go |
| `.rs` | rust |
| `.java` | java |
| `.c`, `.h` | c |
| `.cpp`, `.cc`, `.hpp` | cpp |
| `.rb` | ruby |
| `.php` | php |
| `.xml`, `.svg` | xml |
| `.ini`, `.cfg` | ini |
| `.env` | bash |
| `Dockerfile` | docker |
| (other) | text (plain) |

Theme: `monokai` with line numbers and word-wrap.

## Implementation details

The dashboard is built with:

- [`rich.live.Live`](https://rich.readthedocs.io/en/stable/live.html)
  with `refresh_per_second=12` + `screen=True` + `transient=True`.
- **`screen=True`** enters the terminal's **alt-screen buffer** — the
  dashboard fills the screen and updates IN PLACE (like vim / less /
  gemini-cli's TUI). No scrolling, no flicker, no "muncul mulu".
- **`transient=True`** clears the dashboard when the agent finishes —
  only the final answer prints to stdout, with no leftover dashboard
  above it.
- [`rich.layout.Layout`](https://rich.readthedocs.io/en/stable/layout.html)
  — three-row column split (header / body / footer).
- [`rich.panel.Panel`](https://rich.readthedocs.io/en/stable/panel.html)
  — borders for each panel.
- [`rich.syntax.Syntax`](https://rich.readthedocs.io/en/stable/syntax.html)
  — syntax-highlighted code panels for `write_file` calls.

The `RichLiveChat` class is in
[`gemini_py/live.py`](../gemini_py/live.py).

## Fallback (non-TTY / no rich)

If `rich` is unavailable or stdout isn't a TTY (e.g. piping to a file),
`make_default_live_view()` falls back to `LiveView` — a plain
stderr-streaming variant that prints the same events line-by-line.

This means:

- `gemini-py -p "..." --live` in a TTY → full-screen dashboard
- `gemini-py -p "..." --live | tee log.txt` → plain stderr stream

## From Python

You can drive the live view programmatically:

```python
from gemini_py import GeminiClient
from gemini_py.live import make_default_live_view

client = GeminiClient()
live = make_default_live_view(
    enabled=True,
    live=True,
    model="gemini-3.8-flash",
    max_iterations=25,
)
try:
    result = client.run_agent_turn(
        "Build a simple Express todo API",
        live=True,
        on_event=live.emit,
    )
finally:
    live.close()  # stop the Live display BEFORE printing the answer

print(result.text)  # final answer goes to stdout after the dashboard closes
```

### Event reference

The `on_event` callback receives dicts with a `type` field:

| `type` | Other keys | When emitted |
| --- | --- | --- |
| `iteration` | `n`, `max` | Start of each agent iteration |
| `text_chunk` | `text` | Each streamed text chunk (only when `live=True`) |
| `tool_call` | `name`, `args` | Model is calling a tool |
| `file_write` | `path`, `content` | Emitted before `tool_call` for `write_file` |
| `tool_result` | `name`, `result` (truncated to 500 chars) | Tool returned |
| `done` | `finish_reason`, `n_tool_calls` | Agent finished (final answer / loop / max-iter) |
| `error` | `message` | On errors |

## Custom renderers

You can write your own `on_event` callback to render events however you
like — `RichLiveChat` is just one implementation. For example, to log
events to a file:

```python
import json

def log_event(event):
    with open("agent_events.jsonl", "a") as f:
        f.write(json.dumps(event) + "\n")

result = client.run_agent_turn("...", on_event=log_event)
```

## Caveats

- The dashboard uses `screen=False` (doesn't grab the alt-screen) so
  output scrolls normally. This is intentional — it lets you scroll
  back to see earlier events.
- `Ctrl+C` aborts the agent. The CLI catches `KeyboardInterrupt` and
  exits with code `130`.
- The final answer is printed to stdout AFTER `live.close()` so the
  dashboard doesn't clobber it.
