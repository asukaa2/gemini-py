# CLI Reference

Complete reference for every `gemini-py` flag and subcommand.

## Usage

```
gemini-py [-p PROMPT | QUERY...] [OPTIONS]
gemini-py sessions <list|delete|resume>
gemini-py skills <list|show|enable|disable|enable-all|disable-all>
gemini-py mcp <list|add|remove|enable|disable>
gemini-py login [--api-key KEY | --method gemini|google | ...]
gemini-py logout [--scope user|project]
gemini-py whoami
gemini-py extensions list
```

## Prompt sources

The CLI resolves the prompt in this order:

1. `--prompt / -p` — explicit prompt text. Forces non-interactive mode.
2. `--prompt-interactive / -i` — execute prompt then stop (this Python
   port has no TUI; same as `-p`).
3. Positional `query` — any remaining positional args are joined with
   spaces.
4. stdin — if no `-p` and stdin is piped (not a TTY), the entire stdin
   is appended to the prompt.

## Global options

| Flag | Alias | Type | Default | Description |
| --- | --- | --- | --- | --- |
| `--prompt` | `-p` | string | — | Prompt text (forces non-interactive). |
| `--prompt-interactive` | `-i` | string | — | Run prompt then stop (no TUI in this port). |
| `--model` | `-m` | string | `auto` | Model alias or concrete model name. |
| `--output-format` | `-o` | enum | `text` | `text` \| `json` \| `stream-json`. |
| `--resume` | `-r` | string | — | Resume session by ID, `latest`, or index. |
| `--list-sessions` | | flag | — | List sessions for the current project and exit. |
| `--delete-session` | | string | — | Delete session by index. |
| `--include-directories` | | string | — | Extra workspace dirs (comma-separated). |
| `--sandbox` | `-s` | flag | `false` | (Stub) Run in a sandboxed environment. |
| `--approval-mode` | | enum | `default` | `default` \| `auto_edit` \| `yolo` \| `plan`. |
| `--yolo` | `-y` | flag | — | Alias of `--approval-mode=yolo`. |
| `--skip-trust` | | flag | — | Skip folder trust check (always yes). |
| `--debug` | `-d` | flag | — | Verbose logging. |
| `--version` | `-v` | — | — | Print version and exit. |
| `--help` | `-h` | — | — | Show help and exit. |

## Tool-control options

| Flag | Default | Description |
| --- | --- | --- |
| `--no-tools` | off | Disable all built-in tools (text-only). |
| `--no-shell` | off | Disable only the shell tool. |
| `--enable-web` | off | Enable `web_search` + `web_fetch` tools. |
| `--no-memory` | off | Disable hierarchical `GEMINI.md` auto-memory. |
| `--no-skills` | off | Disable `@skill_name` expansion and skill listing. |
| `--no-subagent` | off | Disable the `delegate_subagent` tool. |
| `--no-mcp` | off | Do not connect to configured MCP servers. |

## Live view

| Flag | Default | Description |
| --- | --- | --- |
| `--live` | off | Render a `rich.live.Live` dashboard with streaming thinking, tool calls, and tool results in real-time. |

## Generation options

| Flag | Default | Description |
| --- | --- | --- |
| `--max-iterations` | `25` | Max tool-call round-trips in agent mode. |
| `--max-output-tokens` | `8192` | Max output tokens per generation. |
| `--temperature` | `1.0` | Sampling temperature. |
| `--top-p` | `0.95` | Top-p nucleus sampling. |
| `--system-prompt` | — | Override the system prompt. Use `-` to read from stdin. |

## Subcommands

Subcommands are detected BEFORE the main parser runs (so the positional
`query` doesn't swallow them). All subcommands have their own `--help`.

### `sessions`

```
gemini-py sessions list
gemini-py sessions delete INDEX
gemini-py sessions resume INDEX|latest
```

Saves conversations to `~/.gemini-py/sessions/<id>.json`. See
[Sessions](./sessions.md).

### `skills`

```
gemini-py skills list
gemini-py skills show NAME
gemini-py skills enable NAME
gemini-py skills disable NAME
gemini-py skills enable-all
gemini-py skills disable-all
```

Discovers skills from `~/.gemini-py/skills/<name>/SKILL.md` and
`./.gemini/skills/<name>/SKILL.md`. See [Skills](./skills.md).

### `mcp`

```
gemini-py mcp list
gemini-py mcp add NAME COMMAND_OR_URL [CMD_ARGS...] \
  [--transport stdio|http|auto] \
  [--env KEY=value ...] \
  [--cwd PATH] \
  [--scope user|project] \
  [--include-tools a,b,c] \
  [--disabled]
gemini-py mcp remove NAME
gemini-py mcp enable NAME
gemini-py mcp disable NAME
```

`cmd_args` uses `nargs=REMAINDER` so flags meant for the subprocess
(like `npx -y`) survive intact. See [MCP Servers](./mcp-servers.md).

### `login`

```
gemini-py login [--api-key KEY] \
  [--method gemini|google] \
  [--project PROJECT_ID] \
  [--location us-central1] \
  [--scope user|project]
```

If `--api-key` is omitted and stdin is a TTY, the key is read via
`getpass` (hidden input). If stdin is piped, the first line is used.

See [Authentication](./authentication.md).

### `logout`

```
gemini-py logout [--scope user|project]
```

Removes the persisted `.env` file and clears the env vars from the
current process.

### `whoami`

```
gemini-py whoami
```

Prints the currently-active credentials (masked). Returns exit code 0
even when no credentials are configured.

### `extensions`

```
gemini-py extensions list
```

Stub — this Python port does not support the TypeScript-style
extensions system. Use Python packages directly.

## Exit codes

| Code | Meaning |
| --- | --- |
| `0` | Success. |
| `1` | General error (API failure, auth not configured, etc.). |
| `2` | Argparse error (bad flag). |
| `42` | Input error (invalid prompt or arguments). |
| `53` | Turn limit exceeded (`--max-iterations` reached). |
| `130` | Ctrl+C interrupted. |

## Output formats

### `text` (default)

Prints the model's final answer text directly to stdout. Tool calls
and intermediate results are NOT printed (use `--live` for that).

### `json`

Prints a single JSON object to stdout with this schema:

```json
{
  "response": "...",
  "stats": {
    "prompt_token_count": 7,
    "candidates_token_count": 42,
    "total_token_count": 49,
    "cached_content_token_count": 0,
    "thoughts_token_count": 0
  },
  "model": "gemini-3.8-flash",
  "finish_reason": "STOP",
  "tool_calls": [],
  "tool_results": [...]
}
```

### `stream-json`

Prints newline-delimited JSON events (JSONL) to stdout. Event types:

- `init` — session metadata (model).
- `message` — user and assistant message chunks.
- `tool_call` — tool call requests with arguments.
- `tool_result` — output from executed tools.
- `result` — final outcome with aggregated stats.

## Examples

### Quick text answer

```bash
gemini-py -p "What's the capital of Indonesia?"
# Jakarta
```

### JSON for scripting

```bash
result=$(gemini-py -p "Get the current weather in Jakarta" -o json --no-tools)
echo "$result" | jq -r .response
```

### Live dashboard with agentic mode

```bash
gemini-py -p "Build a simple Express todo API" --live --yolo
```

### Resume a session

```bash
gemini-py sessions list
# Sessions for /home/user/projects/todo:
#   [0] abc123  updated=2026-10-19 10:30:00  messages=5  model=gemini-3.8-flash

gemini-py -r 0 -p "Add a DELETE endpoint"
```

### Pipe an error log

```bash
python app.py 2>&1 | gemini-py -p "Diagnose the error and suggest a fix" --live
```
