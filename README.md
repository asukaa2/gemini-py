# gemini-py

A Python SDK and headless CLI for **Google Gemini** — a Python port of
[google-gemini/gemini-cli](https://github.com/google-gemini/gemini-cli) that
**uses CLI args, not a TUI/GUI**.

This package wraps the official [`google-genai`](https://pypi.org/project/google-genai/)
SDK and exposes:

1. A **GeminiClient** Python class for programmatic use, and
2. A **`gemini-py` command-line tool** (argparse-based, no GUI) that mirrors
   the most-used flags from the original `gemini` CLI:
   `-p / --prompt`, `-m / --model`, `-o / --output-format`, `-r / --resume`,
   `--list-sessions`, `--approval-mode`, etc.

## Installation

```bash
pip install -e .
# or, after publishing:
pip install gemini-py
```

## Authentication

Same environment variables as the official CLI:

| Method                     | Env vars required                                              |
| -------------------------- | ------------------------------------------------------------- |
| Gemini API Key (default)   | `GEMINI_API_KEY`                                              |
| Vertex AI with ADC         | `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION`, ADC          |
| Vertex AI with SA key      | `GOOGLE_APPLICATION_CREDENTIALS`, `GOOGLE_CLOUD_PROJECT`, ... |
| Google Cloud API key       | `GOOGLE_API_KEY`, `GOOGLE_CLOUD_PROJECT`                      |

`.env` files are auto-loaded from (in order):
1. `./.env`
2. `./.gemini/.env`
3. `~/.gemini/.env`

## CLI usage

```bash
# One-shot non-interactive prompt
gemini-py -p "Explain how HTTP/3 works"

# Pipe stdin into the prompt
cat README.md | gemini-py -p "Summarize this file"

# JSON output for scripting
gemini-py -p "What time is it in Jakarta?" --output-format json

# Streaming JSON events (JSONL)
gemini-py -p "List 3 Python tips" --output-format stream-json

# Choose a model alias (auto, pro, flash, flash-lite) or a concrete model
gemini-py -p "Quick fact" --model flash

# Resume the most recent session
gemini-py -r latest -p "What did we discuss?"

# Disable all built-in tools (text-only)
gemini-py -p "Hello" --no-tools

# Approve tool calls automatically
gemini-py -p "List files" --yolo

# List / delete sessions
gemini-py sessions list
gemini-py sessions delete 0
```

### Full flag reference

| Flag                  | Alias | Description                                            |
| --------------------- | ----- | ----------------------------------------------------- |
| `--prompt`            | `-p`  | Prompt text (forces non-interactive mode)             |
| `--prompt-interactive`| `-i`  | Run prompt then stop (no TUI in this Python port)     |
| `--model`             | `-m`  | Model alias or name (default: `auto`)                |
| `--output-format`     | `-o`  | `text` (default), `json`, or `stream-json`           |
| `--resume`            | `-r`  | Resume session: `latest`, index, or ID              |
| `--list-sessions`     |       | List sessions for current project and exit           |
| `--delete-session`   |       | Delete session by index                              |
| `--include-directories` |     | Extra workspace dirs (comma-separated)               |
| `--sandbox`           |       | (Stub) Sandbox mode                                   |
| `--approval-mode`     |       | `default` / `auto_edit` / `yolo` / `plan`            |
| `--yolo`              | `-y`  | Auto-approve all tool calls                           |
| `--no-tools`          |       | Disable all built-in tools                           |
| `--no-shell`          |       | Disable only the shell tool                          |
| `--enable-web`        |       | Enable web_search + web_fetch tools                  |
| `--system-prompt`     |       | Override system prompt. Use `-` to read from stdin   |
| `--max-iterations`    |       | Max tool-call round-trips (default: 10)              |
| `--max-output-tokens` |       | Max output tokens (default: 8192)                     |
| `--temperature`       |       | Sampling temperature (default: 1.0)                  |
| `--top-p`             |       | Top-p nucleus sampling (default: 0.95)              |
| `--debug`             | `-d`  | Verbose logging                                       |
| `--version`            | `-v` | Print version and exit                                |
| `--help`              | `-h`  | Show help                                             |

## SDK usage

```python
from gemini_py import GeminiClient, GeminiConfig, ModelAlias, run_headless

# One-liner
print(run_headless("Summarize the Python release notes.").text)

# Full control
client = GeminiClient(
    config=GeminiConfig(
        model=ModelAlias.FLASH,         # or a concrete model name
        temperature=0.7,
        enable_shell_tool=True,
        enable_filesystem_tools=True,
    )
)

# Simple generation
r = client.generate_text("Explain what 2 + 2 equals.")
print(r.text, r.usage.to_dict())

# Agentic flow with built-in tools (filesystem + shell + optional web)
result = client.run_agent_turn(
    "List the Python files in this directory and tell me how many there are.",
    max_iterations=5,
)
print(result.text)
for tr in result.tool_results:
    print(f"tool={tr['name']} result={tr['result'][:200]}")

# Streaming
for chunk in client.stream("Tell me a story about a robot."):
    print(chunk, end="", flush=True)
```

## Built-in tools

| Tool                 | Description                                      |
| -------------------- | ------------------------------------------------ |
| `list_directory`     | List files and subdirectories in a directory    |
| `read_file`          | Read UTF-8 text file contents                    |
| `write_file`         | Write content to a file (overwrite)             |
| `grep_search`        | Recursively search files for a regex             |
| `run_shell_command`  | Run a shell command and return stdout/stderr    |
| `web_search`         | Search the web (no API key; basic scraper)      |
| `web_fetch`          | Fetch and strip a web page to plain text         |

## What's **not** ported

This is a focused Python port — the following are intentionally out of scope:

- The interactive TUI (Ink/React-based) — the user explicitly requested CLI args only.
- The extensions system (gemini-cli's TypeScript plugin format).
- The MCP server registry (use `mcp` Python packages directly).
- Sandbox mode (Docker/Podman-based) — flag is accepted but ignored.
- Auto-memory / GEMINI.md hierarchical context (planned).
- Skills / subagents.

## License

Apache-2.0 (same as the upstream gemini-cli).
