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

### One-call `login()` API (v0.3+)

Instead of manually exporting env vars, you can authenticate once and have
the credentials auto-loaded by every future `GeminiClient()` call:

```python
from gemini_py import login, logout, whoami, GeminiClient

login("AIza...")            # writes GEMINI_API_KEY to ~/.gemini-py/.env
                            # (also sets it in the current process)

client = GeminiClient()     # auto-picks up the saved key
print(client.generate_text("Hello!").text)

print(whoami())
# {'method': 'gemini_api_key', 'api_key': 'AIza****...abcd', 'project': None, ...}

logout()                    # removes the persisted credentials
```

For Vertex AI:

```python
login(
    "ya29.abc...",
    method="google",
    project="my-gcp-project",
    location="us-central1",
)
```

CLI equivalents:

```bash
gemini-py login --api-key AIza...           # non-interactive
gemini-py login                              # interactive (hidden prompt)
gemini-py login --method google --project my-proj --api-key ya29.abc...
gemini-py whoami                             # show masked credentials
gemini-py logout                             # clear credentials
```

Credentials are persisted to `~/.gemini-py/.env` (scope=user) or
`./.gemini/.env` (scope=project). Files are chmod'd to 0600 when possible.
The `.env` files are loaded general→specific (so a project `.env` overrides
the user's saved credentials).

`.env` files are auto-loaded from (in order, later overrides earlier):

1. `~/.gemini-py/.env`     ← where `login()` writes by default
2. `~/.gemini/.env`         ← gemini-cli convention
3. `./.gemini/.env`         ← project-scoped
4. `./.env`                 ← project root (most specific)

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

# Authenticate once, use everywhere
gemini-py login --api-key AIza...
gemini-py whoami
gemini-py logout
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
| `--no-memory`         |       | Disable hierarchical GEMINI.md auto-memory          |
| `--no-skills`         |       | Disable @skill_name expansion + skill listing       |
| `--no-subagent`       |       | Disable the delegate_subagent tool                   |
| `--no-mcp`            |       | Do not connect to configured MCP servers             |
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
- Sandbox mode (Docker/Podman-based) — flag is accepted but ignored.

## Ported subsystems (v0.2)

The following features that were originally "not ported" have now been
ported in this release:

### MCP server registry

Connect to external MCP servers (stdio or HTTP) and expose their tools to
the Gemini model as regular function-call tools.

```bash
# Add a stdio server (auto-detected)
gemini-py mcp add filesystem npx -y @modelcontextprotocol/server-filesystem /tmp

# Add an HTTP server
gemini-py mcp add api-server https://example.com/mcp --transport http

# Add with env vars + tool filter
gemini-py mcp add github npx -y @modelcontextprotocol/server-github \
  --env GITHUB_TOKEN=ghp_xxx --include-tools list_repos,get_pr

# List / enable / disable / remove
gemini-py mcp list
gemini-py mcp enable filesystem
gemini-py mcp disable filesystem
gemini-py mcp remove filesystem
```

MCP tool names are namespaced as `mcp_<server>__<tool>` to avoid collisions
with built-in tools. Configs are persisted to `~/.gemini-py/mcp.json`
(scope=user) or `./.gemini/mcp.json` (scope=project).

To skip MCP in a single run: `gemini-py -p "..." --no-mcp`.

### Auto-memory (GEMINI.md hierarchy)

Hierarchical `GEMINI.md` files are loaded and injected into the system
prompt on every agent turn. The search order is:

1. `~/.gemini/GEMINI.md` (user-wide preferences)
2. Each parent of cwd up to home (project hierarchy)
3. `cwd/.gemini/GEMINI.md` (project-scoped)
4. `cwd/GEMINI.md` (most specific, project root)

Each chunk is rendered with a `# Memory: <path>` header. Use
`# gemini-py-ignore` in a file to stop processing from that point.

To disable: `gemini-py -p "..." --no-memory`.

### Skills & subagents

A **skill** is a directory containing a `SKILL.md` file under
`~/.gemini-py/skills/<name>/` or `./.gemini/skills/<name>/`. Skills are
loaded automatically and exposed to the model in two ways:

1. **`@skill_name` expansion** — any `@skill_name` token in your prompt
   is replaced inline with the skill's full content.
2. **`delegate_subagent(skill, task)` tool** — the model can spawn a
   sub-agent that runs a fresh agent turn with the skill's content as its
   system prompt and returns the sub-agent's final answer.

```bash
# Discover skills
gemini-py skills list
gemini-py skills show code-reviewer
gemini-py skills enable code-reviewer
gemini-py skills disable code-reviewer
gemini-py skills enable-all
gemini-py skills disable-all

# Use a skill inline
gemini-py -p "@code-reviewer Please review src/foo.py"

# Let the model delegate
gemini-py -p "Review src/foo.py using the code-reviewer skill"
```

To disable: `--no-skills` (no @-expansion + no skill listing),
`--no-subagent` (no delegate_subagent tool).

### Skill file format

```markdown
# <Skill Name>

A one-paragraph description (becomes the skill's description in the listing).

The rest of the file is the skill's full prompt content. When @skill is
used in a prompt or a subagent is delegated, this content is injected
verbatim.
```

## License

Apache-2.0 (same as the upstream gemini-cli).
