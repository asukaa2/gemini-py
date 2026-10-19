# Quickstart

Minimal, copy-pasteable recipes for the most common things you'll want
to do with `gemini-py`.

## CLI basics

```bash
# One-shot prompt (non-interactive, text output)
gemini-py -p "Explain how HTTP/3 works"

# Pipe stdin into the prompt
cat README.md | gemini-py -p "Summarize this file"

# Choose a model alias (auto, argon, pro, flash, flash-lite, ...)
gemini-py -p "Quick fact" --model flash
gemini-py -p "Complex reasoning" --model argon

# JSON output for scripting
gemini-py -p "What time is it in Jakarta?" --output-format json

# Streaming JSONL events (one event per line)
gemini-py -p "List 3 Python tips" --output-format stream-json

# Live dashboard (rich.live.Live + Layout)
gemini-py -p "Build a simple Flask todo app" --live

# Disable built-in tools (text-only — no filesystem/shell access)
gemini-py -p "Hello" --no-tools

# Auto-approve all tool calls (no confirmation prompts)
gemini-py -p "List files" --yolo

# Specify a concrete model name (skip aliases)
gemini-py -p "Hello" -m gemini-3.1-pro
```

## Python SDK basics

```python
from gemini_py import GeminiClient, GeminiConfig, ModelAlias, run_headless

# One-liner for ad-hoc prompts
print(run_headless("Summarize the Python release notes.").text)

# Or construct a client once and reuse it
client = GeminiClient(
    config=GeminiConfig(
        model=ModelAlias.FLASH,         # alias
        temperature=0.7,
        enable_shell_tool=True,
        enable_filesystem_tools=True,
    )
)

# Simple text generation (no tools)
r = client.generate_text("Explain what 2 + 2 equals.")
print(r.text)
print(r.usage.to_dict())
# {'prompt_token_count': 7, 'candidates_token_count': 5, 'total_token_count': 12, ...}
```

## Agentic flow with tool-calling

```python
result = client.run_agent_turn(
    "List the Python files in this directory and tell me how many there are.",
    max_iterations=5,
)
print(result.text)
for tr in result.tool_results:
    print(f"tool={tr['name']} result={tr['result'][:200]}")
```

By default, `run_agent_turn` enables:
- Auto-memory (hierarchical `GEMINI.md` files)
- Skills (`@skill_name` expansion)
- Subagent tool (`delegate_subagent(skill, task)`)
- Built-in tools (`list_directory`, `read_file`, `write_file`,
  `create_directory`, `grep_search`, `run_shell_command`)

You can opt out individually:

```python
client.run_agent_turn(
    "Just answer this — don't touch my filesystem",
    enable_memory=False,
    enable_skills=False,
    enable_subagent=False,
    # tools=[]  would also disable all built-in tools
)
```

## Live view from Python

```python
from gemini_py import GeminiClient
from gemini_py.live import make_default_live_view

live = make_default_live_view(enabled=True, model="gemini-3.8-flash", max_iterations=25)
try:
    result = client.run_agent_turn(
        "Build a simple todo CLI app",
        live=True,
        on_event=live.emit,
    )
finally:
    live.close()

print(result.text)  # final answer goes to stdout after the dashboard closes
```

## Login / logout / whoami

```python
from gemini_py import login, logout, whoami

login("AIza...")               # writes to ~/.gemini-py/.env
print(whoami())                # masked credentials
logout()                       # removes credentials
```

## Streaming tokens (for interactive REPLs)

```python
for chunk in client.stream("Tell me a story about a robot."):
    print(chunk, end="", flush=True)
```

## Multi-turn chat

```python
from gemini_py import Message, MessageRole

history = [
    Message(role=MessageRole.USER, content="Hello, what's your name?"),
    Message(role=MessageRole.MODEL, content="I'm gemini-py. How can I help?"),
    Message(role=MessageRole.USER, content="What's 2+2?"),
]
result = client.chat(history)
print(result.text)
```

## Sessions

```bash
# After running a prompt, a session is saved automatically.
gemini-py -p "Build a simple app"

# List saved sessions for the current project
gemini-py sessions list

# Resume the most recent one
gemini-py -r latest -p "Add a README to the app you just built"

# Resume by index (from `sessions list`)
gemini-py -r 0 -p "Now add tests"

# Delete a session
gemini-py sessions delete 0
```

## Skills

```bash
# Discover skills (lives in ~/.gemini-py/skills/<name>/SKILL.md
# or ./.gemini/skills/<name>/SKILL.md)
gemini-py skills list

# Show a skill's full prompt content
gemini-py skills show code-reviewer

# Enable / disable / enable-all / disable-all
gemini-py skills enable code-reviewer
gemini-py skills disable-all
```

Use a skill inline with `@skill_name`:

```bash
gemini-py -p "@code-reviewer Please review src/foo.py"
```

Or let the model delegate to a subagent:

```bash
gemini-py -p "Review src/foo.py using the code-reviewer skill"
```

## MCP servers

```bash
# Add a stdio MCP server (auto-detects transport from URL)
gemini-py mcp add filesystem npx -y @modelcontextprotocol/server-filesystem /tmp

# Add an HTTP MCP server
gemini-py mcp add api-server https://example.com/mcp --transport http

# Add with env vars and tool filter
gemini-py mcp add github npx -y @modelcontextprotocol/server-github \
  --env GITHUB_TOKEN=ghp_xxx --include-tools list_repos,get_pr

# List / enable / disable / remove
gemini-py mcp list
gemini-py mcp enable filesystem
gemini-py mcp disable filesystem
gemini-py mcp remove filesystem
```

Once added, MCP tools appear alongside built-in tools (namespaced as
`mcp_<server>__<tool>`).

## Common flag combos

| Goal | Command |
| --- | --- |
| Quick text answer, no tools | `gemini-py -p "..." --no-tools --model flash` |
| Code review with live dashboard | `gemini-py -p "@code-reviewer review src/" --live` |
| Pipe stdin as extra context | `cat log.txt \| gemini-py -p "diagnose this"` |
| JSON output for a CI step | `gemini-py -p "..." -o json --no-tools` |
| Agentic, auto-approve, live | `gemini-py -p "build X" --live --yolo` |
