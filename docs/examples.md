# Examples

Bigger worked examples showing how to use `gemini-py` for real tasks.

## Example 1: Summarize a file

```bash
cat README.md | gemini-py -p "Summarize this file in one paragraph"
```

Or from Python:

```python
from gemini_py import run_headless
text = open("README.md").read()
result = run_headless(f"Summarize this file in one paragraph:\n\n{text}")
print(result.text)
```

## Example 2: Build a small Flask app from scratch

```bash
gemini-py -p "Build a Flask todo API with /todos CRUD endpoints and a SQLite backend" --live --yolo
```

What happens (visible in the live dashboard):

1. Model thinks, then calls `create_directory(path="todo-api/")`.
2. Calls `write_file(path="todo-api/requirements.txt", content="...")` —
   you see the file contents in a syntax-highlighted code panel.
3. Calls `write_file(path="todo-api/app.py", content="...")` — same.
4. Calls `run_shell_command(command="cd todo-api && pip install -r requirements.txt")`.
5. Final answer: "I built a Flask todo API in `todo-api/`. Run it with
   `python todo-api/app.py`."

## Example 3: Diagnose a CI error log

```bash
# Pipe the error log into the agent
python my_app.py 2>&1 | gemini-py -p "Diagnose the error and suggest a fix" --live
```

The model will read the log via stdin (appended to the prompt), then
potentially call `read_file` to inspect the failing source, then
propose a fix.

## Example 4: Review code with a skill

First, set up the `code-reviewer` skill:

```bash
mkdir -p ~/.gemini-py/skills/code-reviewer
cat > ~/.gemini-py/skills/code-reviewer/SKILL.md <<'EOF'
# Code Reviewer

You are a meticulous code reviewer. Look for:
- Bugs and logic errors
- Security issues (SQL injection, XSS, etc.)
- Style violations
- Missing error handling
- Test coverage gaps

For each issue, output:
- File: path/to/file.py:LINE
- Severity: high / medium / low
- Issue: description
- Fix: suggested fix
EOF

# Verify it's discovered
gemini-py skills list
# Skills (1):
#   [enabled ] @code-reviewer  (user)  You are a meticulous code reviewer...
```

Now use it inline:

```bash
gemini-py -p "@code-reviewer Please review src/foo.py" --live
```

Or let the model decide to delegate:

```bash
gemini-py -p "Review src/foo.py using the code-reviewer skill" --live
```

## Example 5: Add an MCP filesystem server

```bash
# Add the filesystem MCP server
gemini-py mcp add filesystem npx -y @modelcontextprotocol/server-filesystem /tmp

# Verify it's connected (no-op for now, but confirms config)
gemini-py mcp list

# Now any prompt can use the filesystem server's tools
gemini-py -p "List the files in /tmp and tell me which is the largest"
```

## Example 6: Stream tokens to a custom UI

```python
from gemini_py import GeminiClient

client = GeminiClient()
for chunk in client.stream("Tell me a story about a robot."):
    print(chunk, end="", flush=True)
print()  # newline at the end
```

## Example 7: Live view with custom event handler

```python
import json
from gemini_py import GeminiClient

def log_event(event):
    """Log every event to a JSONL file."""
    with open("agent_events.jsonl", "a") as f:
        f.write(json.dumps(event) + "\n")
    # Print high-level events to the terminal too
    etype = event.get("type")
    if etype == "tool_call":
        print(f"\n[tool_call] {event['name']} {event['args']}")
    elif etype == "tool_result":
        print(f"[tool_result] {event['name']}: {event['result'][:100]}")
    elif etype == "done":
        print(f"\n[done] finish={event['finish_reason']} tool_calls={event['n_tool_calls']}")

client = GeminiClient()
result = client.run_agent_turn(
    "List Python files in the current dir",
    on_event=log_event,
)
print(result.text)
```

## Example 8: Custom tool (fetch weather)

```python
from gemini_py import GeminiClient
from gemini_py.tools import Tool, build_default_tools

def _fetch_weather(city: str, **_):
    # Replace with a real weather API call
    return f"Weather in {city}: 25°C, sunny"

weather_tool = Tool(
    name="fetch_weather",
    description="Get the current weather for a city.",
    parameters={
        "city": {
            "type": "string",
            "description": "City name.",
            "required": True,
        },
    },
    executor=_fetch_weather,
)

client = GeminiClient()
tools = build_default_tools() + [weather_tool]
result = client.run_agent_turn(
    "What's the weather in Jakarta?",
    tools=tools,
)
print(result.text)
```

## Example 9: JSON output for scripting

```bash
# Get a structured answer
result=$(gemini-py -p "List 5 Python web frameworks" -o json --no-tools)
echo "$result" | jq -r .response

# Or pipe to a file
gemini-py -p "..." -o json > result.json
```

## Example 10: Headless CI step

```yaml
# .github/workflows/review.yml
name: AI Code Review
on: [pull_request]
jobs:
  review:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Install gemini-py
        run: pip install -e .
      - name: Login
        env:
          GEMINI_API_KEY: ${{ secrets.GEMINI_API_KEY }}
        run: gemini-py login --api-key "$GEMINI_API_KEY"
      - name: Review
        run: |
          git diff origin/main...HEAD | gemini-py -p "Review this diff for bugs and security issues" -o json --no-tools > review.json
          cat review.json
```

## Example 11: Multi-turn chat

```python
from gemini_py import GeminiClient, Message, MessageRole

client = GeminiClient()
history = [
    Message(MessageRole.USER, "Hello! What's your name?"),
    Message(MessageRole.MODEL, "I'm gemini-py. How can I help?"),
    Message(MessageRole.USER, "What's 2+2?"),
]
result = client.chat(history)
print(result.text)
# 4
```

## Example 12: Subagent delegation (programmatic)

```python
from gemini_py import GeminiClient
from gemini_py.skills.subagent import delegate_to_subagent
from gemini_py.skills import SkillsLoader

client = GeminiClient()
loader = SkillsLoader()

# Verify the code-reviewer skill is loaded
assert loader.get("code-reviewer") is not None, "Create the skill first"

result = delegate_to_subagent(
    parent_client=client,
    skill_name="code-reviewer",
    task="Review src/foo.py for bugs and security issues",
    max_iterations=6,
)
if result.success:
    print("Sub-agent's review:")
    print(result.text)
    print(f"\n(Sub-agent made {len(result.tool_results)} tool calls)")
else:
    print(f"Sub-agent failed: {result.error}")
```

## Example 13: Auto-project scaffolding

Just ask the AI to build something — it will auto-create a project
folder:

```bash
gemini-py -p "Build a simple CLI todo app in Python" --live --yolo
```

What happens:

1. Model: "I'll create a `todo-cli/` folder..."
2. Calls `create_directory(path="todo-cli/")`.
3. Calls `write_file(path="todo-cli/main.py", content="...")`.
4. Calls `write_file(path="todo-cli/README.md", content="...")`.
5. Final answer: "Built `todo-cli/`. Run it with `python todo-cli/main.py`."

The system prompt explicitly instructs the model to auto-create a
project folder when no directory is specified.

## Example 14: Combine everything

```bash
# Set up auth once
gemini-py login --api-key AIza...

# Add skills + MCP servers
mkdir -p ~/.gemini-py/skills/code-reviewer
echo "# Code Reviewer\nYou review code for bugs." > ~/.gemini-py/skills/code-reviewer/SKILL.md
gemini-py mcp add filesystem npx -y @modelcontextprotocol/server-filesystem /tmp

# Set up project GEMINI.md
mkdir -p .gemini
echo "# Project: my-app\nUse Flask + pytest." > .gemini/GEMINI.md

# Run with everything
gemini-py -p "Review my Flask app for bugs and write a fix" --live --yolo
```

The agent will:
1. Load `GEMINI.md` auto-memory (project conventions).
2. List available skills (including `code-reviewer`).
3. Connect to the filesystem MCP server (tools appear as `mcp_filesystem__*`).
4. Likely delegate the review to `code-reviewer` via `delegate_subagent`.
5. Apply the fix using `write_file`.
