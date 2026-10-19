# Tools

`gemini-py` comes with a set of built-in tools the model can call during
an agent turn. You can also write custom tools and integrate external
MCP servers.

## Built-in tools

| Tool | Description |
| --- | --- |
| `list_directory(path)` | List files and subdirectories in a directory. |
| `read_file(path)` | Read UTF-8 text file contents. |
| `write_file(path, content)` | Write content to a file. Parent directories are auto-created. |
| `create_directory(path)` | Create a directory (and any missing parents). |
| `grep_search(pattern, path)` | Recursively search files for a regex pattern. |
| `run_shell_command(command, timeout)` | Run a shell command and return stdout/stderr. |
| `web_search(query)` | Search the web (basic scraper, no API key). |
| `web_fetch(url)` | Fetch and strip a web page to plain text. |
| `delegate_subagent(skill, task)` | Spawn a focused subagent driven by a skill. |

The `web_search` and `web_fetch` tools are OFF by default (to avoid
unexpected network calls). Enable with `--enable-web` (CLI) or
`enable_web=True` (`build_default_tools()`).

## Tool control flags (CLI)

| Flag | Effect |
| --- | --- |
| `--no-tools` | Disable ALL built-in tools (text-only mode). |
| `--no-shell` | Disable only `run_shell_command`. |
| `--enable-web` | Enable `web_search` + `web_fetch`. |
| `--no-subagent` | Disable `delegate_subagent`. |
| `--no-mcp` | Don't connect to configured MCP servers. |

## Tool control from Python

```python
from gemini_py import GeminiClient, GeminiConfig
from gemini_py.tools import build_default_tools

client = GeminiClient(config=GeminiConfig(
    enable_shell_tool=True,        # default
    enable_filesystem_tools=True,  # default
    enable_web_search=False,       # default
    enable_web_fetch=False,        # default
))

# Or override the tool list entirely:
tools = build_default_tools(
    enable_shell=True,
    enable_filesystem=True,
    enable_web=True,
)
result = client.run_agent_turn("List Python files", tools=tools)
```

## How tool-calling works

`GeminiClient.run_agent_turn()` runs a loop:

1. Send the conversation + tool declarations to the model.
2. The model returns text and/or function calls.
3. For each function call:
   - Dispatch to the right tool executor (built-in, MCP, or subagent).
   - Capture the result.
   - Append the result back to the conversation as a `function_response`.
4. Repeat until the model returns text without function calls (final
   answer), or `max_iterations` is reached.

The agent loop also:
- Preserves the original `Part` objects so `thought_signature` survives
  (the Gemini API requires this on function-call parts).
- Forwards the `id` field from `function_call` to `function_response`
  so the API can link request ↔ response.
- Detects when the model is stuck repeating the same call (same name +
  sorted args, 3+ times) and breaks with `finish_reason="loop_detected"`.

## Writing a custom tool

A `Tool` is a dataclass with a `name`, `description`, JSON-schema
`parameters`, and an `executor` callable.

```python
from gemini_py.tools import Tool, build_default_tools, tool_declarations

def _fetch_weather(city: str, **_):
    # Your weather API call here
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

# Combine with built-in tools
tools = build_default_tools() + [weather_tool]
result = client.run_agent_turn("What's the weather in Jakarta?", tools=tools)
```

### Conventions

- The `parameters` dict uses a per-property `"required": True` flag (a
  gemini-py convention). `Tool.to_google_decl()` lifts these into a
  top-level `required: [list]` and strips the per-property flag —
  required by the google-genai SDK's pydantic schema.
- The `executor` is called with the model's `args` as kwargs. Unknown
  kwargs are swallowed via `**_` (we recommend this pattern).
- Return a string from the executor. Non-string return values are
  coerced to `str()`.
- For errors, return a string starting with `"Error:"` — the live view
  color-codes these red.

## Direct tool usage (no model)

You can also call the tools directly from Python without going through
the model:

```python
from gemini_py.tools import build_default_tools, find_tool
from gemini_py.tools.shell import run_shell_command
from gemini_py.tools.websearch import web_search, web_fetch

# Direct shell
print(run_shell_command("echo hello"))

# Direct web
print(web_search("Python 3.13 features"))
print(web_fetch("https://example.com"))

# Or via the Tool registry
tools = build_default_tools()
read_file = find_tool(tools, "read_file")
print(read_file.executor(path="README.md"))
```

## MCP integration

External MCP servers expose their own tools. Once you add a server
(see [MCP Servers](./mcp-servers.md)), its tools appear alongside
the built-in ones, namespaced as `mcp_<server>__<tool>`.

```bash
# Add the filesystem MCP server
gemini-py mcp add filesystem npx -y @modelcontextprotocol/server-filesystem /tmp

# Now in any agent turn, the model can call mcp_filesystem__read_file etc.
gemini-py -p "List files in /tmp"
```

## Tool declarations format

Internally, tools are converted to the `google.genai` SDK's
`function_declarations` format:

```python
{
    "name": "read_file",
    "description": "Read the UTF-8 text contents of a file.",
    "parameters": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the file."}
        },
        "required": ["path"],
    },
}
```

Use `Tool.to_google_decl()` to get this form for a single tool, or
`tool_declarations(tools)` for the whole list wrapped in the
`{"function_declarations": [...]}` envelope.

## Subagent tool

The `delegate_subagent(skill, task)` tool is special: it spawns a
fresh `GeminiClient.run_agent_turn()` call with the skill's content as
the system prompt and returns the sub-agent's final answer.

```python
# Python — direct subagent invocation
from gemini_py.skills.subagent import delegate_to_subagent
from gemini_py import GeminiClient

client = GeminiClient()
result = delegate_to_subagent(
    parent_client=client,
    skill_name="code-reviewer",
    task="Review src/foo.py",
)
print(result.text)
```

The subagent tool is only exposed when at least one skill is enabled.
Disable with `--no-subagent` (CLI) or `enable_subagent=False` (Python).

See [Skills](./skills.md) for the full skills system.
