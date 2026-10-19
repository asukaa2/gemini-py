# MCP Servers

`gemini-py` can connect to external MCP (Model Context Protocol)
servers and expose their tools to the Gemini model alongside the
built-in ones.

This is a **client-side** port — we don't host MCP servers (the
original TypeScript gemini-cli does). Instead, we connect to servers
you provide and surface their tools.

## Adding an MCP server

### stdio server (most common)

```bash
# Basic
gemini-py mcp add filesystem npx -y @modelcontextprotocol/server-filesystem /tmp

# With env vars
gemini-py mcp add github npx -y @modelcontextprotocol/server-github \
  --env GITHUB_TOKEN=ghp_xxx

# With tool filter (only expose these tools)
gemini-py mcp add github npx -y @modelcontextprotocol/server-github \
  --include-tools list_repos,get_pr

# Save to project scope (./.gemini/mcp.json instead of ~/.gemini-py/mcp.json)
gemini-py mcp add filesystem npx -y ... --scope project

# Add but leave disabled
gemini-py mcp add my-server node server.js --disabled
```

The `--transport` flag defaults to `auto`, which detects HTTP from the
`http://` / `https://` prefix. Force a transport with
`--transport stdio|http`.

### HTTP server

```bash
gemini-py mcp add api-server https://example.com/mcp --transport http
```

### Command + args + flags

`mcp add` uses `nargs=REMAINDER` for the `cmd_args` positional, so
flags meant for the subprocess (like `npx -y`) survive intact:

```bash
# -y is passed to npx, not eaten by argparse
gemini-py mcp add filesystem npx -y @modelcontextprotocol/server-filesystem /tmp
```

## Managing servers

```bash
gemini-py mcp list
# MCP servers (2):
#   [enabled ] filesystem  [stdio] npx -y @modelcontextprotocol/server-filesystem /tmp
#   [enabled ] api-server  [http]  https://example.com/mcp

gemini-py mcp enable filesystem
gemini-py mcp disable filesystem
gemini-py mcp remove filesystem
```

## How MCP tools appear to the model

MCP tools are namespaced as `mcp_<server>__<tool>` to avoid collisions
with built-in tools. For example, the filesystem server's `read_file`
becomes `mcp_filesystem__read_file`.

When the model calls `mcp_filesystem__read_file(args)`, the agent loop:

1. Detects the `mcp_` prefix.
2. Routes the call to the connected `McpClient`.
3. The `McpClient` finds the right server session.
4. Calls the underlying `ClientSession.call_tool()`.
5. Returns the result text back to the model.

## Using MCP servers in an agent turn

Once a server is added (and enabled), it's automatically connected at
the start of every CLI agent run (unless `--no-mcp` is passed).

```bash
# Add the filesystem MCP server
gemini-py mcp add filesystem npx -y @modelcontextprotocol/server-filesystem /tmp

# Now any prompt can call its tools
gemini-py -p "List the files in /tmp"
```

### From Python

```python
from gemini_py import GeminiClient
from gemini_py.mcp import McpRegistry, McpClient

client = GeminiClient()
registry = McpRegistry()
mcp = McpClient(registry)
mcp.start()  # connect to all enabled servers

try:
    result = client.run_agent_turn(
        "List the files in /tmp",
        mcp_client=mcp,
    )
    print(result.text)
finally:
    mcp.stop()  # close all sessions
```

## Config file locations

Server configs are persisted to:

- `~/.gemini-py/mcp.json` (scope=user, default)
- `./.gemini/mcp.json` (scope=project)

Both are loaded on every run; project scope overrides user scope on
name conflicts.

### File format

```json
{
  "servers": [
    {
      "name": "filesystem",
      "transport": "stdio",
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
      "env": {},
      "cwd": null,
      "scope": "user",
      "include_tools": [],
      "enabled": true
    },
    {
      "name": "api-server",
      "transport": "http",
      "url": "https://example.com/mcp",
      "scope": "user",
      "include_tools": [],
      "enabled": true
    }
  ]
}
```

You can edit these files directly — the registry re-reads on every
`McpRegistry()` construction.

## Transports supported

| Transport | When to use |
| --- | --- |
| `stdio` | Default. Spawn a subprocess and communicate via stdin/stdout. Most MCP servers use this. |
| `http` | Connect to an HTTP server. Use for remote servers or shared services. |

Auto-detection: if the first non-flag arg looks like a URL (starts
with `http://` or `https://`), transport defaults to `http`.

## Tool discovery

When `McpClient.start()` runs, it:

1. For each enabled server, opens a connection (subprocess for stdio,
   HTTP session for http).
2. Calls `ClientSession.initialize()`.
3. Calls `ClientSession.list_tools()` to enumerate the server's tools.
4. Filters out tools not in `include_tools` (if set).
5. Stores the tool declarations in a cache.

When the model calls a tool, `McpClient.call_tool(full_name, args)`:

1. Splits `full_name` into `server_name` and `tool_name`.
2. Looks up the server's session.
3. Calls `session.call_tool(tool_name, args)`.
4. Extracts text from the response's `content` (handles TextContent,
   ImageContent, and StructuredContent as fallback).

## When things go wrong

| Symptom | Fix |
| --- | --- |
| `MCP server '<name>' failed to start: ...` | Check the command is on PATH, env vars are set, and the server actually runs. |
| Tool names don't appear | `gemini-py mcp list` to confirm `enabled`. Try `mcp enable <name>`. |
| `unknown MCP tool 'mcp_xxx__yyy'` | The server was disabled or removed mid-run. Restart the agent. |
| HTTP server gives 401 | Set auth headers (we don't yet expose a flag for this — file an issue). |

## What's NOT yet ported

The upstream gemini-cli's MCP integration also includes:

- **MCP resources** (read-only resources exposed by servers)
- **MCP prompts** (prompt templates exposed by servers)
- **MCP server hosting** (running your OWN MCP server)

We only support the **client side** (consuming tools from external
servers). If you need resources / prompts / hosting, file an issue.

## See also

- [MCP specification](https://modelcontextprotocol.io/)
- [`mcp` Python package](https://pypi.org/project/mcp/)
- [Tools](./tools.md) — the built-in tools
