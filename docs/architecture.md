# Architecture

How `gemini-py` is structured internally.

## Module layout

```
gemini_py/
├── __init__.py         — public exports + version
├── __main__.py         — `python -m gemini_py` entry
├── auth.py             — AuthConfig: env vars + .env loading
├── client.py           — GeminiClient (main client)
├── config.py           — GeminiConfig + ModelAlias enum
├── cli.py              — argparse CLI (top-level + subcommands)
├── live.py             — RichLiveChat + LiveView (live dashboard)
├── login.py            — login()/logout()/whoami() persistent credentials
├── mcp/
│   └── __init__.py     — McpRegistry + McpClient + McpServerConfig
├── memory.py           — MemoryLoader (hierarchical GEMINI.md)
├── models.py           — Message, GenerateResult, UsageStats dataclasses
├── output.py           — OutputFormat + format_result()
├── prompts.py          — default system prompt builder
├── sessions.py         — SessionStore + SessionRecord
├── skills/
│   ├── __init__.py     — SkillsLoader + @skill_name expansion
│   └── subagent.py     — delegate_to_subagent()
└── tools/
    ├── __init__.py     — Tool dataclass + registry + execute_tool_call
    ├── filesystem.py   — path-safe resolution helpers
    ├── shell.py        — run_shell_command()
    └── websearch.py    — web_search() + web_fetch()
```

## The agent loop

`GeminiClient.run_agent_turn()` is the core flow:

```
┌──────────────────────────────────────────────────────────────┐
│ 1. Build tools list (built-in + subagent + MCP declarations) │
│ 2. Build system instruction:                                  │
│      default system prompt                                   │
│      + GEMINI.md auto-memory                                 │
│      + extra context (stdin)                                 │
│      + skills summary                                        │
│ 3. Expand @skill_name tokens in the user's prompt            │
│ 4. contents = [{"role": "user", "parts": [{"text": prompt}]}]│
│                                                               │
│ 5. for iteration in range(max_iterations):                   │
│      A. emit "iteration" event                                │
│      B. Call model.generate_content (or _stream)             │
│      C. Extract text + function_calls from response          │
│      D. If no function_calls: emit "done", return final answer│
│      E. Loop detection: break if same call 3x                │
│      F. Append model_parts (original Parts, preserve         │
│         thought_signature) to contents                      │
│      G. For each function_call:                              │
│           - emit "tool_call" event                           │
│           - For write_file: emit "file_write" event          │
│           - Dispatch to built-in / MCP / subagent            │
│           - emit "tool_result" event                        │
│      H. Append function_response parts to contents          │
│      I. Continue loop                                       │
│                                                               │
│ 6. If max_iterations reached:                                │
│      emit "error", return diagnostic                         │
└──────────────────────────────────────────────────────────────┘
```

## Tool dispatch

`GeminiClient._dispatch_tool_call()` routes by name:

1. **Built-in tool**: name matches a `Tool` in our `tools` list →
   `execute_tool_call(tools, fc)`.
2. **Subagent tool**: `delegate_subagent` is registered as a built-in
   tool when skills are enabled.
3. **MCP tool**: name starts with `mcp_` and an McpClient is provided
   → `mcp_client.call_tool(name, args)`.
4. **Unknown**: return an error string.

Each branch builds a `{"function_response": {"name", "response", "id"}}`
dict that gets appended to the conversation.

## Streaming path (`live=True`)

When `live=True`, step B uses `generate_content_stream` instead of
`generate_content`:

1. Stream chunks from the SDK.
2. For each chunk:
   - Extract text → emit `text_chunk` event (for live rendering).
   - Extract function_call parts → accumulate (preserve original Part
     for thought_signature).
3. Synthesize a `_StreamedResponse` object that quacks like the
   non-streaming response (has `.candidates[].content.parts[]` +
   `.usage_metadata`).
4. The rest of the agent loop is unchanged.

## Live view rendering

`RichLiveChat` (in `gemini_py/live.py`):

1. On construction, starts a `rich.live.Live` context with
   `refresh_per_second=12`.
2. Internal state: `self.state` (header info) +
   `self._body_renderables` (list of accumulated renderables).
3. `emit(event)` mutates state and appends renderables. The Live
   display auto-refreshes 12x/sec via the callable renderer.
4. `_render()` builds a `Layout` with three rows (header / body /
   footer) and returns it.
5. `close()` calls `self.live.stop()` to release the terminal.

## Path resolution (lazy defaults)

All "default path" module-level constants (`USER_CREDENTIALS_FILE`,
`USER_SKILLS_DIR`, `USER_CONFIG_PATH`, etc.) are `None` and resolved
lazily via helper functions like `_default_user_credentials_file()`.
This is important because module-level constants are evaluated at
import time, which breaks `monkeypatch.setenv("HOME", ...)` in tests.

## .env file loading order

`_env_search_paths()` returns (in order):

1. `Path.cwd() / ".env"` (most specific)
2. `Path.cwd() / ".gemini" / ".env"`
3. `Path.home() / ".gemini" / ".env"`
4. `Path.home() / ".gemini-py" / ".env"` (least specific — where
   `login()` writes)

`_load_env_files()` iterates `reversed(paths)` so the most general
file loads first and the most specific wins (later loads override
earlier).

## Token usage tracking

`UsageStats.from_metadata()` extracts:

- `prompt_token_count`
- `candidates_token_count`
- `total_token_count`
- `cached_content_token_count`
- `thoughts_token_count`

Set on every `GenerateResult`. Used by `--output-format json` for the
`stats` field.

## Loop detection

In `run_agent_turn`, for each iteration we compute a fingerprint per
call:

```python
f"{name}:{json.dumps(args, sort_keys=True)}"
```

We keep the last 6 fingerprints and count duplicates. If any fingerprint
appears 3+ times, we treat it as a loop and break with
`finish_reason="loop_detected"`.

This is a safety valve for the common failure mode where the model
re-issues the same call (e.g. `run_shell_command(command="ls")` every
iteration) because the result confuses it.

## Session storage

`SessionStore` writes JSON files to `~/.gemini-py/sessions/<id>.json`.
Each file contains:

- `session_id`, `project_dir`, `created_at`, `updated_at`, `model`
- `messages` (list of dicts with `role` + `parts[]`)
- `config` (the `GeminiConfig` at creation time, serialized as dict)

The CLI creates a session record on every run but doesn't yet
automatically append the model's response — see
[Sessions](./sessions.md#what-gets-saved) for the caveat.

## Authentication resolution

`AuthConfig.from_env()`:

1. Load `.env` files (general→specific, later wins).
2. Read `GEMINI_API_KEY`, `GOOGLE_API_KEY`,
   `GOOGLE_APPLICATION_CREDENTIALS`, `GOOGLE_CLOUD_PROJECT`,
   `GOOGLE_CLOUD_LOCATION` from the environment.
3. Resolve method:
   - `GEMINI_API_KEY` set → `AuthMethod.GEMINI_API_KEY`
   - `GOOGLE_API_KEY` set → `AuthMethod.GOOGLE_API_KEY`
   - `GOOGLE_APPLICATION_CREDENTIALS` set → `AuthMethod.VERTEX_AI_ADC`
   - else → `AuthMethod.VERTEX_AI_ADC` (will fail if no ADC available)
4. `to_client_kwargs()` builds the kwargs for `google.genai.Client()`.

## Dependencies

| Package | Required | Used for |
| --- | --- | --- |
| `google-genai` | yes | The Gemini API client SDK |
| `python-dotenv` | yes | `.env` file loading |
| `rich` | yes | Live dashboard + colored output |
| `mcp` | yes (since v0.2) | MCP client |

Optional / dev:

| Package | Used for |
| --- | --- |
| `pytest` | Tests |
| `pytest-mock` | Tests |
| `google-auth` | Vertex AI ADC (`pip install gemini-py[vertex]`) |

## Test layout

```
tests/
├── __init__.py
├── test_smoke.py          — basic SDK + CLI parser smoke tests
├── test_advanced.py       — MCP, memory, skills, sessions
├── test_agent_loop.py     — id linkage, loop detection, CLI defaults
├── test_login.py          — login/logout/whoami + .env merging
└── test_live.py           — LiveView + RichLiveChat + create_directory
```

109 tests as of v0.5.1. Run with:

```bash
pytest tests/ -v
```

## Design decisions

### Why argparse, not click / typer?

The user explicitly requested "cli args biar enak jangan GUI" — argparse
is stdlib, has no extra deps, and handles the surface area well.

### Why no async by default?

The agent loop is sync. MCP client calls use `asyncio.run()` to bridge
the async `mcp.ClientSession` calls. This is simpler than requiring
`asyncio.run()` at the call site.

### Why is `required` a per-property flag, not a top-level list?

It's a gemini-py convention to make tool definitions read more
naturally:

```python
{
    "path": {"type": "string", "description": "...", "required": True},
    "content": {"type": "string", "description": "...", "required": True},
}
```

vs. the JSON-schema standard:

```python
{
    "path": {"type": "string", "description": "..."},
    "content": {"type": "string", "description": "..."},
    # "required": ["path", "content"]   ← separate
}
```

`Tool.to_google_decl()` lifts the per-property flags into the
top-level list AND removes them from the property specs (otherwise
the google-genai SDK's pydantic schema validation rejects them).

### Why is `~/.gemini-py/` separate from `~/.gemini/`?

`~/.gemini/` is the gemini-cli convention (TypeScript). We use
`~/.gemini-py/` for our own state (sessions, skills, MCP configs,
login credentials) to avoid collisions if the user has both installed.

The `.env` search path includes both, so .env files in either location
work.

## Contributing

See [Contributing](./contributing.md) for how to extend the SDK.
