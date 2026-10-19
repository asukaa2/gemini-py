# Troubleshooting

Common errors and how to fix them.

## Authentication errors

### `Error: Authentication not configured: GEMINI_API_KEY is not set`

You haven't logged in. Fix:

```bash
# Get a key from https://aistudio.google.com/app/apikey
gemini-py login --api-key AIza...
```

Or set the env var directly:

```bash
export GEMINI_API_KEY="AIza..."
```

### `Error: Authentication not configured: GOOGLE_CLOUD_PROJECT must be set for Vertex AI`

You have Vertex AI ADC env vars set but no project ID. Fix:

```bash
export GOOGLE_CLOUD_PROJECT="your-project-id"
export GOOGLE_CLOUD_LOCATION="us-central1"
```

Or unset them and use `GEMINI_API_KEY` instead:

```bash
unset GOOGLE_APPLICATION_CREDENTIALS
unset GOOGLE_CLOUD_PROJECT
unset GOOGLE_CLOUD_LOCATION
gemini-py login --api-key AIza...
```

### `whoami()` returns None

Either you haven't logged in, or the credentials file is unreadable.
Check:

```bash
ls -la ~/.gemini-py/.env
# -rw------- 1 user user 40 Oct 19 10:30 /home/user/.gemini-py/.env
```

If the file is missing, re-run `login()`.

If the file exists but `whoami()` still returns None, the file may be
malformed. Inspect:

```bash
cat ~/.gemini-py/.env
# GEMINI_API_KEY="AIza..."
```

## API errors

### `400 INVALID_ARGUMENT. Function call is missing a thought_signature`

This was a bug in v0.4.0 and earlier. Fixed in v0.4.1+. Upgrade:

```bash
pip install --upgrade -e .
```

The agent loop now preserves the original `Part` objects so the
`thought_signature` field survives the round-trip back to the API.

### `400 INVALID_ARGUMENT. tools.function_declarations.X.parameters.properties.Y.required Input should be a valid list`

This was a bug in v0.4.0 and earlier. Fixed in v0.4.1+. Upgrade.

The `Tool.to_google_decl()` method was building the JSON schema with
`"required": True` inside each property spec, but the google-genai
SDK expects `required` to be a LIST at the parameters level. Now we
strip the per-property flag and lift to the top-level list.

### `Error: agent iteration failed: 5 validation errors for GenerateContentConfig`

This usually means your `Tool.parameters` dict has the wrong shape.
The schema must be:

```python
{
    "param_name": {
        "type": "string",
        "description": "...",
        # NOT "required": True here — that's a per-property flag
        # we strip in to_google_decl(). You CAN use it as a
        # gemini-py convention; it gets lifted to the top-level list.
    }
}
```

The `required` flag at the property level is a gemini-py convenience.
`Tool.to_google_decl()` lifts these into a top-level
`"required": ["param_name"]` list and removes them from the
property specs. So you can use it; just don't be surprised when it's
not in the output.

### `Error: agent iteration failed: 429 RESOURCE_EXHAUSTED`

You've hit the rate limit. Slow down or upgrade your API plan. For
CI/CD, add retries with exponential backoff.

### `(agent hit the max-iterations limit of 25 without producing a final answer)`

The model is stuck in a tool-call loop. Possible causes:

1. **The model is confused by a tool's output** — check the
   `tool_results` to see what the model was getting back.
2. **The task is genuinely complex** — raise `--max-iterations`:
   ```bash
   gemini-py -p "..." --max-iterations 50
   ```
3. **The model is too small** — switch to a more capable model:
   ```bash
   gemini-py -p "..." --model argon  # gemini-4-argon
   ```
4. **`id` linkage is broken** — fixed in v0.4.1+. Upgrade.

The agent also has automatic loop detection: if the model makes the
exact same call (same name + same args) 3+ times in a row, it stops
with `finish_reason="loop_detected"`.

## MCP errors

### `MCP server '<name>' failed to start: ...`

The subprocess couldn't be spawned. Check:

1. The command is on PATH: `which npx`
2. The package exists: `npx -y @modelcontextprotocol/server-filesystem /tmp`
   (try running it manually)
3. Env vars are set: `--env GITHUB_TOKEN=...`
4. The MCP server itself logs to stderr — run with `--debug` to see:

   ```bash
   gemini-py -p "..." --debug 2>&1 | grep MCP
   ```

### `Error: mcp package not installed. Run 'pip install mcp'.`

Install the optional MCP dependency:

```bash
pip install mcp
# or
pip install -e ".[mcp]"   # if we had an extras_require — we don't yet
```

Actually, `mcp` is a core dependency in v0.5.0+. Just upgrade:

```bash
pip install --upgrade -e .
```

### Tools from an MCP server don't appear

- `gemini-py mcp list` — is the server enabled?
- Try `gemini-py mcp enable <name>`.
- Check for typos in the server name.
- Restart the agent — the connection is made at agent-turn start.

## Skills errors

### `Error: no skill named '<name>'`

The skill directory doesn't exist or `SKILL.md` is missing. Check:

```bash
ls ~/.gemini-py/skills/<name>/SKILL.md
# /home/user/.gemini-py/skills/<name>/SKILL.md
```

Or check discovery:

```bash
gemini-py skills list
# Should list all discovered skills
```

### `@skill_name` doesn't expand

- The skill might be disabled. Run `gemini-py skills enable <name>`.
- The skill might not be discovered. Check `skills list`.
- You passed `--no-skills`. Remove that flag.

### `delegate_subagent` tool not exposed

The subagent tool only appears when at least one skill is enabled.
Either enable a skill or pass `--no-subagent` to suppress the
warning.

## Session errors

### `Error: session not found for 'latest'`

You have no sessions yet for the current project. Run a prompt first:

```bash
gemini-py -p "Hello"
# (creates a session)

gemini-py -r latest -p "What did I just say?"
```

### `Error: invalid session index: N`

You passed an index that's out of range. Run `gemini-py sessions list`
to see valid indices (they start at 0).

## Live view issues

### The dashboard flickers / jumps around

This can happen if your terminal is too narrow. Try widening the
window or reducing the font size. The dashboard uses
`refresh_per_second=12` which should be smooth on most terminals.

### The dashboard shows nothing

- Make sure stdout is a TTY: `tty` should print `/dev/pts/X`.
- If you're piping output (`gemini-py ... | tee log.txt`), the
  dashboard auto-falls back to plain stderr streaming.
- Try `gemini-py -p "..." --live --debug` to see what's happening.

### The dashboard clobbers the final answer

Fixed in v0.5.1+. The CLI now calls `live_view.close()` BEFORE
printing the final answer to stdout. Upgrade.

## Memory errors

### Auto-memory doesn't load

- The file must be named `GEMINI.md` (uppercase, no `.txt`).
- Check the search paths:
  ```bash
  ls ~/.gemini/GEMINI.md
  ls ./.gemini/GEMINI.md
  ls ./GEMINI.md
  ```
- Pass `--debug` to see verbose output about memory loading.

### `# gemini-py-ignore` doesn't work

Make sure it's on its own line:

```markdown
# Project rules
Test with pytest.

# gemini-py-ignore
These are personal notes...
```

The directive is matched as a substring within a line, so it must
appear exactly as `# gemini-py-ignore` (with the `#` prefix).

## Login errors

### `LoginError: api_key is required (got empty value)`

You passed an empty string. Make sure you have a real API key.

### `LoginError: project is required when method='google' (Vertex AI)`

For Vertex AI, you must pass `--project PROJECT_ID`:

```bash
gemini-py login --api-key ya29... --method google --project my-proj
```

### `LoginError: unknown method 'xxx'. Use 'gemini' or 'google'.`

`method` must be `"gemini"` (default, for AI Studio API keys) or
`"google"` (for Google Cloud API keys / Vertex AI).

### `LoginError: unknown scope 'xxx'. Use 'user' or 'project'.`

`scope` must be `"user"` (writes to `~/.gemini-py/.env`, default) or
`"project"` (writes to `./.gemini/.env`).

## Misc

### Where are the logs?

`gemini-py` writes verbose output to **stderr**. To capture both
stdout (final answer) and stderr (logs) separately:

```bash
gemini-py -p "..." --debug > answer.txt 2> debug.log
```

### How do I see what's being sent to the API?

Set `--debug`. We don't yet dump the full request/response JSON, but
you can patch the client to print it:

```python
import json
from gemini_py import GeminiClient

client = GeminiClient()

# Wrap generate_content
_orig = client._client.models.generate_content
def _wrapped(*args, **kwargs):
    print("REQUEST:", json.dumps(kwargs, default=str, indent=2))
    resp = _orig(*args, **kwargs)
    print("RESPONSE:", resp)
    return resp
client._client.models.generate_content = _wrapped

# Now run
client.run_agent_turn("Hello")
```

### The model is making the same tool call repeatedly

See "(agent hit the max-iterations limit)" above. The agent loop
has built-in loop detection that breaks after 3 identical calls.

### I'm hitting `400 INVALID_ARGUMENT` with custom tools

Likely your `Tool.parameters` schema is malformed. The google-genai
SDK is strict. Common issues:

- Forgetting `"type": "string"` on a property
- Putting `"required": True` inside a property (gemini-py handles
  this automatically via `to_google_decl()`; but if you build the
  dict by hand, remove the flag)
- Using unsupported JSON-schema features (e.g. `$ref`, `oneOf`)

Test with:

```python
from gemini_py.tools import build_default_tools
for t in build_default_tools():
    print(t.to_google_decl())
```

### How do I report a bug?

Open an issue at <https://github.com/asukaa2/gemini-cli-py/issues>
with:

1. The exact command you ran.
2. The exact error message.
3. The output of `gemini-py --version`.
4. The output of `gemini-py whoami`.
5. The output of `gemini-py -p "..." --debug 2>&1 | head -50`.
