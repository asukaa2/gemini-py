# Sessions

`gemini-py` saves every agent turn as a session, so you can resume a
conversation later.

## Where sessions live

Sessions are stored as JSON files under:

- `~/.gemini-py/sessions/<session_id>.json` (default)

Each file contains:

```json
{
  "session_id": "abc123",
  "project_dir": "/home/user/my-project",
  "created_at": 1729246800.0,
  "updated_at": 1729246900.0,
  "model": "gemini-3.8-flash",
  "messages": [
    {"role": "user", "parts": [{"text": "Hello"}]},
    {"role": "model", "parts": [{"text": "Hi! How can I help?"}]}
  ],
  "config": {
    "model": "auto",
    "temperature": 1.0,
    "top_p": 0.95,
    ...
  }
}
```

## Listing sessions

```bash
gemini-py sessions list
```

Output (for the current project directory):

```
Sessions for /home/user/my-project:
  [0] abc123  updated=2026-10-19 10:30:00  messages=5  model=gemini-3.8-flash
  [1] def456  updated=2026-10-18 14:22:11  messages=2  model=gemini-3.8-flash
```

Sessions are scoped to the current project directory (the dir you ran
`gemini-py` from). To see sessions from another project, `cd` there
first.

## Resuming sessions

### By index

```bash
# Resume the most recent session
gemini-py -r latest -p "What did we discuss?"

# Resume session at index 0 (from `sessions list`)
gemini-py -r 0 -p "Add a README to the app you just built"

# Resume by session ID
gemini-py -r abc123 -p "Finish this PR"
```

### From the `sessions` subcommand

```bash
gemini-py sessions resume 0
# Prints the session's last few messages, doesn't run anything.
# Use `-r` to actually resume + continue.
```

## Deleting sessions

```bash
# Delete by index
gemini-py sessions delete 0

# Or via the global flag
gemini-py --delete-session 0
```

## From Python

```python
from gemini_py import SessionStore, Message, MessageRole

store = SessionStore()

# Create a new session
rec = store.create(model="gemini-3.8-flash")
print(rec.session_id)

# Append messages
store.append_message(rec.session_id, Message(MessageRole.USER, "Hello"))
store.append_message(rec.session_id, Message(MessageRole.MODEL, "Hi!"))

# Get the latest session for the current project
latest = store.latest()
print(latest.session_id, len(latest.messages))

# List all sessions for the current project
for r in store.list_all(project_dir="/home/user/my-project"):
    print(r.session_id, r.updated_at, len(r.messages))

# Delete
store.delete(rec.session_id)
```

### Custom storage location

```python
store = SessionStore(base_dir=Path("/custom/sessions/dir"))
```

## What gets saved

When you run a prompt via the CLI:

```bash
gemini-py -p "Build a todo app"
```

The flow is:

1. If `-r` was passed, load the existing session's messages into the
   conversation.
2. Otherwise, create a new session record.
3. Run the agent turn.
4. The CLI doesn't yet persist the model's responses back to the
   session file automatically — this is on the roadmap. For now, the
   session is created empty and only the user prompt is recorded.

> **Note**: Session persistence is currently a stub — we save the
> session record but don't automatically append the model's response.
> To fully persist a multi-turn conversation, use the Python API:
>
> ```python
> store.append_message(rec.session_id, Message(MessageRole.MODEL, result.text))
> ```

## Session resume + skills / memory

When you resume a session, the agent loop:

1. Loads the session's messages.
2. Builds the system prompt fresh (including `GEMINI.md` auto-memory
   and skills summary).
3. Sends the full conversation history to the model.

This means your `GEMINI.md` and skills are always up-to-date when
resuming — they're not part of the session file.

## Session file format

The `config` field stores the `GeminiConfig` at creation time. When
resuming, we use the **current** `--model` / `--temperature` etc. from
the CLI (or the SDK config), not the saved ones. This is intentional —
you might want to switch models mid-conversation.

## Comparing to gemini-cli

The upstream gemini-cli's session system has more features:

- Compression (drop old turns when the context gets long)
- Forking (branch a session)
- Search across sessions
- Per-session token usage tracking

`gemini-py` is simpler: one file per session, full message history,
no compression. If you need those features, file an issue.

## See also

- [CLI Reference](./cli-reference.md) — `sessions` subcommand details
- [SDK Reference](./sdk-reference.md) — `SessionStore` class
