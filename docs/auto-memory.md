# Auto-Memory (GEMINI.md hierarchy)

`gemini-py` automatically loads hierarchical `GEMINI.md` files and
injects them as system context on every agent turn. This is a Python
port of gemini-cli's auto-memory feature.

## Search order

Files are loaded general → specific. More specific files override
less specific ones (later loads win).

1. `~/.gemini/GEMINI.md` — user-wide preferences
2. Each parent of cwd up to home — project hierarchy
3. `cwd/.gemini/GEMINI.md` — project-scoped, version-controlled
4. `cwd/GEMINI.md` — project root, most specific

Each loaded chunk is rendered with a `# Memory: <path>` header so the
model knows where each piece came from.

## Why hierarchy?

Different scopes want different rules:

- **User-wide** (`~/.gemini/GEMINI.md`) — your global preferences:
  "Always use type hints. Prefer pytest. Don't apologize."
- **Project-level** (`cwd/GEMINI.md`) — project conventions:
  "This project uses Flask + SQLAlchemy. Tests go in `tests/`."
- **Subdirectory** (parent dirs up to home) — when you're working in a
  monorepo subdir, the parent project's rules should still apply.

## Example `~/.gemini/GEMINI.md`

```markdown
# Global preferences

- Always use Python type hints.
- Prefer pytest for testing.
- Don't apologize in your responses.
- When asked to "explain X", provide a 1-paragraph TL;DR first, then
  the details.
- For file paths, use the project root as the reference.
```

## Example `cwd/GEMINI.md`

```markdown
# Project: my-todo-api

## Stack

- Backend: Flask 3.0
- DB: SQLAlchemy (PostgreSQL)
- Tests: pytest + pytest-cov

## Conventions

- All API routes are under `/api/v1/`.
- Use snake_case for Python, camelCase for JSON.
- Response envelopes: `{"data": ..., "meta": {...}}` on success,
  `{"error": {"code": ..., "message": ...}}` on error.

## Don't

- Don't add a new ORM model without a migration.
- Don't import from `app` directly — use the `create_app()` factory.
```

## Disabling auto-memory

```bash
# Single-run opt-out
gemini-py -p "Hello" --no-memory

# From Python
client.run_agent_turn("...", enable_memory=False)
```

## The `# gemini-py-ignore` directive

Use this to stop processing a file from a certain point:

```markdown
# Project rules
Test with pytest.

# gemini-py-ignore
These are personal notes the agent shouldn't see:
- TODO: refactor auth
- NOTE: the FOO env var is broken on macOS
```

Everything after `# gemini-py-ignore` is dropped before injection.

## Programmatic API

```python
from gemini_py.memory import MemoryLoader, MemoryConfig, load_memory

# Load and print what would be injected
context = load_memory()
print(context)

# With overrides
loader = MemoryLoader(MemoryConfig(
    enabled=True,
    user_dir=Path("/custom/.gemini"),
    project_dir=Path.cwd(),
    extra_files=[Path("/tmp/extra.md")],
))
print(loader.load())
```

### `MemoryConfig`

```python
@dataclass
class MemoryConfig:
    enabled: bool = True
    user_dir: Optional[Path] = None       # defaults to ~/.gemini
    project_dir: Optional[Path] = None    # defaults to cwd
    extra_files: List[Path] = []           # CLI override
```

### `MemoryChunk`

```python
@dataclass
class MemoryChunk:
    path: Path
    content: str
    source_kind: str   # 'user' | 'parent' | 'project' | 'extra'

    def render(self) -> str  # "# Memory: <path>\n<content>\n"
```

## How it integrates with the agent loop

In `GeminiClient.run_agent_turn()`:

1. Build the system prompt (default or `--system-prompt` override).
2. If `enable_memory=True`, call `load_memory()`.
3. If memory is non-empty, append `# Project context (from GEMINI.md)`
   followed by the loaded chunks.
4. Append extra context (e.g. stdin).
5. Append the skills summary (if `enable_skills=True`).
6. Pass the full system instruction to the model.

Errors during memory loading are caught and logged to stderr — they
don't crash the agent turn.

## Where to put your `GEMINI.md`

| File | What it's for |
| --- | --- |
| `~/.gemini/GEMINI.md` | Personal global preferences |
| `~/.gemini-py/GEMINI.md` | Same — also picked up (legacy compat) |
| `<project>/GEMINI.md` | Project root conventions |
| `<project>/.gemini/GEMINI.md` | Same — checked into git |
| `<parent>/GEMINI.md` | Inherited when working in a subdir |

## Comparison with `.cursorrules`, `.github/copilot-instructions.md`, `CLAUDE.md`

`GEMINI.md` is the Gemini-flavored equivalent. The format and intent
are the same: a markdown file with project rules + conventions that
the AI assistant should follow.

`gemini-py` only auto-loads `GEMINI.md` files (not the others). If you
want to also load `.cursorrules`, symlink it:

```bash
ln -s .cursorrules GEMINI.md
```

Or include it via the `extra_files` parameter (not exposed on the CLI
yet — use the Python API).

## What's NOT yet ported

The upstream gemini-cli has these advanced features we haven't ported:

- `@path/to/file.md` imports — referencing other files from GEMINI.md.
- `.geminiignore` patterns — separate file-ignore system.
- Hierarchical scope merging rules (we just concat all files; the
  original has more nuanced priority).

If you need these, open an issue.
