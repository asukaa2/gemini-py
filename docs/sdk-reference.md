# Python SDK Reference

Complete reference for every public class and function in `gemini_py`.

## Top-level imports

```python
from gemini_py import (
    GeminiClient,        # main client class
    AuthConfig,           # auth configuration
    AuthMethod,           # enum: GEMINI_API_KEY | GOOGLE_API_KEY | VERTEX_AI_ADC | AUTO
    GeminiConfig,         # generation config
    ModelAlias,           # enum of model aliases
    GenerateResult,       # dataclass for one-shot generation results
    Message,              # dataclass for chat messages
    MessageRole,          # enum: USER | MODEL | SYSTEM | TOOL
    UsageStats,           # token-usage dataclass
    SessionStore,         # persistent conversation store
    SessionRecord,        # one saved session
    OutputFormat,         # enum: TEXT | JSON | STREAM_JSON
    format_result,        # function: GenerateResult + OutputFormat -> str
    run_headless,         # one-shot helper: run_headless(prompt) -> GenerateResult
    login,                # persist credentials
    logout,               # clear credentials
    whoami,               # show currently-active credentials (masked)
    LoginResult,          # dataclass returned by login()
    LoginError,           # exception type for login()/logout() errors
)
```

## `GeminiClient`

The main client. Wraps `google.genai.Client` and provides high-level
methods.

### Construction

```python
client = GeminiClient(
    auth: Optional[AuthConfig] = None,    # defaults to AuthConfig.from_env()
    config: Optional[GeminiConfig] = None, # defaults to GeminiConfig()
)
```

Construction triggers `AuthConfig.from_env()` which:
1. Reads `GEMINI_API_KEY`, `GOOGLE_API_KEY`, `GOOGLE_APPLICATION_CREDENTIALS`,
   `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION` from the environment.
2. Loads `.env` files in order: `~/.gemini-py/.env`, `~/.gemini/.env`,
   `./.gemini/.env`, `./.env` (later files override earlier ones).

Raises `GeminiError` if no credentials are configured.

### Methods

#### `generate_text(prompt, *, config=None, system=None) -> GenerateResult`

One-shot text generation (no tools, no agent loop).

```python
r = client.generate_text("Explain HTTP/3.", system="Be concise.")
print(r.text, r.usage.to_dict())
```

#### `chat(messages, *, config=None) -> GenerateResult`

Multi-turn chat with a message history.

```python
from gemini_py import Message, MessageRole
history = [
    Message(MessageRole.USER, "Hi!"),
    Message(MessageRole.MODEL, "Hello!"),
    Message(MessageRole.USER, "What's 2+2?"),
]
r = client.chat(history)
```

#### `stream(prompt, *, config=None) -> Iterator[str]`

Stream text chunks for a prompt. Yields incremental deltas.

```python
for chunk in client.stream("Tell me a story."):
    print(chunk, end="", flush=True)
```

#### `run_agent_turn(prompt, *, ...) -> GenerateResult`

Agentic flow with tool-calling. The model can call tools; we execute
them and feed back the results until the model produces a final text
response or `max_iterations` is reached.

Full signature:

```python
def run_agent_turn(
    prompt: str,
    *,
    tools: Optional[List[Tool]] = None,           # default: built-in tools
    config: Optional[GeminiConfig] = None,        # override config
    max_iterations: int = 10,                     # max tool-call round-trips
    extra_context: Optional[str] = None,          # appended to system prompt
    on_event: Optional[Callable[[dict], None]] = None,  # event callback
    enable_memory: bool = True,                   # GEMINI.md auto-memory
    enable_skills: bool = True,                   # @skill_name expansion
    enable_subagent: bool = True,                 # delegate_subagent tool
    mcp_client: Optional[Any] = None,             # an McpClient (already .start()-ed)
    skills_loader: Optional[Any] = None,          # pre-built SkillsLoader
    live: bool = False,                            # use generate_content_stream
) -> GenerateResult
```

Events emitted to `on_event`:

- `{"type": "iteration", "n": N, "max": M}`
- `{"type": "text_chunk", "text": "..."}` (only when `live=True`)
- `{"type": "tool_call", "name": "...", "args": {...}}`
- `{"type": "file_write", "path": "...", "content": "..."}` (for `write_file` calls)
- `{"type": "tool_result", "name": "...", "result": "..."}`
- `{"type": "done", "finish_reason": "...", "n_tool_calls": N}`
- `{"type": "error", "message": "..."}`

#### `run_headless(prompt, *, ...) -> GenerateResult`

Headless one-shot, mirrors `gemini -p`. Convenience wrapper around
`run_agent_turn`.

```python
result = client.run_headless("Summarize this", tools=[], output_format="json")
```

## `AuthConfig`

```python
@dataclass
class AuthConfig:
    method: AuthMethod = AuthMethod.AUTO
    api_key: Optional[str] = None
    vertex_project: Optional[str] = None
    vertex_location: Optional[str] = "us-central1"
    vertex_credentials_path: Optional[str] = None
```

### Class methods

#### `AuthConfig.from_env(load_dotenv_files=True) -> AuthConfig`

Build an `AuthConfig` from environment variables + `.env` files.

#### `to_client_kwargs() -> dict`

Convert to kwargs accepted by `google.genai.Client`. Raises `ValueError`
if a required field is missing (e.g. `GOOGLE_CLOUD_PROJECT` for Vertex AI).

#### `describe() -> str`

Human-readable summary.

## `GeminiConfig`

```python
@dataclass
class GeminiConfig:
    model: str = "gemini-3.8-flash"          # alias or concrete name
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 64
    max_output_tokens: int = 8192
    system_instruction: Optional[str] = None
    enable_shell_tool: bool = True
    enable_filesystem_tools: bool = True
    enable_web_search: bool = False
    enable_web_fetch: bool = False
    safety_settings: list = []
```

### Methods

- `resolve_model() -> str` — alias → concrete model name.
- `to_generate_content_config() -> dict` — for `google.genai` SDK.
- `to_dict() -> dict` / `from_dict(d) -> GeminiConfig` — for session
  persistence.

## `ModelAlias`

```python
class ModelAlias(str, Enum):
    AUTO          # "auto"          -> "gemini-3.8-flash"
    ARGON         # "argon"         -> "gemini-4-argon"
    PRO           # "pro"           -> "gemini-3.1-pro"
    FLASH         # "flash"         -> "gemini-3.8-flash"
    FLASH_LITE    # "flash-lite"    -> "gemini-3.5-flash-lite"
    FLASH_LATEST  # "flash-latest"  -> "gemini-flash-latest"
    OMNI          # "omni"          -> "gemini-omni-1.1-flash"
    TRANSCRIBE    # "transcribe"    -> "gemini-3.5-transcribe"
    TTS           # "tts"           -> "gemini-3.8-flash-tts"
```

### Methods

- `resolve() -> str` — alias → concrete model name.
- `is_alias(value: str) -> bool` — check if a string is a known alias.
- `all_aliases() -> dict` — `{alias: resolved_model}` for help text.

See [Models](./models.md) for the full table.

## `Message`

```python
@dataclass
class Message:
    role: MessageRole
    content: Union[str, List[Dict[str, Any]]]
    tool_calls: List[Dict[str, Any]] = []
    tool_call_id: Optional[str] = None
```

### Methods

- `to_dict() -> dict` / `from_dict(d) -> Message`.

## `MessageRole`

```python
class MessageRole(str, Enum):
    USER = "user"
    MODEL = "model"
    SYSTEM = "system"
    TOOL = "tool"
```

## `GenerateResult`

```python
@dataclass
class GenerateResult:
    text: str
    usage: UsageStats
    model: str
    finish_reason: str          # e.g. "STOP", "loop_detected", "max_iterations_reached"
    tool_calls: List[Dict[str, Any]]
    tool_results: List[Dict[str, Any]]
    raw: Any                    # the underlying google.genai response
```

### Methods

- `to_dict() -> dict` — for JSON serialization.

## `UsageStats`

```python
@dataclass
class UsageStats:
    prompt_token_count: int = 0
    candidates_token_count: int = 0
    total_token_count: int = 0
    cached_content_token_count: int = 0
    thoughts_token_count: int = 0
```

### Class methods

- `from_metadata(metadata) -> UsageStats` — extract from a google.genai
  `usage_metadata` object.

### Methods

- `to_dict() -> dict`.

## `SessionStore`

```python
class SessionStore:
    def __init__(self, base_dir: Optional[Path] = None): ...
```

Persists sessions as JSON files under `~/.gemini-py/sessions/`.

### Methods

- `create(*, project_dir=None, config=None, model="") -> SessionRecord`
- `get(session_id) -> Optional[SessionRecord]`
- `latest(project_dir=None) -> Optional[SessionRecord]`
- `list_all(project_dir=None) -> List[SessionRecord]`
- `update(rec) -> None`
- `delete(session_id) -> bool`
- `append_message(session_id, msg: Message) -> None`

See [Sessions](./sessions.md).

## `SessionRecord`

```python
@dataclass
class SessionRecord:
    session_id: str
    project_dir: str
    created_at: float
    updated_at: float
    model: str
    messages: List[Dict[str, Any]]
    config: Dict[str, Any]
```

### Methods

- `to_dict() -> dict` / `from_dict(d) -> SessionRecord`.

## `OutputFormat`

```python
class OutputFormat(str, Enum):
    TEXT = "text"
    JSON = "json"
    STREAM_JSON = "stream-json"
```

## `format_result(result, fmt) -> str`

Format a `GenerateResult` for terminal output. Used by the CLI.

## `run_headless(prompt, **kwargs) -> GenerateResult`

Module-level convenience for `GeminiClient().run_headless(...)`.

```python
from gemini_py import run_headless
print(run_headless("Summarize this.").text)
```

## `login(api_key, *, method="gemini", project=None, location="us-central1", scope="user") -> LoginResult`

Persist credentials. See [Authentication](./authentication.md).

```python
login("AIza...")
# For Vertex AI:
login("ya29...", method="google", project="my-proj", location="us-east1")
```

## `logout(*, scope="user") -> bool`

Remove persisted credentials. Returns True if a file was deleted.

## `whoami() -> Optional[dict]`

Returns a dict with `method`, `api_key` (masked), `project`, `location`,
`source`, or None if no credentials are configured.

## `LoginResult`

```python
@dataclass
class LoginResult:
    method: str
    api_key: str
    credentials_path: Path
    scope: str
    project: Optional[str] = None
    location: Optional[str] = None
```

### Methods

- `describe() -> str` — friendly one-paragraph summary.

## `LoginError`

Raised by `login()` and `logout()` on bad input (empty key, missing
project for Vertex, unknown scope/method, etc.).

## Providers (v0.6.0+)

The SDK supports multiple LLM providers via the `Provider` abstraction.
See [Providers](./providers.md) for the full guide.

### `Provider` (ABC)

Abstract base class. Subclass it to write a custom adapter:

```python
from gemini_py.providers import Provider

class MyProvider(Provider):
    def generate_content(self, *, model, contents, config):
        # Build a response-shaped object (see gemini_py/_response.py).
        ...
    def generate_content_stream(self, *, model, contents, config):
        # Yield response chunks.
        ...
    def list_models(self):
        return ["my-model-1", "my-model-2"]
    def supports_tools(self) -> bool:
        return False  # text-only mode
    @property
    def name(self) -> str:
        return "my-provider"
```

### `make_provider(name, *, api_key=None, base_url=None, model=None, auth=None) -> Provider`

Factory function for the built-in providers:

```python
from gemini_py.providers import make_provider

provider = make_provider("ollama", model="llama3.2")
client = GeminiClient(provider=provider)
```

Names: `gemini` (default), `openai`, `anthropic`, `ollama`,
`transformers`. (`custom` requires direct construction — call
`CustomProvider(generate_fn=...)`.)

### Built-in provider classes

| Class | Backend | Tools? | Extra deps |
| --- | --- | --- | --- |
| `GeminiProvider` | `google.genai.Client` | ✅ | `google-genai` |
| `OpenAIProvider` | `openai` SDK | ❌ | `pip install openai` |
| `AnthropicProvider` | `anthropic` SDK | ❌ | `pip install anthropic` |
| `OllamaProvider` | HTTP to local server | ❌ | none |
| `TransformersProvider` | HuggingFace transformers | ✅ (parses markers) | `pip install transformers torch` |
| `CustomProvider` | user-supplied callable | ❌ | none |

### `GeminiClient(provider=...)`

Accepts:
- `None` (default) → builds a `GeminiProvider` from `AuthConfig.from_env()`.
- A provider name string: `"gemini" | "openai" | "anthropic" | "ollama" | "transformers"`.
- A `Provider` instance directly.

```python
# Default (Gemini)
client = GeminiClient()

# By name
client = GeminiClient(provider="ollama")

# By instance
from gemini_py import OllamaProvider
client = GeminiClient(provider=OllamaProvider(default_model="llama3.2"))
```

### `CustomProvider` (text-only, callable-based)

```python
from gemini_py.providers.custom import CustomProvider

def my_generate(prompt: str, *, model: str, **config) -> str:
    return f"[{model}] Echo: {prompt}"

provider = CustomProvider(
    generate_fn=my_generate,
    models=["my-model"],
    name="echo",
)
client = GeminiClient(provider=provider)
print(client.generate_text("Hello").text)
```

Streaming variant: pass `stream_fn` instead of (or in addition to)
`generate_fn`. The callable receives the user's last message text
(extracted from the conversation history).

### Response-shape helpers (`gemini_py._response`)

Provider adapters use these to build response objects:

- `text_response(text, usage=...)` — build a non-streaming text response.
- `text_chunk(text)` — build a streaming chunk carrying incremental text.
- `text_with_tool_calls_response(text, tool_calls, usage=..., raw=...)` —
  build a response carrying both leftover text + `_FunctionCallPart`s
  for each parsed tool call (used by TransformersProvider).

### Submodules

The package also exposes these submodules for advanced use:

- `gemini_py.live.LiveView` — plain stderr streaming fallback.
- `gemini_py.live.RichLiveChat` — full-screen `rich.live.Live` dashboard.
- `gemini_py.live.make_default_live_view(...)` — factory.
- `gemini_py.mcp.McpRegistry`, `McpClient`, `McpServerConfig` — MCP.
- `gemini_py.memory.MemoryLoader`, `load_memory(...)` — auto-memory.
- `gemini_py.skills.SkillsLoader`, `expand_skills_in_prompt(...)` — skills.
- `gemini_py.skills.subagent.delegate_to_subagent(...)` — subagent.
- `gemini_py.tools.Tool`, `build_default_tools(...)`, `execute_tool_call(...)`.
- `gemini_py.tools.shell.run_shell_command(...)` — direct shell tool.
- `gemini_py.tools.websearch.web_search`, `web_fetch` — direct web tools.
