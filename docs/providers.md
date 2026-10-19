# Providers

`gemini-py` supports multiple LLM providers via the `Provider` abstraction.
The default is `gemini` (Google AI Studio / Vertex AI). You can switch to
OpenAI, Anthropic, Ollama (local), Transformers (local), or define a
custom provider.

## At a glance

| Provider | Backend | Function-calling? | Extra deps |
| --- | --- | --- | --- |
| `gemini` (default) | `google.genai.Client` | ✅ | `google-genai` (installed) |
| `openai` | `openai` Python SDK | ❌ (text-only) | `pip install openai` |
| `anthropic` | `anthropic` Python SDK | ❌ (text-only) | `pip install anthropic` |
| `ollama` | HTTP to local Ollama | ❌ (text-only) | none (uses urllib) |
| `transformers` | HuggingFace transformers | ✅ (parses markers) | `pip install transformers torch` |
| `custom` | user-supplied callable | ❌ (text-only) | none |

**Note**: OpenAI, Anthropic, and Ollama providers currently support text
generation only. Transformers provider supports function-calling via
output-text parsing. We're working on adapters for OpenAI/Anthropic
native tool-calling. When you run `run_agent_turn` with a non-Gemini
provider that doesn't support function-calling, the CLI auto-disables
tools and prints a warning.

## Picking a provider from the CLI

```bash
# Default (Gemini)
gemini-py -p "Hello"

# OpenAI
gemini-py -p "Hello" --provider openai --provider-model gpt-4o-mini --api-key sk-...

# Anthropic
gemini-py -p "Hello" --provider anthropic --provider-model claude-3-5-sonnet-20241022 --api-key sk-ant-...

# Ollama (local)
gemini-py -p "Hello" --provider ollama --provider-model llama3.2
# Or pull a coder model first: ollama pull qwen2.5-coder
gemini-py -p "Write a Python function to reverse a list" --provider ollama --provider-model qwen2.5-coder

# Transformers (local, no API)
gemini-py -p "Hello" --provider transformers --provider-model HuggingFaceTB/SmolLM2-360M
```

## OpenAI-compatible APIs

The `openai` provider supports ANY OpenAI-compatible API by setting
`--provider-base-url`. Examples:

```bash
# DeepSeek
gemini-py -p "Hello" --provider openai --provider-model deepseek-chat \
  --provider-base-url https://api.deepseek.com --api-key sk-...

# Together
gemini-py -p "Hello" --provider openai --provider-model meta-llama/Llama-3-8B-chat-hf \
  --provider-base-url https://api.together.xyz/v1 --api-key ...

# Groq
gemini-py -p "Hello" --provider openai --provider-model llama-3.1-8b-instant \
  --provider-base-url https://api.groq.com/openai/v1 --api-key ...

# vLLM (local)
gemini-py -p "Hello" --provider openai --provider-model meta-llama/Llama-3-8B \
  --provider-base-url http://localhost:8000/v1 --api-key dummy

# LM Studio (local)
gemini-py -p "Hello" --provider openai --provider-model local-model \
  --provider-base-url http://localhost:1234/v1 --api-key dummy
```

## From Python

```python
from gemini_py import GeminiClient
from gemini_py.providers import make_provider

# Option 1: Use the factory
provider = make_provider("ollama", model="llama3.2")
client = GeminiClient(provider=provider)
print(client.generate_text("Hello").text)

# Option 2: Pass a provider name string
client = GeminiClient(provider="openai")
# (api_key read from OPENAI_API_KEY env var)

# Option 3: Construct the provider class directly
from gemini_py import OpenAIProvider
provider = OpenAIProvider(api_key="sk-...", default_model="gpt-4o")
client = GeminiClient(provider=provider)
```

## Custom provider

The `CustomProvider` lets you plug in any function that takes
`(prompt: str, *, model: str, **config)` and returns text:

```python
from gemini_py import GeminiClient
from gemini_py.providers.custom import CustomProvider

def my_generate(prompt: str, *, model: str, **config) -> str:
    # Replace with your real backend (REST, gRPC, IPC, whatever)
    return f"[{model}] You said: {prompt}"

provider = CustomProvider(
    generate_fn=my_generate,
    models=["my-model-1", "my-model-2"],
    name="my-backend",
)
client = GeminiClient(provider=provider)
print(client.generate_text("Hello").text)
# [my-model-1] You said: Hello
```

Streaming variant:

```python
def my_stream(prompt: str, *, model: str, **config):
    for word in f"[{model}] You said: {prompt}".split():
        yield word + " "

provider = CustomProvider(
    stream_fn=my_stream,
    models=["my-model"],
    name="streaming-backend",
)
client = GeminiClient(provider=provider)
for chunk in client.stream("Hello"):
    print(chunk, end="", flush=True)
print()
```

The callable receives the user's last message text (extracted from the
conversation history). Multi-turn conversation state is NOT passed to
the callable — each call is treated as a fresh prompt.

## Ollama setup

1. Install Ollama from <https://ollama.com>.
2. Start the server: `ollama serve` (or it's already running as a
   service).
3. Pull a model: `ollama pull llama3.2` (or `qwen2.5-coder`,
   `mistral`, `phi3`, etc.)
4. Run gemini-py:

   ```bash
   gemini-py -p "Hello" --provider ollama --provider-model llama3.2
   ```

The Ollama provider uses HTTP directly (via `urllib.request`) — no
extra Python deps. Default base URL is `http://localhost:11434`; override
with `--provider-base-url`.

## Transformers setup

1. Install the heavy deps: `pip install transformers torch`
2. The first run downloads model weights from HuggingFace Hub (can be
   several GB). Default model is `HuggingFaceTB/SmolLM2-360M` (small
   enough for CPU inference).
3. Run:

   ```bash
   gemini-py -p "Hello" --provider transformers --provider-model HuggingFaceTB/SmolLM2-360M
   ```

For better quality, try larger models:

```bash
gemini-py -p "Hello" --provider transformers --provider-model Qwen/Qwen2.5-1.5B
gemini-py -p "Hello" --provider transformers --provider-model microsoft/Phi-3-mini-4k-instruct
```

GPU is auto-detected via `torch.cuda.is_available()`. To force CPU,
unset `CUDA_VISIBLE_DEVICES`.

### Function-calling with Transformers (v0.6.1+)

The Transformers provider supports function-calling via output-text
parsing. When you run an agent turn with tools enabled, the provider:

1. Converts the Google `function_declarations` to HuggingFace tools
   format.
2. Tries `tokenizer.apply_chat_template(messages, tools=hf_tools, ...)`.
3. If the template doesn't accept `tools=` (some models), falls back
   to injecting a system message describing the tools.
4. Generates text.
5. Parses tool-call markers out of the model's output text.
6. Returns `_FunctionCallPart` objects for each detected call.

Recognized tool-call marker formats:

- `<|tool_call|>{...}<|/tool_call|>`  (Hermes / Llama-3 style)
- `【{...}】`  (Qwen2.5 style)
- `[TOOL_CALLS] {...}`  (Mistral style)
- ` ```tool\n{...}\n``` ` or ` ```json\n{...}\n``` ` (fenced code block)
- bare `{"name": "...", "arguments": {...}}` JSON

The JSON can use `"name"` + `"arguments"` (standard) or `"name"` +
`"args"` (alias). The `arguments` field can be either a dict or a JSON
string (Qwen format).

Recommended models for function-calling:

- `Qwen/Qwen2.5-1.5B-Instruct` — works well, small enough for CPU.
- `Qwen/Qwen2.5-Coder-7B-Instruct` — coding + tools.
- `meta-llama/Llama-3.1-8B-Instruct` — Hermes-style tool calls.
- `NousResearch/Hermes-2-Pro-Llama-3-8B` — explicit Hermes format.
- `mistralai/Mistral-Nemo-Instruct-2407` — Mistral-style.

Example:

```bash
gemini-py -p "List the Python files in the current directory" \
  --provider transformers \
  --provider-model Qwen/Qwen2.5-1.5B-Instruct \
  --live
```

The agent loop will:
1. Call `list_directory(path=".")` (model emits a tool call).
2. Receive the directory listing as a tool result.
3. Filter for `.py` files in the response.
4. Return a final answer.

The quality of tool-calling depends heavily on the model. Small models
(360M / 1B params) often fail to produce valid tool-call markers. For
reliable function-calling, use at least 7B-parameter models.

## The Provider protocol

```python
from gemini_py.providers import Provider

class MyProvider(Provider):
    def generate_content(self, *, model, contents, config):
        # Build a response-shaped object. See gemini_py/_response.py.
        from gemini_py._response import _StreamedResponse
        text = ...
        return _StreamedResponse(parts=[], text=text, usage_metadata=None, raw=None)

    def generate_content_stream(self, *, model, contents, config):
        # Yield _StreamedResponse chunks (incremental text).
        for chunk in [...]:
            yield _StreamedResponse(parts=[], text=chunk, usage_metadata=None, raw=None)

    def list_models(self):
        return ["my-model-1", "my-model-2"]

    def supports_tools(self) -> bool:
        return False  # text-only mode

    @property
    def name(self) -> str:
        return "my-provider"
```

The response shape must quack like the google.genai response:
- `.candidates[].content.parts[].text` (str | None)
- `.candidates[].content.parts[].function_call` (None for text-only providers)
- `.usage_metadata` (Optional, has `prompt_token_count` etc.)

Use the helpers in `gemini_py/_response.py`:
- `text_response(text, usage=...)` — build a non-streaming response
- `text_chunk(text)` — build a streaming chunk

## Provider factory

```python
from gemini_py.providers import make_provider

provider = make_provider(
    name="openai",  # 'gemini' | 'openai' | 'anthropic' | 'ollama' | 'transformers'
    api_key="sk-...",  # for openai/anthropic
    base_url="https://api.deepseek.com",  # for openai/anthropic/ollama
    model="deepseek-chat",  # default model
    auth=AuthConfig,  # for gemini only
)
```

## Auth per provider

| Provider | Env vars |
| --- | --- |
| `gemini` | `GEMINI_API_KEY` (or `login()`) |
| `openai` | `OPENAI_API_KEY` |
| `anthropic` | `ANTHROPIC_API_KEY` |
| `ollama` | (none — local server) |
| `transformers` | (none — local model) |
| `custom` | (none — your callable handles auth) |

CLI flags `--api-key` and `--provider-base-url` override the env vars.

## Limitations of non-Gemini providers

1. **No function-calling** — the agent loop will not call tools. The
   model's text answer is returned directly. The CLI auto-disables
   tools when you pick a non-Gemini provider.
2. **No MCP integration** — MCP tools aren't exposed to the model.
3. **No thought signatures** — those are Gemini-specific.
4. **Multi-turn chat works** (for `chat()`) because the contents are
   converted to the provider's chat-message format.
5. **Streaming works** — text is chunked into `_StreamedResponse` chunks.

If you need tool-calling with a non-Gemini provider, file an issue —
adapters for OpenAI/Anthropic native function-calling are on the roadmap.
