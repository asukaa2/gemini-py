# Models

`gemini-py` ships with model aliases that map to the latest AI Studio
model lineup (as of October 2026). You can also pass any concrete model
name directly.

## Aliases (Oct 2026 AI Studio lineup)

| Alias | Resolves to | Description |
| --- | --- | --- |
| `auto` (default) | `gemini-3.8-flash` | AI Studio recommended for new projects |
| `argon` | `gemini-4-argon` | Most capable (GA through Dec 31, 2026) |
| `pro` | `gemini-3.1-pro` | Advanced reasoning for coding and complex tasks |
| `flash` | `gemini-3.8-flash` | Frontier intelligence built for speed |
| `flash-lite` | `gemini-3.5-flash-lite` | Low-latency, cost-effective subagent |
| `flash-latest` | `gemini-flash-latest` | Google's rolling "latest Flash" alias |
| `omni` | `gemini-omni-1.1-flash` | Video generation and editing |
| `transcribe` | `gemini-3.5-transcribe` | Speech-to-text |
| `tts` | `gemini-3.8-flash-tts` | Text-to-speech |

Source: <https://ai.google.dev/gemini-api/docs/changelog> (Oct 13, 2026 entry).

## Using aliases

### From the CLI

```bash
# Default (auto -> gemini-3.8-flash)
gemini-py -p "Hello"

# Pick a specific alias
gemini-py -p "Quick fact" --model flash
gemini-py -p "Complex reasoning" --model argon
gemini-py -p "Cheap and fast" --model flash-lite
```

### From Python

```python
from gemini_py import GeminiClient, GeminiConfig, ModelAlias

# Use an alias
client = GeminiClient(config=GeminiConfig(model=ModelAlias.ARGON))

# Or as a string
client = GeminiClient(config=GeminiConfig(model="flash"))

# Or a concrete model name (no alias resolution)
client = GeminiClient(config=GeminiConfig(model="gemini-3.1-pro"))
```

## Picking the right model

| Use case | Recommended alias |
| --- | --- |
| Default / not sure | `auto` |
| Complex coding / math / agentic | `argon` |
| Long-context reasoning | `pro` |
| High-volume / cost-sensitive | `flash-lite` |
| Want the latest Flash whatever it is | `flash-latest` |
| Video generation | `omni` |
| Transcribe audio | `transcribe` |
| Generate speech | `tts` |

## Checking what an alias resolves to

```python
from gemini_py import ModelAlias

# Resolve one
print(ModelAlias.ARGON.resolve())  # gemini-4-argon

# Check if a string is a known alias
print(ModelAlias.is_alias("argon"))            # True
print(ModelAlias.is_alias("gemini-4-argon"))  # False (it's a concrete name)

# List all aliases
for alias, model in ModelAlias.all_aliases().items():
    print(f"{alias:15s} -> {model}")
```

## Concrete model names

You can always pass a concrete model name instead of an alias. Use this
when you want to pin a specific version (e.g. for reproducibility).

```bash
gemini-py -p "Hello" -m gemini-4-argon
gemini-py -p "Hello" -m gemini-3.8-flash
gemini-py -p "Hello" -m gemini-3.1-pro
gemini-py -p "Hello" -m gemini-3.5-flash-lite
gemini-py -p "Hello" -m gemini-flash-latest
gemini-py -p "Hello" -m gemini-omni-1.1-flash
gemini-py -p "Hello" -m gemini-3.5-transcribe
gemini-py -p "Hello" -m gemini-3.8-flash-tts
```

## Default model

`GeminiConfig.model` defaults to `ModelAlias.AUTO.resolve()` =
`gemini-3.8-flash`. This is the AI Studio-recommended default for new
projects as of Oct 2026.

## Updating the alias table

The alias table is in
[`gemini_py/config.py`](../gemini_py/config.py) — `class ModelAlias`.
To add or update an alias, edit the `mapping` dict in `ModelAlias.resolve()`.

We track AI Studio's changelog
(<https://ai.google.dev/gemini-api/docs/changelog>) and update the table
on each release.

## Notes

- The Google Gen AI Python SDK (`google-genai`) is updated independently.
  gemini-py's `pyproject.toml` requires `google-genai>=1.0.0` — newer
  versions add support for newer models automatically.
- `flash-latest` (`gemini-flash-latest`) is a Google-managed alias that
  always points to the newest Flash model. Use this if you want to
  "always use the latest" without code changes.
- Some models (like `tts`, `transcribe`, `omni`) are specialized —
  calling `generate_text` on them may not produce text output. Use the
  appropriate SDK methods for those tasks.
