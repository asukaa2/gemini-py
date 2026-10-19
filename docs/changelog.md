# Changelog

All notable changes to `gemini-py` are documented here. Versions follow
[Semantic Versioning](https://semver.org/).

## [0.6.3] — 2026-10-19

### Added
- `docs/assets/` folder with SVG screenshots of the live view dashboard:
  - `live-view-thinking.svg` — mid-generation state
  - `live-view-tool-call.svg` — tool call in progress
  - `live-view-tool-result.svg` — tool result received
  - `live-view-file-write.svg` — write_file code panel with syntax highlighting
  - `live-view-done.svg` — finished state
  - `live-view-ansi.txt` — raw ANSI text for terminal preview
- `scripts/generate_live_view_screenshots.py` — reproducible script that
  builds 5 dashboard scenarios and exports them as SVG via
  `rich.console.Console.export_svg()`. Run with
  `python scripts/generate_live_view_screenshots.py`.
- `docs/live-view.md` now embeds all 5 screenshots in dedicated sections
  at the top of the doc, with descriptive captions.

### Changed
- Bumped version to 0.6.3.

## [0.6.2] — 2026-10-19

### Fixed
- `RichLiveChat` now uses `screen=True` (alt-screen buffer) +
  `transient=True`. The dashboard fills the terminal and updates IN
  PLACE (like vim / less / gemini-cli's TUI) — no scrolling, no
  flicker, no "muncul mulu" repeating renders. When the agent
  finishes, the dashboard disappears entirely and only the final
  answer prints to stdout.

### Changed
- `docs/live-view.md` updated to document the new `screen=True` mode.
- `docs/cli-reference.md` gained a "Provider selection" section with
  `--provider`, `--provider-model`, `--provider-base-url`, `--api-key`.
- `docs/sdk-reference.md` gained a "Providers" section documenting
  `Provider` ABC, `make_provider()`, all built-in provider classes,
  `GeminiClient(provider=...)`, and `CustomProvider` usage.
- Bumped version to 0.6.2.

## [0.6.1] — 2026-10-19

### Added
- Function-calling support for the TransformersProvider. The provider now
  converts Google function_declarations to HuggingFace tools format,
  injects them via `tokenizer.apply_chat_template(tools=...)` (or as a
  system message fallback), and parses tool-call markers out of the
  model's output text.
- Tool-call marker parser (`_parse_tool_calls`) recognizes:
  - `<|tool_call|>{...}<|/tool_call|>`  (Hermes / Llama-3)
  - `【{...}】`  (Qwen2.5)
  - `[TOOL_CALLS] {...}`  (Mistral)
  - ` ```tool\n{...}\n``` ` (fenced code block)
  - bare `{"name": "...", "arguments": {...}}` JSON
- `text_with_tool_calls_response()` helper in `_response.py` for building
  responses with both text + function_call parts.
- `contents_to_chat_messages()` now converts function_call parts (becomes
  assistant message with `tool_calls`) and function_response parts
  (becomes 'tool' role message with `tool_call_id`).
- `_extract_part_function_call()` / `_extract_part_function_response()`
  helpers in `_converters.py` for both dict-style and object-style parts.
- `TransformersProvider.supports_tools()` returns True.
- 25 new tests in `tests/test_transformers_tools.py` covering parser
  (all marker formats), normalization (args/arguments alias, string
  args, function/name alias), converter updates (function_call +
  function_response parts), and the `text_with_tool_calls_response`
  helper.

### Changed
- `TransformersProvider` now reads `config["tools"]` (Google
  function_declarations format) and converts to HF tools via
  `_tools_to_hf_format()`.
- Bumped version to 0.6.1.

## [0.6.0] — 2026-10-19

### Added
- Multi-provider system: pick the LLM backend via `--provider NAME`.
- `Provider` abstract base class + `make_provider(name, ...)` factory.
- Built-in providers:
  - `GeminiProvider` (default) — wraps `google.genai.Client`. Full
    function-calling / streaming / thought-signature support.
  - `OpenAIProvider` — uses `openai` Python SDK. Works with any
    OpenAI-compatible API (DeepSeek, Together, Groq, vLLM, LM Studio)
    via `--provider-base-url`.
  - `AnthropicProvider` — uses `anthropic` Python SDK.
  - `OllamaProvider` — HTTP to local Ollama server. No extra deps.
  - `TransformersProvider` — runs HuggingFace models locally. Lazy-loaded.
  - `CustomProvider` — takes a user-supplied callable.
- `gemini_py/providers/_converters.py` with `contents_to_text`,
  `extract_user_prompt`, `contents_to_chat_messages`,
  `contents_to_ollama_messages`.
- `gemini_py/_response.py` (new module) with the response-shape helper
  classes (`_StreamedResponse`, `_Candidate`, `_Content`, `_TextPart`,
  `_FunctionCall`, `_UsageMetadata`, `text_response`, `text_chunk`).
  Moved from `client.py` so provider adapters can import them without
  circular deps.
- New CLI flags: `--provider`, `--provider-model`, `--provider-base-url`,
  `--api-key`.
- `GeminiClient(provider=...)` accepts a Provider instance, a name
  string, or None (default = Gemini).
- Auto-disable tools when the provider doesn't support function-calling
  (with a stderr warning).
- `docs/providers.md` — new doc covering all providers + setup.

### Changed
- `GeminiClient` now calls `self._provider.generate_content(...)` instead
  of `self._client.models.generate_content(...)`. The provider protocol
  is uniform across all backends.
- Bumped version to 0.6.0.

## [0.5.1] — 2026-10-19

### Added
- `RichLiveChat` class: full-screen `rich.live.Live` + `Layout`
  dashboard with three panels (header / streaming transcript / footer).
- `make_default_live_view()` factory that prefers `RichLiveChat` when
  stdout is a TTY, falls back to `LiveView` otherwise.

### Changed
- `--live` flag now uses `RichLiveChat` (full-screen dashboard) instead
  of plain stderr streaming.
- CLI calls `live_view.close()` BEFORE printing the final answer to
  stdout, so the dashboard doesn't clobber output.

## [0.5.0] — 2026-10-19

### Added
- Live view (`--live` flag): stream the model's thinking + tool calls
  + tool results in real-time via `rich`.
- `gemini_py/live.py` with `LiveView` class.
- `run_agent_turn(live=True)` uses `generate_content_stream` and emits
  `text_chunk` events live.
- Special `file_write` event for `write_file` calls — the live view
  renders them as a syntax-highlighted code panel (lexer auto-detected
  from file extension).
- `_StreamedResponse` helper class to accumulate streaming chunks.
- `create_directory(path)` tool for explicit dir creation.
- System prompt now instructs the model to auto-create a project
  folder when no directory is specified.

### Changed
- `write_file` description updated to mention auto-creation of parent dirs.

## [0.4.1] — 2026-10-19

### Fixed
- Strip per-property `"required": True` flag from `Tool.to_google_decl()`
  output. The google-genai SDK expects `required` to be a LIST at the
  parameters level, not a bool inside each property spec.
- Preserve `Part.thought_signature` on function-call parts by passing
  the original Part objects back to the API (instead of rebuilding from
  just name+args).
- Forward `function_call.id` from `execute_tool_call()` to the
  `function_response` so the API can link request ↔ response. Same for
  the MCP and unknown-tool dispatch branches.
- Add loop detection: if the model makes the exact same call (same name
  + sorted args) 3+ times in a row, break with
  `finish_reason="loop_detected"`.

### Changed
- CLI `--max-iterations` default bumped from 10 → 25.
- Genuine max-iterations fallback message now explains causes and
  suggests `--max-iterations 25` or `--model argon`.

## [0.4.0] — 2026-10-19

### Added
- Update model aliases to AI Studio Oct 2026 lineup.
- New aliases: `argon` (`gemini-4-argon`), `flash-latest`
  (`gemini-flash-latest`), `omni` (`gemini-omni-1.1-flash`),
  `transcribe` (`gemini-3.5-transcribe`), `tts` (`gemini-3.8-flash-tts`).
- `ModelAlias.all_aliases()` classmethod for introspection.

### Changed
- `auto` alias now resolves to `gemini-3.8-flash` (was `gemini-2.5-pro`)
  — AI Studio's recommended default for new projects.
- `pro` → `gemini-3.1-pro` (was `gemini-2.5-pro`).
- `flash` → `gemini-3.8-flash` (was `gemini-2.5-flash`).
- `flash-lite` → `gemini-3.5-flash-lite` (was `gemini-2.5-flash-lite`).

## [0.3.0] — 2026-10-19

### Added
- `login()`, `logout()`, `whoami()` Python API for persistent
  credentials.
- CLI subcommands: `gemini-py login [--api-key ...] [--method gemini|google]
  [--project ...] [--location ...] [--scope user|project]`,
  `gemini-py logout`, `gemini-py whoami`.
- `LoginResult` dataclass with `describe()` for friendly confirmation.
- `LoginError` exception type.
- Added `~/.gemini-py/.env` to the .env search paths.

### Changed
- `_load_env_files()` now MERGES all .env files general→specific (was:
  load first found only). This is necessary so `login()`'s saved user
  credentials work even when a project .env exists.

## [0.2.0] — 2026-10-19

### Added
- MCP server registry: `gemini_py/mcp/__init__.py` with `McpRegistry`,
  `McpClient`, `McpServerConfig`. Supports stdio + HTTP transports.
- CLI subcommands: `mcp add/list/remove/enable/disable` with
  `--transport`, `--env`, `--cwd`, `--scope`, `--include-tools`.
- Auto-memory: `gemini_py/memory.py` with `MemoryLoader` and
  hierarchical `GEMINI.md` search.
- Skills + subagent: `gemini_py/skills/__init__.py` with `SkillsLoader`
  + `@skill_name` expansion; `gemini_py/skills/subagent.py` with
  `delegate_to_subagent()`.
- CLI subcommands: `skills list/show/enable/disable/enable-all/disable-all`.
- New CLI flags: `--no-memory`, `--no-skills`, `--no-subagent`, `--no-mcp`.
- `delegate_subagent(skill, task)` tool exposed when skills are enabled.

## [0.1.0] — 2026-10-18

### Initial release
- Python port of google-gemini/gemini-cli — headless/CLI-only.
- `GeminiClient` with `generate_text()`, `chat()`, `stream()`,
  `run_agent_turn()`, `run_headless()`.
- `AuthConfig` with `.from_env()` — supports `GEMINI_API_KEY`,
  `GOOGLE_API_KEY`, Vertex AI ADC.
- `GeminiConfig` + `ModelAlias` enum.
- `SessionStore` for persistent conversation sessions.
- Built-in tools: `list_directory`, `read_file`, `write_file`,
  `grep_search`, `run_shell_command`, `web_search`, `web_fetch`.
- CLI: argparse-based with `-p/-i/-m/-o/-r/--list-sessions/--delete-session/
  --approval-mode/--yolo/--no-tools/--no-shell/--enable-web/
  --system-prompt/--max-iterations/--max-output-tokens/--temperature/--top-p`.
- Subcommands: `sessions`, `extensions`, `mcp` (stubs at the time).
- Output formats: `text`, `json`, `stream-json`.
- 17 unit tests.
