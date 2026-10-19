# gemini-py Documentation

Welcome to the **gemini-py** documentation. This is a Python SDK and CLI
for Google Gemini — a Python port of
[google-gemini/gemini-cli](https://github.com/google-gemini/gemini-cli)
that uses **CLI args, not a GUI/TUI**, and integrates the latest AI Studio
model lineup (as of October 2026).

## What you'll find here

| Document | What it covers |
| --- | --- |
| [Getting Started](./getting-started.md) | Install, set up auth, run your first prompt |
| [Quickstart](./quickstart.md) | Minimal Python and CLI examples |
| [CLI Reference](./cli-reference.md) | Every flag and subcommand |
| [Python SDK Reference](./sdk-reference.md) | Public classes and functions |
| [Authentication](./authentication.md) | `login()`, env vars, `.env` files, Vertex AI |
| [Models](./models.md) | Model aliases (Oct 2026 lineup) + how to pick |
| [Providers](./providers.md) | Multi-provider: OpenAI, Anthropic, Ollama, Transformers, custom |
| [Tools](./tools.md) | Built-in tools, writing custom tools, MCP integration |
| [Live View](./live-view.md) | The `rich.live.Live` dashboard |
| [Skills](./skills.md) | `@skill_name` expansion + subagent delegation |
| [Auto-Memory](./auto-memory.md) | Hierarchical `GEMINI.md` context |
| [MCP Servers](./mcp-servers.md) | Adding external MCP servers |
| [Sessions](./sessions.md) | Save, list, resume, delete conversations |
| [Examples](./examples.md) | Recipes for common tasks |
| [Troubleshooting](./troubleshooting.md) | Common errors and fixes |
| [Changelog](./changelog.md) | Version history |
| [Architecture](./architecture.md) | Module structure and design |
| [Contributing](./contributing.md) | How to extend the SDK |

## What gemini-py is

A focused Python port of the official `gemini-cli` (TypeScript + Ink
TUI). The original gemini-cli is a huge monorepo with hundreds of files
covering terminal UI, node-pty sandboxing, MCP server hosting, A2A
protocol, VSCode integration, telemetry, and more. Many of those
subsystems are deeply tied to the JS/TS ecosystem.

This port captures the **core agent loop** in a clean, pip-installable
Python package:

- Chat with Gemini (with streaming)
- Function-calling / tool execution (built-in: filesystem, shell, web)
- Hierarchical `GEMINI.md` auto-memory
- `@skill_name` expansion + subagent delegation
- MCP server registry (client-side, uses the `mcp` Python package)
- Persistent credentials via `login()` / `logout()` / `whoami()`
- Session management (save / list / resume / delete)
- Live dashboard using `rich.live.Live` (`--live` flag)
- AI Studio's latest model lineup (Oct 2026): `gemini-4-argon`,
  `gemini-3.8-flash`, `gemini-3.5-flash-lite`, etc.

## What gemini-py is NOT

- No interactive TUI (the user explicitly requested CLI args only).
- No TypeScript-style extensions system (use Python packages directly).
- No Docker/Podman sandbox (the `--sandbox` flag is accepted but ignored).
- No A2A protocol server (planned but not yet ported).

## Quick links

- **Repo**: https://github.com/asukaa2/gemini-cli-py
- **Upstream gemini-cli**: https://github.com/google-gemini/gemini-cli
- **Google Gen AI SDK**: https://github.com/googleapis/python-genai
- **AI Studio changelog**: https://ai.google.dev/gemini-api/docs/changelog

## License

Apache-2.0 (same as the upstream gemini-cli).
