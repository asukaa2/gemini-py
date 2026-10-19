# Contributing

How to extend `gemini-py`.

## Dev setup

```bash
git clone https://github.com/asukaa2/gemini-cli-py.git
cd gemini-cli-py
pip install -e ".[dev]"   # installs pytest + pytest-mock too
```

Run the test suite:

```bash
pytest tests/ -v
# 109 passed in 2.5s
```

## Project layout

See [Architecture](./architecture.md) for the full module breakdown.

The code is in `gemini_py/`, tests in `tests/`, docs in `docs/`.

## Code style

- Python 3.9+ (uses `from __future__ import annotations`).
- `dataclass` for value types.
- Type hints everywhere.
- Docstrings on every public class and function (markdown-friendly).
- No emojis in code or output (per the project's conventions).
- No trailing whitespace.
- Tests for every new feature.

## Adding a new built-in tool

1. Add a `_t_<name>() -> Tool` factory in
   [`gemini_py/tools/__init__.py`](../gemini_py/tools/__init__.py).
2. Register it in `build_default_tools()` (gated by an
   `enable_<name>` flag, default to off unless it's filesystem/shell).
3. Add a CLI flag `--no-<name>` or `--enable-<name>` to opt in/out.
4. Update the system prompt in
   [`gemini_py/prompts.py`](../gemini_py/prompts.py) to mention the
   new tool.
5. Add tests in `tests/test_smoke.py` (tool registry) and
   `tests/test_advanced.py` (tool behavior).

Example:

```python
def _t_fetch_weather() -> Tool:
    def _exec(city: str, **_):
        return _call_weather_api(city)
    return Tool(
        name="fetch_weather",
        description="Get the current weather for a city.",
        parameters={
            "city": {
                "type": "string",
                "description": "City name.",
                "required": True,
            },
        },
        executor=_exec,
    )
```

## Adding a new model alias

1. Edit the `mapping` dict in
   `ModelAlias.resolve()` in
   [`gemini_py/config.py`](../gemini_py/config.py).
2. Add a new `ModelAlias` enum member if it's a new alias.
3. Update `docs/models.md` table.
4. Update `tests/test_smoke.py::test_model_alias_resolves`.

## Adding a new CLI flag

1. Add an `add_argument()` call in
   `build_parser()` in
   [`gemini_py/cli.py`](../gemini_py/cli.py).
2. Wire it into the `run_agent_turn()` call in `main()`.
3. Update `docs/cli-reference.md`.
4. Add a test in `tests/test_smoke.py::test_cli_parser_flags_parse`
   or a new test function.

## Adding a new subcommand

Subcommands need special handling because of argparse's positional +
subparser conflict. The pattern is:

1. Add a `_add_<name>_subparser(subparsers)` function that defines the
   subcommand + its options.
2. Register it in `_build_subcommand_parser()`.
3. Add the name to the `SUBCOMMANDS` set in `main()`.
4. Add a dispatch branch (`if args.command == "<name>": return _cmd_<name>(args)`).
5. Implement `_cmd_<name>(args) -> int`.
6. Update `docs/cli-reference.md`.
7. Add a test that calls `main(["<name>", "list"])` (or whatever the
   subcommand's primary sub is).

## Extending the live view

To add a new event type:

1. Add an `elif etype == "new_event":` branch to both `LiveView.emit()`
   and `RichLiveChat.emit()` in
   [`gemini_py/live.py`](../gemini_py/live.py).
2. Implement the render logic in each class.
3. Emit the event from `run_agent_turn()` in
   [`gemini_py/client.py`](../gemini_py/client.py).
4. Add a test in `tests/test_live.py`.

## Updating the model alias table

When AI Studio ships new models, update
[`gemini_py/config.py`](../gemini_py/config.py)'s `ModelAlias` mapping.
Check the changelog at <https://ai.google.dev/gemini-api/docs/changelog>
and bump the version with a changelog entry.

## Running tests

```bash
# Full suite
pytest tests/ -v

# Just the smoke tests
pytest tests/test_smoke.py -v

# With coverage (if pytest-cov installed)
pytest tests/ --cov=gemini_py --cov-report=term-missing
```

Tests use `monkeypatch` to redirect `HOME` and `cwd`, so they don't
touch the user's real `~/.gemini-py/` directory.

## Commit message conventions

Follow this format:

```
<type>: <subject>

<body — explain what + why>

<optional: footer with breaking changes, related issues>

Author: <your name> <<your email>>
```

Types: `Add`, `Fix`, `Update`, `Remove`, `Refactor`, `Docs`.

Examples:
- `Add: live view with rich.live.Live dashboard`
- `Fix: strip per-property required flag from tool declarations`
- `Update: model aliases to AI Studio Oct 2026 lineup`
- `Docs: complete docs/ folder with 16 documents`

## Tests for new features

Every new feature MUST come with tests. Aim for at least:

- One happy-path test
- One error-path test
- One edge case (empty input, missing field, etc.)
- One CLI smoke test if applicable

Use `pytest` fixtures (`tmp_path`, `monkeypatch`) heavily — don't
write to the real `~/.gemini-py/` directory.

## Releasing

1. Bump `__version__` in
   [`gemini_py/__init__.py`](../gemini_py/__init__.py) and
   `version = "..."` in
   [`pyproject.toml`](../pyproject.toml).
2. Update [`docs/changelog.md`](./changelog.md) with the new version
   and changes.
3. Update [`tests/test_smoke.py::test_version`](../tests/test_smoke.py)
   to assert the new version.
4. Run `pytest tests/` to confirm.
5. Commit + push.
6. Tag: `git tag v0.6.0 && git push origin v0.6.0` (if you want a
   GitHub release).

## License

Apache-2.0 (same as the upstream gemini-cli). Any contributions you
make are licensed under the same terms.

## Code of conduct

Be respectful. Be specific. Be helpful. Don't break tests.
