# Getting Started

This guide walks you through installing `gemini-py`, authenticating, and
running your first prompt in under five minutes.

## Prerequisites

- Python 3.9 or newer (3.12+ recommended).
- A Google account with access to the Gemini API.

You can verify Python is installed and recent enough with:

```bash
python3 --version
# Python 3.12.x
```

## Step 1: Install

Install from source (the package is pip-installable):

```bash
git clone https://github.com/asukaa2/gemini-cli-py.git
cd gemini-cli-py
pip install -e .
```

Or, if you've downloaded the tarball:

```bash
tar xzf gemini-py.tar.gz
cd gemini-py
pip install -e .
```

This installs:

- The `gemini_py` Python package (importable from anywhere).
- Two console-script entry points: `gemini-py` and `gemini` (both point
  to the same CLI).

Verify the install:

```bash
gemini-py --version
# gemini-py 0.5.1
```

## Step 2: Authenticate

The simplest way is to call the `login()` function once. This persists
your API key to `~/.gemini-py/.env` (chmod 0600) so all future
`GeminiClient()` calls pick it up automatically.

### Option A: From Python

```python
from gemini_py import login

# Get your key from https://aistudio.google.com/app/apikey
login("AIza...")
```

### Option B: From the CLI

```bash
# Interactive (hidden prompt)
gemini-py login

# Non-interactive (CI/CD, scripts)
gemini-py login --api-key AIza...
```

### Option C: Environment variable

```bash
export GEMINI_API_KEY="AIza..."
```

The `.env` files are auto-loaded (see
[Authentication](./authentication.md) for the full search order).

## Step 3: Verify

Check that login worked:

```bash
gemini-py whoami
# Method:  gemini_api_key
# API key: AIza****...abcd
```

Or from Python:

```python
from gemini_py import whoami
print(whoami())
# {'method': 'gemini_api_key', 'api_key': 'AIza****...abcd', ...}
```

## Step 4: Run your first prompt

### CLI (non-interactive, headless)

```bash
gemini-py -p "Explain how HTTP/3 differs from HTTP/2"
```

### CLI (with live dashboard)

```bash
gemini-py -p "Build a simple Flask todo app" --live
```

The `--live` flag renders a `rich.live.Live` dashboard with three
panels (header / streaming transcript / footer) that updates 12x/sec.

### Python SDK

```python
from gemini_py import GeminiClient

client = GeminiClient()
result = client.generate_text("Explain how HTTP/3 differs from HTTP/2")
print(result.text)
print(result.usage.to_dict())
```

## Step 5: Use tools (agentic mode)

By default, every CLI invocation runs in agent mode with built-in
tools enabled (`list_directory`, `read_file`, `write_file`,
`create_directory`, `grep_search`, `run_shell_command`).

```bash
gemini-py -p "List all Python files in this directory and tell me how many there are"
```

The model will call `list_directory` and `grep_search` to investigate,
then return a final answer.

## Step 6: Pipe stdin into a prompt

```bash
cat README.md | gemini-py -p "Summarize this file in one paragraph"
```

Or feed an error log:

```bash
python my_app.py 2>&1 | gemini-py -p "What's causing this error?"
```

## Where to go next

- [Quickstart](./quickstart.md) — minimal recipes for common tasks.
- [CLI Reference](./cli-reference.md) — every flag and subcommand.
- [SDK Reference](./sdk-reference.md) — every public class and function.
- [Examples](./examples.md) — bigger worked examples.
- [Troubleshooting](./troubleshooting.md) — if something breaks.
