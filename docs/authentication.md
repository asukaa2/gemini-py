# Authentication

`gemini-py` supports the same auth methods as the official `gemini-cli`,
plus a one-call Python `login()` API.

## At a glance

| Method | Env vars required | Best for |
| --- | --- | --- |
| Gemini API Key | `GEMINI_API_KEY` | Personal use, AI Studio users |
| Vertex AI with ADC | `GOOGLE_CLOUD_PROJECT`, ADC | Google Cloud users with `gcloud` |
| Vertex AI with SA key | `GOOGLE_APPLICATION_CREDENTIALS`, `GOOGLE_CLOUD_PROJECT` | CI/CD, headless |
| Google Cloud API key | `GOOGLE_API_KEY`, `GOOGLE_CLOUD_PROJECT` | When you can't use ADC |

## The `login()` API (recommended)

The simplest way to authenticate. Persists your API key to
`~/.gemini-py/.env` (chmod 0600) so every future `GeminiClient()` call
picks it up automatically.

### From Python

```python
from gemini_py import login, logout, whoami, GeminiClient

# Get a key from https://aistudio.google.com/app/apikey
login("AIza...")            # writes GEMINI_API_KEY to ~/.gemini-py/.env
                            # (also sets it in the current process)

client = GeminiClient()     # auto-picks up the saved key
print(client.generate_text("Hello!").text)

print(whoami())
# {'method': 'gemini_api_key', 'api_key': 'AIza****...abcd', ...}

logout()                    # removes the persisted credentials
```

### From the CLI

```bash
# Interactive (hidden prompt via getpass)
gemini-py login

# Non-interactive (CI/CD, scripts)
gemini-py login --api-key AIza...

# For Vertex AI (Google Cloud API key)
gemini-py login --api-key ya29... \
  --method google \
  --project my-gcp-project \
  --location us-central1

# Persist to project-scoped ./.gemini/.env instead of ~/.gemini-py/.env
gemini-py login --api-key AIza... --scope project

# Show currently-active credentials (masked)
gemini-py whoami

# Clear persisted credentials
gemini-py logout
gemini-py logout --scope project
```

### `login()` reference

```python
def login(
    api_key: str,
    *,
    method: str = "gemini",       # "gemini" | "google"
    project: Optional[str] = None, # required when method="google"
    location: str = "us-central1",
    scope: str = "user",          # "user" | "project"
) -> LoginResult
```

- `method="gemini"` (default) — Google AI Studio API key. Writes
  `GEMINI_API_KEY` to the .env file.
- `method="google"` — Vertex AI / Google Cloud API key. Writes
  `GOOGLE_API_KEY`, `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION`.
  `project` is required.

### `logout()` reference

```python
def logout(*, scope: str = "user") -> bool
```

Removes the `.env` file at the chosen path and clears the env vars
from the current process. Returns `True` if a file was deleted.

### `whoami()` reference

```python
def whoami() -> Optional[dict]
```

Returns a dict with `method`, `api_key` (masked as `AIza****...abcd`),
`project`, `location`, `source`, or `None` if no credentials are
configured.

## Environment variables

You can also set credentials directly without calling `login()`:

```bash
# Google AI Studio API key (recommended)
export GEMINI_API_KEY="AIza..."

# OR: Vertex AI with ADC (requires gcloud auth application-default login)
export GOOGLE_CLOUD_PROJECT="your-project-id"
export GOOGLE_CLOUD_LOCATION="us-central1"
gcloud auth application-default login

# OR: Vertex AI with service account JSON
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/keyfile.json"
export GOOGLE_CLOUD_PROJECT="your-project-id"

# OR: Google Cloud API key (for Vertex AI)
export GOOGLE_API_KEY="your-google-api-key"
export GOOGLE_CLOUD_PROJECT="your-project-id"
```

### Precedence

When multiple methods are configured, the order is:

1. `GEMINI_API_KEY` (Google AI Studio) — wins if set.
2. `GOOGLE_API_KEY` (Vertex AI API key) — used if `GEMINI_API_KEY` is unset.
3. `GOOGLE_APPLICATION_CREDENTIALS` (Vertex AI ADC) — fallback.

## `.env` file auto-loading

`AuthConfig.from_env()` loads `.env` files from these paths (general →
specific; later files override earlier):

1. `~/.gemini-py/.env` ← where `login()` writes by default
2. `~/.gemini/.env` ← gemini-cli convention
3. `./.gemini/.env` ← project-scoped
4. `./.env` ← project root (most specific)

### Example `~/.gemini-py/.env`

```bash
GEMINI_API_KEY="AIzaSy..."

# Optional: override default model
GEMINI_MODEL="flash"
```

### Project-scoped override

If you want a project to use a different key than your user-wide one,
create `./.env` or `./.gemini/.env` in the project root with your
override. It will take precedence over `~/.gemini-py/.env`.

## File permissions

`login()` chmod's the credentials file to `0600` (owner read/write only).
If you create the file manually, do the same:

```bash
chmod 600 ~/.gemini-py/.env
```

## Security notes

- API keys are sensitive. Treat them like passwords.
- The `whoami()` output is masked (`AIza****...abcd`) so it's safe to
  share for debugging.
- The full API key is NEVER logged or printed by gemini-py itself.
- If you accidentally commit an API key to git, revoke it at
  https://aistudio.google.com/app/apikey immediately.

## Running in Google Cloud environments

When running inside Google Cloud Shell, Compute Engine, Cloud Run, or
GKE, gemini-py typically authenticates automatically using ADC from the
environment's metadata server. No manual configuration needed — just
make sure `GOOGLE_CLOUD_PROJECT` is set.

## Headless mode

For headless / CI environments where there's no interactive prompt:

```bash
# Option 1: Set env vars in the CI config
export GEMINI_API_KEY="${{ secrets.GEMINI_API_KEY }}"

# Option 2: Pipe the key to login via stdin
echo "${{ secrets.GEMINI_API_KEY }}" | gemini-py login

# Option 3: Pass --api-key directly
gemini-py login --api-key "${{ secrets.GEMINI_API_KEY }}"
```

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `Error: Authentication not configured: GOOGLE_CLOUD_PROJECT must be set` | Either set `GEMINI_API_KEY`, or run `login()` first, or set `GOOGLE_CLOUD_PROJECT` + ADC. |
| `Error: Authentication not configured: GEMINI_API_KEY is not set` | Get a key from https://aistudio.google.com/app/apikey and run `login()`. |
| `whoami()` returns None but I set `GEMINI_API_KEY` in my shell | Check that the env var is actually exported (`echo $GEMINI_API_KEY`). |
| Vertex AI: `API keys are not supported by this API` | Your org restricts API keys. Use ADC + `GOOGLE_APPLICATION_CREDENTIALS` instead. |
