"""Persistent login API for gemini-py.

Lets users authenticate once and have the credentials auto-loaded by every
future GeminiClient() call. Mirrors the gemini-cli "Use Gemini API key" flow
but as a Python function call instead of an interactive picker.

Example:
    from gemini_py import login, logout

    login("AIza...")           # writes GEMINI_API_KEY to ~/.gemini-py/.env
    # ... from now on, GeminiClient() picks it up automatically ...

    logout()                   # remove the persisted credentials

For Vertex AI:
    login(
        "ya29.abc...",
        method="google",
        project="my-gcp-project",
        location="us-central1",
    )

Credentials are persisted to `~/.gemini-py/.env` (scope=user) or
`./.gemini/.env` (scope=project). The file is read by `AuthConfig.from_env()`
via the .env search path.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# Where credentials are persisted (computed lazily so monkeypatching
# HOME/cwd in tests works as expected).
USER_CREDENTIALS_FILE = None  # type: ignore
PROJECT_CREDENTIALS_FILE = None  # type: ignore


def _user_credentials_file() -> Path:
    return Path.home() / ".gemini-py" / ".env"


def _project_credentials_file() -> Path:
    return Path.cwd() / ".gemini" / ".env"


class LoginError(Exception):
    """Raised when login() or logout() encounters a problem."""


@dataclass
class LoginResult:
    """Result of a login() call."""

    method: str
    api_key: str
    credentials_path: Path
    scope: str
    project: Optional[str] = None
    location: Optional[str] = None

    def describe(self) -> str:
        """Human-readable summary."""
        masked = _mask(self.api_key)
        if self.method == "gemini":
            return (
                f"Logged in with Gemini API key {masked}.\n"
                f"  Saved to: {self.credentials_path}\n"
                f"  Scope:    {self.scope}\n"
                f"  Future GeminiClient() calls will pick this up automatically."
            )
        if self.method == "google":
            return (
                f"Logged in with Google Cloud API key {masked} "
                f"(Vertex AI, project={self.project}, location={self.location}).\n"
                f"  Saved to: {self.credentials_path}\n"
                f"  Scope:    {self.scope}\n"
                f"  Future GeminiClient() calls will pick this up automatically."
            )
        return f"Logged in (method={self.method})."


# ---------------------------------------------------------------------------
# login() / logout()
# ---------------------------------------------------------------------------


def login(
    api_key: str,
    *,
    method: str = "gemini",
    project: Optional[str] = None,
    location: str = "us-central1",
    scope: str = "user",
) -> LoginResult:
    """Persist credentials for future GeminiClient() calls.

    Args:
        api_key: The API key to save.
            - For `method="gemini"` (default): your Google AI Studio API key
              (get one from https://aistudio.google.com/app/apikey).
            - For `method="google"`: a Google Cloud API key for Vertex AI.
        method: Auth method. `"gemini"` (default) or `"google"` (Vertex AI).
        project: Google Cloud project ID. **Required** when method="google".
        location: Vertex AI region. Defaults to "us-central1". Ignored for
            method="gemini".
        scope: Where to persist credentials. `"user"` (default) writes to
            `~/.gemini-py/.env`; `"project"` writes to `./.gemini/.env`.

    Returns:
        LoginResult with details about the persisted credentials.

    Raises:
        LoginError: if the api_key is empty or project is missing for
            method="google".

    Side effects:
        - Writes the .env file at the chosen path (creating parent dirs).
        - Sets the matching env var in the current process so subsequent
          code in the SAME process can use the credentials immediately
          (without re-importing).
    """
    if not api_key or not api_key.strip():
        raise LoginError("api_key is required (got empty value).")
    api_key = api_key.strip()

    if method == "gemini":
        env_var = "GEMINI_API_KEY"
    elif method == "google":
        env_var = "GOOGLE_API_KEY"
        if not project:
            raise LoginError(
                "project is required when method='google' (Vertex AI)."
            )
    else:
        raise LoginError(
            f"unknown method '{method}'. Use 'gemini' or 'google'."
        )

    # Pick the credentials file based on scope.
    if scope == "user":
        path = _user_credentials_file()
    elif scope == "project":
        path = _project_credentials_file()
    else:
        raise LoginError(
            f"unknown scope '{scope}'. Use 'user' or 'project'."
        )

    # Build the .env contents. We REPLACE any existing entries for the same
    # env var so the file stays clean if login() is called multiple times.
    entries: dict = {env_var: api_key}
    if method == "google":
        entries["GOOGLE_CLOUD_PROJECT"] = project  # type: ignore[assignment]
        entries["GOOGLE_CLOUD_LOCATION"] = location

    _write_env_file(path, entries)

    # Also set in the current process so GeminiClient() works immediately.
    os.environ[env_var] = api_key
    if method == "google":
        os.environ["GOOGLE_CLOUD_PROJECT"] = project  # type: ignore[assignment]
        os.environ["GOOGLE_CLOUD_LOCATION"] = location

    return LoginResult(
        method=method,
        api_key=api_key,
        credentials_path=path,
        scope=scope,
        project=project if method == "google" else None,
        location=location if method == "google" else None,
    )


def logout(*, scope: str = "user") -> bool:
    """Remove persisted credentials.

    Args:
        scope: Which credentials file to clear. `"user"` (default) clears
            `~/.gemini-py/.env`; `"project"` clears `./.gemini/.env`.

    Returns:
        True if a credentials file was deleted, False if there was nothing
        to delete.

    Side effects:
        - Removes the .env file at the chosen path.
        - Unsets GEMINI_API_KEY / GOOGLE_API_KEY / GOOGLE_CLOUD_PROJECT /
          GOOGLE_CLOUD_LOCATION from the current process's environment.
    """
    if scope == "user":
        path = _user_credentials_file()
    elif scope == "project":
        path = _project_credentials_file()
    else:
        raise LoginError(
            f"unknown scope '{scope}'. Use 'user' or 'project'."
        )

    deleted = False
    if path.is_file():
        try:
            path.unlink()
            deleted = True
        except Exception as e:
            raise LoginError(f"failed to remove {path}: {e}") from e

    # Clear from current process env as well.
    for var in (
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "GOOGLE_CLOUD_PROJECT",
        "GOOGLE_CLOUD_LOCATION",
    ):
        os.environ.pop(var, None)

    return deleted


def whoami() -> Optional[dict]:
    """Return currently-active credentials (read from env / .env files).

    Returns None if no credentials are configured. Otherwise a dict with
    keys: method, api_key (masked), project, location, source (path).
    """
    # Force a reload from disk so we see what's persisted.
    try:
        from .auth import AuthConfig
        cfg = AuthConfig.from_env(load_dotenv_files=True)
    except Exception:
        return None
    if cfg.api_key is None:
        return None
    return {
        "method": cfg.method.value,
        "api_key": _mask(cfg.api_key),
        "project": cfg.vertex_project,
        "location": cfg.vertex_location,
        "source": str(cfg.describe()),
    }


# ---------------------------------------------------------------------------
# .env file helpers
# ---------------------------------------------------------------------------


def _write_env_file(path: Path, entries: dict) -> None:
    """Write `entries` to `path`, preserving any unrelated lines.

    For each KEY in `entries`, removes any existing `KEY=...` line and
    appends a fresh `KEY=value` line at the end. If the file doesn't exist,
    creates it.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    # Read existing lines (if file exists).
    existing_lines: list[str] = []
    if path.is_file():
        try:
            existing_lines = path.read_text(encoding="utf-8").splitlines()
        except Exception:
            existing_lines = []

    # Drop lines for any key we're about to write, then append fresh ones.
    keys_to_replace = set(entries.keys())
    kept_lines = [
        line for line in existing_lines
        if not _env_line_matches(line, keys_to_replace)
    ]
    new_lines = list(kept_lines)
    if new_lines and new_lines[-1].strip():
        new_lines.append("")
    for k, v in entries.items():
        new_lines.append(f'{k}="{v}"')

    path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    # Tighten permissions on the credentials file when possible.
    try:
        path.chmod(0o600)
    except Exception:
        pass


def _env_line_matches(line: str, keys: set) -> bool:
    """True if `line` is an assignment to one of `keys` (e.g. 'KEY=value')."""
    if "=" not in line:
        return False
    key = line.split("=", 1)[0].strip()
    # Support 'export KEY=...' form too.
    if key.startswith("export "):
        key = key[len("export "):].strip()
    return key in keys


def _mask(s: str) -> str:
    """Return s with its middle masked for safe display."""
    if not s:
        return ""
    if len(s) <= 8:
        return "*" * len(s)
    return s[:4] + "*" * (len(s) - 8) + s[-4:]
