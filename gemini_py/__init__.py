"""gemini-py: A Python SDK and CLI for Google Gemini.

A Python port of https://github.com/google-gemini/gemini-cli — headless/CLI-only
(no TUI, no GUI). Uses argparse for the command-line interface.

Public API:
    GeminiClient        — main client class for the Gemini API
    AuthConfig          — authentication configuration
    GeminiConfig        — model / generation configuration
    SessionStore        — persistent conversation sessions
    run_headless        — one-shot headless helper that mirrors `gemini -p`

Example:
    from gemini_py import GeminiClient, AuthConfig
    client = GeminiClient(AuthConfig.from_env())
    print(client.generate_text("Explain what 2 + 2 equals."))
"""

from .auth import AuthConfig, AuthMethod
from .client import GeminiClient
from .config import GeminiConfig, ModelAlias
from .models import GenerateResult, Message, MessageRole, UsageStats
from .sessions import SessionStore, SessionRecord
from .output import OutputFormat, format_result

__version__ = "0.2.0"
__all__ = [
    "GeminiClient",
    "AuthConfig",
    "AuthMethod",
    "GeminiConfig",
    "ModelAlias",
    "GenerateResult",
    "Message",
    "MessageRole",
    "UsageStats",
    "SessionStore",
    "SessionRecord",
    "OutputFormat",
    "format_result",
    "run_headless",
]

from .client import run_headless
