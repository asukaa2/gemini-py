"""Output formatters for headless mode.

Mirrors gemini-cli's `--output-format` flag:
    text        : plain text (default)
    json        : single JSON object with response + stats
    stream-json : newline-delimited JSON events (JSONL)
"""

from __future__ import annotations

import json
import sys
from enum import Enum
from typing import Any, Dict, Iterator


class OutputFormat(str, Enum):
    TEXT = "text"
    JSON = "json"
    STREAM_JSON = "stream-json"


def format_result(result, fmt: OutputFormat) -> str:
    """Format a GenerateResult for terminal output.

    Args:
        result: A GenerateResult instance.
        fmt: One of OutputFormat.TEXT / .JSON / .STREAM_JSON.

    Returns:
        A string ready to be printed.
    """
    if fmt == OutputFormat.JSON:
        return json.dumps(result.to_dict(), indent=2, ensure_ascii=False)
    if fmt == OutputFormat.STREAM_JSON:
        # For a one-shot, emit init/message/result as JSONL.
        lines: list[str] = []
        lines.append(json.dumps({"type": "init", "model": result.model}, ensure_ascii=False))
        lines.append(
            json.dumps(
                {"type": "result", "response": result.text, "stats": result.usage.to_dict()},
                ensure_ascii=False,
            )
        )
        return "\n".join(lines)
    # text — default
    return result.text


def print_stream(event: Dict[str, Any]) -> None:
    """Print a single stream-json event to stdout."""
    sys.stdout.write(json.dumps(event, ensure_ascii=False) + "\n")
    sys.stdout.flush()
