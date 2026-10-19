"""System prompts used by gemini-py.

These are simplified equivalents of the gemini-cli system prompt, focused on
the agent behaviors we actually support (file ops, shell, web).
"""

from __future__ import annotations

import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


def build_system_prompt(
    *,
    cwd: Optional[Path] = None,
    enable_tools: bool = True,
    extra_context: Optional[str] = None,
) -> str:
    """Build the default system prompt for the agent.

    Args:
        cwd: Working directory to expose to the model.
        enable_tools: Whether to mention tools available.
        extra_context: Extra context to append (e.g., stdin, file contents).
    """
    cwd = cwd or Path.cwd()
    now = datetime.now().isoformat(timespec="seconds")
    plat = platform.platform()
    py = f"Python {sys.version.split()[0]}"

    parts = [
        "You are gemini-py, an interactive CLI agent powered by Google Gemini.",
        "You assist the user with software engineering, file editing, web search,",
        "and general-purpose tasks directly from the terminal.",
        "",
        f"# Environment",
        f"- Working directory: {cwd}",
        f"- Platform: {plat}",
        f"- Runtime: {py}",
        f"- Current time: {now}",
        "",
    ]

    if enable_tools:
        parts.extend(
            [
                "# Available tools",
                "When you need to perform an action, call one of these tools:",
                "- `list_directory(path)`: List entries in a directory.",
                "- `read_file(path)`: Read a UTF-8 text file's contents.",
                "- `write_file(path, content)`: Write content to a file. Parent directories are auto-created.",
                "- `create_directory(path)`: Create a directory (and any missing parents).",
                "- `grep_search(pattern, path)`: Recursively search for a regex.",
                "- `run_shell_command(command)`: Run a shell command and return output.",
                "- `web_search(query)`: Search the web for up-to-date info.",
                "- `web_fetch(url)`: Fetch the contents of a URL.",
                "",
                "# Creating code projects",
                "When the user asks you to create code, files, or an app but does NOT",
                "specify a directory, you MUST automatically create a project folder",
                "first (use a short, descriptive name based on the request, e.g.",
                "`hello-world/`, `login-page/`, `todo-api/`) and put ALL files",
                "inside it. Use `create_directory` to scaffold the structure (and",
                "subfolders like `src/`, `tests/`, `docs/` as needed) before",
                "calling `write_file`. `write_file` also auto-creates parent dirs,",
                "so you can directly write to `my-project/src/main.py` if you prefer.",
                "Always tell the user the final folder name when you're done.",
                "",
                "Prefer tool calls over asking the user to do things. When done,",
                "give a concise summary of what you did.",
                "",
            ]
        )

    if extra_context:
        parts.append("# Additional context")
        parts.append(extra_context)
        parts.append("")

    return "\n".join(parts)
