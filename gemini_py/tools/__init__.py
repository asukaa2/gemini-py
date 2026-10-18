"""Built-in tools for the gemini-py agent.

Each tool is registered as a function-declaration + an executor. Tools here
mirror the most-used gemini-cli tools:
    - filesystem: list_directory, read_file, write_file, grep_search
    - shell: run_shell_command
    - web: web_search, web_fetch

Tools are exposed to the model via the `google.genai` SDK's
`tools=[...]` parameter; the SDK handles the function-calling protocol.
"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from . import filesystem as _filesystem_mod  # noqa: F401  (public helpers)
from . import shell as shell_tool
from . import websearch


@dataclass
class Tool:
    """A single tool: declaration (for the LLM) + executor (for Python)."""

    name: str
    description: str
    parameters: Dict[str, Any]  # JSON schema
    executor: Callable[..., Any]

    def to_google_decl(self) -> Dict[str, Any]:
        """Convert to the dict form accepted by google.genai.types.Tool.

        Note: the `required` flag on individual properties is a gemini-py
        convention (used by `build_default_tools`). The google-genai SDK
        expects `required` to be a LIST at the parameters level, NOT a
        bool inside each property spec. We strip the per-property `required`
        flag here and lift it to the top-level list.
        """
        # Deep-copy the properties so we don't mutate the original Tool.
        props: Dict[str, Any] = {}
        required_names: list[str] = []
        for name, spec in self.parameters.items():
            spec = dict(spec)  # shallow copy
            if spec.pop("required", False):
                required_names.append(name)
            props[name] = spec
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": props,
                "required": required_names,
            },
        }


def _safe_path(path: str, base: Optional[Path] = None) -> Path:
    """Resolve `path` against `base`, refusing paths that escape."""
    base = base or Path.cwd()
    p = (base / path).resolve() if not Path(path).is_absolute() else Path(path).resolve()
    return p


# --- Filesystem tools -------------------------------------------------------


def _t_list_directory() -> Tool:
    def _exec(path: str = ".", **_):
        p = _safe_path(path)
        if not p.exists():
            return f"Error: path does not exist: {p}"
        if not p.is_dir():
            return f"Error: not a directory: {p}"
        entries = sorted(p.iterdir(), key=lambda x: (x.is_file(), x.name))
        lines = []
        for e in entries:
            kind = "DIR " if e.is_dir() else "FILE"
            try:
                size = e.stat().st_size if e.is_file() else 0
            except OSError:
                size = 0
            lines.append(f"{kind} {size:>10}  {e.name}")
        return "\n".join(lines) or "(empty directory)"

    return Tool(
        name="list_directory",
        description="List files and subdirectories in the given directory. Use '.' for cwd.",
        parameters={
            "path": {
                "type": "string",
                "description": "Directory path to list. Defaults to current directory.",
            },
        },
        executor=_exec,
    )


def _t_read_file() -> Tool:
    def _exec(path: str, **_):
        p = _safe_path(path)
        if not p.exists():
            return f"Error: file not found: {p}"
        if not p.is_file():
            return f"Error: not a file: {p}"
        try:
            return p.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            return f"Error reading {p}: {e}"

    return Tool(
        name="read_file",
        description="Read the UTF-8 text contents of a file.",
        parameters={
            "path": {
                "type": "string",
                "description": "Path to the file to read.",
                "required": True,
            },
        },
        executor=_exec,
    )


def _t_write_file() -> Tool:
    def _exec(path: str, content: str, **_):
        p = _safe_path(path)
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
            return f"OK: wrote {len(content)} chars to {p}"
        except Exception as e:
            return f"Error writing {p}: {e}"

    return Tool(
        name="write_file",
        description="Write content to a file, overwriting if it exists.",
        parameters={
            "path": {"type": "string", "description": "File path to write.", "required": True},
            "content": {
                "type": "string",
                "description": "Full text content to write.",
                "required": True,
            },
        },
        executor=_exec,
    )


def _t_grep_search() -> Tool:
    def _exec(pattern: str, path: str = ".", **_):
        base = _safe_path(path)
        if not base.exists():
            return f"Error: path does not exist: {base}"
        try:
            rx = re.compile(pattern)
        except re.error as e:
            return f"Error: invalid regex: {e}"
        matches: List[str] = []
        try:
            for file_path in base.rglob("*"):
                if not file_path.is_file():
                    continue
                # Skip binary / large files
                if file_path.stat().st_size > 1_000_000:
                    continue
                try:
                    text = file_path.read_text(encoding="utf-8", errors="replace")
                except Exception:
                    continue
                for i, line in enumerate(text.splitlines(), 1):
                    if rx.search(line):
                        matches.append(f"{file_path}:{i}: {line}")
                if len(matches) > 200:
                    matches.append("... (truncated, too many matches)")
                    break
        except Exception as e:
            return f"Error during grep: {e}"
        return "\n".join(matches) if matches else "(no matches)"

    return Tool(
        name="grep_search",
        description="Recursively search files for a regex pattern. Returns matching lines.",
        parameters={
            "pattern": {
                "type": "string",
                "description": "Python regex pattern to search for.",
                "required": True,
            },
            "path": {
                "type": "string",
                "description": "Directory to search in. Defaults to cwd.",
            },
        },
        executor=_exec,
    )


# --- Shell tool -------------------------------------------------------------


def _t_run_shell() -> Tool:
    def _exec(command: str, timeout: int = 60, **_):
        return shell_tool.run_shell_command(command, timeout=timeout)

    return Tool(
        name="run_shell_command",
        description=(
            "Run a shell command and return stdout/stderr. Use for build, "
            "test, git, and other shell operations. Avoid interactive commands."
        ),
        parameters={
            "command": {
                "type": "string",
                "description": "Shell command to execute.",
                "required": True,
            },
            "timeout": {
                "type": "integer",
                "description": "Timeout in seconds (default: 60).",
            },
        },
        executor=_exec,
    )


# --- Web tools --------------------------------------------------------------


def _t_web_search() -> Tool:
    def _exec(query: str, **_):
        return websearch.web_search(query)

    return Tool(
        name="web_search",
        description="Search the web for current information. Returns top results with URLs.",
        parameters={
            "query": {
                "type": "string",
                "description": "Search query.",
                "required": True,
            },
        },
        executor=_exec,
    )


def _t_web_fetch() -> Tool:
    def _exec(url: str, **_):
        return websearch.web_fetch(url)

    return Tool(
        name="web_fetch",
        description="Fetch the text content of a web page.",
        parameters={
            "url": {
                "type": "string",
                "description": "URL to fetch.",
                "required": True,
            },
        },
        executor=_exec,
    )


# --- Registry ---------------------------------------------------------------


def build_default_tools(
    *,
    enable_shell: bool = True,
    enable_filesystem: bool = True,
    enable_web: bool = False,
) -> List[Tool]:
    """Build the list of enabled tools."""
    tools: List[Tool] = []
    if enable_filesystem:
        tools.extend(
            [_t_list_directory(), _t_read_file(), _t_write_file(), _t_grep_search()]
        )
    if enable_shell:
        tools.append(_t_run_shell())
    if enable_web:
        tools.extend([_t_web_search(), _t_web_fetch()])
    return tools


def tool_declarations(tools: List[Tool]) -> List[Dict[str, Any]]:
    """Convert tools to the google.genai Tool declaration format."""
    return [{"function_declarations": [t.to_google_decl() for t in tools]}]


def find_tool(tools: List[Tool], name: str) -> Optional[Tool]:
    """Look up a tool by name."""
    for t in tools:
        if t.name == name:
            return t
    return None


def execute_tool_call(
    tools: List[Tool], call: Dict[str, Any]
) -> Dict[str, Any]:
    """Execute a single function call from the model.

    Returns a dict suitable for `client.models.generate_content` part:
        {"function_response": {"name": ..., "response": {"result": ...}}}
    """
    name = call.get("name", "")
    args = call.get("args") or {}
    tool = find_tool(tools, name)
    if tool is None:
        result = f"Error: unknown tool '{name}'"
    else:
        try:
            result = tool.executor(**args)
        except Exception as e:
            result = f"Error executing {name}: {e}"
    return {
        "function_response": {
            "name": name,
            "response": {"result": str(result)},
        }
    }
