"""MCP (Model Context Protocol) server registry and client.

This is a Python port of gemini-cli's MCP integration. Instead of running
an MCP server host (like the TypeScript original does), this module is a
**client** that connects to external MCP servers (stdio or HTTP) and
exposes their tools to the Gemini model as regular function-call tools.

The registry persists server configs to `~/.gemini-py/mcp.json` (user scope)
or `./.gemini/mcp.json` (project scope), mirroring gemini-cli's `--scope`
flag.

CLI surface (mirrors `gemini mcp add/list/remove`):
    gemini-py mcp add <name> <command> [args...]
    gemini-py mcp add <name> <url> --transport http
    gemini-py mcp add <name> <cmd> --env KEY=value --include-tools a,b
    gemini-py mcp remove <name>
    gemini-py mcp list
    gemini-py mcp enable <name>
    gemini-py mcp disable <name>
"""

from __future__ import annotations

import asyncio
import json
import os
import shlex
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# MCP client imports (lazy to avoid hard dep at import time)
try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    try:
        from mcp.client.streamable_http import streamablehttp_client
    except ImportError:  # older mcp versions
        streamablehttp_client = None  # type: ignore
    MCP_AVAILABLE = True
except Exception:  # pragma: no cover - mcp is optional
    MCP_AVAILABLE = False
    ClientSession = None  # type: ignore
    StdioServerParameters = None  # type: ignore
    stdio_client = None  # type: ignore


USER_CONFIG_PATH = None  # computed lazily
PROJECT_CONFIG_PATH = None  # computed lazily


def _default_user_config_path() -> Path:
    return Path.home() / ".gemini-py" / "mcp.json"


def _default_project_config_path() -> Path:
    return Path.cwd() / ".gemini" / "mcp.json"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class McpServerConfig:
    """Configuration for a single MCP server."""

    name: str
    transport: str = "stdio"  # 'stdio' | 'http'
    command: Optional[str] = None  # for stdio
    args: List[str] = field(default_factory=list)  # for stdio
    env: Dict[str, str] = field(default_factory=dict)  # for stdio
    cwd: Optional[str] = None  # for stdio
    url: Optional[str] = None  # for http
    scope: str = "user"  # 'user' | 'project'
    include_tools: List[str] = field(default_factory=list)
    enabled: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "McpServerConfig":
        return cls(
            name=data["name"],
            transport=data.get("transport", "stdio"),
            command=data.get("command"),
            args=data.get("args", []) or [],
            env=data.get("env", {}) or {},
            cwd=data.get("cwd"),
            url=data.get("url"),
            scope=data.get("scope", "user"),
            include_tools=data.get("include_tools", []) or [],
            enabled=data.get("enabled", True),
        )

    def describe(self) -> str:
        if self.transport == "http":
            return f"{self.name}  [http] {self.url}"
        cmd_str = " ".join([self.command or ""] + self.args)
        return f"{self.name}  [stdio] {cmd_str}".rstrip()


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class McpRegistry:
    """Persistent registry of MCP server configs.

    Servers are stored in two files:
      - `~/.gemini-py/mcp.json`     (scope=user)
      - `./.gemini/mcp.json`        (scope=project, current dir only)

    On load, both files are merged; project scope overrides user scope on
    name conflicts.
    """

    def __init__(
        self,
        user_path: Optional[Path] = None,
        project_path: Optional[Path] = None,
    ) -> None:
        self.user_path = user_path or _default_user_config_path()
        self.project_path = project_path or _default_project_config_path()
        self._servers: Dict[str, McpServerConfig] = {}
        self._load()

    # --- I/O -------------------------------------------------------------

    def _load(self) -> None:
        for path, scope in [(self.user_path, "user"), (self.project_path, "project")]:
            if not path.is_file():
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                servers = data.get("servers", []) or []
                for s in servers:
                    cfg = McpServerConfig.from_dict(s)
                    cfg.scope = scope  # tag with where it was loaded from
                    self._servers[cfg.name] = cfg
            except Exception:
                continue

    def _save_scope(self, scope: str) -> None:
        path = self.user_path if scope == "user" else self.project_path
        path.parent.mkdir(parents=True, exist_ok=True)
        servers = [s.to_dict() for s in self._servers.values() if s.scope == scope]
        path.write_text(
            json.dumps({"servers": servers}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    # --- CRUD ------------------------------------------------------------

    def add(self, cfg: McpServerConfig) -> McpServerConfig:
        cfg.scope = cfg.scope or "user"
        self._servers[cfg.name] = cfg
        self._save_scope(cfg.scope)
        return cfg

    def remove(self, name: str) -> bool:
        if name not in self._servers:
            return False
        cfg = self._servers.pop(name)
        self._save_scope(cfg.scope)
        return True

    def get(self, name: str) -> Optional[McpServerConfig]:
        return self._servers.get(name)

    def list(self) -> List[McpServerConfig]:
        return sorted(self._servers.values(), key=lambda s: s.name)

    def set_enabled(self, name: str, enabled: bool) -> bool:
        if name not in self._servers:
            return False
        self._servers[name].enabled = enabled
        self._save_scope(self._servers[name].scope)
        return True

    def enabled_servers(self) -> List[McpServerConfig]:
        return [s for s in self.list() if s.enabled]


# ---------------------------------------------------------------------------
# Live client (connects to servers during an agent turn)
# ---------------------------------------------------------------------------


class McpClient:
    """Synchronous wrapper around the async MCP ClientSession.

    Opens all enabled servers at start, exposes their tools to the Gemini
    model, executes tool calls, and tears down at the end.
    """

    def __init__(self, registry: McpRegistry) -> None:
        if not MCP_AVAILABLE:
            raise RuntimeError(
                "mcp package not installed. Run `pip install mcp`."
            )
        self.registry = registry
        self._sessions: Dict[str, Any] = {}  # name -> ClientSession
        self._cm_stack: List[Any] = []  # exit stacks to clean up
        self._tools_cache: Dict[str, List[Dict[str, Any]]] = {}

    # --- Lifecycle -------------------------------------------------------

    def start(self) -> None:
        """Open connections to all enabled MCP servers (sync wrapper)."""
        if not self.registry.enabled_servers():
            return
        asyncio.run(self._start_async())

    async def _start_async(self) -> None:
        for cfg in self.registry.enabled_servers():
            try:
                await self._open_one(cfg)
            except Exception as e:
                sys.stderr.write(
                    f"[gemini-py] MCP server '{cfg.name}' failed to start: {e}\n"
                )

    async def _open_one(self, cfg: McpServerConfig) -> None:
        if cfg.transport == "http":
            if streamablehttp_client is None:
                raise RuntimeError("HTTP transport not supported by this mcp version")
            http_cm = streamablehttp_client(cfg.url)
            read, write, _ = await http_cm.__aenter__()
            self._cm_stack.append(http_cm)
        else:
            if not cfg.command:
                raise ValueError(f"stdio server '{cfg.name}' missing command")
            env = {**os.environ, **cfg.env}
            params = StdioServerParameters(
                command=cfg.command,
                args=cfg.args,
                env=env,
                cwd=cfg.cwd,
            )
            stdio_cm = stdio_client(params)
            read, write = await stdio_cm.__aenter__()
            self._cm_stack.append(stdio_cm)

        session_cm = ClientSession(read, write)
        session = await session_cm.__aenter__()
        self._cm_stack.append(session_cm)
        await session.initialize()
        self._sessions[cfg.name] = session

    def stop(self) -> None:
        """Close all open MCP sessions."""
        if self._cm_stack:
            try:
                asyncio.run(self._stop_async())
            except Exception:
                pass
        self._sessions.clear()
        self._cm_stack.clear()
        self._tools_cache.clear()

    async def _stop_async(self) -> None:
        # Exit in reverse order of entry.
        for cm in reversed(self._cm_stack):
            try:
                await cm.__aexit__(None, None, None)
            except Exception:
                pass

    # --- Tool discovery --------------------------------------------------

    def list_tools(self) -> List[Dict[str, Any]]:
        """Return all enabled tools across all servers, in google-genai
        function-declaration format. Tool names are namespaced as
        `mcp_<server>__<tool>` to avoid collisions.
        """
        if not self._sessions:
            return []
        if not self._tools_cache:
            asyncio.run(self._list_tools_async())
        # Flatten cache
        decls: List[Dict[str, Any]] = []
        for server_name, tools in self._tools_cache.items():
            for t in tools:
                decl = _mcp_tool_to_google_decl(server_name, t)
                if decl is not None:
                    decls.append(decl)
        return decls

    async def _list_tools_async(self) -> None:
        for name, session in list(self._sessions.items()):
            try:
                result = await session.list_tools()
                cfg = self.registry.get(name)
                include = set(cfg.include_tools) if cfg and cfg.include_tools else None
                tools = []
                for t in result.tools or []:
                    if include is None or t.name in include:
                        tools.append(
                            {
                                "name": t.name,
                                "description": t.description or "",
                                "input_schema": t.inputSchema or {"type": "object", "properties": {}},
                            }
                        )
                self._tools_cache[name] = tools
            except Exception as e:
                sys.stderr.write(
                    f"[gemini-py] MCP list_tools('{name}') failed: {e}\n"
                )
                self._tools_cache[name] = []

    # --- Tool execution --------------------------------------------------

    def call_tool(self, full_name: str, args: Dict[str, Any]) -> str:
        """Execute a namespaced MCP tool call. Returns the result as text."""
        server_name, tool_name = _split_namespaced(full_name)
        if server_name is None:
            return f"Error: invalid MCP tool name '{full_name}'"
        if server_name not in self._sessions:
            return f"Error: MCP server '{server_name}' not connected"
        try:
            return asyncio.run(
                self._call_tool_async(server_name, tool_name, args)
            )
        except Exception as e:
            return f"Error calling MCP tool {full_name}: {e}"

    async def _call_tool_async(
        self, server_name: str, tool_name: str, args: Dict[str, Any]
    ) -> str:
        session = self._sessions[server_name]
        result = await session.call_tool(tool_name, arguments=args or {})
        # Result content is a list of TextContent, ImageContent, etc.
        parts: List[str] = []
        content = getattr(result, "content", None) or []
        for item in content:
            text = getattr(item, "text", None)
            if text:
                parts.append(text)
        if not parts:
            # Fall back to structured result if available
            structured = getattr(result, "structuredContent", None)
            if structured:
                parts.append(json.dumps(structured, ensure_ascii=False))
        return "\n".join(parts) if parts else "(no output)"

    # --- Context manager protocol ---------------------------------------

    def __enter__(self) -> "McpClient":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _namespaced(server_name: str, tool_name: str) -> str:
    """Build a namespaced MCP tool identifier safe for the LLM.

    Each part is sanitized (non-alphanumeric -> underscore) and stripped of
    leading/trailing underscores so that `__` separators stay clean and the
    name round-trips through `_split_namespaced` exactly.
    """
    safe_server = "".join(c if c.isalnum() else "_" for c in server_name).strip("_")
    safe_tool = "".join(c if c.isalnum() else "_" for c in tool_name).strip("_")
    return f"mcp_{safe_server}__{safe_tool}"


def _split_namespaced(full_name: str) -> Tuple[Optional[str], str]:
    """Reverse of `_namespaced`. Returns (server_name, tool_name)."""
    if not full_name.startswith("mcp_"):
        return None, full_name
    rest = full_name[4:]
    if "__" not in rest:
        return None, full_name
    server_part, tool_part = rest.split("__", 1)
    return server_part, tool_part


def _mcp_tool_to_google_decl(
    server_name: str, tool: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """Convert an MCP tool spec to a google-genai function declaration dict."""
    full_name = _namespaced(server_name, tool["name"])
    schema = tool.get("input_schema") or {"type": "object", "properties": {}}
    # google-genai expects `parameters` with `type: object` + `properties`.
    if "type" not in schema:
        schema["type"] = "object"
    props = schema.get("properties", {}) or {}
    required = schema.get("required", []) or []
    return {
        "name": full_name,
        "description": (
            f"[MCP/{server_name}] {tool['description'] or tool['name']}"
        ),
        "parameters": {
            "type": "object",
            "properties": props,
            "required": list(required),
        },
    }


# ---------------------------------------------------------------------------
# CLI helpers (env parsing, etc.)
# ---------------------------------------------------------------------------


def parse_env_pair(s: str) -> Tuple[str, str]:
    """Parse 'KEY=value' into (KEY, value). Raises ValueError on bad input."""
    if "=" not in s:
        raise ValueError(f"invalid --env value (expected KEY=value): {s}")
    k, v = s.split("=", 1)
    return k.strip(), v


def parse_command_tokens(tokens: List[str]) -> Tuple[str, List[str]]:
    """Split a command + args list. The first token is the command, the rest
    are args. We don't shell-split because that loses quoting semantics.
    """
    if not tokens:
        raise ValueError("no command given")
    return tokens[0], tokens[1:]
