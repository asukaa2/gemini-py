"""Filesystem tools (placeholder — actual implementations live in tools/__init__.py).

This module is kept for clarity and so it can host more elaborate filesystem
behaviors in the future (sandboxing, path-traversal protection, etc.).
"""

from __future__ import annotations

from pathlib import Path


def safe_resolve(path: str, base: Path | None = None) -> Path:
    """Resolve `path` against `base`, refusing to escape it."""
    base = base or Path.cwd()
    p = Path(path)
    if not p.is_absolute():
        p = base / p
    return p.resolve()


def is_binary(path: Path, sample_size: int = 4096) -> bool:
    """Heuristic: detect whether a file is binary by reading a sample."""
    try:
        with path.open("rb") as fh:
            chunk = fh.read(sample_size)
    except OSError:
        return True
    return b"\x00" in chunk
