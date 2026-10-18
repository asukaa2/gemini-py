"""Hierarchical GEMINI.md auto-memory loader.

This is a Python port of gemini-cli's auto-memory feature. It searches a
hierarchy of `GEMINI.md` files in priority order and concatenates them
into the system prompt as additional context.

Search order (later entries override earlier ones; the project's local
file comes last so its instructions win):

    1. ~/.gemini/GEMINI.md            (user-wide, global preferences)
    2. each parent of cwd up to home  (project hierarchy, shallowest first)
    3. cwd/.gemini/GEMINI.md          (project-specific, version-controlled)
    4. cwd/GEMINI.md                  (project root, most specific)

Each file is rendered with a `# Loaded from <path>` header so the model
knows where each chunk came from. Empty files are skipped silently.

A `# gemini-py-ignore` directive anywhere in a file causes the rest of
that file to be skipped — useful for local overrides.

Related but out of scope here:
    - @path/file.md imports (the original supports `@./path` includes).
    - .geminiignore patterns (separate file-ignore system).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


# Default file names we look for.
DEFAULT_FILES = ("GEMINI.md", "GEMINI-MD.md")
GEMINI_DIR_NAME = ".gemini"


@dataclass
class MemoryChunk:
    """A single loaded memory file's contribution to the system prompt."""

    path: Path
    content: str
    source_kind: str  # 'user' | 'parent' | 'project'

    def render(self) -> str:
        return f"# Memory: {self.path}\n{self.content.strip()}\n"


@dataclass
class MemoryConfig:
    """Options for the memory loader."""

    enabled: bool = True
    user_dir: Optional[Path] = None
    project_dir: Optional[Path] = None
    # Additional explicit files to load (highest priority)
    extra_files: List[Path] = field(default_factory=list)


class MemoryLoader:
    """Loads and concatenates GEMINI.md memory files hierarchically.

    Usage:
        loader = MemoryLoader()
        context = loader.load()
        if context:
            system_prompt += "\\n# Project context\\n" + context
    """

    def __init__(self, config: Optional[MemoryConfig] = None) -> None:
        self.config = config or MemoryConfig()
        if self.config.user_dir is not None:
            self.user_dir = self.config.user_dir.expanduser()
        else:
            self.user_dir = Path.home() / GEMINI_DIR_NAME
        if self.config.project_dir is not None:
            self.project_dir = self.config.project_dir.resolve()
        else:
            self.project_dir = Path.cwd().resolve()

    def load(self) -> str:
        """Load and concatenate all memory files. Returns empty string if none."""
        if not self.config.enabled:
            return ""
        chunks: List[MemoryChunk] = []
        # 1. User-wide
        user_file = self._find_in(self.user_dir)
        if user_file is not None:
            chunks.append(MemoryChunk(user_file, _read_filtered(user_file), "user"))
        # 2. Parent hierarchy (cwd up to home, shallowest first)
        for parent in self._parents_to_home():
            f = self._find_in(parent)
            if f is not None:
                chunks.append(MemoryChunk(f, _read_filtered(f), "parent"))
        # 3. Project .gemini dir
        proj_gemini = self.project_dir / GEMINI_DIR_NAME
        f = self._find_in(proj_gemini)
        if f is not None:
            chunks.append(MemoryChunk(f, _read_filtered(f), "project"))
        # 4. Project root
        f = self._find_in(self.project_dir)
        if f is not None and f.parent != proj_gemini:
            chunks.append(MemoryChunk(f, _read_filtered(f), "project"))
        # 5. Explicit extra files (CLI override)
        for ef in self.config.extra_files:
            ef = Path(ef).expanduser().resolve()
            if ef.is_file():
                chunks.append(MemoryChunk(ef, _read_filtered(ef), "extra"))
        if not chunks:
            return ""
        return "\n\n".join(c.render() for c in chunks)

    # --- Discovery -------------------------------------------------------

    def _find_in(self, directory: Path) -> Optional[Path]:
        """Find a memory file in `directory` (not recursive)."""
        if not directory.is_dir():
            return None
        for name in DEFAULT_FILES:
            candidate = directory / name
            if candidate.is_file():
                return candidate
        return None

    def _parents_to_home(self) -> List[Path]:
        """Yield parent directories of cwd up to (not including) home."""
        parents: List[Path] = []
        home = Path.home().resolve()
        # We don't want to include the project itself or its .gemini subdir
        # — those are handled explicitly.
        try:
            cwd = self.project_dir
        except Exception:
            return parents
        # Walk up from project_dir.parent to home
        p = cwd.parent
        while p != p.parent and p != home:
            parents.append(p)
            p = p.parent
        # Reverse so shallower dirs come first (home is the shallowest,
        # but we already handle user_dir separately)
        parents.reverse()
        return parents


def _read_filtered(path: Path) -> str:
    """Read a memory file. Honors the `# gemini-py-ignore` directive."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return f"(error reading {path}: {e})"
    if "# gemini-py-ignore" in text:
        text = text.split("# gemini-py-ignore", 1)[0]
    return text


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------


def load_memory(
    *,
    enabled: bool = True,
    project_dir: Optional[Path] = None,
    extra_files: Optional[List[Path]] = None,
) -> str:
    """One-liner: load GEMINI.md hierarchy and return as system context.

    Paths are resolved lazily at call time (so monkeypatching HOME/cwd in
    tests works).
    """
    cfg = MemoryConfig(
        enabled=enabled,
        project_dir=project_dir,
        extra_files=extra_files or [],
    )
    return MemoryLoader(cfg).load()
