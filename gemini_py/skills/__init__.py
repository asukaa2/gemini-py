"""Skills: user-defined markdown prompt files the agent can activate.

A skill is a directory containing a `SKILL.md` file. The first markdown
paragraph becomes the skill's description; the rest is the prompt content
injected when the skill is activated.

Search order (later wins on conflict):
    1. ~/.gemini-py/skills/   (user-wide)
    2. ./.gemini/skills/      (project-scoped, version-controlled)

CLI surface (mirrors `gemini skills list/enable/disable`):
    gemini-py skills list
    gemini-py skills enable <name>
    gemini-py skills disable <name>
    gemini-py skills show <name>

The agent activates a skill either by:
    - Including `@skill_name` anywhere in the prompt (auto-expanded),
    - Or by calling the `delegate_subagent(skill, task)` tool (which spawns
      a separate GeminiClient with the skill's content as system prompt).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional


USER_SKILLS_DIR = None  # computed lazily from Path.home() at SkillsLoader init
PROJECT_SKILLS_DIR = None  # computed lazily from Path.cwd() at SkillsLoader init
ENABLED_STATE_FILE = None  # computed lazily from Path.home()


def _default_user_skills_dir() -> Path:
    return Path.home() / ".gemini-py" / "skills"


def _default_project_skills_dir() -> Path:
    return Path.cwd() / ".gemini" / "skills"


def _default_state_file() -> Path:
    return Path.home() / ".gemini-py" / "skills_state.json"


@dataclass
class Skill:
    """A loaded skill."""

    name: str
    description: str
    content: str
    file_path: Path
    scope: str  # 'user' | 'project'
    enabled: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "scope": self.scope,
            "enabled": self.enabled,
            "file_path": str(self.file_path),
            "content_preview": self.content[:200],
        }

    def full_prompt(self) -> str:
        """Return the full skill prompt content (for injection)."""
        return self.content


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


class SkillsLoader:
    """Discovers and loads skill SKILL.md files.

    Maintains a per-user enable/disable state file so the user can turn
    skills on/off without deleting them.
    """

    def __init__(
        self,
        user_dir: Optional[Path] = None,
        project_dir: Optional[Path] = None,
        state_file: Optional[Path] = None,
    ) -> None:
        self.user_dir = user_dir or _default_user_skills_dir()
        self.project_dir = project_dir or _default_project_skills_dir()
        self.state_file = state_file or _default_state_file()
        self._state: Dict[str, bool] = {}
        self._load_state()
        self._skills: Dict[str, Skill] = {}
        self._discover()

    # --- Discovery -------------------------------------------------------

    def _discover(self) -> None:
        for base, scope in [(self.user_dir, "user"), (self.project_dir, "project")]:
            if not base.is_dir():
                continue
            for entry in sorted(base.iterdir()):
                if not entry.is_dir():
                    continue
                skill_md = entry / "SKILL.md"
                if not skill_md.is_file():
                    continue
                skill = self._parse_skill(entry.name, skill_md, scope)
                if skill is not None:
                    self._skills[skill.name] = skill

    def _parse_skill(
        self, name: str, path: Path, scope: str
    ) -> Optional[Skill]:
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return None
        # The first markdown paragraph becomes the description.
        description = _extract_description(content)
        enabled = self._state.get(name, True)
        return Skill(
            name=name,
            description=description,
            content=content,
            file_path=path,
            scope=scope,
            enabled=enabled,
        )

    # --- State persistence ----------------------------------------------

    def _load_state(self) -> None:
        if not self.state_file.is_file():
            return
        try:
            data = json.loads(self.state_file.read_text(encoding="utf-8"))
            self._state = {k: bool(v) for k, v in (data.get("skills", {}) or {}).items()}
        except Exception:
            self._state = {}

    def _save_state(self) -> None:
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.state_file.write_text(
            json.dumps({"skills": self._state}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    # --- Public API ------------------------------------------------------

    def list(self) -> List[Skill]:
        return sorted(self._skills.values(), key=lambda s: s.name)

    def get(self, name: str) -> Optional[Skill]:
        return self._skills.get(name)

    def enabled(self) -> List[Skill]:
        return [s for s in self.list() if s.enabled]

    def set_enabled(self, name: str, enabled: bool) -> bool:
        if name not in self._skills:
            return False
        self._skills[name].enabled = enabled
        self._state[name] = enabled
        self._save_state()
        return True

    def enable_all(self) -> None:
        for s in self._skills.values():
            s.enabled = True
            self._state[s.name] = True
        self._save_state()

    def disable_all(self) -> None:
        for s in self._skills.values():
            s.enabled = False
            self._state[s.name] = False
        self._save_state()

    # --- Prompt rendering ------------------------------------------------

    def render_summary(self) -> str:
        """Render a short summary of available skills for the system prompt."""
        skills = self.enabled()
        if not skills:
            return ""
        lines = ["# Available skills (use @skill_name to activate)", ""]
        for s in skills:
            desc = s.description or "(no description)"
            lines.append(f"- @{s.name}: {desc}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# @skill_name expansion in prompts
# ---------------------------------------------------------------------------


_SKILL_AT_PATTERN = re.compile(r"@([a-zA-Z0-9][a-zA-Z0-9_-]*)")


def expand_skills_in_prompt(
    prompt: str, loader: SkillsLoader
) -> str:
    """Replace `@skill_name` tokens in a prompt with the skill's full content.

    Unknown skill references are left untouched (the model may pick them up
    as plain text mentions).
    """
    def repl(m: "re.Match[str]") -> str:
        name = m.group(1)
        skill = loader.get(name)
        if skill is None or not skill.enabled:
            return m.group(0)
        return (
            f"\n--- BEGIN SKILL: {skill.name} ---\n"
            f"{skill.full_prompt()}\n"
            f"--- END SKILL: {skill.name} ---\n"
        )

    return _SKILL_AT_PATTERN.sub(repl, prompt)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_description(content: str) -> str:
    """Pull the first markdown paragraph out of a SKILL.md file."""
    # Skip a leading H1 if present, then take the first non-empty line
    # that isn't a heading or frontmatter.
    lines = content.splitlines()
    in_frontmatter = False
    started_frontmatter = False
    description_lines: List[str] = []
    for line in lines:
        stripped = line.strip()
        # Frontmatter
        if stripped == "---" and not description_lines:
            if not started_frontmatter:
                in_frontmatter = True
                started_frontmatter = True
                continue
            else:
                in_frontmatter = False
                continue
        if in_frontmatter:
            continue
        if not stripped:
            if description_lines:
                break
            continue
        # Skip headings
        if stripped.startswith("#"):
            if description_lines:
                break
            continue
        description_lines.append(stripped)
        if len(description_lines) >= 1 and stripped:
            break
    return " ".join(description_lines)[:300]
