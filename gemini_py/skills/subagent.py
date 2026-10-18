"""Subagent: spawn a child GeminiClient for delegated tasks.

A subagent is a separate `GeminiClient.run_agent_turn()` call with its own
system prompt and (optionally) reduced tool set. The model can delegate to
a subagent via the `delegate_subagent(skill, task)` tool.

Skills define the subagent's role: a skill named `code-reviewer` provides
the system prompt for a code-review subagent. Calling
`delegate_subagent(skill="code-reviewer", task="Review src/foo.py")` runs a
fresh agent turn with that skill's content as the system prompt and returns
the subagent's final text answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .skills import SkillsLoader, Skill


def _skill_to_system_prompt(skill: Skill) -> str:
    """Render a skill's content as a system prompt for a subagent."""
    return (
        "You are a specialized subagent for the parent agent. "
        "Use the skill instructions below as your role and constraints. "
        "Be concise and return a single final answer.\n\n"
        f"{skill.full_prompt()}"
    )


@dataclass
class SubagentResult:
    """Result of a delegated subagent task."""

    skill: str
    task: str
    text: str
    tool_results: List[Dict[str, Any]] = field(default_factory=list)
    success: bool = True
    error: Optional[str] = None


def delegate_to_subagent(
    *,
    parent_client: Any,
    skill_name: str,
    task: str,
    skills_loader: Optional[SkillsLoader] = None,
    max_iterations: int = 6,
    enable_tools: bool = True,
) -> SubagentResult:
    """Run a subagent turn using `skill_name`'s content as the system prompt.

    Args:
        parent_client: A GeminiClient instance (we reuse its auth + config).
        skill_name: The skill to activate for this subagent.
        task: The task to delegate.
        skills_loader: A SkillsLoader (defaults to a fresh one).
        max_iterations: Max tool iterations inside the subagent.
        enable_tools: Whether to give the subagent the standard toolset.

    Returns:
        SubagentResult with the subagent's final answer.
    """
    if skills_loader is None:
        skills_loader = SkillsLoader()
    skill = skills_loader.get(skill_name)
    if skill is None or not skill.enabled:
        return SubagentResult(
            skill=skill_name,
            task=task,
            text="",
            success=False,
            error=f"unknown or disabled skill: {skill_name}",
        )

    # Build a sub-config that uses the skill's content as system_instruction.
    # We import GeminiConfig here to avoid circular imports at module top.
    from ..config import GeminiConfig

    sub_cfg = GeminiConfig(**{**parent_client.config.to_dict()})
    sub_cfg.system_instruction = _skill_to_system_prompt(skill)

    # Subagents get a reduced tool set by default to keep them focused.
    if enable_tools:
        from ..tools import build_default_tools
        sub_tools = build_default_tools(
            enable_shell=parent_client.config.enable_shell_tool,
            enable_filesystem=parent_client.config.enable_filesystem_tools,
            enable_web=(
                parent_client.config.enable_web_search
                or parent_client.config.enable_web_fetch
            ),
        )
    else:
        sub_tools = []

    try:
        result = parent_client.run_agent_turn(
            task,
            tools=sub_tools,
            config=sub_cfg,
            max_iterations=max_iterations,
        )
    except Exception as e:
        return SubagentResult(
            skill=skill_name,
            task=task,
            text="",
            success=False,
            error=str(e),
        )

    return SubagentResult(
        skill=skill_name,
        task=task,
        text=result.text,
        tool_results=result.tool_results,
        success=True,
    )
