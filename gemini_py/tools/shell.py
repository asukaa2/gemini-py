"""Shell execution tool for the gemini-py agent.

`run_shell_command` runs a single shell command and returns combined
stdout/stderr along with the exit code. It is the Python equivalent of
gemini-cli's `run_shell_command` tool.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from typing import Optional


@dataclass
class ShellResult:
    """Result of a shell command execution."""

    exit_code: int
    stdout: str
    stderr: str
    command: str

    def to_text(self) -> str:
        out = []
        out.append(f"$ {self.command}")
        out.append(f"[exit={self.exit_code}]")
        if self.stdout:
            out.append(self.stdout.rstrip())
        if self.stderr:
            out.append(f"[stderr]\n{self.stderr.rstrip()}")
        return "\n".join(out)


def run_shell_command(
    command: str,
    *,
    timeout: int = 60,
    cwd: Optional[str] = None,
    env: Optional[dict] = None,
) -> str:
    """Run a shell command and return combined output as a string.

    Args:
        command: The command to run (passed to `sh -c`).
        timeout: Timeout in seconds. The process is killed if exceeded.
        cwd: Working directory. Defaults to current.
        env: Optional environment override. Inherits current env otherwise.

    Returns:
        A formatted string containing stdout/stderr and the exit code.
    """
    if not command or not command.strip():
        return "Error: empty command"

    try:
        proc = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
            env=env,
            executable="/bin/bash" if sys.platform != "win32" else None,
        )
    except subprocess.TimeoutExpired:
        return f"Error: command timed out after {timeout}s: {command}"
    except FileNotFoundError:
        return f"Error: shell not found for command: {command}"
    except Exception as e:
        return f"Error running command: {e}"

    result = ShellResult(
        exit_code=proc.returncode,
        stdout=proc.stdout or "",
        stderr=proc.stderr or "",
        command=command,
    )
    return result.to_text()
