"""Persistent conversation sessions.

Sessions are stored as JSON files under `~/.gemini-py/sessions/<session_id>.json`.
Each session file contains the full message history and the config used.

Mirrors the gemini-cli session commands:
    gemini -r "latest"           — resume most recent
    gemini -r "<session-id>"    — resume by ID
    gemini --list-sessions      — list sessions for current project
    gemini --delete-session N   — delete by index
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import GeminiConfig
from .models import Message


SESSIONS_DIR = Path.home() / ".gemini-py" / "sessions"


@dataclass
class SessionRecord:
    """A stored session."""

    session_id: str
    project_dir: str
    created_at: float
    updated_at: float
    model: str
    messages: List[Dict[str, Any]] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionRecord":
        return cls(
            session_id=data["session_id"],
            project_dir=data["project_dir"],
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            model=data.get("model", ""),
            messages=data.get("messages", []),
            config=data.get("config", {}),
        )


class SessionStore:
    """Filesystem-backed session store."""

    def __init__(self, base_dir: Optional[Path] = None) -> None:
        self.base_dir = base_dir or SESSIONS_DIR
        self.base_dir.mkdir(parents=True, exist_ok=True)

    # --- CRUD ------------------------------------------------------------

    def create(
        self,
        *,
        project_dir: Optional[str] = None,
        config: Optional[GeminiConfig] = None,
        model: str = "",
    ) -> SessionRecord:
        """Create a new empty session and persist it."""
        sid = uuid.uuid4().hex[:12]
        now = time.time()
        rec = SessionRecord(
            session_id=sid,
            project_dir=project_dir or str(Path.cwd()),
            created_at=now,
            updated_at=now,
            model=model,
            config=config.to_dict() if config else {},
        )
        self._write(rec)
        return rec

    def get(self, session_id: str) -> Optional[SessionRecord]:
        path = self.base_dir / f"{session_id}.json"
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
        return SessionRecord.from_dict(data)

    def latest(self, project_dir: Optional[str] = None) -> Optional[SessionRecord]:
        """Return the most recent session for the given project (or any)."""
        records = self.list_all(project_dir=project_dir)
        if not records:
            return None
        # Sort by updated_at descending
        records.sort(key=lambda r: r.updated_at, reverse=True)
        return records[0]

    def list_all(self, project_dir: Optional[str] = None) -> List[SessionRecord]:
        records: List[SessionRecord] = []
        for path in self.base_dir.glob("*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                rec = SessionRecord.from_dict(data)
                if project_dir and rec.project_dir != project_dir:
                    continue
                records.append(rec)
            except Exception:
                continue
        return records

    def update(self, rec: SessionRecord) -> None:
        rec.updated_at = time.time()
        self._write(rec)

    def delete(self, session_id: str) -> bool:
        path = self.base_dir / f"{session_id}.json"
        if not path.is_file():
            return False
        try:
            path.unlink()
            return True
        except Exception:
            return False

    # --- Helpers ---------------------------------------------------------

    def _write(self, rec: SessionRecord) -> None:
        path = self.base_dir / f"{rec.session_id}.json"
        path.write_text(
            json.dumps(rec.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def append_message(self, session_id: str, msg: Message) -> None:
        """Append a message to a session."""
        rec = self.get(session_id)
        if rec is None:
            return
        rec.messages.append(msg.to_dict())
        self.update(rec)
