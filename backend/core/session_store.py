"""
SQLite-backed conversation persistence store.
Persists sessions, turns, model routing, traces, and deliverables to workspace/sessions.db.
Survives browser refresh and complete application restart.
"""

import json
import logging
import os
import re
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_DEFAULT_DB_PATH = os.getenv(
    "SESSIONS_DB_PATH",
    str(Path(__file__).parent.parent.parent / "workspace" / "sessions.db")
)

_local_lock = threading.Lock()


def _generate_title(prompt: str) -> str:
    """Generate a clean, human-readable session title from the initial prompt."""
    if not prompt or not isinstance(prompt, str):
        return "New Conversation"
    # Take first line
    first_line = prompt.strip().split("\n")[0].strip()
    # Strip markdown headers, leading bullets, or quotes
    cleaned = re.sub(r'^[#*\->\s"\'`]+', '', first_line).strip()
    cleaned = re.sub(r'["\'`]+$', '', cleaned).strip()
    if not cleaned:
        return "New Conversation"
    if len(cleaned) > 48:
        # Cut at word boundary
        truncated = cleaned[:45].rsplit(" ", 1)[0]
        return (truncated or cleaned[:45]) + "..."
    return cleaned


class SessionStore:
    """Thread-safe SQLite storage for conversation persistence."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or _DEFAULT_DB_PATH
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        return conn

    def _init_db(self) -> None:
        with _local_lock, self._get_connection() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS projects (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    project_id TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE SET NULL
                );

                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    model_used TEXT,
                    trace TEXT,
                    deliverables TEXT,
                    created_at REAL NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_sessions_updated_at ON sessions(updated_at DESC);
                CREATE INDEX IF NOT EXISTS idx_sessions_project_id ON sessions(project_id);
                CREATE INDEX IF NOT EXISTS idx_messages_session_id ON messages(session_id, id ASC);
                CREATE INDEX IF NOT EXISTS idx_projects_updated_at ON projects(updated_at DESC);
            """)

    def create_session(
        self,
        session_id: Optional[str] = None,
        title: Optional[str] = None,
        project_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a new conversation session."""
        sid = session_id or str(uuid.uuid4())
        t = title or "New Conversation"
        now = time.time()
        with _local_lock, self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO sessions (id, title, project_id, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (sid, t, project_id, now, now),
            )
        return {
            "id": sid,
            "title": t,
            "project_id": project_id,
            "created_at": now,
            "updated_at": now,
            "message_count": 0,
            "messages": [],
        }

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a session with all its messages."""
        with _local_lock, self._get_connection() as conn:
            row = conn.execute(
                "SELECT id, title, project_id, created_at, updated_at FROM sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
            if not row:
                return None

            msg_rows = conn.execute(
                """
                SELECT id, role, content, model_used, trace, deliverables, created_at
                FROM messages
                WHERE session_id = ?
                ORDER BY id ASC
                """,
                (session_id,),
            ).fetchall()

            messages = []
            for m in msg_rows:
                trace_list = json.loads(m["trace"]) if m["trace"] else []
                deliv_list = json.loads(m["deliverables"]) if m["deliverables"] else []
                messages.append({
                    "id": m["id"],
                    "role": m["role"],
                    "content": m["content"],
                    "model_used": m["model_used"],
                    "trace": trace_list,
                    "deliverables": deliv_list,
                    "created_at": m["created_at"],
                })

            return {
                "id": row["id"],
                "title": row["title"],
                "project_id": row["project_id"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "message_count": len(messages),
                "messages": messages,
            }

    def list_sessions(
        self,
        limit: int = 50,
        offset: int = 0,
        project_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List past sessions ordered by most recently updated."""
        query = """
            SELECT 
                s.id, s.title, s.project_id, s.created_at, s.updated_at,
                COUNT(m.id) as message_count,
                (SELECT content FROM messages WHERE session_id = s.id ORDER BY id DESC LIMIT 1) as last_message
            FROM sessions s
            LEFT JOIN messages m ON s.id = m.session_id
        """
        params: List[Any] = []
        if project_id == "standalone":
            query += " WHERE s.project_id IS NULL"
        elif project_id is not None and project_id != "" and project_id != "all":
            query += " WHERE s.project_id = ?"
            params.append(project_id)

        query += " GROUP BY s.id ORDER BY s.updated_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        with _local_lock, self._get_connection() as conn:
            rows = conn.execute(query, tuple(params)).fetchall()
            return [
                {
                    "id": r["id"],
                    "title": r["title"],
                    "project_id": r["project_id"],
                    "created_at": r["created_at"],
                    "updated_at": r["updated_at"],
                    "message_count": r["message_count"],
                    "last_message": r["last_message"] or "",
                }
                for r in rows
            ]

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        model_used: Optional[str] = None,
        trace: Optional[List[str]] = None,
        deliverables: Optional[List[str]] = None,
        project_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Add a message turn to the session.
        If session does not exist, automatically creates it.
        If it's the first user turn, updates session title.
        """
        now = time.time()
        trace_json = json.dumps(trace) if trace else None
        deliv_json = json.dumps(deliverables) if deliverables else None

        with _local_lock, self._get_connection() as conn:
            # Check if session exists
            cur = conn.execute("SELECT id, title, project_id FROM sessions WHERE id = ?", (session_id,))
            row = cur.fetchone()
            if not row:
                title = _generate_title(content) if role == "user" else "New Conversation"
                conn.execute(
                    """
                    INSERT INTO sessions (id, title, project_id, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (session_id, title, project_id, now, now),
                )
            else:
                # If existing session has default title and this is a user message, update title
                if role == "user" and row["title"] == "New Conversation":
                    new_title = _generate_title(content)
                    conn.execute(
                        "UPDATE sessions SET title = ?, updated_at = ? WHERE id = ?",
                        (new_title, now, session_id),
                    )
                else:
                    conn.execute(
                        "UPDATE sessions SET updated_at = ? WHERE id = ?",
                        (now, session_id),
                    )

            cur = conn.execute(
                """
                INSERT INTO messages (session_id, role, content, model_used, trace, deliverables, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (session_id, role, content, model_used, trace_json, deliv_json, now),
            )
            msg_id = cur.lastrowid

        return {
            "id": msg_id,
            "session_id": session_id,
            "role": role,
            "content": content,
            "model_used": model_used,
            "trace": trace or [],
            "deliverables": deliverables or [],
            "created_at": now,
        }

    def delete_session(self, session_id: str) -> bool:
        """Delete a session and all its messages."""
        with _local_lock, self._get_connection() as conn:
            cur = conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            return cur.rowcount > 0

    def update_session_title(self, session_id: str, title: str) -> bool:
        """Update session title."""
        now = time.time()
        with _local_lock, self._get_connection() as conn:
            cur = conn.execute(
                "UPDATE sessions SET title = ?, updated_at = ? WHERE id = ?",
                (title.strip(), now, session_id),
            )
            return cur.rowcount > 0

    def assign_session_to_project(self, session_id: str, project_id: Optional[str]) -> bool:
        """Move a conversation session to a project, or set to standalone (None)."""
        now = time.time()
        with _local_lock, self._get_connection() as conn:
            # Check session exists
            s_row = conn.execute("SELECT id FROM sessions WHERE id = ?", (session_id,)).fetchone()
            if not s_row:
                return False

            # If project_id provided, verify project exists
            if project_id is not None and project_id != "" and project_id != "standalone":
                p_row = conn.execute("SELECT id FROM projects WHERE id = ?", (project_id,)).fetchone()
                if not p_row:
                    return False
                target_pid = project_id
            else:
                target_pid = None

            cur = conn.execute(
                "UPDATE sessions SET project_id = ?, updated_at = ? WHERE id = ?",
                (target_pid, now, session_id),
            )
            return cur.rowcount > 0

    # ---------- Project Operations ----------

    def create_project(
        self,
        name: str,
        description: Optional[str] = "",
        project_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a new project grouping container."""
        pid = project_id or str(uuid.uuid4())
        desc = description or ""
        now = time.time()
        with _local_lock, self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO projects (id, name, description, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (pid, name.strip(), desc.strip(), now, now),
            )
        return {
            "id": pid,
            "name": name.strip(),
            "description": desc.strip(),
            "created_at": now,
            "updated_at": now,
            "session_count": 0,
        }

    def get_project(self, project_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve project details along with member sessions."""
        with _local_lock, self._get_connection() as conn:
            row = conn.execute(
                """
                SELECT p.id, p.name, p.description, p.created_at, p.updated_at,
                       COUNT(s.id) AS session_count
                FROM projects p
                LEFT JOIN sessions s ON p.id = s.project_id
                WHERE p.id = ?
                GROUP BY p.id
                """,
                (project_id,),
            ).fetchone()
            if not row:
                return None

            sess_rows = conn.execute(
                """
                SELECT id, title, created_at, updated_at
                FROM sessions
                WHERE project_id = ?
                ORDER BY updated_at DESC
                """,
                (project_id,),
            ).fetchall()

            sessions = [
                {
                    "id": sr["id"],
                    "title": sr["title"],
                    "created_at": sr["created_at"],
                    "updated_at": sr["updated_at"],
                }
                for sr in sess_rows
            ]

            return {
                "id": row["id"],
                "name": row["name"],
                "description": row["description"] or "",
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "session_count": row["session_count"],
                "sessions": sessions,
            }

    def list_projects(self) -> List[Dict[str, Any]]:
        """List all projects ordered by most recently updated, each with its chat count."""
        with _local_lock, self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT p.id, p.name, p.description, p.created_at, p.updated_at,
                       COUNT(s.id) AS session_count
                FROM projects p
                LEFT JOIN sessions s ON p.id = s.project_id
                GROUP BY p.id
                ORDER BY p.updated_at DESC
                """
            ).fetchall()
            return [
                {
                    "id": r["id"],
                    "name": r["name"],
                    "description": r["description"] or "",
                    "created_at": r["created_at"],
                    "updated_at": r["updated_at"],
                    "session_count": r["session_count"],
                }
                for r in rows
            ]

    def update_project(
        self,
        project_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> bool:
        """Update a project's name and/or description."""
        now = time.time()
        with _local_lock, self._get_connection() as conn:
            row = conn.execute("SELECT name, description FROM projects WHERE id = ?", (project_id,)).fetchone()
            if not row:
                return False
            new_name = name.strip() if name is not None and name.strip() else row["name"]
            new_desc = description if description is not None else (row["description"] or "")
            cur = conn.execute(
                "UPDATE projects SET name = ?, description = ?, updated_at = ? WHERE id = ?",
                (new_name, new_desc, now, project_id),
            )
            return cur.rowcount > 0

    def delete_project(self, project_id: str) -> bool:
        """
        Delete a project.
        Sessions inside this project will have project_id set to NULL (become standalone).
        """
        with _local_lock, self._get_connection() as conn:
            conn.execute("UPDATE sessions SET project_id = NULL WHERE project_id = ?", (project_id,))
            cur = conn.execute("DELETE FROM projects WHERE id = ?", (project_id,))
            return cur.rowcount > 0

    def clear(self) -> None:
        """Clear all sessions (for testing)."""
        with _local_lock, self._get_connection() as conn:
            conn.execute("DELETE FROM messages")
            conn.execute("DELETE FROM sessions")
            conn.execute("DELETE FROM projects")


# Singleton store instance
_session_store_instance: Optional[SessionStore] = None


def get_session_store(db_path: Optional[str] = None) -> SessionStore:
    """Return singleton session store."""
    global _session_store_instance
    if _session_store_instance is None or (db_path and _session_store_instance.db_path != db_path):
        _session_store_instance = SessionStore(db_path)
    return _session_store_instance
