import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator, Optional

from .core import Session


class Database:
    def __init__(self, db_path: str | Path | None = None):
        self.db_path = Path(db_path) if db_path else Path.home() / ".jixing" / "jixing.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    model_provider TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    system_info TEXT NOT NULL,
                    total_tokens INTEGER DEFAULT 0,
                    total_duration_ms INTEGER DEFAULT 0
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    metrics TEXT,
                    FOREIGN KEY (session_id) REFERENCES sessions(id)
                )
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_messages_session_id ON messages(session_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON messages(timestamp)
            """)
            conn.commit()

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def save_session(self, session: Session):
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO sessions
                (id, model_provider, model_name, created_at, updated_at,
                 system_info, total_tokens, total_duration_ms)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session.id,
                    session.model_provider,
                    session.model_name,
                    session.created_at,
                    session.updated_at,
                    json.dumps(session.system_info),
                    session.total_tokens,
                    session.total_duration_ms,
                ),
            )

            cursor.execute("DELETE FROM messages WHERE session_id = ?", (session.id,))
            for msg in session.messages:
                cursor.execute(
                    """
                    INSERT INTO messages (session_id, role, content, timestamp, metrics)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        session.id,
                        msg["role"],
                        msg["content"],
                        msg["timestamp"],
                        json.dumps(msg.get("metrics")),
                    ),
                )
            conn.commit()

    def get_session(self, session_id: str) -> Optional[Session]:
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))
            row = cursor.fetchone()
            if not row:
                return None

            cursor.execute(
                "SELECT * FROM messages WHERE session_id = ? ORDER BY timestamp", (session_id,)
            )
            messages = []
            for msg_row in cursor.fetchall():
                msg = {
                    "role": msg_row["role"],
                    "content": msg_row["content"],
                    "timestamp": msg_row["timestamp"],
                }
                if msg_row["metrics"]:
                    msg["metrics"] = json.loads(msg_row["metrics"])
                messages.append(msg)

            return Session(
                id=row["id"],
                model_provider=row["model_provider"],
                model_name=row["model_name"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                system_info=json.loads(row["system_info"]),
                messages=messages,
                total_tokens=row["total_tokens"],
                total_duration_ms=row["total_duration_ms"],
            )

    def list_sessions(
        self,
        model_provider: Optional[str] = None,
        model_name: Optional[str] = None,
        limit: int = 100,
    ) -> list[Session]:
        with self._connect() as conn:
            cursor = conn.cursor()
            query = "SELECT * FROM sessions WHERE 1=1"
            params = []
            if model_provider:
                query += " AND model_provider = ?"
                params.append(model_provider)
            if model_name:
                query += " AND model_name LIKE ?"
                params.append(f"%{model_name}%")
            query += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)

            cursor.execute(query, params)
            sessions = []
            for row in cursor.fetchall():
                sessions.append(
                    Session(
                        id=row["id"],
                        model_provider=row["model_provider"],
                        model_name=row["model_name"],
                        created_at=row["created_at"],
                        updated_at=row["updated_at"],
                        system_info=json.loads(row["system_info"]),
                        total_tokens=row["total_tokens"],
                        total_duration_ms=row["total_duration_ms"],
                    )
                )
            return sessions

    def delete_session(self, session_id: str) -> bool:
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            cursor.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            conn.commit()
            return cursor.rowcount > 0

    def search_messages(
        self,
        query: str,
        session_id: Optional[str] = None,
        model_provider: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict]:
        with self._connect() as conn:
            cursor = conn.cursor()
            sql = """
                SELECT m.*, s.model_provider, s.model_name
                FROM messages m
                JOIN sessions s ON m.session_id = s.id
                WHERE m.content LIKE ?
            """
            params = [f"%{query}%"]
            if session_id:
                sql += " AND m.session_id = ?"
                params.append(session_id)
            if model_provider:
                sql += " AND s.model_provider = ?"
                params.append(model_provider)
            sql += " ORDER BY m.timestamp DESC LIMIT ?"
            params.append(limit)

            cursor.execute(sql, params)
            results = []
            for row in cursor.fetchall():
                results.append(
                    {
                        "session_id": row["session_id"],
                        "model_provider": row["model_provider"],
                        "model_name": row["model_name"],
                        "role": row["role"],
                        "content": row["content"],
                        "timestamp": row["timestamp"],
                        "metrics": json.loads(row["metrics"]) if row["metrics"] else None,
                    }
                )
            return results

    def get_stats(self) -> dict[str, Any]:
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) as count FROM sessions")
            total_sessions = cursor.fetchone()["count"]

            cursor.execute("SELECT COUNT(*) as count FROM messages")
            total_messages = cursor.fetchone()["count"]

            cursor.execute("SELECT SUM(total_tokens) as total FROM sessions")
            total_tokens = cursor.fetchone()["total"] or 0

            cursor.execute("SELECT SUM(total_duration_ms) as total FROM sessions")
            total_duration_ms = cursor.fetchone()["total"] or 0

            cursor.execute("""
                SELECT model_provider, COUNT(*) as count
                FROM sessions
                GROUP BY model_provider
            """)
            by_provider = {row["model_provider"]: row["count"] for row in cursor.fetchall()}

            return {
                "total_sessions": total_sessions,
                "total_messages": total_messages,
                "total_tokens": total_tokens,
                "total_duration_ms": total_duration_ms,
                "sessions_by_provider": by_provider,
            }
