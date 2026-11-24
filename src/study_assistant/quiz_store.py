"""SQLite storage for generated quizzes.

Each quiz is stored as one row holding its full JSON (``Quiz.model_dump_json``), so the
answer key survives restarts. A fresh connection is opened per operation, which keeps
the repository safe to share between FastAPI worker threads.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

from study_assistant.models import Quiz

_SCHEMA = """CREATE TABLE IF NOT EXISTS quizzes (
    id TEXT PRIMARY KEY,
    doc_id TEXT,
    created_at TEXT,
    data TEXT
)"""


class QuizRepository:
    """Saves and loads quizzes in a local SQLite file."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as conn, conn:
            conn.execute(_SCHEMA)

    def save(self, quiz: Quiz) -> None:
        """Insert ``quiz``, replacing any stored quiz with the same id."""
        with closing(self._connect()) as conn, conn:
            conn.execute(
                "INSERT OR REPLACE INTO quizzes (id, doc_id, created_at, data) VALUES (?, ?, ?, ?)",
                (quiz.id, quiz.doc_id, quiz.created_at.isoformat(), quiz.model_dump_json()),
            )

    def get(self, quiz_id: str) -> Quiz | None:
        with closing(self._connect()) as conn:
            row = conn.execute("SELECT data FROM quizzes WHERE id = ?", (quiz_id,)).fetchone()
        return Quiz.model_validate_json(row[0]) if row else None

    def list(self, doc_id: str | None = None) -> list[Quiz]:
        """All quizzes (optionally only for ``doc_id``), newest first."""
        query = "SELECT data FROM quizzes"
        params: tuple[str, ...] = ()
        if doc_id is not None:
            query += " WHERE doc_id = ?"
            params = (doc_id,)
        query += " ORDER BY created_at DESC, rowid DESC"
        with closing(self._connect()) as conn:
            rows = conn.execute(query, params).fetchall()
        return [Quiz.model_validate_json(row[0]) for row in rows]

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path, check_same_thread=False)
