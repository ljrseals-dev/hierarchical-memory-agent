"""SQLite + FTS5 persistence (offline-first)."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import List, Optional

from .atoms import Atom


class AtomStore:
    """Append-only atom store with FTS5 full-text search."""

    def __init__(self, db_path: str | Path = "senses_memory.db"):
        self.db_path = str(db_path)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        cur = self._conn.cursor()
        cur.executescript(
            """
            CREATE TABLE IF NOT EXISTS atoms (
                atom_id   TEXT PRIMARY KEY,
                content   TEXT NOT NULL,
                modality  TEXT,
                confidence REAL,
                source_session TEXT,
                timestamp TEXT,
                provenance TEXT,
                metadata  TEXT
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS atoms_fts
                USING fts5(content, content=atoms, content_rowid=rowid);
            CREATE TRIGGER IF NOT EXISTS atoms_ai AFTER INSERT ON atoms BEGIN
                INSERT INTO atoms_fts(rowid, content) VALUES (new.rowid, new.content);
            END;
            """
        )
        self._conn.commit()

    def add(self, atoms: List[Atom]) -> None:
        cur = self._conn.cursor()
        for a in atoms:
            cur.execute(
                """
                INSERT OR IGNORE INTO atoms
                (atom_id, content, modality, confidence, source_session, timestamp, provenance, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    a.atom_id,
                    a.content,
                    a.modality,
                    a.confidence,
                    a.source_session,
                    a.timestamp.isoformat(),
                    json.dumps(a.provenance),
                    json.dumps(a.metadata),
                ),
            )
        self._conn.commit()

    def search(self, query: str, limit: int = 20) -> List[Atom]:
        cur = self._conn.cursor()
        try:
            rows = cur.execute(
                """
                SELECT a.* FROM atoms_fts f
                JOIN atoms a ON a.rowid = f.rowid
                WHERE atoms_fts MATCH ?
                LIMIT ?
                """,
                (query, limit),
            ).fetchall()
        except sqlite3.OperationalError:
            # FTS query syntax issues → fallback
            rows = cur.execute(
                "SELECT * FROM atoms WHERE content LIKE ? LIMIT ?",
                (f"%{query}%", limit),
            ).fetchall()
        return [self._row_to_atom(r) for r in rows]

    def all_atoms(self) -> List[Atom]:
        cur = self._conn.cursor()
        rows = cur.execute("SELECT * FROM atoms").fetchall()
        return [self._row_to_atom(r) for r in rows]

    def _row_to_atom(self, row: sqlite3.Row) -> Atom:
        return Atom.from_dict(
            {
                "atom_id": row["atom_id"],
                "content": row["content"],
                "modality": row["modality"],
                "confidence": row["confidence"],
                "source_session": row["source_session"],
                "timestamp": row["timestamp"],
                "provenance": json.loads(row["provenance"] or "{}"),
                "metadata": json.loads(row["metadata"] or "{}"),
            }
        )

    def close(self) -> None:
        self._conn.close()
