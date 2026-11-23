"""SQLite + FTS5 persistence (offline-first)."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import List

from .atoms import Atom


class AtomStore:
    """Append-only atom store with FTS5 full-text search and soft upsert."""

    def __init__(self, db_path: str | Path = "senses_memory.db"):
        self.db_path = str(db_path)
        # :memory: must stay on same connection
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
                USING fts5(content, content='atoms', content_rowid='rowid');
            """
        )
        # Triggers: recreate safely
        cur.execute("DROP TRIGGER IF EXISTS atoms_ai")
        cur.execute("DROP TRIGGER IF EXISTS atoms_ad")
        cur.execute("DROP TRIGGER IF EXISTS atoms_au")
        cur.executescript(
            """
            CREATE TRIGGER atoms_ai AFTER INSERT ON atoms BEGIN
                INSERT INTO atoms_fts(rowid, content) VALUES (new.rowid, new.content);
            END;
            CREATE TRIGGER atoms_ad AFTER DELETE ON atoms BEGIN
                INSERT INTO atoms_fts(atoms_fts, rowid, content)
                    VALUES ('delete', old.rowid, old.content);
            END;
            CREATE TRIGGER atoms_au AFTER UPDATE ON atoms BEGIN
                INSERT INTO atoms_fts(atoms_fts, rowid, content)
                    VALUES ('delete', old.rowid, old.content);
                INSERT INTO atoms_fts(rowid, content) VALUES (new.rowid, new.content);
            END;
            """
        )
        self._conn.commit()

    def add(self, atoms: List[Atom]) -> None:
        """Insert or replace (supports pin/forget metadata updates)."""
        cur = self._conn.cursor()
        for a in atoms:
            cur.execute(
                """
                INSERT INTO atoms
                (atom_id, content, modality, confidence, source_session, timestamp, provenance, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(atom_id) DO UPDATE SET
                    content=excluded.content,
                    modality=excluded.modality,
                    confidence=excluded.confidence,
                    source_session=excluded.source_session,
                    timestamp=excluded.timestamp,
                    provenance=excluded.provenance,
                    metadata=excluded.metadata
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
            # Quote tokens for FTS5 safety
            safe = " ".join(f'"{t}"' for t in query.split() if t)
            if not safe:
                return []
            rows = cur.execute(
                """
                SELECT a.* FROM atoms_fts f
                JOIN atoms a ON a.rowid = f.rowid
                WHERE atoms_fts MATCH ?
                LIMIT ?
                """,
                (safe, limit),
            ).fetchall()
        except sqlite3.OperationalError:
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
