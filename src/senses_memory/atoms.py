"""Atomic information units (V₀)."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional


@dataclass
class Atom:
    """A self-contained atomic information unit produced by extraction α."""

    content: str
    atom_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    modality: str = "text"          # 5-Senses modality tag (text, visual, auditory, …)
    confidence: float = 1.0
    source_session: Optional[str] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    provenance: dict[str, Any] = field(default_factory=dict)
    embedding: Optional[list[float]] = None   # optional; filled by downstream indexers
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "atom_id": self.atom_id,
            "content": self.content,
            "modality": self.modality,
            "confidence": self.confidence,
            "source_session": self.source_session,
            "timestamp": self.timestamp.isoformat(),
            "provenance": self.provenance,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Atom":
        ts = d.get("timestamp")
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts)
        return cls(
            atom_id=d["atom_id"],
            content=d["content"],
            modality=d.get("modality", "text"),
            confidence=float(d.get("confidence", 1.0)),
            source_session=d.get("source_session"),
            timestamp=ts or datetime.now(timezone.utc),
            provenance=d.get("provenance") or {},
            metadata=d.get("metadata") or {},
        )
