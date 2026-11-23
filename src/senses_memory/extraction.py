"""Extraction operator α: raw experience → atoms (V₀)."""

from __future__ import annotations

import re
from typing import Iterable, List, Optional

from .atoms import Atom


def _split_into_propositions(text: str) -> List[str]:
    """Lightweight proposition splitter.

    Prefer dense, self-contained sentences over tiny fragments.
    This is intentionally simple so it can be replaced by an LLM extractor later.
    """
    # Split on sentence boundaries while keeping reasonable density
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    atoms = []
    for p in parts:
        p = p.strip()
        if len(p) < 12:
            continue
        atoms.append(p)
    if not atoms and text.strip():
        atoms = [text.strip()]
    return atoms


def extract(
    experience: str | Iterable[str],
    *,
    source_session: Optional[str] = None,
    modality: str = "text",
    confidence: float = 1.0,
    provenance: Optional[dict] = None,
) -> List[Atom]:
    """α — Extraction operator.

    Input:  raw experience (conversation turns, tool outputs, observations, documents)
    Output: list of Atom (self-contained propositions / events / facts)
    """
    if isinstance(experience, str):
        texts = [experience]
    else:
        try:
            texts = list(experience)
        except TypeError as exc:
            raise TypeError("experience must be a string or iterable of strings") from exc

    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
        raise TypeError("confidence must be a number between 0 and 1")
    if not 0.0 <= float(confidence) <= 1.0:
        raise ValueError("confidence must be between 0 and 1")
    if not isinstance(modality, str) or not modality.strip():
        raise ValueError("modality must be a non-empty string")

    atoms: List[Atom] = []
    prov = dict(provenance or {})
    for text in texts:
        if not isinstance(text, str):
            raise TypeError("every experience item must be a string")
        for prop in _split_into_propositions(text):
            atoms.append(
                Atom(
                    content=prop,
                    modality=modality,
                    confidence=confidence,
                    source_session=source_session,
                    provenance=dict(prov),
                )
            )
    return atoms
