"""Coarsening operator C = (π, ρ)."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

from .atoms import Atom


@dataclass
class Representative:
    """A node at a coarsened level."""

    rep_id: str
    content: str                          # the representative text (summary or label)
    level: int
    child_ids: List[str] = field(default_factory=list)   # atom_ids or lower-level rep_ids
    strategy: str = "summary"             # "summary" | "label" | "centroid"
    metadata: dict = field(default_factory=dict)


def _simple_hash(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]


def partition_by_keyword(atoms: Sequence[Atom], max_groups: int = 8) -> Dict[int, List[Atom]]:
    """π — simple keyword / first-noun style partition (deterministic, offline).

    Real systems would use embeddings or LLM clustering. This keeps the
    pipeline runnable with zero external dependencies.
    """
    buckets: Dict[str, List[Atom]] = {}
    for a in atoms:
        # crude key: first contentful word
        words = [w.lower() for w in a.content.split() if len(w) > 3]
        key = words[0] if words else "misc"
        buckets.setdefault(key, []).append(a)

    # merge smallest buckets if too many
    items = sorted(buckets.items(), key=lambda kv: -len(kv[1]))
    groups: Dict[int, List[Atom]] = {}
    for i, (_, group) in enumerate(items[:max_groups]):
        groups[i] = group
    if len(items) > max_groups:
        overflow = []
        for _, group in items[max_groups:]:
            overflow.extend(group)
        if overflow:
            groups[max_groups] = overflow
    return groups


def represent_summary(group: Sequence[Atom]) -> str:
    """ρ — produce a self-sufficient summary from a group of atoms."""
    if not group:
        return ""
    if len(group) == 1:
        return group[0].content
    # Concatenate with light compression: keep first sentence of each, join
    parts = []
    for a in group[:6]:
        sent = a.content.split(".")[0].strip()
        if sent:
            parts.append(sent)
    return " | ".join(parts)


def coarsen(
    atoms: Sequence[Atom],
    *,
    level: int = 1,
    partition_fn: Optional[Callable] = None,
    represent_fn: Optional[Callable] = None,
    max_groups: int = 8,
) -> List[Representative]:
    """C = (π, ρ) — one level of coarsening.

    Returns a list of Representative nodes whose child_ids point at the
    input atoms (or previous-level reps).
    """
    π = partition_fn or (lambda ats: partition_by_keyword(ats, max_groups=max_groups))
    ρ = represent_fn or represent_summary

    groups = π(atoms)
    reps: List[Representative] = []
    for gid, group in groups.items():
        content = ρ(group)
        rep_id = f"L{level}-{_simple_hash(content + str(gid))}"
        child_ids = [a.atom_id for a in group]
        reps.append(
            Representative(
                rep_id=rep_id,
                content=content,
                level=level,
                child_ids=child_ids,
                strategy="summary",
            )
        )
    return reps
