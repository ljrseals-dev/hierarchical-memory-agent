"""Hierarchy construction and navigation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from .atoms import Atom
from .coarsening import Representative, coarsen


@dataclass
class Hierarchy:
    """H = (G₀, C₁, …, C_L) — multi-level memory hierarchy."""

    atoms: Dict[str, Atom] = field(default_factory=dict)
    levels: Dict[int, List[Representative]] = field(default_factory=dict)
    # parent_id → list of child_ids (atoms or lower reps)
    edges: Dict[str, List[str]] = field(default_factory=dict)

    def add_atoms(self, new_atoms: Sequence[Atom]) -> None:
        for a in new_atoms:
            self.atoms[a.atom_id] = a

    def build_levels(self, max_level: int = 3, max_groups: int = 8) -> None:
        """Iterate coarsening to produce L1 … L_max."""
        if not self.atoms:
            return
        current: Sequence = list(self.atoms.values())
        for level in range(1, max_level + 1):
            reps = coarsen(current, level=level, max_groups=max_groups)
            self.levels[level] = reps
            for r in reps:
                self.edges[r.rep_id] = list(r.child_ids)
            # next iteration coarsens the representatives themselves
            # (treat them as pseudo-atoms for the next π/ρ)
            current = [
                Atom(content=r.content, atom_id=r.rep_id, metadata={"level": level})
                for r in reps
            ]

    def children_of(self, node_id: str) -> List[str]:
        return self.edges.get(node_id, [])

    def get_atom(self, atom_id: str) -> Optional[Atom]:
        return self.atoms.get(atom_id)

    def all_representatives(self) -> List[Representative]:
        out: List[Representative] = []
        for lvl in sorted(self.levels.keys()):
            out.extend(self.levels[lvl])
        return out
