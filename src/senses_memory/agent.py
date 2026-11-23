"""High-level MemoryAgent implementing Hermes-style hooks and tools."""

from __future__ import annotations

from typing import List, Optional

from .atoms import Atom
from .extraction import extract
from .hierarchy import Hierarchy
from .store import AtomStore
from .traversal import TraversalMode, TraversalResult, traverse


class MemoryAgent:
    """Hermes-style hierarchical memory agent.

    All operations are compositions of (α, C, τ).
    """

    def __init__(
        self,
        db_path: str = "senses_memory.db",
        max_levels: int = 3,
        default_budget: int = 256,
    ):
        self.store = AtomStore(db_path)
        self.hierarchy = Hierarchy()
        self.max_levels = max_levels
        self.default_budget = default_budget
        # load any existing atoms
        existing = self.store.all_atoms()
        if existing:
            self.hierarchy.add_atoms(existing)
            self.hierarchy.build_levels(max_level=max_levels)

    # ----- Hermes-style hooks -----

    def on_memory_write(
        self,
        experience: str | List[str],
        *,
        source_session: Optional[str] = None,
        modality: str = "text",
    ) -> List[Atom]:
        """α then incremental coarsening."""
        atoms = extract(experience, source_session=source_session, modality=modality)
        self.store.add(atoms)
        self.hierarchy.add_atoms(atoms)
        # rebuild hierarchy (simple incremental strategy for v0.1)
        self.hierarchy.build_levels(max_level=self.max_levels)
        return atoms

    def on_session_start(self, query_hint: str = "") -> TraversalResult:
        """Load relevant representatives into the session primer."""
        return self.memory_search(query_hint or "session start", budget=self.default_budget)

    def on_session_end(self) -> None:
        """Crystallise session atoms into the persistent hierarchy."""
        # already persisted on every write; rebuild is enough for v0.1
        self.hierarchy.build_levels(max_level=self.max_levels)

    # ----- Hermes-style tools -----

    def memory_search(
        self,
        query: str,
        budget: Optional[int] = None,
        mode: TraversalMode = TraversalMode.TOP_DOWN,
    ) -> TraversalResult:
        """τ — retrieve atoms under a token budget."""
        return traverse(
            self.hierarchy,
            query,
            budget=budget or self.default_budget,
            mode=mode,
        )

    def memory_expand(self, rep_or_atom_id: str) -> List[Atom]:
        """Expand a representative or return a single atom."""
        if rep_or_atom_id in self.hierarchy.atoms:
            return [self.hierarchy.atoms[rep_or_atom_id]]
        child_ids = self.hierarchy.children_of(rep_or_atom_id)
        return [self.hierarchy.atoms[cid] for cid in child_ids if cid in self.hierarchy.atoms]

    def memory_pin(self, atom_id: str) -> bool:
        """Mark an atom as high-priority (stored in metadata)."""
        atom = self.hierarchy.get_atom(atom_id)
        if not atom:
            return False
        atom.metadata["pinned"] = True
        self.store.add([atom])  # upsert
        return True

    def memory_forget(self, atom_id: str) -> bool:
        """Soft-forget: mark rather than delete (never discard raw atoms)."""
        atom = self.hierarchy.get_atom(atom_id)
        if not atom:
            return False
        atom.metadata["forgotten"] = True
        self.store.add([atom])
        return True

    def memory_reason(self, query: str, budget: Optional[int] = None) -> str:
        """Compositional query: traverse then concatenate evidence."""
        result = self.memory_search(query, budget=budget)
        if not result.atoms:
            return "No relevant memory found."
        lines = [f"- {a.content}" for a in result.atoms]
        return "Relevant memory:\n" + "\n".join(lines)

    def close(self) -> None:
        self.store.close()
