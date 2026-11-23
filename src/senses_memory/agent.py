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
        self._traversal_log: list[dict] = []  # for self-improvement loop
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
        self.hierarchy.build_levels(max_level=self.max_levels)
        return atoms

    def on_session_start(self, query_hint: str = "") -> TraversalResult:
        """Load relevant representatives into the session primer."""
        return self.memory_search(query_hint or "session start", budget=self.default_budget)

    def on_session_end(self) -> None:
        """Crystallise session atoms into the persistent hierarchy."""
        self.hierarchy.build_levels(max_level=self.max_levels)

    # ----- Hermes-style tools -----

    def memory_search(
        self,
        query: str,
        budget: Optional[int] = None,
        mode: TraversalMode = TraversalMode.TOP_DOWN,
    ) -> TraversalResult:
        """τ — retrieve atoms under a token budget."""
        result = traverse(
            self.hierarchy,
            query,
            budget=self.default_budget if budget is None else budget,
            mode=mode,
        )
        # Self-improvement signal: log low-yield traversals
        n_atoms = len(self.hierarchy.atoms)
        yield_ratio = len(result.atoms) / max(1, n_atoms)
        self._traversal_log.append(
            {
                "query": query,
                "mode": mode.value,
                "selected": len(result.atoms),
                "budget_used": result.budget_used,
                "yield_ratio": yield_ratio,
            }
        )
        return result

    def memory_expand(self, rep_or_atom_id: str) -> List[Atom]:
        """Expand a representative or return a single atom."""
        if rep_or_atom_id in self.hierarchy.atoms:
            atom = self.hierarchy.atoms[rep_or_atom_id]
            if atom.metadata.get("forgotten"):
                return []
            return [atom]
        # Representatives can point to other representatives at higher levels.
        # Expand recursively so callers always receive the original V₀ atoms.
        found: List[Atom] = []
        pending = [rep_or_atom_id]
        visited: set[str] = set()
        while pending:
            node_id = pending.pop()
            if node_id in visited:
                continue
            visited.add(node_id)
            atom = self.hierarchy.atoms.get(node_id)
            if atom is not None:
                if not atom.metadata.get("forgotten"):
                    found.append(atom)
                continue
            pending.extend(reversed(self.hierarchy.children_of(node_id)))
        return found

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
        # Rebuild so forgotten atoms leave the hierarchy routing structure
        self.hierarchy.build_levels(max_level=self.max_levels)
        return True

    def memory_reason(self, query: str, budget: Optional[int] = None) -> str:
        """Compositional query: traverse then concatenate evidence."""
        result = self.memory_search(query, budget=budget)
        if not result.atoms:
            return "No relevant memory found."
        lines = [f"- {a.content}" for a in result.atoms]
        return "Relevant memory:\n" + "\n".join(lines)

    def review_traversal_failures(self, min_yield: float = 0.05) -> list[dict]:
        """Self-improvement loop (Hermes trait): surface low-yield queries.

        Callers can use this to decide whether to re-cluster / re-summarise
        a region of the hierarchy.
        """
        return [e for e in self._traversal_log if e["yield_ratio"] < min_yield]

    def stats(self) -> dict:
        s = self.hierarchy.stats()
        s["traversal_log_size"] = len(self._traversal_log)
        return s

    def close(self) -> None:
        self.store.close()
