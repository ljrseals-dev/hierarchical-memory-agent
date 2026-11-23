"""Traversal operator τ: (H, q, B) → selected atoms."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Set

from .atoms import Atom
from .hierarchy import Hierarchy


class TraversalMode(str, Enum):
    TOP_DOWN = "top_down"
    COLLAPSED = "collapsed"


@dataclass
class TraversalResult:
    atoms: List[Atom]
    paths: List[List[str]]          # hierarchy path for each selected atom
    budget_used: int
    mode: TraversalMode


def _is_forgotten(atom: Atom) -> bool:
    return bool(atom.metadata.get("forgotten"))


def _is_pinned(atom: Atom) -> bool:
    return bool(atom.metadata.get("pinned"))


def _relevance(query: str, text: str, *, pinned: bool = False) -> float:
    """Cheap offline relevance score (token overlap). Pinned atoms get a boost."""
    q = set(query.lower().split())
    t = set(text.lower().split())
    base = len(q & t) / len(q) if q else 0.0
    if pinned:
        base += 0.35
    return base


def _token_estimate(text: str) -> int:
    return max(1, len(text.split()))


def traverse(
    hierarchy: Hierarchy,
    query: str,
    budget: int = 256,
    mode: TraversalMode = TraversalMode.TOP_DOWN,
) -> TraversalResult:
    """τ — Traversal operator.

    Selects a subset of atoms under a token budget, using the hierarchy
    to avoid flat retrieval over all atoms. Forgotten atoms are excluded.
    """
    if isinstance(budget, bool) or not isinstance(budget, int):
        raise TypeError("budget must be an integer")
    if budget < 0:
        raise ValueError("budget cannot be negative")
    if mode == TraversalMode.TOP_DOWN:
        return _top_down(hierarchy, query, budget)
    if mode == TraversalMode.COLLAPSED:
        return _collapsed(hierarchy, query, budget)
    raise ValueError(f"unsupported traversal mode: {mode!r}")


def _eligible_atoms(h: Hierarchy) -> dict[str, Atom]:
    return {aid: a for aid, a in h.atoms.items() if not _is_forgotten(a)}


def _top_down(h: Hierarchy, query: str, budget: int) -> TraversalResult:
    selected: List[Atom] = []
    paths: List[List[str]] = []
    used = 0
    visited: Set[str] = set()
    eligible = _eligible_atoms(h)

    levels = sorted(h.levels.keys(), reverse=True)
    if not levels:
        return _flat(h, query, budget)

    candidates: list[tuple[float, str, list[str]]] = []
    top = levels[0]
    for rep in h.levels.get(top, []):
        candidates.append((_relevance(query, rep.content), rep.rep_id, [rep.rep_id]))
    candidates.sort(key=lambda x: x[0], reverse=True)

    while candidates and used < budget:
        score, node_id, path = candidates.pop(0)
        if node_id in visited:
            continue
        visited.add(node_id)
        children = h.children_of(node_id)
        atom_children = sorted(
            (cid for cid in children if cid in eligible),
            key=lambda aid: _relevance(
                query, eligible[aid].content, pinned=_is_pinned(eligible[aid])
            ),
            reverse=True,
        )
        for aid in atom_children:
            if used >= budget:
                break
            atom = eligible[aid]
            cost = _token_estimate(atom.content)
            if used + cost > budget:
                continue
            selected.append(atom)
            paths.append(path + [aid])
            used += cost
        for cid in children:
            if cid in eligible:
                continue
            text = cid
            for reps in h.levels.values():
                for r in reps:
                    if r.rep_id == cid:
                        text = r.content
                        break
            candidates.append((_relevance(query, text), cid, path + [cid]))
        candidates.sort(key=lambda x: x[0], reverse=True)

    return TraversalResult(atoms=selected, paths=paths, budget_used=used, mode=TraversalMode.TOP_DOWN)


def _collapsed(h: Hierarchy, query: str, budget: int) -> TraversalResult:
    eligible = _eligible_atoms(h)
    scored: list[tuple[float, str, object]] = []
    for reps in h.levels.values():
        for rep in reps:
            scored.append((_relevance(query, rep.content), rep.rep_id, rep))
    scored.sort(key=lambda x: x[0], reverse=True)

    selected: List[Atom] = []
    paths: List[List[str]] = []
    used = 0
    seen: Set[str] = set()

    for _score, _rid, rep in scored:
        if used >= budget:
            break
        # Carry the complete hierarchy path while walking descendants.
        stack = [(child_id, [rep.rep_id, child_id]) for child_id in reversed(rep.child_ids)]
        while stack and used < budget:
            node_id, path = stack.pop()
            if node_id in eligible:
                if node_id in seen:
                    continue
                atom = eligible[node_id]
                cost = _token_estimate(atom.content)
                if used + cost > budget:
                    continue
                selected.append(atom)
                paths.append(path)
                used += cost
                seen.add(node_id)
                continue
            children = h.children_of(node_id)
            stack.extend((cid, path + [cid]) for cid in reversed(children))

    # Sort atoms and their paths together so the result stays aligned.
    paired = sorted(
        zip(selected, paths),
        key=lambda pair: (
            0 if _is_pinned(pair[0]) else 1,
            -_relevance(query, pair[0].content, pinned=_is_pinned(pair[0])),
        ),
    )
    selected = [atom for atom, _path in paired]
    paths = [path for _atom, path in paired]
    return TraversalResult(atoms=selected, paths=paths, budget_used=used, mode=TraversalMode.COLLAPSED)


def _flat(h: Hierarchy, query: str, budget: int) -> TraversalResult:
    eligible = list(_eligible_atoms(h).values())
    ranked = sorted(
        eligible,
        key=lambda a: _relevance(query, a.content, pinned=_is_pinned(a)),
        reverse=True,
    )
    selected = []
    used = 0
    for a in ranked:
        cost = _token_estimate(a.content)
        if used + cost > budget:
            continue
        selected.append(a)
        used += cost
    return TraversalResult(
        atoms=selected,
        paths=[[a.atom_id] for a in selected],
        budget_used=used,
        mode=TraversalMode.TOP_DOWN,
    )
