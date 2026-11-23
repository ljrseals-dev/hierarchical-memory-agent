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
    if not q:
        base = 0.0
    else:
        base = len(q & t) / len(q)
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
    to avoid flat retrieval over all atoms.
    Forgotten atoms are excluded. Pinned atoms are preferentially retained.
    """
    if mode == TraversalMode.TOP_DOWN:
        return _top_down(hierarchy, query, budget)
    return _collapsed(hierarchy, query, budget)


def _eligible_atoms(h: Hierarchy) -> dict[str, Atom]:
    return {aid: a for aid, a in h.atoms.items() if not _is_forgotten(a)}


def _top_down(h: Hierarchy, query: str, budget: int) -> TraversalResult:
    """Start at coarsest level, route via representatives, expand promising branches."""
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
        score = _relevance(query, rep.content)
        candidates.append((score, rep.rep_id, [rep.rep_id]))

    candidates.sort(key=lambda x: x[0], reverse=True)

    while candidates and used < budget:
        score, node_id, path = candidates.pop(0)
        if node_id in visited:
            continue
        visited.add(node_id)

        children = h.children_of(node_id)
        if not children:
            continue

        atom_children = [cid for cid in children if cid in eligible]
        if atom_children:
            ranked = sorted(
                atom_children,
                key=lambda aid: _relevance(
                    query, eligible[aid].content, pinned=_is_pinned(eligible[aid])
                ),
                reverse=True,
            )
            for aid in ranked:
                if used >= budget:
                    break
                atom = eligible[aid]
                cost = _token_estimate(atom.content)
                if used + cost > budget:
                    continue
                selected.append(atom)
                paths.append(path + [aid])
                used += cost
        else:
            for cid in children:
                text = cid
                for reps in h.levels.values():
                    for r in reps:
                        if r.rep_id == cid:
                            text = r.content
                            break
                s = _relevance(query, text)
                candidates.append((s, cid, path + [cid]))
            candidates.sort(key=lambda x: x[0], reverse=True)

    return TraversalResult(
        atoms=selected, paths=paths, budget_used=used, mode=TraversalMode.TOP_DOWN
    )


def _collapsed(h: Hierarchy, query: str, budget: int) -> TraversalResult:
    """Search representatives across levels in parallel, then expand selectively."""
    eligible = _eligible_atoms(h)
    scored: list[tuple[float, str, object]] = []
    for level, reps in h.levels.items():
        for r in reps:
            scored.append((_relevance(query, r.content), r.rep_id, r))
    scored.sort(key=lambda x: x[0], reverse=True)

    selected: List[Atom] = []
    paths: List[List[str]] = []
    used = 0
    seen: Set[str] = set()

    for score, _rid, rep in scored:
        if used >= budget:
            break
        stack = list(rep.child_ids)
        path_prefix = [rep.rep_id]
        while stack and used < budget:
            nid = stack.pop()
            if nid in eligible and nid not in seen:
                atom = eligible[nid]
                cost = _token_estimate(atom.content)
                if used + cost > budget:
                    continue
                selected.append(atom)
                paths.append(path_prefix + [nid])
                used += cost
                seen.add(nid)
            else:
                stack.extend(h.children_of(nid))

    # Prefer pinned atoms if we still have budget headroom (re-rank final list)
    selected.sort(
        key=lambda a: (
            0 if _is_pinned(a) else 1,
            -_relevance(query, a.content, pinned=_is_pinned(a)),
        )
    )
    return TraversalResult(
        atoms=selected, paths=paths, budget_used=used, mode=TraversalMode.COLLAPSED
    )


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
