"""Traversal operator τ: (H, q, B) → selected atoms."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Set

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


def _relevance(query: str, text: str) -> float:
    """Cheap offline relevance score (token overlap). Replace with embeddings later."""
    q = set(query.lower().split())
    t = set(text.lower().split())
    if not q:
        return 0.0
    return len(q & t) / len(q)


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
    """
    if mode == TraversalMode.TOP_DOWN:
        return _top_down(hierarchy, query, budget)
    return _collapsed(hierarchy, query, budget)


def _top_down(h: Hierarchy, query: str, budget: int) -> TraversalResult:
    """Start at coarsest level, route via representatives, expand promising branches."""
    selected: List[Atom] = []
    paths: List[List[str]] = []
    used = 0
    visited: Set[str] = set()

    levels = sorted(h.levels.keys(), reverse=True)
    if not levels:
        # no hierarchy yet — fall back to flat
        return _flat(h, query, budget)

    # beam of (score, node_id, path)
    candidates = []
    top = levels[0]
    for rep in h.levels.get(top, []):
        score = _relevance(query, rep.content)
        candidates.append((score, rep.rep_id, [rep.rep_id]))

    candidates.sort(reverse=True)

    while candidates and used < budget:
        score, node_id, path = candidates.pop(0)
        if node_id in visited:
            continue
        visited.add(node_id)

        children = h.children_of(node_id)
        if not children:
            continue

        # if children are atoms, select the best ones under remaining budget
        atom_children = [cid for cid in children if cid in h.atoms]
        if atom_children:
            ranked = sorted(
                atom_children,
                key=lambda aid: _relevance(query, h.atoms[aid].content),
                reverse=True,
            )
            for aid in ranked:
                if used >= budget:
                    break
                atom = h.atoms[aid]
                cost = _token_estimate(atom.content)
                if used + cost > budget:
                    continue
                selected.append(atom)
                paths.append(path + [aid])
                used += cost
        else:
            # children are lower-level representatives — expand
            for cid in children:
                # find the rep object to score
                text = cid
                for reps in h.levels.values():
                    for r in reps:
                        if r.rep_id == cid:
                            text = r.content
                            break
                s = _relevance(query, text)
                candidates.append((s, cid, path + [cid]))
            candidates.sort(reverse=True)

    return TraversalResult(atoms=selected, paths=paths, budget_used=used, mode=TraversalMode.TOP_DOWN)


def _collapsed(h: Hierarchy, query: str, budget: int) -> TraversalResult:
    """Search representatives across levels in parallel, then expand selectively."""
    scored = []
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
        # expand this representative's atom descendants
        stack = list(rep.child_ids)
        path_prefix = [rep.rep_id]
        while stack and used < budget:
            nid = stack.pop()
            if nid in h.atoms and nid not in seen:
                atom = h.atoms[nid]
                cost = _token_estimate(atom.content)
                if used + cost > budget:
                    continue
                selected.append(atom)
                paths.append(path_prefix + [nid])
                used += cost
                seen.add(nid)
            else:
                stack.extend(h.children_of(nid))

    return TraversalResult(atoms=selected, paths=paths, budget_used=used, mode=TraversalMode.COLLAPSED)


def _flat(h: Hierarchy, query: str, budget: int) -> TraversalResult:
    ranked = sorted(
        h.atoms.values(),
        key=lambda a: _relevance(query, a.content),
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
