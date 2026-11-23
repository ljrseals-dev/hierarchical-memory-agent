# 5-Senses Hierarchical Memory

**Offline-first hierarchical memory agent implementing the formal (α, C, τ) framework.**

This repository implements a Hermes-style three-layer hierarchical memory system grounded in the paper *[Toward a Theory of Hierarchical Memory for Language Agents](https://arxiv.org/abs/2603.21564)*.

It is designed for the **ljrseals-dev/5-Senses** project — a classical AI laboratory for reliable, offline AI systems.

> Full master prompt / system definition: [`docs/MASTER_PROMPT.md`](docs/MASTER_PROMPT.md)

---

## The Three Operators

| Operator | Symbol | Role |
|----------|--------|------|
| **Extraction** | α | Raw experience → atomic information units (self-contained propositions) |
| **Coarsening** | C = (π, ρ) | Partition atoms + assign representatives → multi-level hierarchy |
| **Traversal** | τ | Query + budget → selected atoms under token constraints |

Every memory interaction is expressible as a composition of these three operators.

## Memory Layers (Hermes-style)

| Layer | Role | Timescale | Paper mapping |
|-------|------|-----------|---------------|
| **L1 – Working / Session** | Current conversation, active task state | Within-session | Atoms (V₀) |
| **L2 – Persistent / Episodic** | Cross-session facts, preferences, decisions | Across sessions | Coarsened level G₁ |
| **L3 – Long-term / Semantic** | Skills, abstractions, reusable procedures | Durable | Higher levels G₂…G_L |

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e .

# Run a simple demo
python -m senses_memory.demo
```

## Core Design Rules

1. **Never discard raw atoms.** Coarsening is lossy; the original V₀ graph remains retrievable.
2. On every new experience → run α, then incremental coarsening when thresholds are hit.
3. Before every response → run τ with a token budget (typically 20–30 % of context).
4. Self-improvement loop: review traversal failures and revise π/ρ for that region.
5. Tools exposed in Hermes style: `memory_search`, `memory_expand`, `memory_pin`, `memory_forget`, `memory_reason`.

## Project Structure

```
5-senses-memory/
├── src/senses_memory/
│   ├── __init__.py
│   ├── atoms.py          # Atom data model + store
│   ├── extraction.py     # α operator
│   ├── coarsening.py     # C = (π, ρ) operator
│   ├── traversal.py      # τ operator (top-down + collapsed)
│   ├── hierarchy.py      # Hierarchy construction & navigation
│   ├── store.py          # SQLite + FTS5 persistence
│   ├── agent.py          # High-level memory agent
│   └── demo.py           # Runnable demonstration
├── config/
│   └── default.yaml      # Operator configuration
├── docs/
│   └── MASTER_PROMPT.md  # Full system definition
├── tests/
├── pyproject.toml
├── README.md
└── LICENSE
```

## Status

This is a working foundation that implements the formal (α, C, τ) pipeline with:

- Append-only atom store
- Multi-level hierarchy with parent–child edges
- Top-down refinement and collapsed-search traversal modes
- SQLite + FTS5 persistence (offline-first)
- Clear separation of operators so design variants can be compared

## License

MIT
