"""Runnable demonstration of the (α, C, τ) pipeline."""

from __future__ import annotations

from .agent import MemoryAgent
from .traversal import TraversalMode


def main() -> None:
    print("=== 5-Senses Hierarchical Memory Demo ===\n")
    agent = MemoryAgent(db_path=":memory:", max_levels=3, default_budget=120)

    # Simulate experiences (α)
    experiences = [
        "The user prefers dark mode and keyboard-driven workflows. They work primarily on macOS.",
        "Project Prism is a multi-model coding agent. It routes tasks by category: planning, engineering, visual, review, quick.",
        "Verification modes are Explore, Build, and Ship. Ship requires evidence before marking a task done.",
        "Cost tracking is first-class. Every task records token and dollar cost against a workspace budget.",
        "The hierarchical memory paper defines three operators: extraction alpha, coarsening C equals pi and rho, and traversal tau.",
        "Offline reliability is a hard requirement. SQLite plus FTS5 is the default persistence layer.",
    ]

    print("1. Writing experiences (α → atoms → coarsening)…")
    for exp in experiences:
        atoms = agent.on_memory_write(exp, source_session="demo-session")
        print(f"   + {len(atoms)} atom(s) from: {exp[:60]}…")

    print(f"\n   Hierarchy levels: {sorted(agent.hierarchy.levels.keys())}")
    for lvl, reps in agent.hierarchy.levels.items():
        print(f"   L{lvl}: {len(reps)} representatives")

    print("\n2. Traversal (τ) — top-down for query: 'verification and cost'")
    result = agent.memory_search("verification and cost", mode=TraversalMode.TOP_DOWN)
    print(f"   Selected {len(result.atoms)} atoms, budget used: {result.budget_used}")
    for a in result.atoms:
        print(f"   • {a.content[:90]}{'…' if len(a.content) > 90 else ''}")

    print("\n3. memory_reason compositional query")
    print(agent.memory_reason("What are the verification modes and how does cost tracking work?"))

    print("\n4. Collapsed search mode")
    result2 = agent.memory_search("offline SQLite", mode=TraversalMode.COLLAPSED)
    print(f"   Selected {len(result2.atoms)} atoms")
    for a in result2.atoms:
        print(f"   • {a.content[:90]}{'…' if len(a.content) > 90 else ''}")

    agent.close()
    print("\n=== Demo complete ===")


if __name__ == "__main__":
    main()
