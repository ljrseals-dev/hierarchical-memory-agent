"""Smoke tests for the (α, C, τ) pipeline."""

from senses_memory import extract, MemoryAgent
from senses_memory.traversal import TraversalMode


def test_extraction_produces_atoms():
    atoms = extract("The system must remain offline-first. Cost tracking is mandatory.")
    assert len(atoms) >= 1
    assert all(a.content for a in atoms)


def test_agent_end_to_end():
    agent = MemoryAgent(db_path=":memory:", max_levels=2, default_budget=80)
    agent.on_memory_write("Prism uses Explore, Build, and Ship verification modes.")
    agent.on_memory_write("All memory operations are compositions of alpha, C, and tau.")
    result = agent.memory_search("verification", mode=TraversalMode.TOP_DOWN)
    assert isinstance(result.atoms, list)
    agent.close()
