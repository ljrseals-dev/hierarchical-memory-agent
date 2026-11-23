"""Tests for the (α, C, τ) pipeline — Boy Scouts hardened."""

from senses_memory import extract, MemoryAgent
from senses_memory.traversal import TraversalMode
from senses_memory.coarsening import coarsen
from senses_memory.hierarchy import Hierarchy
from senses_memory.atoms import Atom


def test_extraction_produces_atoms():
    atoms = extract("The system must remain offline-first. Cost tracking is mandatory.")
    assert len(atoms) >= 1
    assert all(a.content for a in atoms)
    assert all(a.atom_id for a in atoms)


def test_extraction_preserves_dense_propositions():
    text = "Ship mode requires evidence. Explore mode is fast."
    atoms = extract(text)
    assert len(atoms) >= 2


def test_coarsen_produces_representatives():
    atoms = extract(
        "Verification modes are Explore Build and Ship. "
        "Cost tracking is first-class. Offline reliability is required."
    )
    reps = coarsen(atoms, level=1, max_groups=4)
    assert len(reps) >= 1
    assert all(r.content for r in reps)
    assert all(r.child_ids for r in reps)


def test_hierarchy_progressive_compression():
    atoms = extract(
        "Alpha extracts atoms. Coarsening builds hierarchy. Traversal selects under budget. "
        "Pinned atoms are preferred. Forgotten atoms are excluded. Offline first is the rule."
    )
    h = Hierarchy()
    h.add_atoms(atoms)
    h.build_levels(max_level=3, max_groups=6)
    stats = h.stats()
    assert stats["n_atoms"] == len(atoms)
    # Higher levels should not explode in size
    if 1 in h.levels and 2 in h.levels:
        assert len(h.levels[2]) <= len(h.levels[1]) + 1


def test_agent_end_to_end():
    agent = MemoryAgent(db_path=":memory:", max_levels=2, default_budget=80)
    agent.on_memory_write("Prism uses Explore, Build, and Ship verification modes.")
    agent.on_memory_write("All memory operations are compositions of alpha, C, and tau.")
    result = agent.memory_search("verification", mode=TraversalMode.TOP_DOWN)
    assert isinstance(result.atoms, list)
    agent.close()


def test_forgotten_atoms_excluded_from_traversal():
    agent = MemoryAgent(db_path=":memory:", max_levels=2, default_budget=200)
    atoms = agent.on_memory_write(
        "Secret credential is never stored. Public preference is dark mode."
    )
    # forget the first atom
    target = atoms[0]
    assert agent.memory_forget(target.atom_id)
    result = agent.memory_search("Secret credential dark mode", budget=200)
    contents = [a.content for a in result.atoms]
    assert target.content not in contents
    agent.close()


def test_pinned_atoms_preferred():
    agent = MemoryAgent(db_path=":memory:", max_levels=2, default_budget=30)
    a1 = agent.on_memory_write("Low priority note about the weather today.")
    a2 = agent.on_memory_write("Critical: verification modes are Explore Build Ship.")
    agent.memory_pin(a2[0].atom_id)
    result = agent.memory_search("verification weather modes", budget=15)
    # With tight budget, pinned critical atom should appear
    ids = {a.atom_id for a in result.atoms}
    assert a2[0].atom_id in ids
    agent.close()


def test_memory_reason_returns_text():
    agent = MemoryAgent(db_path=":memory:", max_levels=2, default_budget=80)
    agent.on_memory_write("Ship requires evidence before marking a task done.")
    text = agent.memory_reason("evidence")
    assert "Relevant memory" in text or "Ship" in text
    agent.close()


def test_self_improvement_log():
    agent = MemoryAgent(db_path=":memory:", max_levels=2, default_budget=50)
    agent.on_memory_write("Something about cats and dogs.")
    agent.memory_search("completely unrelated quantum chromodynamics")
    failures = agent.review_traversal_failures(min_yield=0.5)
    assert isinstance(failures, list)
    agent.close()


def test_stats():
    agent = MemoryAgent(db_path=":memory:", max_levels=2, default_budget=50)
    agent.on_memory_write("One fact. Two fact.")
    s = agent.stats()
    assert s["n_atoms"] >= 1
    agent.close()



def test_partition_never_exceeds_max_groups():
    atoms = [
        Atom(content=f"Topic{i} contains a sufficiently long proposition.", atom_id=f"a{i}")
        for i in range(12)
    ]
    reps = coarsen(atoms, level=1, max_groups=3)
    assert len(reps) <= 3
    assert {child for rep in reps for child in rep.child_ids} == {a.atom_id for a in atoms}


def test_memory_expand_recursively_reaches_original_atoms():
    agent = MemoryAgent(db_path=":memory:", max_levels=3, default_budget=200)
    atoms = agent.on_memory_write(
        [
            "Alpha extraction stores atomic propositions with complete provenance.",
            "Beta traversal selects relevant memories under a strict token budget.",
            "Gamma coarsening groups propositions into durable semantic summaries.",
            "Delta persistence retains original atoms across application restarts.",
        ]
    )
    highest_level = max(agent.hierarchy.levels)
    roots = agent.hierarchy.levels[highest_level]
    expanded = [
        atom
        for root in roots
        for atom in agent.memory_expand(root.rep_id)
    ]
    assert {a.atom_id for a in expanded} == {a.atom_id for a in atoms}
    agent.close()


def test_explicit_zero_budget_is_not_replaced_by_default():
    agent = MemoryAgent(db_path=":memory:", max_levels=2, default_budget=100)
    agent.on_memory_write("A sufficiently long memory proposition for retrieval.")
    result = agent.memory_search("memory", budget=0)
    assert result.atoms == []
    assert result.budget_used == 0
    agent.close()


def test_extraction_rejects_invalid_confidence_and_items():
    import pytest

    with pytest.raises(ValueError):
        extract("A sufficiently long proposition.", confidence=1.5)
    with pytest.raises(TypeError):
        extract(["A sufficiently long proposition.", 42])

def test_collapsed_traversal_paths_remain_aligned_after_pinned_sort():
    agent = MemoryAgent(db_path=":memory:", max_levels=3, default_budget=500)
    atoms = agent.on_memory_write([
        "Alpha topic covers hierarchy traversal and evidence preservation.",
        "Bravo topic covers SQLite persistence and atomic writes.",
        "Charlie topic covers pinned memories and retrieval budgets.",
        "Delta topic covers FTS indexing and keyword search.",
        "Echo topic covers traversal paths and representative expansion.",
        "Foxtrot topic covers forgotten records and soft deletion.",
        "Golf topic covers offline-first configuration and operation.",
        "Hotel topic covers extraction confidence and provenance.",
    ])
    agent.memory_pin(atoms[-1].atom_id)
    result = agent.memory_search("topic traversal persistence", budget=500, mode=TraversalMode.COLLAPSED)
    assert len(result.atoms) == len(result.paths)
    assert all(path[-1] == atom.atom_id for atom, path in zip(result.atoms, result.paths))
    agent.close()


def test_atom_store_rebuilds_search_index_on_reopen():
    import tempfile
    from senses_memory.store import AtomStore

    with tempfile.NamedTemporaryFile(suffix=".sqlite") as tmp:
        store = AtomStore(tmp.name)
        store.add(extract("Durable index rebuild should preserve searchability after restart."))
        store.close()
        reopened = AtomStore(tmp.name)
        assert reopened.search("searchability")
        reopened.close()


def test_atom_batch_rolls_back_on_serialization_failure():
    import pytest
    from senses_memory.store import AtomStore

    store = AtomStore(":memory:")
    valid = Atom(content="First valid item", atom_id="valid")
    invalid = Atom(content="Second item", atom_id="invalid", metadata={"bad": object()})
    with pytest.raises(TypeError):
        store.add([valid, invalid])
    assert store.all_atoms() == []
    store.close()
