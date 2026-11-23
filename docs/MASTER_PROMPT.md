Here is a master prompt for building a Hermes-style hierarchical memory agent for ljrseals-dev/5-Senses, grounded in the formal three-operator framework from Toward a Theory of Hierarchical Memory for Language Agents.

---

Master Prompt: ljrseals-dev/5-Senses Hierarchical Memory Agent

You are building a Hermes-style memory agent for the ljrseals-dev/5-Senses project. The agent must implement a three-layer hierarchical memory consistent with Hermes Agent conventions—typically short-term / session, medium-term / persistent, and long-term / skills or knowledge memory—and must formalise every memory operation using the paper’s (α, C, τ) pipeline.

1. Memory Architecture (Hermes-Style Layers)

Layer Hermes Role Timescale Paper Mapping
L1 – Working / Session Current conversation context, active task state, recent observations Within-session Atoms (V₀) produced by extraction α; traversal τ selects atoms for the context window.
L2 – Persistent / Episodic Cross-session facts, user preferences, project decisions; SQLite + FTS5 searchable Across sessions Coarsened level (G₁): atoms are partitioned by π and assigned representative summaries by ρ; traversal can enter here for fast recall.
L3 – Long-term / Semantic Consolidated knowledge, skills, abstractions, reusable procedures Durable Higher coarsened levels (G₂ … G_L): iterated coarsening produces thematic/semantic hierarchies; traversal can route top-down from here.

The agent must capture at full granularity, coarsen iteratively, and reconstruct on demand via traversal.

2. Formal Operators

Implement every memory pipeline as a composition of three operators.

2.1 Extraction (α)

· Input: raw experience—conversation turns, tool outputs, observations, documents.
· Output: a graph G₀ = (U₀, E₀) of atomic information units.
· Each atom must be self-contained: a short natural-language proposition, event, fact, or sensory observation (the “5-Senses” modality tag may be stored as metadata, but the atom remains textual for LLM reasoning).
· Atoms should be dense and retrieval-ready; avoid splitting into fragments that lose meaning without context.
· Store provenance: source_session, timestamp, modality, confidence.

2.2 Coarsening (C = (π, ρ))

· Partition function π: U → [m] groups atoms into m < |U| clusters.
· Representative function ρ: 2^U → F assigns one representative to each group.
· The representative must be self-sufficient enough for the intended traversal strategy: if traversal will use top-down routing, ρ should produce informative summaries; if collapsed search will be used, ρ may be more referential.
· Hierarchy construction: iterate coarsening to build H = (G₀, C₁, …, C_L). Each level ℓ may use a different grouping criterion:
  · Level 1: temporal or episodic grouping.
  · Level 2: semantic or entity-based grouping.
  · Level 3: thematic or skill-based grouping.
· Store parent–child edges so the hierarchy is a navigable DAG or tree.

2.3 Traversal (τ)

· Signature: τ: (H, q, B) → S ⊆ V₀, where q is the query and B is the token budget.
· Implement at least two traversal modes:
  1. Top-down refinement: start at a coarse level, use representatives to route to relevant branches, expand only promising nodes until the budget is filled.
  2. Collapsed search: search representatives across multiple levels in parallel and include the most relevant units, expanding selectively.
· Traversal must respect the coarsening–traversal coupling: do not spend budget on referential labels when a self-sufficient representative could answer the query directly; do not expand nodes whose summaries already contain the needed evidence.
· Return the selected atoms plus their provenance and hierarchy path so the agent can cite or drill down later.

3. Agent Behaviour Rules

1. Never discard raw atoms. Coarsening is lossy, but the original V₀ graph must remain retrievable for audit and re-expansion.
2. On every new experience: run α to create atoms, then trigger incremental coarsening if the atom set in a group exceeds a threshold.
3. Before every response: run τ with the current query and a token budget (e.g., 20–30% of context window) to inject relevant memory. Use the hierarchy to avoid flat retrieval over all atoms.
4. Self-improvement loop (Hermes trait): periodically review traversal failures. If a query required expanding many nodes or missed relevant atoms, adjust π/ρ for that region of the hierarchy (re-cluster or re-summarise). Record the change as a new coarsening version.
5. Expose memory tools in the Hermes style: memory_search, memory_expand, memory_pin, memory_forget, memory_reason (for compositional queries over the hierarchy). These tools should internally use τ and navigate H rather than performing flat vector search.

4. Data Structures & Storage

· Atom store: append-only table/collection with fields atom_id, content, embedding, provenance, modality, created_at.
· Hierarchy store: nodes for representatives, edges parent → child, plus level metadata and the operator configuration used (π strategy, ρ strategy, version).
· Traversal cache: optional cache of recent (query, budget) → atom_ids to reduce repeated traversal cost.
· Persistence: SQLite + FTS5 for atoms and representatives (Hermes-compatible), with optional vector index for similarity-based π and ρ.

5. Integration with Hermes Agent Runtime

· Plugin/provider shape: package the memory system as a Hermes memory provider or plugin, analogous to excel_line or hyatlas-memory. Register hooks:
  · on_session_start: load relevant representatives into the session primer.
  · on_memory_write: extract atoms and trigger incremental coarsening.
  · on_session_end: crystallise session atoms into the persistent hierarchy.
· Config: allow the operator choices (π clustering method, ρ summarisation model, traversal strategy) to be specified in YAML/JSON so the system can be compared across design variants.

6. Success Criteria

The agent is complete when:

· Every memory interaction can be described as an (α, C, τ) composition.
· The hierarchy contains at least three levels with distinct coarsening criteria.
· Traversal under a fixed token budget retrieves atoms that a flat search would miss, and avoids spending budget on uninformative representatives.
· The agent demonstrates the Hermes self-improvement loop by revising its coarsening strategy after observed retrieval failures.