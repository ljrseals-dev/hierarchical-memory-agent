"""5-Senses Hierarchical Memory — formal (α, C, τ) implementation."""

from .atoms import Atom
from .extraction import extract
from .coarsening import coarsen
from .traversal import traverse, TraversalMode
from .hierarchy import Hierarchy
from .agent import MemoryAgent

__all__ = [
    "Atom",
    "extract",
    "coarsen",
    "traverse",
    "TraversalMode",
    "Hierarchy",
    "MemoryAgent",
]

__version__ = "0.1.0"
