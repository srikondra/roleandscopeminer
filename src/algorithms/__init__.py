from .base import RoleAlgorithm
from .registry import AlgorithmRegistry

# Import concrete implementations so they self-register
from . import louvain  # noqa: F401
from . import nmf      # noqa: F401
try:
    from . import leiden   # noqa: F401
except ImportError:
    pass  # leidenalg/igraph not installed — Leiden silently unavailable

__all__ = ["RoleAlgorithm", "AlgorithmRegistry"]
