"""
Central algorithm registry.

Usage
-----
Registering a new algorithm (in its own module):

    from .registry import AlgorithmRegistry
    from .base import RoleAlgorithm

    @AlgorithmRegistry.register
    class SpectralAlgorithm(RoleAlgorithm):
        name = "spectral"
        ...

Retrieving and running an algorithm (in the pipeline):

    algo = AlgorithmRegistry.get("spectral")
    assignments = algo.fit(matrix, user_index, cfg.spectral)
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import RoleAlgorithm


class AlgorithmRegistry:
    _registry: dict[str, type[RoleAlgorithm]] = {}

    @classmethod
    def register(cls, algo_cls: type) -> type:
        """Decorator — registers algo_cls under algo_cls.name."""
        if not algo_cls.name:
            raise ValueError(f"{algo_cls.__name__} must define a non-empty `name` class attribute")
        cls._registry[algo_cls.name] = algo_cls
        return algo_cls

    @classmethod
    def get(cls, name: str) -> RoleAlgorithm:
        """Instantiate and return a registered algorithm by name."""
        if name not in cls._registry:
            raise KeyError(
                f"Algorithm '{name}' is not registered. "
                f"Available: {sorted(cls._registry)}"
            )
        return cls._registry[name]()

    @classmethod
    def available(cls) -> dict[str, type[RoleAlgorithm]]:
        """Return {name: class} for all registered algorithms."""
        return dict(cls._registry)

    @classmethod
    def names(cls) -> list[str]:
        return list(cls._registry)
