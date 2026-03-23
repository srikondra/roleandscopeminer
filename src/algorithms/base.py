"""
Abstract base class for role mining algorithms.

To add a new algorithm:
  1. Subclass RoleAlgorithm
  2. Set class-level `name` and `description`
  3. Implement `fit()` and `config_schema`
  4. Decorate with @AlgorithmRegistry.register
  5. Import the module in algorithms/__init__.py

That's it — the UI and pipeline pick it up automatically.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import NamedTuple

import pandas as pd
from scipy.sparse import csr_matrix


class AlgorithmFitResult(NamedTuple):
    """
    Return type for RoleAlgorithm.fit().

    primary     : ritsid → role_id, exactly one row per user (dominant/best role).
    memberships : ritsid → role_id, one row per (user, role) pair — a user may
                  appear in multiple rows when they have significant membership in
                  more than one role (e.g. NMF soft-clustering above threshold).
                  For hard-partition algorithms (Louvain, Leiden) this equals primary.
    """
    primary:     pd.DataFrame   # columns: ritsid, role_id  (1:1)
    memberships: pd.DataFrame   # columns: ritsid, role_id  (1:N)


class RoleAlgorithm(ABC):
    """
    Contract every role mining algorithm must satisfy.

    The pipeline calls fit() on the *residual* matrix (Tier 1 and Tier 2
    columns already stripped) and expects an AlgorithmFitResult back.
    """

    #: Unique short identifier — used in output filenames and UI toggles.
    name: str = ""

    #: Human-readable description shown in the UI.
    description: str = ""

    @abstractmethod
    def fit(
        self,
        matrix: csr_matrix,
        user_index: list[str],
        algo_cfg,           # typed sub-config from PipelineConfig (e.g. LouvainConfig)
    ) -> AlgorithmFitResult:
        """
        Cluster users into role candidates.

        Parameters
        ----------
        matrix      : n_users × n_grants sparse binary matrix (residual only)
        user_index  : ritsid list matching matrix row order
        algo_cfg    : algorithm-specific config dataclass from PipelineConfig

        Returns
        -------
        AlgorithmFitResult
            primary     : one row per user — their single best/dominant role
            memberships : one row per (user, role) pair — every role the user
                          has significant membership in; equals primary for
                          hard-partition algorithms
        """
        ...

    @property
    @abstractmethod
    def config_schema(self) -> list[dict]:
        """
        Describes the algorithm's tuning knobs so the UI can render them
        without knowing anything about the algorithm internals.

        Each entry is a dict with keys:
            key     : str   — matches the field name in the algo's Config model
            label   : str   — human-readable widget label
            type    : str   — "slider" | "number" | "checkbox"
            default : any   — default value
            min     : float — (slider only)
            max     : float — (slider only)
            step    : float — (slider only)
            help    : str   — tooltip text (optional)
        """
        ...
