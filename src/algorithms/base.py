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

import pandas as pd
from scipy.sparse import csr_matrix


class RoleAlgorithm(ABC):
    """
    Contract every role mining algorithm must satisfy.

    The pipeline calls fit() on the *residual* matrix (Tier 1 and Tier 2
    columns already stripped) and expects a DataFrame with columns [ritsid, role_id] back.
    A user may appear in multiple rows if they belong to more than one role.
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
    ) -> pd.DataFrame:
        """
        Cluster users into role candidates.

        Parameters
        ----------
        matrix      : n_users × n_grants sparse binary matrix (residual only)
        user_index  : ritsid list matching matrix row order
        algo_cfg    : algorithm-specific config dataclass from PipelineConfig

        Returns
        -------
        pd.DataFrame : columns=[ritsid, role_id] — one row per (employee, role) membership.
                       A single employee may appear in multiple rows if they belong to multiple roles.
                       Every ritsid in user_index must appear in at least one row.
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
