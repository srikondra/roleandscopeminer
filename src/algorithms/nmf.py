"""
Non-negative Matrix Factorization (NMF) role mining algorithm.

Decomposes the user-entitlement matrix into latent role components.
Each user is assigned to every component whose normalised W-row fraction meets membership_threshold. The dominant component is always included.

Best for  : discovering overlapping / soft-boundary roles; scales well to large datasets.
Trade-off : Latent components are less directly interpretable than graph clusters.
            Sensitive to the choice of n_roles; auto-detection via SVD elbow helps.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import NMF as _NMF

from .base import RoleAlgorithm
from .registry import AlgorithmRegistry

log = logging.getLogger("role_miner.nmf")


def _auto_n_roles(matrix: csr_matrix, min_r: int, max_r: int) -> int:
    """Estimate number of roles via SVD explained-variance elbow."""
    try:
        from scipy.sparse.linalg import svds
        k       = min(max_r + 5, min(matrix.shape) - 1)
        _, s, _ = svds(matrix.astype(float), k=k)
        s       = np.sort(s)[::-1]
        diffs   = np.diff(s)
        elbow   = int(np.argmin(diffs[:max_r])) + 1
        return max(min_r, min(elbow, max_r))
    except Exception:
        return max(min_r, min(10, max_r))


@AlgorithmRegistry.register
class NMFAlgorithm(RoleAlgorithm):
    name        = "nmf"
    description = "Non-negative Matrix Factorization (NMF) — discovers latent role components"

    def fit(self, matrix: csr_matrix, user_index: list[str], algo_cfg) -> pd.DataFrame:
        n_roles = algo_cfg.n_roles or _auto_n_roles(
            matrix, algo_cfg.min_roles, algo_cfg.max_roles
        )
        log.info("Running NMF with %d roles (seed=%d) …", n_roles, algo_cfg.seed)

        model = _NMF(
            n_components=n_roles,
            max_iter=algo_cfg.max_iter,
            random_state=algo_cfg.seed,
            init="nndsvda",
        )
        W         = model.fit_transform(matrix.astype(float))   # users × roles
        threshold = getattr(algo_cfg, "membership_threshold", 0.10)

        rows = []
        for i, ritsid in enumerate(user_index):
            row_sum  = W[i].sum()
            w_norm   = W[i] / max(row_sum, 1e-10)
            dominant = int(np.argmax(W[i]))
            assigned = False
            for j, frac in enumerate(w_norm):
                if frac >= threshold:
                    rows.append({"ritsid": ritsid, "role_id": f"NMFRole{j + 1}"})
                    assigned = True
            if not assigned:                          # always assign at least the dominant role
                rows.append({"ritsid": ritsid, "role_id": f"NMFRole{dominant + 1}"})

        assignments = pd.DataFrame(rows)
        n_roles_out = assignments["role_id"].nunique()
        n_multi     = (assignments.groupby("ritsid").size() > 1).sum()
        log.info("NMF: %d roles discovered  |  %d users with multiple role memberships",
                 n_roles_out, n_multi)
        return assignments

    @property
    def config_schema(self) -> list[dict]:
        return [
            {
                "key":     "n_roles",
                "label":   "Number of Roles (0 = auto-detect)",
                "type":    "number",
                "default": 0,
                "help":    "Set to 0 for automatic detection via SVD elbow method.",
            },
            {
                "key":     "min_roles",
                "label":   "Min Roles (auto mode)",
                "type":    "number",
                "default": 3,
                "help":    "Lower bound when auto-detecting the number of roles.",
            },
            {
                "key":     "max_roles",
                "label":   "Max Roles (auto mode)",
                "type":    "number",
                "default": 30,
                "help":    "Upper bound when auto-detecting the number of roles.",
            },
            {
                "key":     "max_iter",
                "label":   "Max Iterations",
                "type":    "number",
                "default": 500,
                "help":    "Maximum NMF optimisation iterations.",
            },
            {
                "key":     "seed",
                "label":   "Random Seed",
                "type":    "number",
                "default": 42,
                "help":    "Set for reproducible decomposition.",
            },
            {
                "key":     "membership_threshold",
                "label":   "Membership Threshold",
                "type":    "slider",
                "default": 0.10, "min": 0.0, "max": 1.0, "step": 0.01,
                "help":    "Minimum normalised W-row fraction for a role to be assigned. "
                           "Lower values allow broader multi-role membership.",
            },
        ]
