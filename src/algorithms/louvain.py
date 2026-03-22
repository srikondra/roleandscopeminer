"""
Louvain Graph Community Detection algorithm.

Builds a Jaccard-similarity user graph in RAM-safe batches, then partitions
it with the Louvain algorithm.  Each community becomes one business role candidate.

Best for  : tight, interpretable user communities with near-identical entitlement sets.
Trade-off : O(n²) similarity pass — can be slow / memory-heavy for very large populations.
            Reduce LouvainConfig.batch_size or disable on datasets > ~100 K users.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

try:
    import networkx as nx
    import community as community_louvain
except ImportError as exc:
    raise ImportError(
        "Louvain requires networkx and python-louvain. "
        "Run: pip install networkx python-louvain"
    ) from exc

from .base import RoleAlgorithm
from .registry import AlgorithmRegistry

log = logging.getLogger("role_miner.louvain")


@AlgorithmRegistry.register
class LouvainAlgorithm(RoleAlgorithm):
    name        = "louvain"
    description = "Graph Community Detection (Louvain) — finds tightly-knit user communities"

    def fit(self, matrix: csr_matrix, user_index: list[str], algo_cfg) -> pd.DataFrame:
        n          = matrix.shape[0]
        batch_size = algo_cfg.batch_size
        k          = min(15, n - 1)

        # Row-normalise for cosine / Jaccard approximation
        norms = np.asarray(matrix.sum(axis=1)).flatten()
        norms[norms == 0] = 1.0
        mat_norm = matrix.multiply(1.0 / norms[:, None]).tocsr()

        log.info("Building user similarity graph (%d users, batch=%d) …", n, batch_size)
        G = nx.Graph()
        G.add_nodes_from(range(n))

        n_batches = (n + batch_size - 1) // batch_size
        for b in range(n_batches):
            start     = b * batch_size
            end       = min(start + batch_size, n)
            sim_batch = (mat_norm[start:end] @ mat_norm.T).toarray()

            for local_i, i in enumerate(range(start, end)):
                row    = sim_batch[local_i]
                row[i] = 0.0                                     # no self-loops
                top_k  = np.argpartition(row, -k)[-k:]
                for j in top_k:
                    if row[j] > 0:
                        G.add_edge(i, j, weight=float(row[j]))

            del sim_batch                                        # free immediately

            if (b + 1) % 50 == 0 or (b + 1) == n_batches:
                log.info("  batch %d/%d  (users %d–%d)", b + 1, n_batches, start, end - 1)

        log.info("Running Louvain (resolution=%.2f, seed=%d) …",
                 algo_cfg.resolution, algo_cfg.seed)
        partition = community_louvain.best_partition(
            G,
            resolution=algo_cfg.resolution,
            random_state=algo_cfg.seed,
        )
        rows = [
            {"ritsid": user_index[i], "role_id": f"LouvainRole{cid + 1}"}
            for i, cid in partition.items()
        ]
        assignments = pd.DataFrame(rows)
        log.info("Louvain: %d clusters discovered", assignments["role_id"].nunique())
        return assignments

    @property
    def config_schema(self) -> list[dict]:
        return [
            {
                "key":     "resolution",
                "label":   "Resolution",
                "type":    "slider",
                "default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1,
                "help":    "Higher values produce more, smaller clusters.",
            },
            {
                "key":     "batch_size",
                "label":   "Similarity Batch Size",
                "type":    "number",
                "default": 2000,
                "help":    "Rows processed per similarity pass. "
                           "Each batch uses batch_size × n_users × 4 bytes of RAM.",
            },
            {
                "key":     "seed",
                "label":   "Random Seed",
                "type":    "number",
                "default": 42,
                "help":    "Set for reproducible partitioning.",
            },
        ]
