"""
Leiden Graph Community Detection algorithm.

Builds a weighted cosine-similarity user graph in RAM-safe batches, then
partitions it with the Leiden algorithm via leidenalg + igraph.

Best for  : tight, well-connected user communities; strictly better partition
            quality guarantees than Louvain (no internally disconnected communities).
Trade-off : Same O(n²) similarity pass as Louvain — can be slow / memory-heavy
            for very large populations.  Requires: pip install leidenalg igraph
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

try:
    import igraph as ig
    import leidenalg
except ImportError as exc:
    raise ImportError(
        "Leiden requires igraph and leidenalg. "
        "Run: pip install leidenalg igraph"
    ) from exc

from .base import AlgorithmFitResult, RoleAlgorithm
from .registry import AlgorithmRegistry

log = logging.getLogger("role_miner.leiden")


@AlgorithmRegistry.register
class LeidenAlgorithm(RoleAlgorithm):
    name        = "leiden"
    description = "Graph Community Detection (Leiden) — well-connected communities, stricter than Louvain"

    def fit(self, matrix: csr_matrix, user_index: list[str], algo_cfg) -> pd.DataFrame:
        n          = matrix.shape[0]
        batch_size = algo_cfg.batch_size
        k          = min(15, n - 1)

        # Row-normalise for cosine similarity
        norms = np.asarray(matrix.sum(axis=1)).flatten()
        norms[norms == 0] = 1.0
        mat_norm = matrix.multiply(1.0 / norms[:, None]).tocsr()

        log.info("Building user similarity graph (%d users, batch=%d) …", n, batch_size)

        edges   = []
        weights = []

        n_batches = (n + batch_size - 1) // batch_size
        for b in range(n_batches):
            start     = b * batch_size
            end       = min(start + batch_size, n)
            sim_batch = (mat_norm[start:end] @ mat_norm.T).toarray()

            for local_i, i in enumerate(range(start, end)):
                row    = sim_batch[local_i]
                row[i] = 0.0                            # no self-loops
                top_k  = np.argpartition(row, -k)[-k:]
                for j in top_k:
                    if row[j] > 0 and i < j:           # undirected: store once
                        edges.append((i, j))
                        weights.append(float(row[j]))

            del sim_batch

            if (b + 1) % 50 == 0 or (b + 1) == n_batches:
                log.info("  batch %d/%d  (users %d–%d)", b + 1, n_batches, start, end - 1)

        log.info("Running Leiden (resolution=%.2f, seed=%d) …",
                 algo_cfg.resolution, algo_cfg.seed)

        G = ig.Graph(n=n, edges=edges, directed=False)
        G.es["weight"] = weights

        partition = leidenalg.find_partition(
            G,
            leidenalg.RBConfigurationVertexPartition,
            weights="weight",
            resolution_parameter=algo_cfg.resolution,
            seed=algo_cfg.seed,
        )

        rows = [
            {"ritsid": user_index[i], "role_id": f"LeidenRole{cid + 1}"}
            for cid, members in enumerate(partition)
            for i in members
        ]
        assignments = pd.DataFrame(rows)
        log.info("Leiden: %d clusters discovered", assignments["role_id"].nunique())
        return AlgorithmFitResult(primary=assignments, memberships=assignments.copy())

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
