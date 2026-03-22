"""
Cluster analysis: role profiling.

RoleProfiler
------------
Characterises each discovered cluster by its HR attribute mix.
Produces:
  role_profiles.csv      — one row per cluster: dominant segment/geo, top job dims
  role_entitlements.csv  — one row per (cluster, grant): prevalence, is_core flag
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from .config import PipelineConfig

log = logging.getLogger("role_miner.analysis")


# ── RoleProfiler ───────────────────────────────────────────────────────────────

def _top_values(series: pd.Series, n: int = 3) -> str:
    counts = series.dropna().replace("", pd.NA).dropna().value_counts()
    return " | ".join(counts.head(n).index.tolist())


class RoleProfiler:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg

    def analyze(
        self,
        assignments: pd.DataFrame,
        df: pd.DataFrame,
        matrix: csr_matrix,
        user_index: list[str],
        grant_index: list[str],
        method_prefix: str,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Profile every cluster meeting MIN_CLUSTER_SIZE. Returns (profiles, entitlements)."""
        profiles   = []
        all_grants = []
        min_size   = self.cfg.min_cluster_size

        for cid, group in assignments.groupby("role_id"):
            members = group["ritsid"].tolist()
            if len(members) < min_size:
                continue
            prof, grants = self._profile_cluster(
                cid, members, df, matrix, user_index, grant_index
            )
            if prof:
                profiles.append(prof)
                all_grants.extend(grants)

        profiles_df     = pd.DataFrame(profiles)
        entitlements_df = pd.DataFrame(all_grants)
        if not entitlements_df.empty:
            entitlements_df["_csiid"] = entitlements_df["grant_id"].str.split("|").str[0]
            entitlements_df = (
                entitlements_df
                .sort_values(["cluster_id", "_csiid", "prevalence"],
                             ascending=[True, True, False])
                .drop(columns=["_csiid"])
                .reset_index(drop=True)
            )
        log.info("[%s] Profiled %d roles (min_size=%d)",
                 method_prefix, len(profiles_df), min_size)
        return profiles_df, entitlements_df

    def _profile_cluster(
        self,
        cluster_id: str,
        members: list[str],
        df: pd.DataFrame,
        matrix: csr_matrix,
        user_index: list[str],
        grant_index: list[str],
    ) -> tuple[dict | None, list[dict]]:
        cfg = self.cfg
        u_idx       = {u: i for i, u in enumerate(user_index)}
        member_rows = [u_idx[m] for m in members if m in u_idx]
        if not member_rows:
            return None, []

        sub          = matrix[member_rows, :]
        grant_counts = np.asarray(sub.sum(axis=0)).flatten()
        prevalence   = grant_counts / len(member_rows)

        core_mask  = prevalence >= 0.50
        core_count = int(core_mask.sum())

        top_grant_rows = [
            {
                "cluster_id": cluster_id,
                "grant_id":   grant_index[j],
                "prevalence": round(float(prevalence[j]), 3),
                "is_core":    bool(core_mask[j]),
            }
            for j in np.where(prevalence > 0)[0]
        ]
        top_grant_rows.sort(key=lambda r: -r["prevalence"])

        hr_sub = (
            df[df["ritsid"].isin(set(members))]
            .drop_duplicates("ritsid")
        )

        seg_cols = [c for c in cfg.segment_cols    if c in hr_sub.columns]
        geo_cols = [c for c in cfg.geo_cols        if c in hr_sub.columns]
        job_cols = [c for c in cfg.job_cols        if c in hr_sub.columns]
        geo_d    = [c for c in cfg.geo_direct_cols if c in hr_sub.columns]

        dominant_segment = self._dominant(hr_sub, seg_cols)
        dominant_geo     = self._dominant(hr_sub, geo_cols)

        profile_row = {
            "cluster_id":       cluster_id,
            "member_count":     len(members),
            "core_grant_count": core_count,
            "dominant_segment": dominant_segment,
            "dominant_geo":     dominant_geo,
        }
        for col in geo_d + job_cols:
            profile_row[f"top_{col}"] = _top_values(hr_sub[col])

        return profile_row, top_grant_rows

    @staticmethod
    def _dominant(hr_sub: pd.DataFrame, cols: list[str], threshold: float = 0.40) -> str:
        for col in cols:
            vals = hr_sub[col].replace("", pd.NA).dropna()
            if not vals.empty:
                top      = vals.value_counts().idxmax()
                coverage = (vals == top).sum() / len(hr_sub)
                if coverage >= threshold:
                    return top
        return ""
