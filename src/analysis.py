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


# ── App-scope summary (Phase A) ─────────────────────────────────────────────────

def build_app_scope_summary(entitlements_df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarise role entitlements by (cluster_id, csiid).

    For each discovered role, groups all grants by their CSIID to show which
    applications contribute to the role, at what depth, and at what prevalence.
    This is Phase A of CSIID-aware analysis — pure post-processing, no algorithm change.

    Returns a DataFrame with columns:
        cluster_id, csiid, grant_count, core_grant_count,
        avg_prevalence, max_prevalence, top_tranids
    """
    if entitlements_df is None or entitlements_df.empty:
        return pd.DataFrame()

    df = entitlements_df.copy()
    df["_csiid"]  = df["grant_id"].str.split("|").str[0]
    df["_tranid"] = df["grant_id"].str.split("|").str[1]

    summary = (
        df.groupby(["cluster_id", "_csiid"])
        .agg(
            grant_count      = ("grant_id",   "count"),
            core_grant_count = ("is_core",    "sum"),
            avg_prevalence   = ("prevalence", "mean"),
            max_prevalence   = ("prevalence", "max"),
        )
        .reset_index()
        .rename(columns={"_csiid": "csiid"})
    )

    # Top-3 tranids per (cluster_id, csiid) by prevalence for readability
    top_tranids = (
        df.sort_values("prevalence", ascending=False)
          .groupby(["cluster_id", "_csiid"])["_tranid"]
          .apply(lambda s: " | ".join(s.head(3).tolist()))
          .reset_index()
          .rename(columns={"_csiid": "csiid", "_tranid": "top_tranids"})
    )

    summary = summary.merge(top_tranids, on=["cluster_id", "csiid"], how="left")
    summary["avg_prevalence"] = summary["avg_prevalence"].round(3)
    summary["max_prevalence"] = summary["max_prevalence"].round(3)
    summary = summary.sort_values(
        ["cluster_id", "grant_count"], ascending=[True, False]
    ).reset_index(drop=True)

    log.info(
        "App-scope summary: %d (role, csiid) pairs across %d roles",
        len(summary),
        summary["cluster_id"].nunique(),
    )
    return summary


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
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Profile every cluster meeting min_cluster_size.

        Returns (profiles, entitlements, unassigned_users, orphan_grants).

        unassigned_users : users whose cluster was too small to form a role.
        orphan_grants    : grants whose best prevalence across all profiled roles
                           is below orphan_grant_max_role_prevalence; these do not
                           belong cleanly to any role.
        """
        profiles        = []
        all_grants      = []
        min_size        = self.cfg.min_cluster_size
        orphan_thresh   = self.cfg.hierarchy.orphan_grant_max_role_prevalence

        unassigned_ritsids: list[str] = []

        for cid, group in assignments.groupby("role_id"):
            members = group["ritsid"].tolist()
            if len(members) < min_size:
                unassigned_ritsids.extend(members)
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

        # ── Unassigned users ───────────────────────────────────────────────────
        unassigned_df = pd.DataFrame({"ritsid": unassigned_ritsids})
        if not unassigned_df.empty and df is not None:
            hr_cols = ["ritsid"] + [
                c for c in ("userid", "empid", "jobfunctiondescription",
                            "jobfamilydescription", "region", "country")
                if c in df.columns
            ]
            hr_map = df.drop_duplicates("ritsid").set_index("ritsid")[
                [c for c in hr_cols if c != "ritsid"]
            ]
            unassigned_df = unassigned_df.join(hr_map, on="ritsid", how="left")

        # ── Orphan grants ──────────────────────────────────────────────────────
        # Grant is orphan if its max prevalence across all profiled roles < threshold,
        # OR if it is held exclusively by unassigned users.
        orphan_df = pd.DataFrame()
        if not entitlements_df.empty:
            best_prev = (
                entitlements_df
                .groupby("grant_id")["prevalence"]
                .max()
                .reset_index()
                .rename(columns={"prevalence": "best_role_prevalence"})
            )
            orphan_by_prev = best_prev[
                best_prev["best_role_prevalence"] < orphan_thresh
            ].copy()
            orphan_by_prev["reason"] = "low_role_prevalence"

            # Grants held only by unassigned users (not in any profiled role at all)
            all_role_grant_ids = set(entitlements_df["grant_id"].unique())
            if unassigned_ritsids and matrix is not None:
                u_idx    = {u: i for i, u in enumerate(user_index)}
                ua_rows  = [u_idx[r] for r in unassigned_ritsids if r in u_idx]
                if ua_rows:
                    ua_sub     = matrix[ua_rows, :]
                    ua_counts  = np.asarray(ua_sub.sum(axis=0)).flatten()
                    ua_gids    = {grant_index[j] for j in np.where(ua_counts > 0)[0]}
                    ua_only    = ua_gids - all_role_grant_ids
                    if ua_only:
                        ua_df = pd.DataFrame({
                            "grant_id":            sorted(ua_only),
                            "best_role_prevalence": 0.0,
                            "reason":              "unassigned_users_only",
                        })
                        orphan_by_prev = pd.concat(
                            [orphan_by_prev, ua_df], ignore_index=True
                        ).drop_duplicates(subset="grant_id")

            orphan_df = orphan_by_prev.sort_values("best_role_prevalence").reset_index(drop=True)

        log.info(
            "[%s] Profiled %d roles (min_size=%d) | %d unassigned users | %d orphan grants",
            method_prefix, len(profiles_df), min_size,
            len(unassigned_df), len(orphan_df),
        )
        return profiles_df, entitlements_df, unassigned_df, orphan_df

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
