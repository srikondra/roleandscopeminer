"""
Cluster analysis: role profiling and scope attribute discovery.

RoleProfiler
------------
Characterises each discovered cluster by its HR attribute mix.
Produces:
  role_profiles.csv      — one row per cluster: dominant segment/geo, top job dims
  role_entitlements.csv  — one row per (cluster, grant): prevalence, is_core flag

ScopeDiscovery
--------------
Pass 2 — for each role cluster, identifies grants in a mid-prevalence band
(not universal, not rare) and tests every HR attribute for lift:

    lift(G, A=V) = P(has G | A=V) / P(has G | A≠V)

When best lift ≥ SCOPE_LIFT_THRESHOLD the grant is scope-specific and
(A, V) is its scope attribute.  Grants sharing the same best (A, V) are
grouped into one scope variant.
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


# ── ScopeDiscovery ─────────────────────────────────────────────────────────────

def _lift(has_grant: np.ndarray, in_group: np.ndarray) -> float:
    in_mask  = in_group.astype(bool)
    out_mask = ~in_mask
    p_in  = has_grant[in_mask].mean()  if in_mask.sum()  > 0 else 0.0
    p_out = has_grant[out_mask].mean() if out_mask.sum() > 0 else 0.0
    return 0.0 if p_in == 0 else p_in / max(p_out, 0.01)


def _scope_hr_attrs(df: pd.DataFrame, cfg: PipelineConfig) -> list[str]:
    candidates = []
    for col in cfg.geo_direct_cols + cfg.geo_cols + cfg.segment_cols + cfg.job_cols:
        if col in df.columns and df[col].replace("", pd.NA).notna().any():
            candidates.append(col)
    return candidates


class ScopeDiscovery:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg

    def discover(
        self,
        assignments: pd.DataFrame,
        df: pd.DataFrame,
        matrix: csr_matrix,
        user_index: list[str],
        grant_index: list[str],
        method_prefix: str,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        cfg          = self.cfg
        scope_cfg    = cfg.scope
        min_size     = cfg.min_cluster_size
        u_idx        = {u: i for i, u in enumerate(user_index)}
        hr_lookup    = df.drop_duplicates("ritsid").set_index("ritsid")
        scope_attrs  = _scope_hr_attrs(df, cfg)
        userid_map   = hr_lookup["userid"].to_dict() if "userid" in hr_lookup.columns else {}
        empid_map    = hr_lookup["empid"].to_dict()  if "empid"  in hr_lookup.columns else {}

        all_profiles = []
        all_members  = []
        scope_counter = 0

        for cid, group in assignments.groupby("role_id"):
            members = group["ritsid"].tolist()
            if len(members) < min_size:
                continue

            member_rows = [u_idx[m] for m in members if m in u_idx]
            if not member_rows:
                continue

            sub      = matrix[member_rows, :]
            g_counts = np.asarray(sub.sum(axis=0)).flatten()
            prev     = g_counts / len(member_rows)

            cand_mask = (prev >= scope_cfg.min_prevalence) & (prev <= scope_cfg.max_prevalence)
            cand_idxs = np.where(cand_mask)[0]
            if len(cand_idxs) == 0:
                continue

            hr_sub = hr_lookup.reindex([m for m in members if m in hr_lookup.index])

            scope_grant_map: dict[tuple, list] = {}
            scope_lift_map:  dict[tuple, list] = {}

            for j in cand_idxs:
                has_g = np.array(sub[:, j].toarray().flatten() > 0, dtype=float)
                has_g_aligned = np.array(
                    [has_g[member_rows.index(u_idx[m])] if m in u_idx else 0.0
                     for m in hr_sub.index],
                    dtype=float,
                )

                best_lift, best_attr, best_val = 0.0, None, None

                for attr in scope_attrs:
                    if attr not in hr_sub.columns:
                        continue
                    col_vals = hr_sub[attr].replace("", pd.NA).fillna("__missing__")
                    for val in col_vals.unique():
                        if val == "__missing__":
                            continue
                        in_grp = (col_vals == val).values.astype(float)
                        if in_grp.sum() < scope_cfg.min_group_size:
                            continue
                        l = _lift(has_g_aligned, in_grp)
                        if l > best_lift:
                            best_lift, best_attr, best_val = l, attr, val

                if best_lift >= scope_cfg.lift_threshold and best_attr is not None:
                    key = (best_attr, best_val)
                    scope_grant_map.setdefault(key, []).append(grant_index[j])
                    scope_lift_map.setdefault(key,  []).append(round(best_lift, 2))

            for (attr, val), grants in scope_grant_map.items():
                scope_id   = f"{cid}_S{scope_counter}"
                scope_counter += 1
                avg_lift   = round(float(np.mean(scope_lift_map[(attr, val)])), 2)
                col_vals   = hr_sub[attr].replace("", pd.NA).fillna("__missing__")
                scope_mbrs = [m for m in hr_sub.index if col_vals.get(m) == val]

                if len(scope_mbrs) < scope_cfg.min_group_size:
                    continue

                all_profiles.append({
                    "template_cluster_id": cid,
                    "scope_id":            scope_id,
                    "scope_attribute":     attr,
                    "scope_value":         val,
                    "member_count":        len(scope_mbrs),
                    "scope_grant_count":   len(grants),
                    "scope_grants":        " | ".join(grants),
                    "avg_lift":            avg_lift,
                })
                for m in scope_mbrs:
                    all_members.append({
                        "ritsid":              m,
                        "userid":              userid_map.get(m, ""),
                        "empid":               empid_map.get(m, ""),
                        "template_cluster_id": cid,
                        "scope_id":            scope_id,
                        "scope_attribute":     attr,
                        "scope_value":         val,
                    })

        scope_profiles = pd.DataFrame(all_profiles)
        scope_members  = pd.DataFrame(all_members)

        n_scoped = scope_profiles["template_cluster_id"].nunique() if not scope_profiles.empty else 0
        log.info("[%s] Scope discovery: %d variants across %d role templates",
                 method_prefix, len(scope_profiles), n_scoped)
        return scope_profiles, scope_members
