"""
Role hierarchy: Tier 1/2 discovery and business role sub-tier decomposition.

TierDiscovery
-------------
Classifies grants into:
  Tier 1 — Staff root (globally prevalent, adguid-absent grants)
  Tier 2 — Staff with Tech Access (prevalent among tech users, mid-band)
  Residual — everything else, passed to clustering algorithms

BusinessRoleHierarchy
---------------------
For each cluster produced by Louvain / NMF, sorts grants by prevalence and
splits them into ordered sub-tiers wherever a consecutive prevalence drop
exceeds HIER_BUSINESS_GAP_THRESHOLD.

  Sub-tier 1  Core       (highest-prevalence grants)
  Sub-tier 2  Extended   (next band after first gap)
  Sub-tier 3  Specialist (next band after second gap)
  Sub-tier N  Level N    (subsequent bands)
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from .config import PipelineConfig, TierResult

log = logging.getLogger("role_miner.hierarchy")

_SUB_LABELS = {1: "Core", 2: "Extended", 3: "Specialist"}


# ── TierDiscovery ──────────────────────────────────────────────────────────────

class TierDiscovery:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg

    def discover(
        self,
        df: pd.DataFrame,
        matrix: csr_matrix,
        user_index: list[str],
        grant_index: list[str],
    ) -> TierResult:
        cfg     = self.cfg
        n_users = len(user_index)
        u_idx   = {u: i for i, u in enumerate(user_index)}

        # Global prevalence — always computed (used in hierarchy_rows)
        global_counts = np.asarray(matrix.sum(axis=0)).flatten()
        global_prev   = global_counts / n_users

        # ── Tier identification ────────────────────────────────────────────────
        predefined = self._load_tier_definitions(df, grant_index)

        if predefined is not None:
            staff_grants, tech_baseline_grants = predefined
            log.info("Tier 1 Staff grants    : %d  (pre-defined file)", len(staff_grants))
            log.info("Tier 2 Tech Baseline   : %d  (pre-defined file)", len(tech_baseline_grants))
        else:
            staff_grants, tech_baseline_grants = self._dynamic_discovery(
                df, matrix, user_index, u_idx, grant_index, global_prev, cfg
            )

        # ── Residual matrix for Tier-3 clustering ─────────────────────────────
        exclude              = staff_grants | tech_baseline_grants
        keep_cols            = [j for j, g in enumerate(grant_index) if g not in exclude]
        residual_matrix      = matrix[:, keep_cols].tocsr()
        residual_grant_index = [grant_index[j] for j in keep_cols]
        log.info("Tier 3 residual grants : %d  (stripped %d)", len(residual_grant_index), len(exclude))

        # ── hierarchy_rows — expand normalized grants → original source grants ─
        hierarchy_rows = self._build_hierarchy_rows(
            df, staff_grants, tech_baseline_grants, grant_index, global_prev
        )

        return TierResult(
            staff_grants=staff_grants,
            tech_baseline_grants=tech_baseline_grants,
            hierarchy_rows=hierarchy_rows,
            residual_matrix=residual_matrix,
            residual_grant_index=residual_grant_index,
        )

    def _load_tier_definitions(
        self,
        df: pd.DataFrame,
        grant_index: list[str],
    ) -> tuple[set, set] | None:
        path_str = self.cfg.tier_definitions_file
        if not path_str:
            return None

        path = Path(path_str)
        if not path.exists():
            log.warning("TIER_DEFINITIONS_FILE '%s' not found — dynamic discovery", path)
            return None

        tier_df = pd.read_csv(path, dtype=str).fillna("")
        tier_df.columns = tier_df.columns.str.lower().str.strip()
        missing = {"tier", "tranid"} - set(tier_df.columns)
        if missing:
            raise ValueError(f"tier_definitions CSV is missing columns: {missing}")

        tier1_tranids = set(tier_df[tier_df["tier"].str.strip() == "1"]["tranid"].str.strip())
        tier2_tranids = set(tier_df[tier_df["tier"].str.strip() == "2"]["tranid"].str.strip())

        grant_tranid = (
            df[["grant_id", "tranid"]]
            .drop_duplicates("grant_id")
            .set_index("grant_id")["tranid"]
            .to_dict()
        )

        tier1_grants = {g for g in grant_index if grant_tranid.get(g) in tier1_tranids}
        tier2_grants = {
            g for g in grant_index
            if grant_tranid.get(g) in tier2_tranids and g not in tier1_grants
        }

        log.info(
            "Tier definitions from '%s': "
            "%d Tier-1 tranids → %d grants | %d Tier-2 tranids → %d grants",
            path, len(tier1_tranids), len(tier1_grants), len(tier2_tranids), len(tier2_grants),
        )

        unmatched1 = tier1_tranids - {grant_tranid.get(g) for g in tier1_grants}
        unmatched2 = tier2_tranids - {grant_tranid.get(g) for g in tier2_grants}
        if unmatched1:
            log.warning("Tier-1 tranids with no matching grants: %s", ", ".join(sorted(unmatched1)))
        if unmatched2:
            log.warning("Tier-2 tranids with no matching grants: %s", ", ".join(sorted(unmatched2)))

        return tier1_grants, tier2_grants

    def _dynamic_discovery(
        self, df, matrix, user_index, u_idx, grant_index, global_prev, cfg
    ) -> tuple[set, set]:
        hier = cfg.hierarchy

        # Tier 1 — Staff root: adguid-absent grants above prevalence floor
        no_adguid_users = set(df[df["adguid"].eq("")]["ritsid"].unique())
        no_adguid_rows  = [u_idx[u] for u in no_adguid_users if u in u_idx]

        no_adguid_counts = (
            np.asarray(matrix[no_adguid_rows, :].sum(axis=0)).flatten()
            if no_adguid_rows else np.zeros(len(grant_index))
        )

        staff_mask   = (no_adguid_counts > 0) & (global_prev >= hier.staff_min_prevalence)
        staff_grants = {grant_index[j] for j in np.where(staff_mask)[0]}
        log.info("Tier 1 Staff grants    : %d  (global prev ≥ %.0f%%)",
                 len(staff_grants), hier.staff_min_prevalence * 100)

        # Tier 2 — Tech Baseline: adguid-present users, mid-prevalence band
        tech_users = set(df[df["adguid"].ne("")]["ritsid"].unique())
        tech_rows  = [u_idx[u] for u in tech_users if u in u_idx]
        n_tech     = len(tech_rows)

        tech_prev = (
            np.asarray(matrix[tech_rows, :].sum(axis=0)).flatten() / n_tech
            if tech_rows else np.zeros(len(grant_index))
        )

        tech_mask            = (
            ~staff_mask
            & (tech_prev >= hier.tech_baseline_min_prevalence)
            & (tech_prev <= hier.tech_baseline_max_prevalence)
        )
        tech_baseline_grants = {grant_index[j] for j in np.where(tech_mask)[0]}
        log.info("Tier 2 Tech Baseline   : %d  (tech prev %.0f%%–%.0f%%)",
                 len(tech_baseline_grants),
                 hier.tech_baseline_min_prevalence * 100,
                 hier.tech_baseline_max_prevalence * 100)

        return staff_grants, tech_baseline_grants

    def _build_hierarchy_rows(
        self,
        df: pd.DataFrame,
        staff_grants: set,
        tech_baseline_grants: set,
        grant_index: list[str],
        global_prev: np.ndarray,
    ) -> list[dict]:
        grant_meta = (
            df[["grant_id", "descrtx", "appname"]]
            .drop_duplicates("grant_id")
            .set_index("grant_id")
        )
        g_idx_map = {g: j for j, g in enumerate(grant_index)}

        orig_expansion: dict[str, list[str]] = {}
        if "grant_id_orig" in df.columns:
            for norm_gid, grp in (
                df[["grant_id", "grant_id_orig"]]
                .drop_duplicates()
                .groupby("grant_id")["grant_id_orig"]
            ):
                orig_expansion[norm_gid] = grp.tolist()

        rows = []
        for tier_num, tier_name, grant_set in (
            (1, "Staff",                  staff_grants),
            (2, "Staff with Tech Access", tech_baseline_grants),
        ):
            for g in grant_set:
                j       = g_idx_map.get(g)
                prev    = round(float(global_prev[j]), 3) if j is not None else 0.0
                meta    = grant_meta.loc[g] if g in grant_meta.index else {}
                descrtx = meta.get("descrtx", "") if isinstance(meta, dict) else meta["descrtx"]
                appname = meta.get("appname",  "") if isinstance(meta, dict) else meta["appname"]

                for orig_gid in orig_expansion.get(g, [g]):
                    rows.append({
                        "tier":                tier_num,
                        "tier_name":           tier_name,
                        "grant_id":            orig_gid,
                        "grant_id_normalized": g,
                        "prevalence":          prev,
                        "descrtx":             descrtx,
                        "appname":             appname,
                    })
        return rows


# ── BusinessRoleHierarchy ──────────────────────────────────────────────────────

def _canonical_role_name(
    cluster_id: str,
    profile_row: dict,
    core_grants: list[dict],
    grant_meta: pd.DataFrame,
) -> str:
    """Synthesise a human-readable name from HR signals and top app."""
    parts = []

    seg = (profile_row.get("dominant_segment") or "").strip()
    if seg:
        parts.append(seg)

    for key in ("top_jobfunctiondescription", "top_jobdescription",
                "top_jobfamilydescription", "top_jobcode"):
        raw = (profile_row.get(key) or "").strip()
        if raw:
            first = raw.split("|")[0].strip()
            if first:
                parts.append(first)
                break

    if len(parts) < 2 and core_grants:
        for g in core_grants:
            gid = g["grant_id"]
            if gid in grant_meta.index:
                app = str(grant_meta.loc[gid]["appname"]).strip()
                csi = gid.split("|")[0] if "|" in gid else gid
                if app and app != csi:
                    parts.append(app)
                    break

    if not parts:
        return f"Business Role {cluster_id}"

    deduped = [parts[0]]
    for p in parts[1:]:
        if p.lower() != deduped[-1].lower():
            deduped.append(p)
    return " - ".join(deduped[:3])


class BusinessRoleHierarchy:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg

    def discover(
        self,
        assignments: pd.DataFrame,
        profiles_df: pd.DataFrame | None,
        df: pd.DataFrame,
        cluster_matrix: csr_matrix,
        user_index: list[str],
        cluster_grant_index: list[str],
        method_prefix: str,
    ) -> pd.DataFrame:
        cfg        = self.cfg
        gap_thresh = cfg.hierarchy.business_gap_threshold
        min_prev   = cfg.hierarchy.business_min_prevalence
        min_grants = cfg.hierarchy.business_min_tier_grants
        min_size   = cfg.min_cluster_size

        u_idx = {u: i for i, u in enumerate(user_index)}
        grant_meta = (
            df[["grant_id", "descrtx", "appname"]]
            .drop_duplicates("grant_id")
            .set_index("grant_id")
        )

        profile_lookup: dict[str, dict] = {}
        if profiles_df is not None and not profiles_df.empty:
            for rec in profiles_df.to_dict("records"):
                profile_lookup[rec["cluster_id"]] = rec

        rows        = []
        n_roles_out = 0

        for cid, group in assignments.groupby("role_id"):
            members = group["ritsid"].tolist()
            if len(members) < min_size:
                continue

            member_rows = [u_idx[m] for m in members if m in u_idx]
            if not member_rows:
                continue

            sub          = cluster_matrix[member_rows, :]
            grant_counts = np.asarray(sub.sum(axis=0)).flatten()
            prevalence   = grant_counts / len(member_rows)

            keep = np.where(prevalence >= min_prev)[0]
            if len(keep) == 0:
                continue

            order       = keep[np.argsort(-prevalence[keep])]
            sorted_prev = prevalence[order]

            core_grants = [
                {"grant_id": cluster_grant_index[j], "prevalence": float(prevalence[j])}
                for j in np.where(prevalence >= 0.50)[0]
            ]

            profile_row = profile_lookup.get(cid, {"cluster_id": cid})
            role_name   = _canonical_role_name(cid, profile_row, core_grants, grant_meta)

            sub_tier       = 1
            grants_in_tier = 0

            for rank, j in enumerate(order):
                gid      = cluster_grant_index[j]
                prev_val = float(sorted_prev[rank])

                if rank > 0:
                    gap = float(sorted_prev[rank - 1]) - prev_val
                    if gap >= gap_thresh and grants_in_tier >= min_grants:
                        sub_tier       += 1
                        grants_in_tier  = 0

                tier_label    = _SUB_LABELS.get(sub_tier, f"Level {sub_tier}")
                sub_role_name = role_name if sub_tier == 1 else f"{role_name} - {tier_label}"

                meta    = grant_meta.loc[gid] if gid in grant_meta.index else None
                descrtx = str(meta["descrtx"]) if meta is not None else ""
                appname = str(meta["appname"])  if meta is not None else ""

                rows.append({
                    "method":           method_prefix,
                    "cluster_id":       cid,
                    "role_name":        role_name,
                    "parent_role_name": "" if sub_tier == 1 else role_name,
                    "member_count":     len(members),
                    "sub_tier":         sub_tier,
                    "sub_role_name":    sub_role_name,
                    "grant_rank":       rank + 1,
                    "grant_id":         gid,
                    "prevalence":       round(prev_val, 3),
                    "descrtx":          descrtx,
                    "appname":          appname,
                })
                grants_in_tier += 1

            n_roles_out += 1

        result = pd.DataFrame(rows)
        if not result.empty:
            deduped = (
                result
                .sort_values(["cluster_id", "sub_tier"], ascending=[True, True])
                .drop_duplicates(subset=["cluster_id", "sub_tier"])
            )
            tier_label = deduped["sub_tier"].map(
                lambda t: _SUB_LABELS.get(t, f"Level{t}"))
            sub_cid = deduped["cluster_id"].where(
                deduped["sub_tier"] == 1,
                deduped["cluster_id"] + "-" + tier_label)
            result = pd.DataFrame({
                "cluster_id":        sub_cid.values,
                "parent_cluster_id": deduped["cluster_id"].where(
                    deduped["sub_tier"] > 1, "").values,
            })
            log.info("[%s] Business roles: %d roles  (gap=%.0f%%  floor=%.0f%%)",
                     method_prefix, n_roles_out, gap_thresh * 100, min_prev * 100)
        else:
            log.info("[%s] Business roles: 0 roles (check MIN_CLUSTER_SIZE / data)", method_prefix)
        return result
