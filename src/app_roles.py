"""
AppRoleDiscovery — Phase B & D: CSIID-scoped app-role discovery.

Phase B — Exact-match fingerprinting (small CSIIDs, ≤ max_grants_exact_match):
  1. Group grant columns by CSIID.
  2. Fingerprint each user's access pattern (exact grant set).
  3. Patterns with ≥ min_users_per_pattern users → one app_role bundle.
  4. Replace raw grant columns with a single app_role column.

Phase D — Fuzzy discovery for large CSIIDs (> max_grants_exact_match):
  D.1 — Prevalence-anchored tiers:
    Compute per-grant prevalence, build cumulative tier grant sets at fixed
    thresholds (e.g. ≥0.70, ≥0.40, ≥0.20), assign each user to the HIGHEST
    tier where their coverage ≥ phase_d_min_tier_coverage.
    Falls back to D.2 when the prevalence distribution is flat (< 2 useful tiers).
  D.2 — Per-CSIID k-means:
    Auto-detect k via inertia elbow (range 2..phase_d_max_k).
    Assign users to closest centroid. Core grants per cluster = those held by
    ≥ 50 % of cluster members.

The resulting combined matrix (app_role cols + residual raw grant cols) is fed
to the downstream clustering algorithms instead of the flat raw-grant matrix.

app_role_id naming: "{csiid}|_ar{N:02d}"
  "|" separator preserves Phase-A CSIID extraction (split("|")[0]) for both
  raw grant_ids and app_role_ids.

Phase E (future): soft-membership NMF per CSIID, aligned with patent Role DAG.
"""
from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse import hstack as sparse_hstack

from .config import AppRoleResult, PipelineConfig

log = logging.getLogger("role_miner.app_roles")


class AppRoleDiscovery:
    """
    CSIID-scoped app-role discovery (Phase B + Phase D).

    Inserts between TierDiscovery (step 5) and algorithm clustering (step 6).
    Mutates cluster_matrix and cluster_grant_index so that all downstream code
    is CSIID-aware without any algorithm changes.
    """

    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg

    def discover(
        self,
        matrix:      csr_matrix,
        user_index:  list[str],
        grant_index: list[str],
    ) -> AppRoleResult:
        ar_cfg     = self.cfg.app_role
        min_users  = ar_cfg.min_users_per_pattern
        max_grants = ar_cfg.max_grants_exact_match

        n_users  = len(user_index)
        n_grants = len(grant_index)

        if n_users == 0 or n_grants == 0:
            empty = pd.DataFrame()
            return AppRoleResult(
                app_role_profiles=empty,
                user_app_assignments=empty,
                partial_users=pd.DataFrame({"ritsid": user_index}),
                app_role_matrix=matrix.copy(),
                app_role_index=list(grant_index),
            )

        # ── Group grant columns by CSIID ───────────────────────────────────────
        csiid_to_cols: dict[str, list[int]] = {}
        for j, gid in enumerate(grant_index):
            csiid = gid.split("|")[0]
            csiid_to_cols.setdefault(csiid, []).append(j)

        app_role_profiles:    list[dict] = []
        user_app_assignments: list[dict] = []

        # Absorbed indicator lists — built into a sparse mask later
        absorbed_rows_list: list[int] = []
        absorbed_cols_list: list[int] = []

        ar_user_rows: dict[str, list[int]] = {}   # app_role_id → matrix row indices
        app_role_index: list[str] = []

        # ── Per-CSIID discovery ────────────────────────────────────────────────
        for csiid, col_indices in sorted(csiid_to_cols.items()):
            sub = matrix[:, col_indices]                      # n_users × n_csiid_grants

            row_totals  = np.asarray(sub.sum(axis=1)).flatten()
            active_rows = np.where(row_totals > 0)[0]

            if len(active_rows) < min_users:
                continue                                       # too sparse — leave raw

            if len(col_indices) > max_grants:
                # ── Phase D: fuzzy matching for large CSIIDs ──────────────────
                if ar_cfg.phase_d_enabled:
                    self._phase_d_discover(
                        csiid, col_indices, sub, active_rows, user_index, grant_index,
                        app_role_profiles, user_app_assignments, ar_user_rows,
                        app_role_index, absorbed_rows_list, absorbed_cols_list,
                    )
                else:
                    log.debug(
                        "CSIID %s: %d grants > max_grants_exact_match=%d — skipped "
                        "(Phase D disabled)",
                        csiid, len(col_indices), max_grants,
                    )
                continue

            # ── Phase B: exact-match fingerprinting ───────────────────────────
            pattern_to_rows: dict[tuple[int, ...], list[int]] = {}
            for row in active_rows:
                row_vec = sub[row, :]
                pattern = tuple(sorted(int(j) for j in row_vec.nonzero()[1]))
                pattern_to_rows.setdefault(pattern, []).append(int(row))

            role_num = 1
            for pattern, rows in sorted(
                pattern_to_rows.items(), key=lambda x: -len(x[1])
            ):
                if len(rows) < min_users:
                    continue

                app_role_id = f"{csiid}|_ar{role_num:02d}"
                role_num   += 1

                grant_ids = [grant_index[col_indices[j]] for j in pattern]
                app_role_profiles.append({
                    "app_role_id":     app_role_id,
                    "csiid":           csiid,
                    "user_count":      len(rows),
                    "grant_count":     len(grant_ids),
                    "grant_ids":       " | ".join(grant_ids),
                    "discovery_phase": "B",
                })
                app_role_index.append(app_role_id)
                ar_user_rows[app_role_id] = rows

                for row in rows:
                    user_app_assignments.append({
                        "ritsid":      user_index[row],
                        "csiid":       csiid,
                        "app_role_id": app_role_id,
                    })
                    for j in pattern:
                        absorbed_rows_list.append(row)
                        absorbed_cols_list.append(col_indices[j])

        # ── Build app_role column matrix ───────────────────────────────────────
        n_ar = len(app_role_index)
        if n_ar > 0:
            ar_r, ar_c, ar_d = [], [], []
            for jj, ar_id in enumerate(app_role_index):
                for row in ar_user_rows[ar_id]:
                    ar_r.append(row)
                    ar_c.append(jj)
                    ar_d.append(1.0)
            ar_mat = csr_matrix(
                (ar_d, (ar_r, ar_c)),
                shape=(n_users, n_ar),
                dtype=np.float32,
            )
        else:
            ar_mat = None

        # ── Build residual raw grant matrix (zero out absorbed entries) ────────
        if absorbed_rows_list:
            abs_mat = csr_matrix(
                (np.ones(len(absorbed_rows_list), dtype=np.float32),
                 (absorbed_rows_list, absorbed_cols_list)),
                shape=(n_users, n_grants),
            )
            residual = matrix - matrix.multiply(abs_mat)
            residual.eliminate_zeros()
        else:
            residual = matrix.copy()

        # Drop fully-absorbed columns (all-zero after removal)
        col_sums  = np.asarray(residual.sum(axis=0)).flatten()
        kept_cols = np.where(col_sums > 0)[0]
        if len(kept_cols) < n_grants:
            residual        = residual[:, kept_cols]
            raw_grant_index = [grant_index[j] for j in kept_cols]
        else:
            raw_grant_index = list(grant_index)

        # ── Combine: [app_role cols | residual raw grant cols] ─────────────────
        if ar_mat is not None and residual.shape[1] > 0:
            combined       = sparse_hstack([ar_mat, residual], format="csr")
            combined_index = app_role_index + raw_grant_index
        elif ar_mat is not None:
            combined       = ar_mat.tocsr()
            combined_index = app_role_index
        else:
            combined       = residual
            combined_index = raw_grant_index

        # ── Partial users: no app_role assigned across all CSIIDs ─────────────
        if ar_mat is not None:
            assigned_mask = np.asarray(ar_mat.sum(axis=1)).flatten() > 0
        else:
            assigned_mask = np.zeros(n_users, dtype=bool)
        partial_ritsids = [user_index[i] for i in np.where(~assigned_mask)[0]]

        profiles_df    = pd.DataFrame(app_role_profiles)
        assignments_df = pd.DataFrame(user_app_assignments)
        partial_df     = pd.DataFrame({"ritsid": partial_ritsids})

        n_b  = sum(1 for p in app_role_profiles if p.get("discovery_phase") == "B")
        n_d  = n_ar - n_b
        log.info(
            "AppRoleDiscovery: %d app_roles (B=%d D=%d) across %d CSIIDs | "
            "%d absorbed entries | %d partial users | matrix %d×%d → %d×%d",
            n_ar, n_b, n_d,
            profiles_df["csiid"].nunique() if not profiles_df.empty else 0,
            len(absorbed_rows_list),
            len(partial_ritsids),
            n_users, n_grants,
            n_users, len(combined_index),
        )

        return AppRoleResult(
            app_role_profiles=profiles_df,
            user_app_assignments=assignments_df,
            partial_users=partial_df,
            app_role_matrix=combined,
            app_role_index=combined_index,
        )

    # ── Phase D: fuzzy discovery dispatcher ────────────────────────────────────

    def _phase_d_discover(
        self,
        csiid:        str,
        col_indices:  list[int],
        sub:          csr_matrix,
        active_rows:  np.ndarray,
        user_index:   list[str],
        grant_index:  list[str],
        app_role_profiles:    list[dict],
        user_app_assignments: list[dict],
        ar_user_rows:         dict[str, list[int]],
        app_role_index:       list[str],
        absorbed_rows_list:   list[int],
        absorbed_cols_list:   list[int],
    ) -> None:
        """Try Phase D.1 (prevalence tiers); fall back to D.2 (k-means)."""
        success = self._phase_d1_tier_discover(
            csiid, col_indices, sub, active_rows, user_index, grant_index,
            app_role_profiles, user_app_assignments, ar_user_rows,
            app_role_index, absorbed_rows_list, absorbed_cols_list,
        )
        if not success:
            self._phase_d2_kmeans_discover(
                csiid, col_indices, sub, active_rows, user_index, grant_index,
                app_role_profiles, user_app_assignments, ar_user_rows,
                app_role_index, absorbed_rows_list, absorbed_cols_list,
            )

    # ── Phase D.1: prevalence-anchored tier discovery ──────────────────────────

    def _phase_d1_tier_discover(
        self,
        csiid:        str,
        col_indices:  list[int],
        sub:          csr_matrix,
        active_rows:  np.ndarray,
        user_index:   list[str],
        grant_index:  list[str],
        app_role_profiles:    list[dict],
        user_app_assignments: list[dict],
        ar_user_rows:         dict[str, list[int]],
        app_role_index:       list[str],
        absorbed_rows_list:   list[int],
        absorbed_cols_list:   list[int],
    ) -> bool:
        """
        Assign grants to cumulative prevalence tiers; assign users to the highest
        tier where coverage ≥ phase_d_min_tier_coverage.

        Cumulative tiers: tier-k includes all grants with prevalence ≥ threshold-k
        (e.g. tier 1 = grants ≥0.70, tier 2 = grants ≥0.40, tier 3 = grants ≥0.20).
        A user assigned to tier-3 holds most of those high+mid+low grants, making
        them a "full-access" archetype; tier-1 users hold only the common core.

        Returns True when ≥ 2 distinct tiers with ≥ phase_d_min_cluster_size users
        are found (indicating a genuine access-level hierarchy).
        """
        ar_cfg   = self.cfg.app_role
        n_local  = len(col_indices)
        n_active = len(active_rows)

        active_dense = sub[active_rows, :].toarray()          # n_active × n_local
        prev         = active_dense.sum(axis=0) / n_active    # per-grant prevalence

        # Build cumulative tier masks from highest threshold downward
        thresholds = sorted(ar_cfg.phase_d_tier_thresholds, reverse=True)
        tier_masks: list[tuple[float, np.ndarray]] = []       # (threshold, cumulative mask)
        cum_mask = np.zeros(n_local, dtype=bool)
        for thresh in thresholds:
            cum_mask = cum_mask | (prev >= thresh)
            if cum_mask.any():
                tier_masks.append((thresh, cum_mask.copy()))

        if not tier_masks:
            return False

        min_cov = ar_cfg.phase_d_min_tier_coverage
        min_sz  = ar_cfg.phase_d_min_cluster_size

        # For each active user, find the HIGHEST tier where coverage ≥ min_cov.
        # Higher tier index = larger cumulative grant set = more access.
        user_best_tier: dict[int, int] = {}                   # active_idx → tier_idx

        for t_idx, (_, gmask) in enumerate(tier_masks):
            tier_cols = np.where(gmask)[0]
            coverage  = active_dense[:, tier_cols].sum(axis=1) / len(tier_cols)
            for ai in np.where(coverage >= min_cov)[0]:
                ai = int(ai)
                if t_idx > user_best_tier.get(ai, -1):
                    user_best_tier[ai] = t_idx

        # Group active-row indices by their assigned tier
        tier_to_active: defaultdict[int, list[int]] = defaultdict(list)
        for ai, t_idx in user_best_tier.items():
            tier_to_active[t_idx].append(ai)

        useful_tiers = [
            (t_idx, ais) for t_idx, ais in tier_to_active.items()
            if len(ais) >= min_sz
        ]

        if len(useful_tiers) < 2:
            log.debug(
                "CSIID %s (D.1): %d useful tier(s) — prevalence flat, falling to D.2",
                csiid, len(useful_tiers),
            )
            return False

        role_num = 1
        for t_idx, active_indices in sorted(useful_tiers, key=lambda x: -len(x[1])):
            _, gmask  = tier_masks[t_idx]
            tier_cols = np.where(gmask)[0]
            rows      = [int(active_rows[ai]) for ai in active_indices]
            grant_ids = [grant_index[col_indices[j]] for j in tier_cols]

            app_role_id = f"{csiid}|_ar{role_num:02d}"
            role_num   += 1

            app_role_profiles.append({
                "app_role_id":     app_role_id,
                "csiid":           csiid,
                "user_count":      len(rows),
                "grant_count":     len(grant_ids),
                "grant_ids":       " | ".join(grant_ids),
                "discovery_phase": "D1",
            })
            app_role_index.append(app_role_id)
            ar_user_rows[app_role_id] = rows

            for row in rows:
                user_app_assignments.append({
                    "ritsid":      user_index[row],
                    "csiid":       csiid,
                    "app_role_id": app_role_id,
                })
                # Absorb all grants this user actually holds within the CSIID
                for j in sub[row, :].nonzero()[1]:
                    absorbed_rows_list.append(row)
                    absorbed_cols_list.append(col_indices[int(j)])

        log.debug(
            "CSIID %s (D.1): %d tiers → %d users assigned",
            csiid, len(useful_tiers),
            sum(len(ais) for _, ais in useful_tiers),
        )
        return True

    # ── Phase D.2: per-CSIID k-means ──────────────────────────────────────────

    def _phase_d2_kmeans_discover(
        self,
        csiid:        str,
        col_indices:  list[int],
        sub:          csr_matrix,
        active_rows:  np.ndarray,
        user_index:   list[str],
        grant_index:  list[str],
        app_role_profiles:    list[dict],
        user_app_assignments: list[dict],
        ar_user_rows:         dict[str, list[int]],
        app_role_index:       list[str],
        absorbed_rows_list:   list[int],
        absorbed_cols_list:   list[int],
    ) -> None:
        """
        Per-CSIID binary k-means for large CSIIDs where D.1 finds a flat
        prevalence distribution (parallel archetypes at similar access depth).

        Auto-selects k in [2, phase_d_max_k] via inertia elbow (second derivative
        of inertia curve peaks at the elbow k).
        Core grants per cluster = those held by ≥ 50 % of cluster members.
        """
        try:
            from sklearn.cluster import KMeans
        except ImportError:
            log.warning("sklearn unavailable — Phase D.2 skipped for CSIID %s", csiid)
            return

        ar_cfg   = self.cfg.app_role
        min_sz   = ar_cfg.phase_d_min_cluster_size
        n_active = len(active_rows)

        if n_active < min_sz * 2:
            log.debug("CSIID %s (D.2): too few active users (%d) — skipped", csiid, n_active)
            return

        active_dense = sub[active_rows, :].toarray().astype(np.float32)   # n_active × n_local
        max_k = max(2, min(ar_cfg.phase_d_max_k, n_active // min_sz))

        # Auto-select k via inertia elbow
        best_k = 2
        if max_k > 2:
            inertias = []
            for k in range(2, max_k + 1):
                km = KMeans(n_clusters=k, random_state=42, n_init=5, max_iter=100)
                km.fit(active_dense)
                inertias.append(km.inertia_)

            if len(inertias) >= 3:
                # diffs2[i] = curvature at k = i+3; peak = elbow
                diffs2 = np.diff(np.diff(inertias))
                best_k = min(int(np.argmax(diffs2)) + 3, max_k)
            # else best_k stays 2

        km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10, max_iter=300)
        labels   = km_final.fit_predict(active_dense)

        role_num = 1
        for k in range(best_k):
            cluster_ai  = np.where(labels == k)[0]       # indices into active_rows
            if len(cluster_ai) < min_sz:
                continue

            rows        = [int(active_rows[ai]) for ai in cluster_ai]
            cluster_sub = active_dense[cluster_ai, :]    # n_cluster × n_local
            core_prev   = cluster_sub.mean(axis=0)       # prevalence within cluster
            core_cols   = np.where(core_prev >= 0.50)[0]

            if core_cols.size == 0:
                continue

            app_role_id = f"{csiid}|_ar{role_num:02d}"
            role_num   += 1
            grant_ids   = [grant_index[col_indices[j]] for j in core_cols]

            app_role_profiles.append({
                "app_role_id":     app_role_id,
                "csiid":           csiid,
                "user_count":      len(rows),
                "grant_count":     len(grant_ids),
                "grant_ids":       " | ".join(grant_ids),
                "discovery_phase": "D2",
            })
            app_role_index.append(app_role_id)
            ar_user_rows[app_role_id] = rows

            for row in rows:
                user_app_assignments.append({
                    "ritsid":      user_index[row],
                    "csiid":       csiid,
                    "app_role_id": app_role_id,
                })
                # Absorb all grants this user holds within the CSIID
                for j in sub[row, :].nonzero()[1]:
                    absorbed_rows_list.append(row)
                    absorbed_cols_list.append(col_indices[int(j)])

        log.debug(
            "CSIID %s (D.2): k-means k=%d → %d app_roles from %d active users",
            csiid, best_k, role_num - 1, n_active,
        )
