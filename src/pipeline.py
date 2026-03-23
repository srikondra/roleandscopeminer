"""
PipelineRunner — orchestrates the full role mining pipeline.

Execution order
---------------
1.  Load & normalise CSVs              (DataLoader)
2.  Apply population filter            (PopulationFilter)
3.  Build user-entitlement matrix      (build_user_entitlement_matrix)
4.  Generate top-N tranid report       (top_tranid_by_population)
5.  Discover Tier-1 / Tier-2 grants    (TierDiscovery)
6.  For each enabled algorithm:
      a. Cluster users                 (RoleAlgorithm.fit)
      b. Profile clusters              (RoleProfiler.analyze)
      c. Build business role hierarchy (BusinessRoleHierarchy.discover)
7.  Save all outputs to OUTPUT_DIR

progress_callback(step: str, pct: float) is optional — the UI wires this to
a Streamlit progress bar; CLI runs ignore it.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import pandas as pd

from .config import PipelineConfig, AlgorithmResult, PipelineResult
from .data import DataLoader, build_user_entitlement_matrix, top_tranid_by_population
from .hierarchy import TierDiscovery, BusinessRoleHierarchy
from .algorithms.registry import AlgorithmRegistry
from .analysis import RoleProfiler

log = logging.getLogger("role_miner.pipeline")

ProgressFn = Callable[[str, float], None]


class PipelineRunner:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg

    def run(self, progress: Optional[ProgressFn] = None) -> PipelineResult:
        cfg    = self.cfg
        result = PipelineResult()
        t0     = time.perf_counter()

        def _p(step: str, pct: float) -> None:
            log.info("[%3.0f%%] %s", pct * 100, step)
            if progress:
                progress(step, pct)

        # ── 1. Load ────────────────────────────────────────────────────────────
        _p("Loading data …", 0.02)
        df = DataLoader(cfg).load()

        # ── 2. Population filter ───────────────────────────────────────────────
        pf = cfg.population_filter
        if not pf.is_empty:
            before = df["ritsid"].nunique()
            df     = pf.apply(df)
            after  = df["ritsid"].nunique()
            log.info("Population filter: %d → %d users", before, after)

        # ── 3. Matrix ──────────────────────────────────────────────────────────
        _p("Building user-entitlement matrix …", 0.12)
        matrix, user_index, grant_index = build_user_entitlement_matrix(df)
        result.df          = df
        result.matrix      = matrix
        result.user_index  = user_index
        result.grant_index = grant_index

        # ── 4. Top-tranid report ───────────────────────────────────────────────
        _p("Generating top-tranid report …", 0.18)
        result.top_tranids = top_tranid_by_population(
            df, ignore_csiid=cfg.top_tranid_ignore_csiid
        )

        # ── 5. Tier hierarchy ──────────────────────────────────────────────────
        _p("Discovering Tier-1 / Tier-2 grants …", 0.22)
        tier_result = TierDiscovery(cfg).discover(df, matrix, user_index, grant_index)
        result.tier_result  = tier_result
        cluster_matrix      = tier_result.residual_matrix
        cluster_grant_index = tier_result.residual_grant_index

        # ── 6. Algorithms ──────────────────────────────────────────────────────
        enabled   = cfg.enabled_algorithms
        n         = max(len(enabled), 1)
        profiler   = RoleProfiler(cfg)
        biz_hier   = BusinessRoleHierarchy(cfg)

        for i, algo_name in enumerate(enabled):
            base_pct = 0.28 + (i / n) * 0.58
            end_pct  = 0.28 + ((i + 1) / n) * 0.58

            def _ap(step, frac):          # local progress within this algorithm
                _p(f"[{algo_name}] {step}", base_pct + frac * (end_pct - base_pct))

            _ap("Clustering users …", 0.0)
            try:
                algo       = AlgorithmRegistry.get(algo_name)
                algo_cfg   = getattr(cfg, algo_name, None)
                fit_result = algo.fit(cluster_matrix, user_index, algo_cfg)
            except Exception as exc:
                log.error("Algorithm '%s' failed: %s", algo_name, exc, exc_info=True)
                result.algorithm_results[algo_name] = AlgorithmResult(method=algo_name)
                continue

            primary     = fit_result.primary      # 1:1 — one row per user
            memberships = fit_result.memberships  # 1:N — one row per (user, role)

            _ap("Profiling clusters …", 0.35)
            profiles, entitlements, unassigned_users, orphan_grants = profiler.analyze(
                primary, df, cluster_matrix, user_index, cluster_grant_index, algo_name
            )

            _ap("Building business role hierarchy …", 0.60)
            biz_df = biz_hier.discover(
                primary, profiles, df,
                cluster_matrix, user_index, cluster_grant_index, algo_name
            )

            result.algorithm_results[algo_name] = AlgorithmResult(
                method=algo_name,
                assignments=primary,
                memberships=memberships,
                profiles=profiles,
                entitlements=entitlements,
                biz_hierarchy=biz_df,
                unassigned_users=unassigned_users if not unassigned_users.empty else None,
                orphan_grants=orphan_grants if not orphan_grants.empty else None,
            )

        # ── 7. Save ────────────────────────────────────────────────────────────
        _p("Saving outputs …", 0.90)
        self._save_outputs(result)

        result.elapsed_seconds = time.perf_counter() - t0
        _p(f"Done  ({result.elapsed_seconds:.1f}s)", 1.0)
        return result

    # ── Output persistence ─────────────────────────────────────────────────────

    def _save_outputs(self, result: PipelineResult) -> None:
        out = Path(self.cfg.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        def _save(df: pd.DataFrame | None, name: str) -> None:
            if df is not None and not df.empty:
                p = out / f"{name}_{ts}.csv"
                df.to_csv(p, index=False)
                log.info("  Saved %-55s (%d rows)", str(p.name), len(df))

        # Tier hierarchy
        if result.tier_result:
            _save(pd.DataFrame(result.tier_result.hierarchy_rows), "role_hierarchy_tiers")

        _save(result.top_tranids, "top_tranids")

        # Per-algorithm
        for algo_name, ar in result.algorithm_results.items():
            _save(ar.profiles,         f"{algo_name}_role_profiles")
            _save(ar.entitlements,     f"{algo_name}_role_entitlements")
            _save(ar.biz_hierarchy,    f"{algo_name}_business_role_hierarchy")
            _save(ar.memberships,      f"{algo_name}_role_memberships")
            _save(ar.unassigned_users, f"{algo_name}_unassigned_users")
            _save(ar.orphan_grants,    f"{algo_name}_orphan_grants")

        # Unified hierarchy
        unified = result.unified_hierarchy()
        _save(unified, "role_hierarchy_full")

        # User-role assignment map (long format: one row per ritsid × role membership)
        userid_map: dict[str, str] = {}
        empid_map:  dict[str, str] = {}
        if result.df is not None:
            base = result.df.drop_duplicates("ritsid").set_index("ritsid")
            if "userid" in base.columns:
                userid_map = base["userid"].to_dict()
            if "empid" in base.columns:
                empid_map  = base["empid"].to_dict()
        assign_frames = []
        for algo_name, ar in result.algorithm_results.items():
            src = ar.memberships if ar.memberships is not None else ar.assignments
            if src is None or src.empty:
                continue
            a = src.copy()
            a["userid"] = a["ritsid"].map(userid_map).fillna("")
            a["empid"]  = a["ritsid"].map(empid_map).fillna("")
            a["method"] = algo_name
            assign_frames.append(a[["ritsid", "userid", "empid", "method", "role_id"]])
        all_assigns = pd.concat(assign_frames, ignore_index=True) if assign_frames else None
        _save(all_assigns, "user_role_assignments")
