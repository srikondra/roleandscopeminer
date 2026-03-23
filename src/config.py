"""
All configuration models and result dataclasses for the IGA Role Mining pipeline.

Pydantic models (PipelineConfig and sub-configs) live alongside result dataclasses
(TierResult, AlgorithmResult, PipelineResult) so all pipeline contracts are in one place.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from scipy.sparse import csr_matrix


# ── Algorithm sub-configs ──────────────────────────────────────────────────────

class LouvainConfig(BaseModel):
    """Louvain graph community-detection settings."""
    resolution:  float = Field(1.0,  ge=0.1, le=10.0, description="Higher = more, smaller clusters")
    seed:        int   = Field(42,   description="Random seed for reproducibility")
    batch_size:  int   = Field(2000, gt=0,   description="Similarity rows per RAM batch (batch × n_users × 4 B)")


class LeidenConfig(BaseModel):
    """Leiden graph community-detection settings."""
    resolution: float = Field(1.0,  ge=0.1, le=10.0, description="Higher = more, smaller clusters")
    seed:       int   = Field(42,            description="Random seed for reproducibility")
    batch_size: int   = Field(2000, gt=0,    description="Similarity rows per RAM batch (batch × n_users × 4 B)")


class NMFConfig(BaseModel):
    """Non-negative Matrix Factorization settings."""
    n_roles:   Optional[int] = Field(None, description="None = auto-detect via SVD elbow")
    min_roles: int           = Field(3,    gt=0, description="Lower bound for auto-detection")
    max_roles: int           = Field(30,   gt=0, description="Upper bound for auto-detection")
    max_iter:  int           = Field(500,  gt=0, description="Maximum NMF iterations")
    seed:      int           = Field(42,          description="Random seed")
    membership_threshold: float = Field(0.10, ge=0.0, le=1.0,
        description="Minimum normalised W-row fraction for a component to be assigned to a user. "
                    "Set to 0.0 to assign every component; the dominant component is always assigned.")


class HierarchyConfig(BaseModel):
    """Role hierarchy tier discovery settings."""
    # Tier 1 — Staff root
    staff_min_prevalence:          float = Field(0.95, ge=0.0, le=1.0,
        description="Global prevalence floor for Tier-1 Staff grants")
    # Tier 2 — Staff with Tech Access
    tech_baseline_min_prevalence:  float = Field(0.50, ge=0.0, le=1.0,
        description="Tech-user prevalence lower bound for Tier-2 grants")
    tech_baseline_max_prevalence:  float = Field(0.80, ge=0.0, le=1.0,
        description="Tech-user prevalence upper bound for Tier-2 grants")
    # Tier 3 — Business role sub-hierarchy
    business_gap_threshold:        float = Field(0.20, ge=0.0, le=1.0,
        description="Prevalence drop that opens a new sub-tier within a business role")
    business_min_prevalence:       float = Field(0.10, ge=0.0, le=1.0,
        description="Grants below this floor are excluded from hierarchy output")
    business_min_tier_grants:      int   = Field(2, gt=0,
        description="Minimum grants per sub-tier (prevents single-grant micro-tiers)")
    # Orphan / unassigned grants
    orphan_grant_max_role_prevalence: float = Field(0.50, ge=0.0, le=1.0,
        description="A grant is flagged orphan if its prevalence in its best-fit role "
                    "is below this threshold.  0.50 = 'must be held by a majority of "
                    "a role's members to be claimed'; lower = fewer orphans.")


# ── Population filter ──────────────────────────────────────────────────────────

class PopulationFilter(BaseModel):
    """
    Define the staff population to mine roles for.

    Two modes — HR attributes or an explicit user list:
      • HR attributes: intersect any combination of MS/MG org levels,
        job function, region, country.
      • Explicit list: provide ritsids directly (e.g. uploaded from a CSV).
        When ritsids are present, all HR attribute filters are ignored.

    An empty filter (all fields empty) means "all users".
    """
    # Org hierarchy cross-sections
    ms_levels:     dict[str, list[str]] = Field(default_factory=dict,
        description="MS org hierarchy filters: {ms_descr_l01: ['ICG', 'PBWM']}")
    mg_levels:     dict[str, list[str]] = Field(default_factory=dict,
        description="MG geo hierarchy filters: {mg_descr_l01: ['NAM']}")
    # Job dimensions
    job_functions: list[str]            = Field(default_factory=list)
    job_families:  list[str]            = Field(default_factory=list)
    job_codes:     list[str]            = Field(default_factory=list)
    # Geography
    regions:       list[str]            = Field(default_factory=list)
    countries:     list[str]            = Field(default_factory=list)
    # Explicit user list (takes precedence when non-empty)
    ritsids:       list[str]            = Field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not any([
            self.ms_levels, self.mg_levels,
            self.job_functions, self.job_families, self.job_codes,
            self.regions, self.countries,
            self.ritsids,
        ])

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter the merged entitlement+employee frame to the defined population.
        Filtering is ritsid-level: once qualifying ritsids are identified from
        the HR columns, all their entitlement rows are kept.
        """
        if self.is_empty:
            return df

        # Explicit list takes full precedence
        if self.ritsids:
            return df[df["ritsid"].isin(set(self.ritsids))]

        # Build per-user HR frame (one row per user) for predicate evaluation
        hr   = df.drop_duplicates("ritsid").set_index("ritsid")
        mask = pd.Series(True, index=hr.index)

        for col, vals in self.ms_levels.items():
            if col in hr.columns and vals:
                mask &= hr[col].isin(vals)

        for col, vals in self.mg_levels.items():
            if col in hr.columns and vals:
                mask &= hr[col].isin(vals)

        _col_filter(hr, mask, "jobfunctiondescription", self.job_functions)
        _col_filter(hr, mask, "jobfamilydescription",   self.job_families)
        _col_filter(hr, mask, "jobcode",                self.job_codes)
        _col_filter(hr, mask, "region",                 self.regions)
        _col_filter(hr, mask, "country",                self.countries)

        qualifying = set(hr.index[mask])
        return df[df["ritsid"].isin(qualifying)]


def _col_filter(hr: pd.DataFrame, mask: pd.Series, col: str, vals: list) -> None:
    """In-place AND of mask with col.isin(vals) when vals is non-empty."""
    if vals and col in hr.columns:
        mask &= hr[col].isin(vals)


# ── Top-level pipeline config ──────────────────────────────────────────────────

class PipelineConfig(BaseModel):
    """
    Complete configuration for the IGA Role Mining pipeline.

    All sub-configs have sensible defaults so a minimal instantiation
    (just the CSV paths) is enough to run.
    """
    # ── Data sources ──────────────────────────────────────────────────────────
    csv_entitlements: str           = "sample_data/sample_entitlements.csv"
    csv_employees:    str           = "sample_data/sample_employees.csv"
    csv_applications: Optional[str] = "sample_data/sample_applications.csv"
    output_dir:       str           = "output"

    # ── Sampling ──────────────────────────────────────────────────────────────
    sample_size:      Optional[int] = Field(None, description="Limit to N employees (None = all)")
    min_cluster_size: int           = Field(3, gt=0, description="Clusters smaller than this are excluded")

    # ── Entitlement normalisation ─────────────────────────────────────────────
    # Keys are Python regex patterns matched against tranid; values are the
    # canonical replacement.  When a tranid is changed by an alias rule its
    # csiid is automatically dropped from the grant key so that the same
    # logical access from different app instances collapses into one matrix column.
    tranid_aliases: dict[str, str] = Field(
        default_factory=lambda: {
            # Split AD groups: NAM\O365-E3-License, -2, -3, … → one grant
            r"^NAM\\O365-E3-License(-\d+)?$": r"O365-E3-License",
            # Regional internet access variants → one grant
            # r"^(EUR|NAM|LAC)\\app_(EUR|NAM|LAC)_Standard_Internet_Access$": r"app_Standard_Internet_Access",
            # Email-config variants (uncomment and adjust to your data):
            # r"^(IUO|INBOUND|INBOUNDSMALL|INBOUNDOUTBOUND)$": r"INBOUNDOUTBOUND",
        },
        description="Regex-to-canonical tranid normalization rules",
    )

    # ── Tier definitions ──────────────────────────────────────────────────────
    tier_definitions_file: Optional[str] = Field(
        None,
        description="Path to tier_definitions.csv (tier, tranid[, notes]). "
                    "None = use dynamic prevalence-based discovery.",
    )

    # ── Sub-configs ───────────────────────────────────────────────────────────
    hierarchy:         HierarchyConfig  = Field(default_factory=HierarchyConfig)
    louvain:           LouvainConfig    = Field(default_factory=LouvainConfig)
    leiden:            LeidenConfig     = Field(default_factory=LeidenConfig)
    nmf:               NMFConfig        = Field(default_factory=NMFConfig)
    population_filter: PopulationFilter = Field(default_factory=PopulationFilter)

    # ── Algorithm selection ───────────────────────────────────────────────────
    enabled_algorithms: list[str] = Field(
        default_factory=lambda: ["louvain", "leiden", "nmf"],
        description="Ordered list of algorithm names to run (must be registered)",
    )

    # ── HR column names ───────────────────────────────────────────────────────
    segment_cols:    list[str] = Field(
        default_factory=lambda: [f"ms_descr_l{i:02d}" for i in range(1, 11)])
    geo_cols:        list[str] = Field(
        default_factory=lambda: [f"mg_descr_l{i:02d}" for i in range(1, 11)])
    job_cols:        list[str] = Field(
        default_factory=lambda: ["jobcode", "jobdescription",
                                 "jobfamilydescription", "jobfunctiondescription"])
    geo_direct_cols: list[str] = Field(
        default_factory=lambda: ["country", "work_country", "region"])

    # ── Top-tranid report ─────────────────────────────────────────────────────
    top_tranid_ignore_csiid: bool = Field(
        True,
        description="Collapse same-tranid entries from different csiids in the top-tranid report",
    )


# ── Result dataclasses ─────────────────────────────────────────────────────────
# Using dataclasses (not Pydantic) because they hold numpy/pandas objects
# that Pydantic cannot serialise natively.

@dataclass
class TierResult:
    """Output of TierDiscovery: the three-tier grant classification."""
    staff_grants:          set[str]
    tech_baseline_grants:  set[str]
    hierarchy_rows:        list[dict]          # rows for role_hierarchy_tiers.csv
    residual_matrix:       csr_matrix          # matrix with Tier-1+2 cols removed
    residual_grant_index:  list[str]           # grant_ids in residual_matrix column order


@dataclass
class AlgorithmResult:
    """All outputs produced by one role mining algorithm."""
    method:         str
    assignments:      Optional[pd.DataFrame] = None  # ritsid → role_id, one row per user (primary/dominant role)
    memberships:      Optional[pd.DataFrame] = None  # ritsid → role_id, one row per (user, role) pair (1:N)
    profiles:         Optional[pd.DataFrame] = None  # role_profiles.csv
    entitlements:     Optional[pd.DataFrame] = None  # role_entitlements.csv
    biz_hierarchy:    Optional[pd.DataFrame] = None  # business_role_hierarchy.csv
    unassigned_users: Optional[pd.DataFrame] = None  # ritsid of users in sub-threshold clusters
    orphan_grants:    Optional[pd.DataFrame] = None  # grants not core to any role
    @property
    def n_roles(self) -> int:
        if self.profiles is None or self.profiles.empty:
            return 0
        return len(self.profiles)


@dataclass
class PipelineResult:
    """Aggregated output of the full pipeline run."""
    algorithm_results: dict[str, AlgorithmResult] = field(default_factory=dict)
    tier_result:       Optional[TierResult]        = None
    top_tranids:       Optional[pd.DataFrame]      = None
    df:                Optional[pd.DataFrame]      = None   # merged entitlement+HR frame
    matrix:            Optional[csr_matrix]        = None
    user_index:        Optional[list[str]]         = None
    grant_index:       Optional[list[str]]         = None
    elapsed_seconds:   float                       = 0.0

    @property
    def n_users(self) -> int:
        return len(self.user_index) if self.user_index else 0

    @property
    def n_grants(self) -> int:
        return len(self.grant_index) if self.grant_index else 0

    @property
    def n_roles(self) -> int:
        return sum(ar.n_roles for ar in self.algorithm_results.values())

    def unified_hierarchy(self) -> Optional[pd.DataFrame]:
        """
        Combine Tier 1/2 and all business-role hierarchies into one DataFrame
        with a consistent column set — ready for display or export.
        """
        _COLS = [
            "method", "tier", "tier_name", "cluster_id", "role_name", "parent_role_name",
            "member_count", "sub_tier", "sub_role_name", "grant_rank",
            "grant_id", "grant_id_normalized", "prevalence", "descrtx", "appname",
        ]
        frames = []

        if self.tier_result and self.tier_result.hierarchy_rows:
            h = pd.DataFrame(self.tier_result.hierarchy_rows).copy()
            h["method"]        = "Global"
            h["cluster_id"]    = ""
            h["role_name"]     = h["tier_name"]
            h["member_count"]  = pd.NA
            h["sub_tier"]      = pd.NA
            h["sub_role_name"] = h["tier_name"]
            h["grant_rank"]    = pd.NA
            for c in _COLS:
                if c not in h.columns:
                    h[c] = pd.NA
            frames.append(h[_COLS])

        for ar in self.algorithm_results.values():
            if ar.biz_hierarchy is not None and not ar.biz_hierarchy.empty:
                b = ar.biz_hierarchy.copy()
                b["tier"]      = 3
                b["tier_name"] = "Business Role"
                for c in _COLS:
                    if c not in b.columns:
                        b[c] = pd.NA
                frames.append(b[_COLS])

        return pd.concat(frames, ignore_index=True) if frames else None
