"""
IGA Role Mining Module
======================
Based on US 12,309,164 B1 — Kondra et al. (Citigroup Inc.)

Discovers candidate enterprise roles from current entitlement data
using two complementary approaches:
  1. Graph Community Detection (Louvain) — finds natural user clusters
     by shared entitlement structure.
  2. Non-negative Matrix Factorization (NMF) — decomposes the
     user-entitlement matrix into latent role components.

Each discovered role candidate is characterized by:
  - Core entitlement set (what access defines the role)
  - HR attribute profile (reporting hierarchy, geography, job family/function)
  - Coverage and cohesion metrics

Data source: CSV files only (employees, entitlements, applications).

Grant key rule (from ebac_schema.md):
  adguid IS NULL     →  (csiid, tranid, descrtx, entitlecd)
  adguid IS NOT NULL →  (csiid, tranid, descrtx)   ← entitlecd is ignored

Usage:
    python role_miner.py                          # uses defaults in CONFIG
    python role_miner.py --ents PATH --hr PATH    # override CSV paths
    python role_miner.py --help
"""

# ── stdlib ────────────────────────────────────────────────────────────────────
import sys
import argparse
import logging
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")

# ── third-party ───────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import NMF
from sklearn.preprocessing import normalize

try:
    import networkx as nx
except ImportError:
    sys.exit("pip install networkx")

try:
    import community as community_louvain   # python-louvain package
except ImportError:
    sys.exit("pip install python-louvain")

# ── logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("role_miner")

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION — edit paths and tuning knobs before running
# ──────────────────────────────────────────────────────────────────────────────
CONFIG = {
    # ── Input CSV paths ───────────────────────────────────────────────────────
    "CSV_ENTITLEMENTS": "sample_data/sample_entitlements.csv",
    "CSV_EMPLOYEES":    "sample_data/sample_employees.csv",
    "CSV_APPLICATIONS": "sample_data/sample_applications.csv",   # optional

    # ── Sampling (None = use all rows) ────────────────────────────────────────
    # Set to an integer to limit the number of employees processed.
    "SAMPLE_SIZE": None,

    # ── Minimum employees per role candidate ─────────────────────────────────
    # Clusters smaller than this are labelled "micro" and excluded from output.
    "MIN_CLUSTER_SIZE": 3,

    # ── Louvain tuning ────────────────────────────────────────────────────────
    "LOUVAIN_RESOLUTION": 1.0,   # higher = more, smaller clusters
    "LOUVAIN_SEED":       42,
    # Rows of the similarity matrix computed at once.  Each batch uses
    # batch_size × n_users × 4 bytes of RAM.  At 2 000 × 450 000 that is
    # ~3.6 GB — safe on a 38 GB machine.  Reduce if you hit memory pressure.
    "LOUVAIN_BATCH_SIZE": 2000,

    # ── NMF tuning ────────────────────────────────────────────────────────────
    # Set to None to auto-detect (uses SVD elbow method).
    "NMF_N_ROLES":    None,
    "NMF_MIN_ROLES":  3,
    "NMF_MAX_ROLES":  30,
    "NMF_MAX_ITER":   500,
    "NMF_SEED":       42,

    # ── Output directory ──────────────────────────────────────────────────────
    "OUTPUT_DIR": "output",

    # ── HR grouping dimensions used in role profiling ─────────────────────────
    # Columns used to characterise each discovered role cluster.
    "SEGMENT_COLS": [f"ms_descr_l{i:02d}" for i in range(1, 11)],
    "GEO_COLS":     [f"mg_descr_l{i:02d}" for i in range(1, 11)],
    "JOB_COLS": [
        "jobcode",
        "jobdescription",
        "jobfamilydescription",
        "jobfunctiondescription",
    ],
    "GEO_DIRECT_COLS": ["country", "work_country", "region"],

    # ── Scope discovery (Pass 2) ──────────────────────────────────────────────
    # A non-core grant is flagged as scope-specific when the lift of any single
    # HR attribute value exceeds this threshold.
    #   lift(G, A=V) = P(has G | A=V)  /  P(has G | A≠V)
    # A lift of 5 means members with attribute value V are 5× more likely to
    # hold grant G than members without it — strong evidence of a scope.
    "SCOPE_LIFT_THRESHOLD":  5.0,

    # Prevalence window for scope candidates.
    # Grants below MIN are too rare (individual exceptions).
    # Grants above MAX are universal template grants — not scoped.
    "SCOPE_MIN_PREVALENCE":  0.10,
    "SCOPE_MAX_PREVALENCE":  0.85,

    # A scope sub-group must contain at least this many members to be reported.
    "SCOPE_MIN_GROUP_SIZE":  2,
 
    # ── Algorithm switches ────────────────────────────────────────────────
    # Louvain builds a 450K × 450K user-similarity matrix which requires
    # significant RAM.  Disable on large datasets (> ~100K employees) unless
    # you have sufficient memory.  NMF runs on the sparse matrix and is safe
    # to leave enabled at any scale.
    "ENABLE_LOUVAIN": True,
    "ENABLE_NMF":     True,

    # ── Role hierarchy tiers ──────────────────────────────────────────────
    # Staff (root): grants where adguid="" AND global prevalence >= threshold.
    "HIER_STAFF_MIN_PREVALENCE":          0.95,

    # Staff-with-Tech-Access: among adguid-present users, grants whose
    # prevalence falls within [MIN, MAX] define the tech-baseline parent role.
    "HIER_TECH_BASELINE_MIN_PREVALENCE":  0.50,
    "HIER_TECH_BASELINE_MAX_PREVALENCE":  0.80,

    # ── Business Role Sub-hierarchy (Pass 3) ──────────────────────────────
    # Minimum drop in prevalence between two consecutive grants (sorted desc)
    # that triggers a new sub-tier boundary within a business role cluster.
    # Example: 0.20 means a 20-point drop opens a new child tier.
    "HIER_BUSINESS_GAP_THRESHOLD":   0.20,
    # Grants below this prevalence floor are excluded from the hierarchy output.
    "HIER_BUSINESS_MIN_PREVALENCE":  0.10,
    # A sub-tier must contain at least this many grants; otherwise it is merged
    # into the previous tier (prevents single-grant micro-tiers).
    "HIER_BUSINESS_MIN_TIER_GRANTS": 2,

    # ── Entitlement normalization (tranid aliasing) ───────────────────────────
    # Maps regex patterns (matched against tranid) → canonical tranid value.
    # Use th^is when the same logical access is split across multiple AD groups
    # due to org size (e.g. NAM\O365-E3-License, -2, -3, -4 all grant the same
    # access and should be treated as one entitlement during role mining).
    #
    # Keys are Python regex full-match patterns; values are the canonical
    # replacement string.  Patterns are tested in order; first match wins.
    #
    # Example:
    #   r"^NAM\\O365-E3-License(-\d+)?$": r"NAM\O365-E3-License"
    #   matches  NAM\O365-E3-License
    #            NAM\O365-E3-License-2
    #            NAM\O365-E3-License-3  … and normalises all to the base name.
    "TRANID_ALIASES": {
        r"^NAM\\O365-E3-License(-\d+)?$": r"NAM\O365-E3-License",
    },
}

# ──────────────────────────────────────────────────────────────────────────────
# 1. DATA LOADING
# ──────────────────────────────────────────────────────────────────────────────

def load_data(cfg: dict) -> pd.DataFrame:
    """
    Load employees, entitlements, and (optionally) applications CSVs,
    merge them, apply the grant key rule, and return a clean flat frame.

    Grant key rule
    --------------
    adguid IS NULL     →  grant_id = csiid|tranid|descrtx|entitlecd
    adguid IS NOT NULL →  grant_id = csiid|tranid|descrtx
    """
    # ── Load raw CSVs ─────────────────────────────────────────────────────────
    log.info("Loading entitlements: %s", cfg["CSV_ENTITLEMENTS"])
    ent = pd.read_csv(cfg["CSV_ENTITLEMENTS"], dtype=str).fillna("")

    log.info("Loading employees:    %s", cfg["CSV_EMPLOYEES"])
    emp = pd.read_csv(cfg["CSV_EMPLOYEES"],    dtype=str).fillna("")

    # Normalise column names to lowercase
    ent.columns = ent.columns.str.lower().str.strip()
    emp.columns = emp.columns.str.lower().str.strip()

    # Optional applications table enriches grant descriptions
    app = None
    if cfg.get("CSV_APPLICATIONS"):
        app_path = Path(cfg["CSV_APPLICATIONS"])
        if app_path.exists():
            log.info("Loading applications: %s", app_path)
            app = pd.read_csv(app_path, dtype=str).fillna("")
            app.columns = app.columns.str.lower().str.strip()

    # ── Validate required columns ─────────────────────────────────────────────
    _require_cols(ent, ["ritsid", "csiid", "tranid", "descrtx", "entitlecd", "adguid"],
                  source="entitlements CSV")
    _require_cols(emp, ["ritsid", "empid", "dept_mgr_geid"],
                  source="employees CSV")

    # ── Normalise split AD groups → canonical tranid ─────────────────────────
    aliases = cfg.get("TRANID_ALIASES", {})
    if aliases:
        original = ent["tranid"].copy()
        for pattern, replacement in aliases.items():
            ent["tranid"] = ent["tranid"].str.replace(
                pattern, replacement, regex=True
            )
        changed = (ent["tranid"] != original).sum()
        if changed:
            log.info("TRANID_ALIASES: normalised %d tranid values to canonical form", changed)
            # Pick up descrtx from the canonical (un-aliased) records so all
            # aliased rows share the same descrtx as the base record.
            canonical_mask = ent["tranid"] == original
            canonical_descrtx = (
                ent[canonical_mask][["tranid", "descrtx"]]
                .drop_duplicates("tranid")
                .set_index("tranid")["descrtx"]
                .to_dict()
            )
            aliased_mask = ~canonical_mask
            ent.loc[aliased_mask, "descrtx"] = (
                ent.loc[aliased_mask, "tranid"]
                .map(canonical_descrtx)
                .fillna(ent.loc[aliased_mask, "descrtx"])
            )

    # ── Apply grant key rule ──────────────────────────────────────────────────
    # adguid="" (empty string after fillna) is treated the same as NULL.
    no_guid  = ent["adguid"].eq("")
    ent["grant_id"] = np.where(
        no_guid,
        ent["csiid"] + "|" + ent["tranid"] + "|" + ent["descrtx"] + "|" + ent["entitlecd"],
        ent["csiid"] + "|" + ent["tranid"] + "|" + ent["descrtx"],
    )
    # entitlecd_eff: populated only when it contributed to the key
    ent["entitlecd_eff"] = np.where(no_guid, ent["entitlecd"], "")

    log.info("Entitlement rows: %d  |  unique grants: %d  |  unique users: %d",
             len(ent), ent["grant_id"].nunique(), ent["ritsid"].nunique())

    # ── Optional sampling ─────────────────────────────────────────────────────
    sample_size = cfg.get("SAMPLE_SIZE")
    if sample_size and sample_size < emp["ritsid"].nunique():
        sampled_ids = emp["ritsid"].drop_duplicates().sample(
            n=sample_size, random_state=42
        )
        emp = emp[emp["ritsid"].isin(sampled_ids)]
        ent = ent[ent["ritsid"].isin(sampled_ids)]
        log.info("Sampled to %d employees", len(emp))

    # ── Join entitlements → employees ─────────────────────────────────────────
    df = ent.merge(emp, on="ritsid", how="inner", suffixes=("_ent", "_emp"))

    # ── Optionally join application name ─────────────────────────────────────
    if app is not None and "csiid" in app.columns and "appname" in app.columns:
        df = df.merge(app[["csiid", "appname"]], on="csiid", how="left")
    else:
        df["appname"] = df["csiid"]   # fall back to csiid if no app table

    log.info("Merged frame: %d rows  |  %d employees  |  %d unique grants",
             len(df), df["ritsid"].nunique(), df["grant_id"].nunique())
    return df


def _require_cols(df: pd.DataFrame, cols: list, source: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{source} is missing columns: {missing}")


# ──────────────────────────────────────────────────────────────────────────────
# 2. USER-ENTITLEMENT MATRIX
# ──────────────────────────────────────────────────────────────────────────────

def build_user_entitlement_matrix(df: pd.DataFrame):
    """
    Build a binary sparse matrix  M[user, grant]
    where M[i,j] = 1 if user i holds grant j.

    Returns
    -------
    matrix     : csr_matrix  (n_users × n_grants)
    user_index : list of ritsid values (row order)
    grant_index: list of grant_id values (column order)
    """
    users  = sorted(df["ritsid"].unique())
    grants = sorted(df["grant_id"].unique())
    u_idx  = {u: i for i, u in enumerate(users)}
    g_idx  = {g: i for i, g in enumerate(grants)}

    pairs = df[["ritsid", "grant_id"]].drop_duplicates()
    row_idx = pairs["ritsid"].map(u_idx).to_numpy(dtype=np.int32)
    col_idx = pairs["grant_id"].map(g_idx).to_numpy(dtype=np.int32)
    data    = np.ones(len(row_idx), dtype=np.float32)
    mat     = csr_matrix((data, (row_idx, col_idx)),
                         shape=(len(users), len(grants)))

    log.info("Matrix: %d users × %d grants  (density %.3f%%)",
             len(users), len(grants),
             100.0 * mat.nnz / (len(users) * len(grants)))
    return mat, users, grants


# ──────────────────────────────────────────────────────────────────────────────
# 3. TOP TRANID BY POPULATION
# ──────────────────────────────────────────────────────────────────────────────

def top_tranid_by_population(df: pd.DataFrame, n: int = 50) -> pd.DataFrame:
    """
    Return the top-n grants ranked by how many distinct employees (ritsid)
    hold them, using the full grant key as the grouping unit.

    Grant key rule (mirrors load_data):
        adguid IS NULL     →  (csiid, tranid, descrtx, entitlecd)
        adguid IS NOT NULL →  (csiid, tranid, descrtx)

    Columns returned
    ----------------
    grant_id        : full grant key string
    csiid           : application identifier
    tranid          : the (possibly normalised) tranid value
    descrtx         : entitlement description
    entitlecd       : entitlement code (blank when adguid is present)
    user_count      : distinct ritsid values holding this grant
    population_pct  : user_count / total_distinct_ritsid  (0–100)
    adguid_null_pct : % of holders where adguid = ""
    """
    # Ensure string comparison works regardless of categorical dtype
    adguid = df["adguid"].astype(str)
    no_guid = adguid.eq("")

    # ── Compute grant_id using the grant key rule ─────────────────────────
    df = df.copy()
    df["grant_id"] = np.where(
        no_guid,
        df["csiid"].astype(str) + "|" + df["tranid"].astype(str) + "|" +
        df["descrtx"].astype(str) + "|" + df["entitlecd"].astype(str),
        df["csiid"].astype(str) + "|" + df["tranid"].astype(str) + "|" +
        df["descrtx"].astype(str),
    )
    df["entitlecd_eff"] = np.where(no_guid, df["entitlecd"].astype(str), "")

    total_users = df["ritsid"].nunique()

    # ── Per-grant population stats ────────────────────────────────────────
    base = df[["ritsid", "grant_id", "adguid"]].drop_duplicates(
        subset=["ritsid", "grant_id"]
    )

    user_counts = (
        base.groupby("grant_id")["ritsid"]
        .nunique()
        .rename("user_count")
    )

    adguid_null = (
        base.groupby("grant_id")["adguid"]
        .apply(lambda s: round(100.0 * s.astype(str).eq("").sum() / len(s), 1))
        .rename("adguid_null_pct")
    )

    stats = (
        pd.concat([user_counts, adguid_null], axis=1)
        .reset_index()
        .sort_values("user_count", ascending=False)
        .head(n)
        .assign(population_pct=lambda d: (d["user_count"] / total_users * 100).round(1))
    )

    top_grant_set = set(stats["grant_id"])

    # ── Attach component columns from the first matching raw row ──────────
    meta_cols = ["grant_id", "csiid", "tranid", "descrtx", "entitlecd_eff"]
    meta = (
        df.loc[df["grant_id"].isin(top_grant_set), meta_cols]
        .drop_duplicates("grant_id")
        .rename(columns={"entitlecd_eff": "entitlecd"})
    )

    result = (
        stats
        .merge(meta, on="grant_id", how="left")
        .sort_values("user_count", ascending=False)
        [["grant_id", "csiid", "tranid", "descrtx", "entitlecd",
          "user_count", "population_pct", "adguid_null_pct"]]
        .reset_index(drop=True)
    )

    log.info("Top-%d grants: highest coverage %.1f%%  lowest %.1f%%",
             n,
             stats["population_pct"].iloc[0],
             stats["population_pct"].iloc[-1])
    return result


# ──────────────────────────────────────────────────────────────────────────────
# 4. ORG GRAPH  (reporting hierarchy)
# ──────────────────────────────────────────────────────────────────────────────

def build_org_graph(df: pd.DataFrame) -> nx.DiGraph:
    """
    Build a directed org graph: manager → direct_report.

    Critical detail (from ebac_schema.md):
      dept_mgr_geid stores the manager's EMPID, not ritsid.
      We need a two-step lookup:
        Step 1 — build empid → ritsid map from employees data.
        Step 2 — resolve dept_mgr_geid (empid) → manager's ritsid.
    """
    hr = df[["ritsid", "empid", "dept_mgr_geid"]].drop_duplicates("ritsid")
    ritsids = set(hr["ritsid"].dropna())

    # Step 1: empid → ritsid lookup
    empid_to_ritsid = (
        hr[["empid", "ritsid"]]
        .dropna(subset=["empid", "ritsid"])
        .drop_duplicates("empid")
        .set_index("empid")["ritsid"]
        .to_dict()
    )

    # Step 2: resolve manager
    edges = hr[["dept_mgr_geid", "ritsid"]].dropna(subset=["dept_mgr_geid"]).copy()
    edges = edges[edges["dept_mgr_geid"] != ""]
    edges["manager_ritsid"] = edges["dept_mgr_geid"].map(empid_to_ritsid)
    edges = edges.dropna(subset=["manager_ritsid"])
    edges = edges[edges["manager_ritsid"].isin(ritsids)]

    G = nx.DiGraph()
    G.add_nodes_from(ritsids)
    G.add_edges_from(zip(edges["manager_ritsid"], edges["ritsid"]))

    resolved = len(edges)
    total    = hr["dept_mgr_geid"].ne("").sum()
    log.info("Org graph: %d nodes  %d edges  (resolved %d/%d managers)",
             G.number_of_nodes(), G.number_of_edges(), resolved, total)
    return G


# ──────────────────────────────────────────────────────────────────────────────
# 4. ROLE HIERARCHY  (Staff root → Tech Baseline → Business roles)
# ──────────────────────────────────────────────────────────────────────────────

def discover_hierarchy_grants(df: pd.DataFrame,
                               matrix: csr_matrix,
                               user_index: list,
                               grant_index: list,
                               cfg: dict) -> dict:
    """
    Identify grants belonging to each tier of the role hierarchy and return
    a residual matrix (with tier-1 and tier-2 grants stripped) for use as
    the clustering input so that business-role discovery is not dominated by
    near-universal tech grants.

    Tier 1 — Staff (root)
        adguid = ""  AND  global prevalence >= HIER_STAFF_MIN_PREVALENCE
        Typically: building access, site badge — no technology entitlements.

    Tier 2 — Staff with Tech Access (tech baseline)
        adguid != ""  AND  prevalence among tech users within
        [HIER_TECH_BASELINE_MIN_PREVALENCE, HIER_TECH_BASELINE_MAX_PREVALENCE]
        Typically: Domain, password reset, O365, VPN, Remote Access.

    Tier 3 — Business roles (residual)
        Everything else — fed into Louvain / NMF clustering.

    Returns
    -------
    dict with keys:
        staff_grants         : set of grant_ids in the Staff root tier
        tech_baseline_grants : set of grant_ids in the Tech Baseline tier
        residual_matrix      : csr_matrix with Tier-1 + Tier-2 columns removed
        residual_grant_index : grant_index list filtered to residual grants only
        hierarchy_rows       : list of dicts for hierarchy_tiers.csv output
    """
    staff_min_prev = cfg["HIER_STAFF_MIN_PREVALENCE"]
    tech_min_prev  = cfg["HIER_TECH_BASELINE_MIN_PREVALENCE"]
    tech_max_prev  = cfg["HIER_TECH_BASELINE_MAX_PREVALENCE"]

    u_idx   = {u: i for i, u in enumerate(user_index)}
    n_users = len(user_index)

    # ── Tier 1: Staff root grants ─────────────────────────────────────────
    no_adguid_users = set(df[df["adguid"].eq("")]["ritsid"].unique())
    no_adguid_rows  = [u_idx[u] for u in no_adguid_users if u in u_idx]

    global_counts = np.asarray(matrix.sum(axis=0)).flatten()
    global_prev   = global_counts / n_users

    no_adguid_counts = (
        np.asarray(matrix[no_adguid_rows, :].sum(axis=0)).flatten()
        if no_adguid_rows else np.zeros(len(grant_index))
    )

    staff_mask   = (no_adguid_counts > 0) & (global_prev >= staff_min_prev)
    staff_grants = {grant_index[j] for j in np.where(staff_mask)[0]}
    log.info("Hierarchy — Tier 1 Staff root grants    : %d  (global prev ≥ %.0f%%)",
             len(staff_grants), staff_min_prev * 100)

    # ── Tier 2: Tech Baseline grants ──────────────────────────────────────
    tech_users = set(df[df["adguid"].ne("")]["ritsid"].unique())
    tech_rows  = [u_idx[u] for u in tech_users if u in u_idx]
    n_tech     = len(tech_rows)

    tech_prev = (
        np.asarray(matrix[tech_rows, :].sum(axis=0)).flatten() / n_tech
        if tech_rows else np.zeros(len(grant_index))
    )

    tech_mask            = (~staff_mask
                            & (tech_prev >= tech_min_prev)
                            & (tech_prev <= tech_max_prev))
    tech_baseline_grants = {grant_index[j] for j in np.where(tech_mask)[0]}
    log.info("Hierarchy — Tier 2 Tech Baseline grants : %d  (tech prev %.0f%%–%.0f%%)",
             len(tech_baseline_grants), tech_min_prev * 100, tech_max_prev * 100)

    # ── Residual matrix for Tier 3 clustering ────────────────────────────
    exclude              = staff_grants | tech_baseline_grants
    keep_cols            = [j for j, g in enumerate(grant_index) if g not in exclude]
    residual_matrix      = matrix[:, keep_cols].tocsr()
    residual_grant_index = [grant_index[j] for j in keep_cols]
    log.info("Hierarchy — Tier 3 residual grants      : %d  (stripped %d)",
             len(residual_grant_index), len(exclude))

    # ── Build hierarchy_rows for CSV output ───────────────────────────────
    # Attach descrtx, appname, and global prevalence.
    grant_meta = (
        df[["grant_id", "descrtx", "appname"]]
        .drop_duplicates("grant_id")
        .set_index("grant_id")
    )
    g_idx_map = {g: j for j, g in enumerate(grant_index)}

    hierarchy_rows = []
    for g in staff_grants:
        j    = g_idx_map.get(g)
        prev = round(float(global_prev[j]), 3) if j is not None else 0.0
        meta = grant_meta.loc[g] if g in grant_meta.index else {}
        hierarchy_rows.append({
            "tier":       1,
            "tier_name":  "Staff",
            "grant_id":   g,
            "prevalence": prev,
            "descrtx":    meta.get("descrtx", "") if isinstance(meta, dict) else meta["descrtx"],
            "appname":    meta.get("appname",  "") if isinstance(meta, dict) else meta["appname"],
        })
    for g in tech_baseline_grants:
        j    = g_idx_map.get(g)
        prev = round(float(global_prev[j]), 3) if j is not None else 0.0
        meta = grant_meta.loc[g] if g in grant_meta.index else {}
        hierarchy_rows.append({
            "tier":       2,
            "tier_name":  "Staff with Tech Access",
            "grant_id":   g,
            "prevalence": prev,
            "descrtx":    meta.get("descrtx", "") if isinstance(meta, dict) else meta["descrtx"],
            "appname":    meta.get("appname",  "") if isinstance(meta, dict) else meta["appname"],
        })

    return {
        "staff_grants":          staff_grants,
        "tech_baseline_grants":  tech_baseline_grants,
        "residual_matrix":       residual_matrix,
        "residual_grant_index":  residual_grant_index,
        "hierarchy_rows":        hierarchy_rows,
    }


# ──────────────────────────────────────────────────────────────────────────────
# PASS 3 — BUSINESS ROLE DISCOVERY WITH SUB-HIERARCHY
# ──────────────────────────────────────────────────────────────────────────────

def _canonical_role_name(
    cluster_id: str,
    profile_row: dict,
    core_grants: list,
    grant_meta: pd.DataFrame,
) -> str:
    """
    Synthesize a human-readable canonical name for a business role cluster.

    Combines up to three distinguishing signals (in priority order):
      1. Dominant org segment  (ms_descr hierarchy level)
      2. Top job function / job description
      3. Top application name from core grants

    Falls back to "Business Role {cluster_id}" when no HR signal is present.
    """
    parts = []

    # 1. Org segment
    seg = (profile_row.get("dominant_segment") or "").strip()
    if seg:
        parts.append(seg)

    # 2. Job dimension — most-specific key first
    for key in ("top_jobfunctiondescription", "top_jobdescription",
                "top_jobfamilydescription", "top_jobcode"):
        raw = (profile_row.get(key) or "").strip()
        if raw:
            first = raw.split("|")[0].strip()   # _top_values returns "A | B | C"
            if first:
                parts.append(first)
                break

    # 3. Top app name from core grants — added when fewer than 2 parts found
    if len(parts) < 2 and core_grants:
        for g in core_grants:
            gid = g["grant_id"]
            if gid in grant_meta.index:
                app = str(grant_meta.loc[gid]["appname"]).strip()
                csi = gid.split("|")[0] if "|" in gid else gid
                if app and app != csi:      # skip when appname fell back to csiid
                    parts.append(app)
                    break

    if not parts:
        return f"Business Role {cluster_id}"

    # Remove consecutive duplicates (case-insensitive)
    deduped = [parts[0]]
    for p in parts[1:]:
        if p.lower() != deduped[-1].lower():
            deduped.append(p)

    return " - ".join(deduped[:3])   # cap at 3 components


def discover_business_role_hierarchy(
    assignments: pd.Series,
    profiles_df: pd.DataFrame,
    df: pd.DataFrame,
    cluster_matrix: csr_matrix,
    user_index: list,
    cluster_grant_index: list,
    cfg: dict,
    method_prefix: str,
) -> pd.DataFrame:
    """
    Pass 3 — Business Role Discovery with Sub-hierarchy Gap Detection.

    For each cluster in ``assignments``:

    1. Compute per-grant prevalence among cluster members.
    2. Generate a canonical role name (HR segment + job function + top app).
    3. Sort grants by prevalence descending; detect *natural gaps* — consecutive
       drops ≥ HIER_BUSINESS_GAP_THRESHOLD — to split into ordered sub-tiers:
         Sub-tier 1  (Core)       highest-prevalence grants
         Sub-tier 2  (Extended)   next band after first gap
         Sub-tier 3  (Specialist) next band after second gap
         Sub-tier N  (Level N)    subsequent bands
       A new sub-tier opens only when the current band already contains at least
       HIER_BUSINESS_MIN_TIER_GRANTS grants (avoids single-grant micro-tiers).

    The resulting structure nests under "Staff with Tech Access" (Tier 2):
        Tier 2 : Staff with Tech Access
          Tier 3 : <role_name>                          ← cluster parent (sub-tier 1)
            Tier 3.2 : <role_name> - Extended           ← sub-tier 2
            Tier 3.3 : <role_name> - Specialist         ← sub-tier 3

    Parameters
    ----------
    assignments         : ritsid → cluster_id  (from run_louvain / run_nmf)
    profiles_df         : DataFrame from analyze_roles (dominant_segment, top_job*)
    df                  : full merged frame (for grant metadata)
    cluster_matrix      : residual sparse matrix (Tier 1 + 2 columns stripped)
    user_index          : row-order list of ritsid
    cluster_grant_index : column-order list of grant_id (residual only)
    cfg                 : CONFIG dict
    method_prefix       : "Louvain" or "NMF"

    Returns
    -------
    pd.DataFrame  columns:
        method, cluster_id, role_name, member_count,
        sub_tier, sub_role_name, grant_rank,
        grant_id, prevalence, descrtx, appname
    """
    gap_thresh = cfg.get("HIER_BUSINESS_GAP_THRESHOLD",   0.20)
    min_prev   = cfg.get("HIER_BUSINESS_MIN_PREVALENCE",  0.10)
    min_grants = cfg.get("HIER_BUSINESS_MIN_TIER_GRANTS", 2)
    min_size   = cfg["MIN_CLUSTER_SIZE"]

    u_idx = {u: i for i, u in enumerate(user_index)}
    grant_meta = (
        df[["grant_id", "descrtx", "appname"]]
        .drop_duplicates("grant_id")
        .set_index("grant_id")
    )

    # Profile lookup for naming: cluster_id → dict of profile columns
    profile_lookup: dict = {}
    if profiles_df is not None and not profiles_df.empty:
        for rec in profiles_df.to_dict("records"):
            profile_lookup[rec["cluster_id"]] = rec

    _SUB_LABELS = {1: "Core", 2: "Extended", 3: "Specialist"}

    rows = []
    n_roles_done = 0

    for cid, members_series in assignments.groupby(assignments):
        members = members_series.index.tolist()
        if len(members) < min_size:
            continue

        member_rows = [u_idx[m] for m in members if m in u_idx]
        if not member_rows:
            continue

        # Per-grant prevalence within this cluster
        sub          = cluster_matrix[member_rows, :]
        grant_counts = np.asarray(sub.sum(axis=0)).flatten()
        prevalence   = grant_counts / len(member_rows)

        # Keep only grants above the floor
        keep = np.where(prevalence >= min_prev)[0]
        if len(keep) == 0:
            continue

        # Sort descending by prevalence
        order       = keep[np.argsort(-prevalence[keep])]
        sorted_prev = prevalence[order]

        # Core grants (≥50%) passed to the name generator
        core_grants = [
            {"grant_id": cluster_grant_index[j], "prevalence": float(prevalence[j])}
            for j in np.where(prevalence >= 0.50)[0]
        ]

        profile_row = profile_lookup.get(cid, {"cluster_id": cid})
        role_name   = _canonical_role_name(cid, profile_row, core_grants, grant_meta)

        # ── Sub-tier assignment via gap detection ─────────────────────────
        sub_tier       = 1
        grants_in_tier = 0

        for rank, j in enumerate(order):
            gid      = cluster_grant_index[j]
            prev_val = float(sorted_prev[rank])

            # Open a new sub-tier when the drop from the previous grant meets
            # the threshold AND the current tier already has enough grants.
            if rank > 0:
                gap = float(sorted_prev[rank - 1]) - prev_val
                if gap >= gap_thresh and grants_in_tier >= min_grants:
                    sub_tier       += 1
                    grants_in_tier  = 0

            tier_label    = _SUB_LABELS.get(sub_tier, f"Level {sub_tier}")
            sub_role_name = (role_name
                             if sub_tier == 1
                             else f"{role_name} - {tier_label}")

            meta    = grant_meta.loc[gid] if gid in grant_meta.index else None
            descrtx = str(meta["descrtx"]) if meta is not None else ""
            appname = str(meta["appname"])  if meta is not None else ""

            rows.append({
                "method":        method_prefix,
                "cluster_id":    cid,
                "role_name":     role_name,
                "member_count":  len(members),
                "sub_tier":      sub_tier,
                "sub_role_name": sub_role_name,
                "grant_rank":    rank + 1,
                "grant_id":      gid,
                "prevalence":    round(prev_val, 3),
                "descrtx":       descrtx,
                "appname":       appname,
            })
            grants_in_tier += 1

        n_roles_done += 1

    result = pd.DataFrame(rows)
    if not result.empty:
        n_sub = int(result.groupby("cluster_id")["sub_tier"].max().sum())
        log.info("[%s] Pass 3 Business Roles: %d roles  %d total sub-tiers  "
                 "(gap=%.0f%%  floor=%.0f%%)",
                 method_prefix, n_roles_done, n_sub,
                 gap_thresh * 100, min_prev * 100)
    else:
        log.info("[%s] Pass 3 Business Roles: 0 roles (check MIN_CLUSTER_SIZE / data)",
                 method_prefix)
    return result


# ──────────────────────────────────────────────────────────────────────────────
# 5. LOUVAIN ROLE MINING
# ──────────────────────────────────────────────────────────────────────────────

def run_louvain(matrix: csr_matrix, user_index: list, cfg: dict) -> pd.Series:
    """
    Build a Jaccard-similarity user graph and run Louvain community detection.
    Returns a pd.Series: ritsid → cluster_id  (string like 'L0', 'L1', …)

    Similarity is computed in row-batches so peak RAM stays at
    batch_size × n_users × 4 bytes rather than n_users² × 4 bytes.
    At LOUVAIN_BATCH_SIZE=2000 and 450K users that is ~3.6 GB per batch.
    """
    log.info("Building user similarity graph for Louvain (batched) …")
    n          = matrix.shape[0]
    batch_size = cfg.get("LOUVAIN_BATCH_SIZE", 2000)
    k          = min(15, n - 1)

    # Row-normalised matrix for cosine/Jaccard approximation
    norms = np.asarray(matrix.sum(axis=1)).flatten()
    norms[norms == 0] = 1
    mat_norm = matrix.multiply(1.0 / norms[:, None]).tocsr()

    G = nx.Graph()
    G.add_nodes_from(range(n))

    n_batches = (n + batch_size - 1) // batch_size
    for b in range(n_batches):
        start = b * batch_size
        end   = min(start + batch_size, n)

        # sim_batch: (end-start) × n_users  — dense, lives briefly then freed
        sim_batch = (mat_norm[start:end] @ mat_norm.T).toarray()

        for local_i, i in enumerate(range(start, end)):
            row        = sim_batch[local_i]
            row[i]     = 0.0          # no self-loops
            top_k      = np.argpartition(row, -k)[-k:]
            for j in top_k:
                if row[j] > 0:
                    G.add_edge(i, j, weight=float(row[j]))

        del sim_batch                 # free batch memory immediately

        if (b + 1) % 50 == 0 or (b + 1) == n_batches:
            log.info("  similarity batch %d/%d  (users %d–%d)",
                     b + 1, n_batches, start, end - 1)

    log.info("Running Louvain (resolution=%.2f) …", cfg["LOUVAIN_RESOLUTION"])
    partition = community_louvain.best_partition(
        G,
        resolution=cfg["LOUVAIN_RESOLUTION"],
        random_state=cfg["LOUVAIN_SEED"],
    )
    assignments = pd.Series(
        {user_index[i]: f"L{cid}" for i, cid in partition.items()}
    )
    log.info("Louvain: %d clusters", assignments.nunique())
    return assignments


# ──────────────────────────────────────────────────────────────────────────────
# 5. NMF ROLE MINING
# ──────────────────────────────────────────────────────────────────────────────

def _auto_n_roles(matrix: csr_matrix, min_r: int, max_r: int) -> int:
    """Estimate number of roles via SVD explained-variance elbow."""
    try:
        from scipy.sparse.linalg import svds
        k    = min(max_r + 5, min(matrix.shape) - 1)
        _, s, _ = svds(matrix.astype(float), k=k)
        s    = np.sort(s)[::-1]
        diffs = np.diff(s)
        elbow = int(np.argmin(diffs[:max_r])) + 1
        return max(min_r, min(elbow, max_r))
    except Exception:
        return max(min_r, min(10, max_r))


def run_nmf(matrix: csr_matrix, user_index: list, cfg: dict) -> pd.Series:
    """
    Decompose user-entitlement matrix with NMF.
    Each user is assigned to the role component with highest activation.
    Returns a pd.Series: ritsid → cluster_id  (string like 'N0', 'N1', …)
    """
    n_roles = cfg.get("NMF_N_ROLES") or _auto_n_roles(
        matrix, cfg["NMF_MIN_ROLES"], cfg["NMF_MAX_ROLES"]
    )
    log.info("Running NMF with %d roles …", n_roles)
    model = NMF(
        n_components=n_roles,
        max_iter=cfg["NMF_MAX_ITER"],
        random_state=cfg["NMF_SEED"],
        init="nndsvda",
    )
    W = model.fit_transform(matrix.astype(float))   # users × roles
    assignments = pd.Series(
        {user_index[i]: f"N{int(np.argmax(W[i]))}" for i in range(len(user_index))}
    )
    log.info("NMF: %d roles", assignments.nunique())
    return assignments


# ──────────────────────────────────────────────────────────────────────────────
# 6. ROLE PROFILING
# ──────────────────────────────────────────────────────────────────────────────

def _top_values(series: pd.Series, n: int = 3) -> str:
    """Return top-n values by frequency as a comma-separated string."""
    counts = series.dropna().replace("", pd.NA).dropna().value_counts()
    return " | ".join(counts.head(n).index.tolist())


def profile_role(cluster_id: str,
                 members: list,
                 df: pd.DataFrame,
                 grant_index: list,
                 matrix: csr_matrix,
                 user_index: list,
                 cfg: dict) -> tuple:
    """
    Characterise a role cluster.

    Returns
    -------
    profile_row : dict  (one row for role_profiles.csv)
    top_grants  : list of dicts  (rows for role_entitlements.csv)
    """
    member_set  = set(members)
    u_idx       = {u: i for i, u in enumerate(user_index)}
    member_rows = [u_idx[m] for m in members if m in u_idx]

    if not member_rows:
        return None, []

    # ── Grant prevalence within cluster ───────────────────────────────────────
    sub = matrix[member_rows, :]
    grant_counts = np.asarray(sub.sum(axis=0)).flatten()
    prevalence   = grant_counts / len(member_rows)

    # Core grants: present in ≥50% of cluster members
    core_mask    = prevalence >= 0.50
    core_count   = int(core_mask.sum())

    top_grant_rows = []
    for j in np.where(prevalence > 0)[0]:
        top_grant_rows.append({
            "cluster_id":   cluster_id,
            "grant_id":     grant_index[j],
            "prevalence":   round(float(prevalence[j]), 3),
            "is_core":      bool(core_mask[j]),
        })
    top_grant_rows.sort(key=lambda r: -r["prevalence"])

    # ── HR attribute profile ──────────────────────────────────────────────────
    hr_sub = df[df["ritsid"].isin(member_set)].drop_duplicates("ritsid")

    seg_cols  = [c for c in cfg["SEGMENT_COLS"] if c in hr_sub.columns]
    geo_cols  = [c for c in cfg["GEO_COLS"]     if c in hr_sub.columns]
    job_cols  = [c for c in cfg["JOB_COLS"]     if c in hr_sub.columns]
    geo_d     = [c for c in cfg["GEO_DIRECT_COLS"] if c in hr_sub.columns]

    # First non-empty segment level that has a majority value
    dominant_segment = ""
    for col in seg_cols:
        vals = hr_sub[col].replace("", pd.NA).dropna()
        if not vals.empty:
            top = vals.value_counts().idxmax()
            coverage = (vals == top).sum() / len(hr_sub)
            if coverage >= 0.40:
                dominant_segment = top
                break

    dominant_geo = ""
    for col in geo_cols:
        vals = hr_sub[col].replace("", pd.NA).dropna()
        if not vals.empty:
            top = vals.value_counts().idxmax()
            coverage = (vals == top).sum() / len(hr_sub)
            if coverage >= 0.40:
                dominant_geo = top
                break

    profile_row = {
        "cluster_id":         cluster_id,
        "member_count":       len(members),
        "core_grant_count":   core_count,
        "dominant_segment":   dominant_segment,
        "dominant_geo":       dominant_geo,
    }

    # Direct geo cols
    for col in geo_d:
        profile_row[f"top_{col}"] = _top_values(hr_sub[col])

    # Job cols
    for col in job_cols:
        profile_row[f"top_{col}"] = _top_values(hr_sub[col])

    return profile_row, top_grant_rows


def analyze_roles(assignments: pd.Series,
                  df: pd.DataFrame,
                  matrix: csr_matrix,
                  user_index: list,
                  grant_index: list,
                  cfg: dict,
                  method_prefix: str) -> tuple:
    """
    Run profile_role() for every cluster that meets MIN_CLUSTER_SIZE.

    Returns (profiles_df, entitlements_df).
    """
    profiles   = []
    all_grants = []
    min_size   = cfg["MIN_CLUSTER_SIZE"]

    for cid, members_series in assignments.groupby(assignments):
        members = members_series.index.tolist()
        if len(members) < min_size:
            continue
        prof, grants = profile_role(
            cid, members, df, grant_index, matrix, user_index, cfg
        )
        if prof:
            profiles.append(prof)
            all_grants.extend(grants)

    profiles_df    = pd.DataFrame(profiles)
    entitlements_df = pd.DataFrame(all_grants)
    log.info("[%s] Profiled %d roles (min_size=%d)", method_prefix,
             len(profiles_df), min_size)
    return profiles_df, entitlements_df


# ──────────────────────────────────────────────────────────────────────────────
# 7. SCOPE DISCOVERY  (Pass 2)
# ──────────────────────────────────────────────────────────────────────────────

def _scope_hr_attrs(df: pd.DataFrame, cfg: dict) -> list:
    """
    Return the ordered list of HR attribute columns to test for scope lift.
    Priority: direct geo → mg hierarchy levels → ms hierarchy levels → job dims.
    Only columns that are actually populated (have at least one non-empty value)
    are included.
    """
    candidates = []
    for col in cfg["GEO_DIRECT_COLS"]:
        if col in df.columns and df[col].replace("", pd.NA).notna().any():
            candidates.append(col)
    for col in cfg["GEO_COLS"] + cfg["SEGMENT_COLS"]:
        if col in df.columns and df[col].replace("", pd.NA).notna().any():
            candidates.append(col)
    for col in cfg["JOB_COLS"]:
        if col in df.columns and df[col].replace("", pd.NA).notna().any():
            candidates.append(col)
    return candidates


def _lift(has_grant: np.ndarray, in_group: np.ndarray) -> float:
    """
    Lift ratio for a single (grant, attribute-value) pair.

        lift = P(grant | in_group)  /  P(grant | not in_group)

    Returns 0.0 when the grant is absent in the sub-group.
    Floors the denominator at 0.01 to avoid division by zero.
    """
    in_mask  = in_group.astype(bool)
    out_mask = ~in_mask
    p_in  = has_grant[in_mask].mean()  if in_mask.sum()  > 0 else 0.0
    p_out = has_grant[out_mask].mean() if out_mask.sum() > 0 else 0.0
    if p_in == 0:
        return 0.0
    return p_in / max(p_out, 0.01)


def discover_scopes(assignments: pd.Series,
                    df: pd.DataFrame,
                    matrix: csr_matrix,
                    user_index: list,
                    grant_index: list,
                    cfg: dict,
                    method_prefix: str) -> tuple:
    """
    Pass 2 — Scope Attribute Discovery.

    For each role cluster found in Pass 1, identify grants whose prevalence
    sits in a mid-range band (not universal, not rare).  For each such grant,
    test every HR attribute and find the value V that maximises:

        lift(G, A=V) = P(has G | A=V) / P(has G | A≠V)

    When the best lift exceeds SCOPE_LIFT_THRESHOLD the grant is flagged as
    scope-specific and (A, V) is its scope attribute.  Grants that share the
    same best (A, V) are grouped into one scope variant.

    Returns
    -------
    scope_profiles  : pd.DataFrame — one row per scope variant
    scope_members   : pd.DataFrame — one row per (member, scope) assignment
    """
    lift_thresh  = cfg["SCOPE_LIFT_THRESHOLD"]
    min_prev     = cfg["SCOPE_MIN_PREVALENCE"]
    max_prev     = cfg["SCOPE_MAX_PREVALENCE"]
    min_grp      = cfg["SCOPE_MIN_GROUP_SIZE"]
    min_size     = cfg["MIN_CLUSTER_SIZE"]

    u_idx        = {u: i for i, u in enumerate(user_index)}
    hr_lookup    = (df[["ritsid"] + [c for c in df.columns if c != "ritsid"]]
                    .drop_duplicates("ritsid")
                    .set_index("ritsid"))
    scope_attrs  = _scope_hr_attrs(df, cfg)

    all_scope_profiles = []
    all_scope_members  = []
    scope_counter      = 0

    for cid, members_series in assignments.groupby(assignments):
        members = members_series.index.tolist()
        if len(members) < min_size:
            continue

        member_rows = [u_idx[m] for m in members if m in u_idx]
        if not member_rows:
            continue

        # Sub-matrix and prevalence for this cluster
        sub       = matrix[member_rows, :]
        g_counts  = np.asarray(sub.sum(axis=0)).flatten()
        prev      = g_counts / len(member_rows)

        # Scope candidate grants: mid-prevalence band
        cand_mask = (prev >= min_prev) & (prev <= max_prev)
        cand_idxs = np.where(cand_mask)[0]
        if len(cand_idxs) == 0:
            continue

        # HR attribute values for this cluster's members
        hr_sub = hr_lookup.reindex([m for m in members if m in hr_lookup.index])

        # For each candidate grant, find best-lift (attr, value)
        # Structure: {(attr, value): [grant_id, ...]}
        scope_grant_map: dict = {}   # (attr, val) → list of grant_ids
        scope_lift_map:  dict = {}   # (attr, val) → list of lifts

        for j in cand_idxs:
            has_g = np.array(sub[:, j].toarray().flatten() > 0, dtype=float)
            # align has_g to hr_sub row order
            has_g_aligned = np.array(
                [has_g[member_rows.index(u_idx[m])]
                 if m in u_idx else 0.0
                 for m in hr_sub.index],
                dtype=float,
            )

            best_lift  = 0.0
            best_attr  = None
            best_val   = None

            for attr in scope_attrs:
                if attr not in hr_sub.columns:
                    continue
                col_vals = hr_sub[attr].replace("", pd.NA).fillna("__missing__")
                for val in col_vals.unique():
                    if val == "__missing__":
                        continue
                    in_grp = (col_vals == val).values.astype(float)
                    if in_grp.sum() < min_grp:
                        continue
                    l = _lift(has_g_aligned, in_grp)
                    if l > best_lift:
                        best_lift = l
                        best_attr = attr
                        best_val  = val

            if best_lift >= lift_thresh and best_attr is not None:
                key = (best_attr, best_val)
                scope_grant_map.setdefault(key, []).append(grant_index[j])
                scope_lift_map.setdefault(key,  []).append(round(best_lift, 2))

        # Build scope variants from grouped (attr, val) keys
        for (attr, val), grants in scope_grant_map.items():
            scope_id   = f"{cid}_S{scope_counter}"
            scope_counter += 1
            avg_lift   = round(float(np.mean(scope_lift_map[(attr, val)])), 2)

            # Members who carry this scope: those with attr=val in HR data
            col_vals   = hr_sub[attr].replace("", pd.NA).fillna("__missing__")
            scope_mbrs = [m for m in hr_sub.index if col_vals.get(m) == val]

            if len(scope_mbrs) < min_grp:
                continue

            all_scope_profiles.append({
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
                all_scope_members.append({
                    "ritsid":              m,
                    "template_cluster_id": cid,
                    "scope_id":            scope_id,
                    "scope_attribute":     attr,
                    "scope_value":         val,
                })

    scope_profiles = pd.DataFrame(all_scope_profiles)
    scope_members  = pd.DataFrame(all_scope_members)

    n_scoped = scope_profiles["template_cluster_id"].nunique() if not scope_profiles.empty else 0
    log.info("[%s] Scope discovery: %d scope variants across %d role templates",
             method_prefix, len(scope_profiles), n_scoped)

    return scope_profiles, scope_members


# ──────────────────────────────────────────────────────────────────────────────
# 8. OUTPUT
# ──────────────────────────────────────────────────────────────────────────────

def save_outputs(louvain_profiles:       pd.DataFrame,
                 louvain_entitlements:    pd.DataFrame,
                 louvain_scope_profiles:  pd.DataFrame,
                 louvain_scope_members:   pd.DataFrame,
                 nmf_profiles:            pd.DataFrame,
                 nmf_entitlements:        pd.DataFrame,
                 nmf_scope_profiles:      pd.DataFrame,
                 nmf_scope_members:       pd.DataFrame,
                 louvain_assignments:     pd.Series,
                 nmf_assignments:         pd.Series,
                 hierarchy_rows:          list,
                 top_tranids:             pd.DataFrame,
                 louvain_biz_hierarchy:   pd.DataFrame,
                 nmf_biz_hierarchy:       pd.DataFrame,
                 cfg: dict):
    out = Path(cfg["OUTPUT_DIR"])
    out.mkdir(parents=True, exist_ok=True)
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")

    def _save(df, name):
        if df is not None and not df.empty:
            p = out / f"{name}_{ts}.csv"
            df.to_csv(p, index=False)
            log.info("Saved %s  (%d rows)", p, len(df))

    # Hierarchy tier grants (Tier 1 = Staff, Tier 2 = Tech Baseline)
    _save(pd.DataFrame(hierarchy_rows), "role_hierarchy_tiers")

    # Top-50 tranid by population coverage
    _save(top_tranids, "top_tranids")

    # Role template outputs
    _save(louvain_profiles,      "louvain_role_profiles")
    _save(louvain_entitlements,  "louvain_role_entitlements")
    _save(nmf_profiles,          "nmf_role_profiles")
    _save(nmf_entitlements,      "nmf_role_entitlements")

    # Business role hierarchy (Pass 3) — per-method detail files
    _save(louvain_biz_hierarchy, "louvain_business_role_hierarchy")
    _save(nmf_biz_hierarchy,     "nmf_business_role_hierarchy")

    # ── Unified role hierarchy  (Tiers 1 → 2 → 3 with sub-tiers) ─────────────
    # Columns shared across all tiers:
    _UNIFIED = [
        "method", "tier", "tier_name", "cluster_id", "role_name", "member_count",
        "sub_tier", "sub_role_name", "grant_rank", "grant_id", "prevalence",
        "descrtx", "appname",
    ]
    unified_frames = []

    hier_df = pd.DataFrame(hierarchy_rows)
    if not hier_df.empty:
        hier_df = hier_df.copy()
        hier_df["method"]        = "Global"
        hier_df["cluster_id"]    = ""
        hier_df["role_name"]     = hier_df["tier_name"]
        hier_df["member_count"]  = pd.NA
        hier_df["sub_tier"]      = pd.NA
        hier_df["sub_role_name"] = hier_df["tier_name"]
        hier_df["grant_rank"]    = pd.NA
        for col in _UNIFIED:
            if col not in hier_df.columns:
                hier_df[col] = pd.NA
        unified_frames.append(hier_df[_UNIFIED])

    for biz_df in (louvain_biz_hierarchy, nmf_biz_hierarchy):
        if biz_df is not None and not biz_df.empty:
            biz = biz_df.copy()
            biz["tier"]      = 3
            biz["tier_name"] = "Business Role"
            for col in _UNIFIED:
                if col not in biz.columns:
                    biz[col] = pd.NA
            unified_frames.append(biz[_UNIFIED])

    if unified_frames:
        unified = pd.concat(unified_frames, ignore_index=True)
        _save(unified, "role_hierarchy_full")

    # Scope outputs
    _save(louvain_scope_profiles, "louvain_scope_profiles")
    _save(louvain_scope_members,  "louvain_scope_members")
    _save(nmf_scope_profiles,     "nmf_scope_profiles")
    _save(nmf_scope_members,      "nmf_scope_members")

    # User-role assignments (both methods side by side)
    base_assignments = louvain_assignments if louvain_assignments is not None else nmf_assignments
    if base_assignments is not None:
        assignments_df = pd.DataFrame({"ritsid": base_assignments.index})
        if louvain_assignments is not None:
            assignments_df["louvain_role"] = louvain_assignments.values
        if nmf_assignments is not None:
            assignments_df["nmf_role"] = assignments_df["ritsid"].map(nmf_assignments.to_dict())

        # Attach louvain scope to assignments where available
        if louvain_scope_members is not None and not louvain_scope_members.empty:
            scope_map = (louvain_scope_members
                         .groupby("ritsid")["scope_id"]
                         .apply(lambda s: " | ".join(sorted(s.unique())))
                         .to_dict())
            assignments_df["louvain_scope"] = assignments_df["ritsid"].map(scope_map)

        _save(assignments_df, "user_role_assignments")

    # Summary text
    summary_path = out / f"summary_{ts}.txt"
    n_lou_scopes = len(louvain_scope_profiles) if louvain_scope_profiles is not None else 0
    n_nmf_scopes = len(nmf_scope_profiles)     if nmf_scope_profiles     is not None else 0
    lines = [
        "IGA Role Mining — Run Summary",
        f"Timestamp : {ts}",
        "=" * 60,
        f"Louvain roles  : {len(louvain_profiles) if louvain_profiles is not None else 'disabled'}",
    ]
    if louvain_profiles is not None and not louvain_profiles.empty:
        lines.append(f"  Avg members    : {louvain_profiles['member_count'].mean():.1f}")
        lines.append(f"  Avg core grants: {louvain_profiles['core_grant_count'].mean():.1f}")
        lines.append(f"  Scope variants : {n_lou_scopes}")
    lines += ["", f"NMF roles      : {len(nmf_profiles) if nmf_profiles is not None else 'disabled'}"]
    if nmf_profiles is not None and not nmf_profiles.empty:
        lines.append(f"  Avg members    : {nmf_profiles['member_count'].mean():.1f}")
        lines.append(f"  Avg core grants: {nmf_profiles['core_grant_count'].mean():.1f}")
        lines.append(f"  Scope variants : {n_nmf_scopes}")
    summary_path.write_text("\n".join(lines))
    log.info("Summary written to %s", summary_path)


# ──────────────────────────────────────────────────────────────────────────────
# 8. PIPELINE
# ──────────────────────────────────────────────────────────────────────────────

def run_pipeline(cfg: dict = None) -> dict:
    """
    End-to-end role mining pipeline.

    Steps
    -----
    1. Load CSVs and build grant keys
    2. Build user-entitlement matrix
    3. Build org graph
    4. Run Louvain community detection  → role templates
    5. Run NMF factorization            → role templates
    6. Profile each role template
    7. Scope discovery (Pass 2)         → scope variants per template
    8. Save outputs

    Returns dict with all intermediate and final results.
    """
    if cfg is None:
        cfg = CONFIG

    log.info("─" * 60)
    log.info("IGA Role Miner — starting")
    log.info("─" * 60)

    # 1. Load
    df = load_data(cfg)

    # 2. Matrix
    matrix, user_index, grant_index = build_user_entitlement_matrix(df)

    # 3. Top tranid by population
    top_tranids = top_tranid_by_population(df)

    # 4. Org graph
    org_graph = build_org_graph(df)

    # 4. Role hierarchy — identify Tier 1 (Staff) + Tier 2 (Tech Baseline)
    #    and strip them from the matrix before clustering so business-role
    #    discovery is not dominated by near-universal tech grants.
    log.info("─" * 40)
    log.info("Role Hierarchy Discovery")
    log.info("─" * 40)
    hier = discover_hierarchy_grants(df, matrix, user_index, grant_index, cfg)
    cluster_matrix      = hier["residual_matrix"]
    cluster_grant_index = hier["residual_grant_index"]

    # 5. Louvain — role templates (on residual matrix)
    louvain_assignments = louvain_profiles = louvain_entitlements = None
    louvain_scope_profiles = louvain_scope_members = None
    if cfg.get("ENABLE_LOUVAIN", False):
        louvain_assignments = run_louvain(cluster_matrix, user_index, cfg)
        louvain_profiles, louvain_entitlements = analyze_roles(
            louvain_assignments, df, cluster_matrix, user_index, cluster_grant_index, cfg, "Louvain"
        )
    else:
        log.info("Louvain disabled (ENABLE_LOUVAIN=False) — skipping")

    # 6. NMF — role templates (on residual matrix)
    nmf_assignments = nmf_profiles = nmf_entitlements = None
    nmf_scope_profiles = nmf_scope_members = None
    if cfg.get("ENABLE_NMF", True):
        nmf_assignments = run_nmf(cluster_matrix, user_index, cfg)
        nmf_profiles, nmf_entitlements = analyze_roles(
            nmf_assignments, df, cluster_matrix, user_index, cluster_grant_index, cfg, "NMF"
        )
    else:
        log.info("NMF disabled (ENABLE_NMF=False) — skipping")

    # Pass 3 — Business Role Hierarchy
    log.info("─" * 40)
    log.info("Pass 3 — Business Role Hierarchy Discovery")
    log.info("─" * 40)
    louvain_biz_hierarchy = nmf_biz_hierarchy = None
    if louvain_assignments is not None and louvain_profiles is not None:
        louvain_biz_hierarchy = discover_business_role_hierarchy(
            louvain_assignments, louvain_profiles, df,
            cluster_matrix, user_index, cluster_grant_index, cfg, "Louvain",
        )
    if nmf_assignments is not None and nmf_profiles is not None:
        nmf_biz_hierarchy = discover_business_role_hierarchy(
            nmf_assignments, nmf_profiles, df,
            cluster_matrix, user_index, cluster_grant_index, cfg, "NMF",
        )

    # Pass 5 — Scope Discovery (HR lift within role clusters)
    log.info("─" * 40)
    log.info("Pass 5 — Scope Discovery (HR lift)")
    log.info("─" * 40)
    if louvain_assignments is not None:
        louvain_scope_profiles, louvain_scope_members = discover_scopes(
            louvain_assignments, df, cluster_matrix, user_index, cluster_grant_index, cfg, "Louvain"
        )
    if nmf_assignments is not None:
        nmf_scope_profiles, nmf_scope_members = discover_scopes(
            nmf_assignments, df, cluster_matrix, user_index, cluster_grant_index, cfg, "NMF"
        )

    # Save all outputs
    save_outputs(
        louvain_profiles,       louvain_entitlements,
        louvain_scope_profiles,  louvain_scope_members,
        nmf_profiles,           nmf_entitlements,
        nmf_scope_profiles,     nmf_scope_members,
        louvain_assignments,    nmf_assignments,
        hier["hierarchy_rows"], top_tranids,
        louvain_biz_hierarchy,  nmf_biz_hierarchy,
        cfg,
    )

    log.info("─" * 60)
    log.info("Done.")
    log.info("─" * 60)

    return {
        "louvain_assignments":     louvain_assignments,
        "nmf_assignments":         nmf_assignments,
        "louvain_profiles":        louvain_profiles,
        "nmf_profiles":            nmf_profiles,
        "louvain_entitlements":    louvain_entitlements,
        "nmf_entitlements":        nmf_entitlements,
        "louvain_biz_hierarchy":   louvain_biz_hierarchy,
        "nmf_biz_hierarchy":       nmf_biz_hierarchy,
        "louvain_scope_profiles":  louvain_scope_profiles,
        "louvain_scope_members":   louvain_scope_members,
        "nmf_scope_profiles":      nmf_scope_profiles,
        "nmf_scope_members":       nmf_scope_members,
        "matrix":                  matrix,
        "user_index":              user_index,
        "grant_index":             grant_index,
        "org_graph":               org_graph,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 9. CLI
# ──────────────────────────────────────────────────────────────────────────────

def load_entitlements_lean(cfg: dict) -> pd.DataFrame:
    """
    Lightweight loader for the --top-tranids path.

    Reads only the 6 columns needed by top_tranid_by_population directly
    from the entitlements CSV — no employee merge, no HR columns.
    All string columns are cast to Categorical to cut RAM usage by ~70%
    versus a full object-dtype load.
    """
    cols_needed = ["ritsid", "tranid", "descrtx", "entitlecd", "csiid", "adguid"]

    log.info("Lean-loading entitlements (columns: %s) …", ", ".join(cols_needed))
    ent = pd.read_csv(
        cfg["CSV_ENTITLEMENTS"],
        dtype=str,
        usecols=lambda c: c.strip().lower() in cols_needed,
    ).fillna("")

    ent.columns = ent.columns.str.lower().str.strip()

    # Ensure all expected columns exist
    for col in cols_needed:
        if col not in ent.columns:
            ent[col] = ""

    # Apply TRANID_ALIASES normalisation — vectorized str.replace per pattern
    aliases = cfg.get("TRANID_ALIASES", {})
    if aliases:
        original = ent["tranid"].copy()
        for pattern, replacement in aliases.items():
            ent["tranid"] = ent["tranid"].str.replace(
                pattern, replacement, regex=True
            )
        changed = (ent["tranid"] != original).sum()
        if changed:
            canonical_mask    = ent["tranid"] == original
            canonical_descrtx = (
                ent.loc[canonical_mask, ["tranid", "descrtx"]]
                .drop_duplicates("tranid")
                .set_index("tranid")["descrtx"]
                .to_dict()
            )
            aliased_mask = ~canonical_mask
            ent.loc[aliased_mask, "descrtx"] = (
                ent.loc[aliased_mask, "tranid"]
                .map(canonical_descrtx)
                .fillna(ent.loc[aliased_mask, "descrtx"])
            )
            log.info("TRANID_ALIASES: normalised %d rows", changed)

    # Cast to Categorical — dramatically reduces memory for repeated strings
    for col in cols_needed:
        ent[col] = ent[col].astype("category")

    log.info("Lean load: %d rows  |  %.1f MB peak estimate",
             len(ent), ent.memory_usage(deep=True).sum() / 1e6)
    return ent


def main():
    parser = argparse.ArgumentParser(description="IGA Role Miner")
    parser.add_argument("--ents",   default=CONFIG["CSV_ENTITLEMENTS"],
                        help="Path to entitlements CSV")
    parser.add_argument("--hr",     default=CONFIG["CSV_EMPLOYEES"],
                        help="Path to employees CSV")
    parser.add_argument("--apps",   default=CONFIG["CSV_APPLICATIONS"],
                        help="Path to applications CSV (optional)")
    parser.add_argument("--out",    default=CONFIG["OUTPUT_DIR"],
                        help="Output directory")
    parser.add_argument("--sample", type=int, default=None,
                        help="Limit to N employees (for testing)")
    parser.add_argument("--top-tranids", type=int, default=None, metavar="N",
                        help="Run only the top-N tranid report and exit (default 50)")
    args = parser.parse_args()

    cfg = CONFIG.copy()
    cfg["CSV_ENTITLEMENTS"] = args.ents
    cfg["CSV_EMPLOYEES"]    = args.hr
    cfg["CSV_APPLICATIONS"] = args.apps
    cfg["OUTPUT_DIR"]       = args.out
    if args.sample:
        cfg["SAMPLE_SIZE"]  = args.sample

    if args.top_tranids is not None:
        df = load_entitlements_lean(cfg)
        n  = args.top_tranids if args.top_tranids > 0 else 50
        result = top_tranid_by_population(df, n=n)
        out = Path(cfg["OUTPUT_DIR"])
        out.mkdir(parents=True, exist_ok=True)
        ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
        p   = out / f"top_tranids_{ts}.csv"
        result.to_csv(p, index=False)
        log.info("Saved %s  (%d rows)", p, len(result))
        return

    run_pipeline(cfg)


if __name__ == "__main__":
    main()
