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
from scipy.sparse import lil_matrix, csr_matrix
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

    mat = lil_matrix((len(users), len(grants)), dtype=np.float32)
    for _, row in df[["ritsid", "grant_id"]].drop_duplicates().iterrows():
        mat[u_idx[row["ritsid"]], g_idx[row["grant_id"]]] = 1.0

    log.info("Matrix: %d users × %d grants  (density %.3f%%)",
             len(users), len(grants),
             100.0 * mat.nnz / (len(users) * len(grants)))
    return mat.tocsr(), users, grants


# ──────────────────────────────────────────────────────────────────────────────
# 3. ORG GRAPH  (reporting hierarchy)
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
# 4. LOUVAIN ROLE MINING
# ──────────────────────────────────────────────────────────────────────────────

def run_louvain(matrix: csr_matrix, user_index: list, cfg: dict) -> pd.Series:
    """
    Build a Jaccard-similarity user graph and run Louvain community detection.
    Returns a pd.Series: ritsid → cluster_id  (string like 'L0', 'L1', …)
    """
    log.info("Building user similarity graph for Louvain …")
    n = matrix.shape[0]
    # Jaccard similarity via dot product on normalised binary vectors
    norms = np.asarray(matrix.sum(axis=1)).flatten()
    norms[norms == 0] = 1
    mat_norm = matrix.multiply(1.0 / norms[:, None])

    G = nx.Graph()
    G.add_nodes_from(range(n))

    # Sparse dot: M_norm × M_norm^T  — keep top-k neighbours per user
    sim = (mat_norm @ mat_norm.T).tocsr()
    k   = min(15, n - 1)
    for i in range(n):
        row     = sim.getrow(i).toarray().flatten()
        row[i]  = 0          # no self-loops
        top_k   = np.argpartition(row, -k)[-k:]
        for j in top_k:
            if row[j] > 0:
                G.add_edge(i, j, weight=float(row[j]))

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
                 cfg: dict):
    out = Path(cfg["OUTPUT_DIR"])
    out.mkdir(parents=True, exist_ok=True)
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")

    def _save(df, name):
        if df is not None and not df.empty:
            p = out / f"{name}_{ts}.csv"
            df.to_csv(p, index=False)
            log.info("Saved %s  (%d rows)", p, len(df))

    # Role template outputs
    _save(louvain_profiles,      "louvain_role_profiles")
    _save(louvain_entitlements,  "louvain_role_entitlements")
    _save(nmf_profiles,          "nmf_role_profiles")
    _save(nmf_entitlements,      "nmf_role_entitlements")

    # Scope outputs
    _save(louvain_scope_profiles, "louvain_scope_profiles")
    _save(louvain_scope_members,  "louvain_scope_members")
    _save(nmf_scope_profiles,     "nmf_scope_profiles")
    _save(nmf_scope_members,      "nmf_scope_members")

    # User-role assignments (both methods side by side)
    assignments_df = pd.DataFrame({
        "ritsid":         louvain_assignments.index,
        "louvain_role":   louvain_assignments.values,
    })
    if nmf_assignments is not None:
        nmf_map = nmf_assignments.to_dict()
        assignments_df["nmf_role"] = assignments_df["ritsid"].map(nmf_map)

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
        f"Louvain roles  : {len(louvain_profiles)}",
    ]
    if not louvain_profiles.empty:
        lines.append(f"  Avg members    : {louvain_profiles['member_count'].mean():.1f}")
        lines.append(f"  Avg core grants: {louvain_profiles['core_grant_count'].mean():.1f}")
        lines.append(f"  Scope variants : {n_lou_scopes}")
    lines += ["", f"NMF roles      : {len(nmf_profiles)}"]
    if not nmf_profiles.empty:
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

    # 3. Org graph
    org_graph = build_org_graph(df)

    # 4. Louvain — role templates
    louvain_assignments = run_louvain(matrix, user_index, cfg)
    louvain_profiles, louvain_entitlements = analyze_roles(
        louvain_assignments, df, matrix, user_index, grant_index, cfg, "Louvain"
    )

    # 5. NMF — role templates
    nmf_assignments = run_nmf(matrix, user_index, cfg)
    nmf_profiles, nmf_entitlements = analyze_roles(
        nmf_assignments, df, matrix, user_index, grant_index, cfg, "NMF"
    )

    # 6. Scope discovery — Pass 2
    log.info("─" * 40)
    log.info("Pass 2 — Scope Discovery")
    log.info("─" * 40)
    louvain_scope_profiles, louvain_scope_members = discover_scopes(
        louvain_assignments, df, matrix, user_index, grant_index, cfg, "Louvain"
    )
    nmf_scope_profiles, nmf_scope_members = discover_scopes(
        nmf_assignments, df, matrix, user_index, grant_index, cfg, "NMF"
    )

    # 7. Save
    save_outputs(
        louvain_profiles,      louvain_entitlements,
        louvain_scope_profiles, louvain_scope_members,
        nmf_profiles,          nmf_entitlements,
        nmf_scope_profiles,    nmf_scope_members,
        louvain_assignments,   nmf_assignments,
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
    args = parser.parse_args()

    cfg = CONFIG.copy()
    cfg["CSV_ENTITLEMENTS"] = args.ents
    cfg["CSV_EMPLOYEES"]    = args.hr
    cfg["CSV_APPLICATIONS"] = args.apps
    cfg["OUTPUT_DIR"]       = args.out
    if args.sample:
        cfg["SAMPLE_SIZE"]  = args.sample

    run_pipeline(cfg)


if __name__ == "__main__":
    main()
