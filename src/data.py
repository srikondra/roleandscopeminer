"""
Data layer: normalisation, loading, matrix construction, and top-tranid report.

Public API
----------
Normalizer                  — apply TRANID_ALIASES and build grant_id columns
DataLoader                  — read CSVs, validate, merge, return flat frame
build_user_entitlement_matrix — sparse binary (n_users × n_grants) matrix
top_tranid_by_population    — ranked tranid coverage report

Grant key rule (from ebac_schema.md)
-------------------------------------
adguid IS NULL     →  grant_id = csiid|tranid|descrtx|entitlecd
adguid IS NOT NULL →  grant_id = csiid|tranid|descrtx   (entitlecd ignored)

TRANID_ALIASES twist
--------------------
When a tranid is changed by an alias rule, its csiid is automatically dropped
from the grant key.  This collapses the same logical access provisioned from
different regional app instances (different csiids, same tranid/access) into a
single matrix column — without requiring the user to enumerate every csiid.

grant_id_orig
-------------
Captured *before* aliases run.  Stored on the frame so that hierarchy output
can publish real source grant references rather than the synthetic canonical
tranid produced by TRANID_ALIASES.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from .config import PipelineConfig

log = logging.getLogger("role_miner.data")


# ── Normalizer ─────────────────────────────────────────────────────────────────

class Normalizer:
    """Apply TRANID_ALIASES and construct grant_id / grant_id_orig columns."""

    def __init__(self, tranid_aliases: dict[str, str]):
        self.aliases = tranid_aliases

    def apply(self, ent: pd.DataFrame) -> pd.DataFrame:
        """
        Mutate (copy-on-write) the entitlements frame in place:
          1. Capture grant_id_orig (pre-alias snapshot)
          2. Apply TRANID_ALIASES; set tranid_aliased flag
          3. Harmonise descrtx for aliased rows
          4. Construct grant_id per the grant key rule

        Returns the enriched frame.
        """
        ent = ent.copy()

        # ── 1. Snapshot pre-alias grant_id ────────────────────────────────────
        no_guid_pre          = ent["adguid"].eq("")
        ent["grant_id_orig"] = np.where(
            no_guid_pre,
            ent["csiid"] + "|" + ent["tranid"] + "|" + ent["descrtx"] + "|" + ent["entitlecd"],
            ent["csiid"] + "|" + ent["tranid"] + "|" + ent["descrtx"],
        )

        # ── 2. Apply TRANID_ALIASES ────────────────────────────────────────────
        ent["tranid_aliased"] = False

        if self.aliases:
            original = ent["tranid"].copy()
            for pattern, replacement in self.aliases.items():
                ent["tranid"] = ent["tranid"].str.replace(pattern, replacement, regex=True)

            changed      = (ent["tranid"] != original).sum()
            aliased_mask = ent["tranid"] != original

            if changed:
                log.info("TRANID_ALIASES: normalised %d tranid values", changed)
                ent["tranid_aliased"] = aliased_mask

                # ── 3. Harmonise descrtx: aliased rows adopt the canonical descrtx
                canonical_descrtx = (
                    ent[~aliased_mask][["tranid", "descrtx"]]
                    .drop_duplicates("tranid")
                    .set_index("tranid")["descrtx"]
                    .to_dict()
                )
                ent.loc[aliased_mask, "descrtx"] = (
                    ent.loc[aliased_mask, "tranid"]
                    .map(canonical_descrtx)
                    .fillna(ent.loc[aliased_mask, "descrtx"])
                )

        # ── 4. Construct grant_id ──────────────────────────────────────────────
        no_guid   = ent["adguid"].eq("")
        csiid_pfx = np.where(ent["tranid_aliased"], "", ent["csiid"] + "|")

        ent["grant_id"] = np.where(
            no_guid,
            csiid_pfx + ent["tranid"] + "|" + ent["descrtx"] + "|" + ent["entitlecd"],
            csiid_pfx + ent["tranid"] + "|" + ent["descrtx"],
        )
        ent["entitlecd_eff"] = np.where(no_guid, ent["entitlecd"], "")

        return ent


# ── DataLoader ─────────────────────────────────────────────────────────────────

def _require_cols(df: pd.DataFrame, cols: list[str], source: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")


class DataLoader:
    """Load, validate, normalise, and merge all input CSVs."""

    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg

    def load(self) -> pd.DataFrame:
        cfg = self.cfg

        # ── Load raw CSVs ──────────────────────────────────────────────────────
        log.info("Loading entitlements : %s", cfg.csv_entitlements)
        ent = pd.read_csv(cfg.csv_entitlements, dtype=str).fillna("")
        ent.columns = ent.columns.str.lower().str.strip()

        log.info("Loading employees    : %s", cfg.csv_employees)
        emp = pd.read_csv(cfg.csv_employees, dtype=str).fillna("")
        emp.columns = emp.columns.str.lower().str.strip()

        app = self._load_applications()

        # ── Validate ───────────────────────────────────────────────────────────
        _require_cols(ent, ["ritsid", "csiid", "tranid", "descrtx", "entitlecd", "adguid"],
                      source="entitlements CSV")
        _require_cols(emp, ["ritsid", "userid", "dept_mgr_geid"],
                      source="employees CSV")

        # ── Normalise ──────────────────────────────────────────────────────────
        ent = Normalizer(cfg.tranid_aliases).apply(ent)

        log.info("Entitlements: %d rows | %d unique grants | %d unique users",
                 len(ent), ent["grant_id"].nunique(), ent["ritsid"].nunique())

        # ── Optional sampling ──────────────────────────────────────────────────
        if cfg.sample_size and cfg.sample_size < emp["ritsid"].nunique():
            sampled = emp["ritsid"].drop_duplicates().sample(
                n=cfg.sample_size, random_state=42
            )
            emp = emp[emp["ritsid"].isin(sampled)]
            ent = ent[ent["ritsid"].isin(sampled)]
            log.info("Sampled to %d employees", emp["ritsid"].nunique())

        # ── Merge ──────────────────────────────────────────────────────────────
        df = ent.merge(emp, on="ritsid", how="inner", suffixes=("_ent", "_emp"))

        if app is not None and {"csiid", "appname"}.issubset(app.columns):
            df = df.merge(app[["csiid", "appname"]], on="csiid", how="left")
        else:
            df["appname"] = df["csiid"]         # fall back to csiid

        log.info("Merged frame: %d rows | %d employees | %d unique grants",
                 len(df), df["ritsid"].nunique(), df["grant_id"].nunique())
        return df

    def _load_applications(self) -> pd.DataFrame | None:
        if not self.cfg.csv_applications:
            return None
        p = Path(self.cfg.csv_applications)
        if not p.exists():
            return None
        log.info("Loading applications : %s", p)
        app = pd.read_csv(p, dtype=str).fillna("")
        app.columns = app.columns.str.lower().str.strip()
        return app


# ── Matrix builder ─────────────────────────────────────────────────────────────

def build_user_entitlement_matrix(
    df: pd.DataFrame,
) -> tuple[csr_matrix, list[str], list[str]]:
    """
    Build a binary sparse matrix M where M[i, j] = 1 iff user i holds grant j.

    Returns
    -------
    matrix      : csr_matrix   (n_users × n_grants)
    user_index  : list[str]    ritsid values, row order
    grant_index : list[str]    grant_id values, column order
    """
    users  = sorted(df["ritsid"].unique())
    grants = sorted(df["grant_id"].unique())
    u_idx  = {u: i for i, u in enumerate(users)}
    g_idx  = {g: i for i, g in enumerate(grants)}

    pairs   = df[["ritsid", "grant_id"]].drop_duplicates()
    row_idx = pairs["ritsid"].map(u_idx).to_numpy(dtype=np.int32)
    col_idx = pairs["grant_id"].map(g_idx).to_numpy(dtype=np.int32)
    data    = np.ones(len(row_idx), dtype=np.float32)

    mat = csr_matrix((data, (row_idx, col_idx)), shape=(len(users), len(grants)))

    log.info("Matrix: %d users × %d grants  (density %.4f%%)",
             len(users), len(grants),
             100.0 * mat.nnz / (len(users) * len(grants)))
    return mat, users, grants


# ── Top-tranid report ──────────────────────────────────────────────────────────

def top_tranid_by_population(
    df: pd.DataFrame,
    n: int = 50,
    ignore_csiid: bool = True,
) -> pd.DataFrame:
    """
    Return the top-n grants ranked by distinct-employee coverage.

    ignore_csiid=True  (default)
        Group by (tranid, descrtx, entitlecd_eff) — csiid collapsed.
        Same access from multiple app instances appears as one row.
        Extra columns: csiid_count, csiids.

    ignore_csiid=False
        Group by the full grant key (csiid + tranid + descrtx + entitlecd_eff).
        Each app instance appears separately.
    """
    df          = df.copy()
    no_guid     = df["adguid"].astype(str).eq("")
    total_users = df["ritsid"].nunique()

    df["entitlecd_eff"] = np.where(no_guid, df["entitlecd"].astype(str), "")

    if ignore_csiid:
        df["_gkey"] = (
            df["tranid"].astype(str) + "|"
            + df["descrtx"].astype(str) + "|"
            + df["entitlecd_eff"].astype(str)
        )
        base = df[["ritsid", "_gkey", "csiid", "adguid"]].drop_duplicates(
            subset=["ritsid", "_gkey"]
        )

        user_counts  = base.groupby("_gkey")["ritsid"].nunique().rename("user_count")
        csiid_counts = base.groupby("_gkey")["csiid"].nunique().rename("csiid_count")
        csiid_list   = (
            base.groupby("_gkey")["csiid"]
            .apply(lambda s: " | ".join(sorted(s.astype(str).unique())))
            .rename("csiids")
        )
        adguid_null  = (
            base.groupby("_gkey")["adguid"]
            .apply(lambda s: round(100.0 * s.astype(str).eq("").sum() / len(s), 1))
            .rename("adguid_null_pct")
        )

        stats = (
            pd.concat([user_counts, csiid_counts, csiid_list, adguid_null], axis=1)
            .reset_index()
            .sort_values("user_count", ascending=False)
            .head(n)
            .assign(population_pct=lambda d: (d["user_count"] / total_users * 100).round(1))
        )

        meta = (
            df.loc[df["_gkey"].isin(stats["_gkey"]),
                   ["_gkey", "tranid", "descrtx", "entitlecd_eff"]]
            .drop_duplicates("_gkey")
            .rename(columns={"entitlecd_eff": "entitlecd"})
        )

        result = (
            stats
            .merge(meta, on="_gkey", how="left")
            .sort_values("user_count", ascending=False)
            [["tranid", "descrtx", "entitlecd", "user_count", "population_pct",
              "csiid_count", "csiids", "adguid_null_pct"]]
            .reset_index(drop=True)
        )

    else:
        df["_gkey"] = np.where(
            no_guid,
            df["csiid"].astype(str) + "|" + df["tranid"].astype(str) + "|"
            + df["descrtx"].astype(str) + "|" + df["entitlecd"].astype(str),
            df["csiid"].astype(str) + "|" + df["tranid"].astype(str) + "|"
            + df["descrtx"].astype(str),
        )
        base = df[["ritsid", "_gkey", "adguid"]].drop_duplicates(subset=["ritsid", "_gkey"])

        user_counts = base.groupby("_gkey")["ritsid"].nunique().rename("user_count")
        adguid_null = (
            base.groupby("_gkey")["adguid"]
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

        meta = (
            df.loc[df["_gkey"].isin(stats["_gkey"]),
                   ["_gkey", "csiid", "tranid", "descrtx", "entitlecd_eff"]]
            .drop_duplicates("_gkey")
            .rename(columns={"entitlecd_eff": "entitlecd"})
        )

        result = (
            stats
            .merge(meta, on="_gkey", how="left")
            .sort_values("user_count", ascending=False)
            [["csiid", "tranid", "descrtx", "entitlecd",
              "user_count", "population_pct", "adguid_null_pct"]]
            .reset_index(drop=True)
        )

    log.info("Top-%d tranids (ignore_csiid=%s): %.1f%% – %.1f%% coverage",
             n, ignore_csiid,
             result["population_pct"].iloc[0],
             result["population_pct"].iloc[-1])
    return result
