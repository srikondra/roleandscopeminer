"""
IGA Role Miner — Streamlit Web UI

Run with:
    streamlit run src/ui/app.py
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

# Ensure the project root is on sys.path when launched directly
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import streamlit as st

from src.algorithms.registry import AlgorithmRegistry
from src.config import (
    HierarchyConfig, LeidenConfig, LouvainConfig, NMFConfig,
    PipelineConfig, PipelineResult, PopulationFilter,
)
from src.data import DataLoader
from src.pipeline import PipelineRunner

# ── Page setup ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="IGA Role Miner",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Minimal CSS polish ─────────────────────────────────────────────────────────
st.markdown("""
<style>
  .block-container { padding-top: 1.5rem; }
  .stMetric label { font-size: 0.8rem; color: #888; }
  div[data-testid="stSidebarNav"] { display: none; }
  .sidebar-section { margin-bottom: 0.3rem; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def _load_data_cached(ents: str, hr: str, apps: str | None,
                      sample: int | None) -> pd.DataFrame:
    """Load and cache the merged frame. Re-runs only when inputs change."""
    cfg = PipelineConfig(
        csv_entitlements=ents,
        csv_employees=hr,
        csv_applications=apps,
        sample_size=sample,
    )
    return DataLoader(cfg).load()


def _unique_vals(df: pd.DataFrame, col: str) -> list[str]:
    if col not in df.columns:
        return []
    return sorted(df[col].replace("", pd.NA).dropna().unique().tolist())


def _algo_widget(algo_name: str, algo_cls, cfg_obj) -> object:
    """Render config_schema widgets for one algorithm. Returns updated config."""
    schema = algo_cls().config_schema
    updates = {}
    for knob in schema:
        key   = knob["key"]
        label = knob["label"]
        help_ = knob.get("help", "")
        cur   = getattr(cfg_obj, key, knob["default"])
        if cur is None:
            cur = knob["default"]

        if knob["type"] == "slider":
            updates[key] = st.slider(
                label, min_value=knob["min"], max_value=knob["max"],
                value=float(cur), step=knob["step"],
                help=help_, key=f"{algo_name}_{key}",
            )
        elif knob["type"] == "checkbox":
            updates[key] = st.checkbox(label, value=bool(cur),
                                        help=help_, key=f"{algo_name}_{key}")
        else:
            updates[key] = st.number_input(
                label, value=int(cur) if isinstance(cur, int) else float(cur),
                help=help_, key=f"{algo_name}_{key}",
            )
    return cfg_obj.model_copy(update=updates)


def _download_btn(df: pd.DataFrame | None, label: str, filename: str) -> None:
    if df is None or df.empty:
        st.caption(f"_No data for {label}_")
        return
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    st.download_button(
        label=f"⬇ {label}",
        data=buf.getvalue(),
        file_name=filename,
        mime="text/csv",
        use_container_width=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.title("🏢 IGA Role Miner")
    st.caption("Based on US 12,309,164 B1 — Kondra et al.")
    st.divider()

    # ── 1. Data Sources ────────────────────────────────────────────────────────
    with st.expander("📁  Data Sources", expanded=True):
        ents_path   = st.text_input("Entitlements CSV",  "sample_data/sample_entitlements.csv", key="p_ents")
        hr_path     = st.text_input("Employees CSV",     "sample_data/sample_employees.csv",    key="p_hr")
        apps_path   = st.text_input("Applications CSV",  "sample_data/sample_applications.csv", key="p_apps")
        tier_defs   = st.text_input("Tier Definitions CSV (optional)", "", key="p_tierdefs",
                                     help="tier,tranid[,notes] — leave blank for dynamic discovery")
        sample_size = st.number_input("Sample Size (0 = all users)", min_value=0, value=0,
                                       step=100, key="p_sample",
                                       help="Limit employees for quick testing")
        load_btn    = st.button("🔄  Load / Refresh Data", use_container_width=True, type="secondary")

    sample_val: int | None = int(sample_size) if sample_size > 0 else None

    if load_btn:
        with st.spinner("Loading data …"):
            try:
                df_raw = _load_data_cached(
                    ents_path, hr_path, apps_path or None, sample_val
                )
                st.session_state["df_raw"] = df_raw
                st.session_state.pop("pipeline_result", None)
                st.success(
                    f"Loaded {df_raw['ritsid'].nunique():,} users · "
                    f"{df_raw['grant_id'].nunique():,} grants"
                )
            except Exception as exc:
                st.error(f"Load failed: {exc}")

    df_raw: pd.DataFrame | None = st.session_state.get("df_raw")

    # ── 2. Population Filter ───────────────────────────────────────────────────
    with st.expander("👥  Population Filter"):
        pop_mode = st.radio(
            "Filter by",
            ["All Staff", "HR Attributes", "Upload User List"],
            key="pop_mode",
            horizontal=True,
        )

        pop_filter = PopulationFilter()

        if pop_mode == "HR Attributes" and df_raw is not None:
            st.caption("Select one or more values per dimension (AND logic across dimensions).")

            # MS hierarchy levels — show only levels with data
            cfg_tmp = PipelineConfig()
            ms_selections: dict[str, list[str]] = {}
            for lvl in cfg_tmp.segment_cols:
                vals = _unique_vals(df_raw, lvl)
                if vals:
                    chosen = st.multiselect(
                        lvl.replace("_", " ").title(), vals, key=f"ms_{lvl}"
                    )
                    if chosen:
                        ms_selections[lvl] = chosen

            mg_selections: dict[str, list[str]] = {}
            with st.expander("MG Geo Hierarchy"):
                for lvl in cfg_tmp.geo_cols:
                    vals = _unique_vals(df_raw, lvl)
                    if vals:
                        chosen = st.multiselect(
                            lvl.replace("_", " ").title(), vals, key=f"mg_{lvl}"
                        )
                        if chosen:
                            mg_selections[lvl] = chosen

            jf = st.multiselect("Job Function",
                                 _unique_vals(df_raw, "jobfunctiondescription"),
                                 key="pop_jf")
            jfam = st.multiselect("Job Family",
                                   _unique_vals(df_raw, "jobfamilydescription"),
                                   key="pop_jfam")
            reg  = st.multiselect("Region",  _unique_vals(df_raw, "region"),  key="pop_reg")
            cty  = st.multiselect("Country", _unique_vals(df_raw, "country"), key="pop_cty")

            pop_filter = PopulationFilter(
                ms_levels=ms_selections,
                mg_levels=mg_selections,
                job_functions=jf,
                job_families=jfam,
                regions=reg,
                countries=cty,
            )

            # Live preview
            if not pop_filter.is_empty and df_raw is not None:
                n_filtered = df_raw[df_raw["ritsid"].isin(
                    pop_filter.apply(df_raw)["ritsid"].unique()
                )]["ritsid"].nunique()
                st.info(f"Filter matches **{n_filtered:,}** users")

        elif pop_mode == "Upload User List":
            uploaded = st.file_uploader(
                "Upload CSV or TXT with ritsid / userid column",
                type=["csv", "txt"], key="pop_upload"
            )
            if uploaded:
                try:
                    up_df = pd.read_csv(uploaded, dtype=str)
                    id_col = next(
                        (c for c in up_df.columns if c.lower() in ("ritsid", "userid", "user_id")),
                        up_df.columns[0]
                    )
                    ritsids = up_df[id_col].dropna().str.strip().tolist()
                    pop_filter = PopulationFilter(ritsids=ritsids)
                    st.success(f"Loaded {len(ritsids):,} user IDs from '{id_col}'")
                except Exception as exc:
                    st.error(f"Could not parse upload: {exc}")
        else:
            st.caption("No filter — all users in the dataset will be mined.")

    # ── 3. Algorithms ──────────────────────────────────────────────────────────
    with st.expander("🧮  Algorithms"):
        all_algos   = AlgorithmRegistry.available()
        enabled_algos: list[str] = []
        algo_configs = {}

        louvain_cfg = LouvainConfig()
        nmf_cfg     = NMFConfig()

        for algo_name, algo_cls in all_algos.items():
            enabled = st.checkbox(
                f"**{algo_name.upper()}** — {algo_cls.description}",
                value=(algo_name in ("louvain", "nmf")),
                key=f"algo_en_{algo_name}",
            )
            if enabled:
                enabled_algos.append(algo_name)
                _cfg_map = {"louvain": LouvainConfig, "leiden": LeidenConfig, "nmf": NMFConfig}
                default_cfg = _cfg_map.get(algo_name, LouvainConfig)()
                with st.expander(f"⚙ {algo_name} settings"):
                    algo_configs[algo_name] = _algo_widget(algo_name, algo_cls, default_cfg)

        if not enabled_algos:
            st.warning("Select at least one algorithm.")

    # ── 4. Hierarchy Config ────────────────────────────────────────────────────
    with st.expander("🏗️  Hierarchy Settings"):
        tier_mode = st.radio(
            "Tier 1 / 2 discovery",
            ["Dynamic (prevalence thresholds)", "Pre-defined file"],
            key="tier_mode",
        )

        hier_cfg = HierarchyConfig()
        tier_defs_file: str | None = None

        if tier_mode == "Dynamic (prevalence thresholds)":
            t1_prev = st.slider("Tier-1 Staff min prevalence",
                                0.50, 1.0, 0.95, 0.01, key="t1_prev",
                                help="Global prevalence floor for Tier-1 grants")
            t2_min  = st.slider("Tier-2 Tech Baseline min prevalence",
                                0.10, 1.0, 0.50, 0.01, key="t2_min")
            t2_max  = st.slider("Tier-2 Tech Baseline max prevalence",
                                0.10, 1.0, 0.80, 0.01, key="t2_max")
            hier_cfg = hier_cfg.model_copy(update=dict(
                staff_min_prevalence=t1_prev,
                tech_baseline_min_prevalence=t2_min,
                tech_baseline_max_prevalence=t2_max,
            ))
        else:
            tier_file_input = st.text_input(
                "tier_definitions.csv path",
                tier_defs if tier_defs else "",
                key="tier_file_widget",
            )
            tier_defs_file = tier_file_input or None

        st.caption("**Business Role sub-tiers (Pass 3)**")
        gap  = st.slider("Sub-tier gap threshold", 0.05, 0.5, 0.20, 0.05, key="biz_gap",
                         help="Prevalence drop that opens a new sub-tier")
        floor = st.slider("Min grant prevalence floor", 0.01, 0.5, 0.10, 0.01, key="biz_floor")

        st.caption("**Orphan / Unassigned grants**")
        orphan_thresh = st.slider(
            "Orphan grant threshold", 0.10, 1.0, 0.50, 0.05, key="orphan_thresh",
            help=(
                "A grant is flagged as orphan/unassigned if its prevalence in its "
                "best-fit role is below this value.  "
                "0.50 = must be held by a majority of a role's members to be 'claimed'.  "
                "Higher → more grants become orphan."
            ),
        )
        hier_cfg = hier_cfg.model_copy(update=dict(
            business_gap_threshold=gap,
            business_min_prevalence=floor,
            orphan_grant_max_role_prevalence=orphan_thresh,
        ))

    st.divider()

    # ── 6. Run ─────────────────────────────────────────────────────────────────
    run_ready = df_raw is not None and bool(enabled_algos)
    run_btn   = st.button(
        "▶️  Run Role Mining",
        disabled=not run_ready,
        type="primary",
        use_container_width=True,
    )
    if not run_ready and df_raw is None:
        st.caption("Load data first ↑")
    elif not run_ready:
        st.caption("Enable at least one algorithm ↑")


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE EXECUTION
# ══════════════════════════════════════════════════════════════════════════════

if run_btn and run_ready:
    # Build config from UI state
    pipe_cfg = PipelineConfig(
        csv_entitlements=ents_path,
        csv_employees=hr_path,
        csv_applications=apps_path or None,
        sample_size=sample_val,
        tier_definitions_file=tier_defs_file,
        hierarchy=hier_cfg,
        population_filter=pop_filter,
        enabled_algorithms=enabled_algos,
        louvain=algo_configs.get("louvain", LouvainConfig()),
        nmf=algo_configs.get("nmf", NMFConfig()),
    )

    progress_bar  = st.progress(0.0, text="Starting …")
    status_text   = st.empty()

    def _progress_cb(step: str, pct: float) -> None:
        progress_bar.progress(min(pct, 1.0), text=step)
        status_text.caption(step)

    try:
        result: PipelineResult = PipelineRunner(pipe_cfg).run(progress=_progress_cb)
        st.session_state["pipeline_result"] = result
        progress_bar.progress(1.0, text="Complete!")
        status_text.empty()
    except Exception as exc:
        progress_bar.empty()
        st.error(f"Pipeline failed: {exc}")
        st.exception(exc)


# ══════════════════════════════════════════════════════════════════════════════
# RESULTS AREA
# ══════════════════════════════════════════════════════════════════════════════

result: PipelineResult | None = st.session_state.get("pipeline_result")

if result is None and df_raw is None:
    # ── Welcome screen ─────────────────────────────────────────────────────────
    st.markdown("# 🏢 IGA Role Miner")
    st.markdown(
        "Discover candidate enterprise roles from entitlement data using "
        "graph community detection and matrix factorisation.\n\n"
        "**Get started:**\n"
        "1. Enter CSV file paths in **Data Sources** (sidebar) and click **Load**\n"
        "2. Optionally filter the population under **Population Filter**\n"
        "3. Toggle algorithms and adjust settings\n"
        "4. Click **▶️ Run Role Mining**"
    )
    col1, col2, col3 = st.columns(3)
    col1.info("**Tier 1 — Staff**\nNear-universal grants (building access, directory)")
    col2.info("**Tier 2 — Tech Baseline**\nTech-user grants (O365, VPN, domain)")
    col3.info("**Tier 3 — Business Roles**\nDiscovered by Louvain / NMF clustering")

elif result is None and df_raw is not None:
    # ── Data loaded, not yet run ────────────────────────────────────────────────
    st.markdown("## Data loaded — ready to run")
    c1, c2, c3 = st.columns(3)
    c1.metric("Users",   f"{df_raw['ritsid'].nunique():,}")
    c2.metric("Grants",  f"{df_raw['grant_id'].nunique():,}")
    c3.metric("Ent rows", f"{len(df_raw):,}")
    st.info("Configure your population filter and algorithm settings in the sidebar, then click ▶️")

else:
    # ── Results dashboard ──────────────────────────────────────────────────────
    st.markdown("## Results")

    # Metrics bar
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Users",          f"{result.n_users:,}")
    m2.metric("Grants",         f"{result.n_grants:,}")
    m3.metric("Business Roles", f"{result.n_roles:,}")
    m4.metric("Runtime",        f"{result.elapsed_seconds:.1f}s")

    st.divider()

    tabs = st.tabs([
        "🏗️ Tier Hierarchy",
        "🏢 Business Roles",
        "👥 User Assignments",
        "📊 Top Tranids",
        "📥 Downloads",
    ])

    # ── Tab 1: Tier Hierarchy ──────────────────────────────────────────────────
    with tabs[0]:
        if result.tier_result and result.tier_result.hierarchy_rows:
            hier_df = pd.DataFrame(result.tier_result.hierarchy_rows)
            t1 = hier_df[hier_df["tier"] == 1]
            t2 = hier_df[hier_df["tier"] == 2]

            col_a, col_b = st.columns(2)
            with col_a:
                st.subheader(f"Tier 1 — Staff  ({len(t1)} grants)")
                st.dataframe(
                    t1[["grant_id", "prevalence", "descrtx", "appname"]],
                    use_container_width=True, hide_index=True,
                )
            with col_b:
                st.subheader(f"Tier 2 — Staff with Tech Access  ({len(t2)} grants)")
                st.dataframe(
                    t2[["grant_id", "prevalence", "descrtx", "appname"]],
                    use_container_width=True, hide_index=True,
                )
        else:
            st.info("No tier hierarchy data.")

    # ── Tab 2: Business Roles ──────────────────────────────────────────────────
    with tabs[1]:
        for algo_name, ar in result.algorithm_results.items():
            if ar.biz_hierarchy is not None and not ar.biz_hierarchy.empty:
                st.subheader(f"{algo_name.upper()} — {ar.n_roles} roles")
                st.dataframe(
                    ar.biz_hierarchy[["cluster_id", "parent_cluster_id"]],
                    use_container_width=True, hide_index=True,
                )
            else:
                st.info(f"{algo_name.upper()}: no business roles discovered.")

    # ── Tab 3: User Assignments ────────────────────────────────────────────────
    with tabs[2]:
        search = st.text_input("🔍 Search by ritsid", key="assign_search")

        # Primary role table (1:1) — one column per algorithm
        primary_frames = []
        for algo_name, ar in result.algorithm_results.items():
            if ar.assignments is not None and not ar.assignments.empty:
                f = ar.assignments[["ritsid", "role_id"]].rename(
                    columns={"role_id": f"{algo_name}_primary_role"})
                primary_frames.append(f)
        if primary_frames:
            primary_df = primary_frames[0]
            for f in primary_frames[1:]:
                primary_df = primary_df.merge(f, on="ritsid", how="outer")
            if search:
                primary_df = primary_df[primary_df["ritsid"].str.contains(search, case=False, na=False)]
            st.subheader("Primary Role (1 per user)")
            st.dataframe(primary_df, use_container_width=True, hide_index=True)

        # Memberships table (1:N) — long format, one row per (user, role)
        mem_frames = []
        for algo_name, ar in result.algorithm_results.items():
            src = ar.memberships if ar.memberships is not None else ar.assignments
            if src is not None and not src.empty:
                f = src[["ritsid", "role_id"]].copy()
                f["algorithm"] = algo_name
                mem_frames.append(f)
        if mem_frames:
            mem_df = pd.concat(mem_frames, ignore_index=True)[["ritsid", "algorithm", "role_id"]]
            if search:
                mem_df = mem_df[mem_df["ritsid"].str.contains(search, case=False, na=False)]
            n_multi = (mem_df.groupby(["ritsid", "algorithm"]).size() > 1).sum()
            st.subheader(f"All Role Memberships ({n_multi:,} users with multiple roles)")
            st.dataframe(mem_df, use_container_width=True, hide_index=True)

        if not primary_frames and not mem_frames:
            st.info("No assignment data available.")

    # ── Tab 4: Top Tranids ─────────────────────────────────────────────────────
    with tabs[3]:
        if result.top_tranids is not None and not result.top_tranids.empty:
            st.caption(
                "Top tranids by distinct-employee coverage.  "
                "Label tier 1 / tier 2 and save as `tier_definitions.csv`."
            )
            st.dataframe(result.top_tranids, use_container_width=True, hide_index=True)
        else:
            st.info("No top-tranid data.")

    # ── Tab 5: Downloads ────────────────────────────────────────────────────────
    with tabs[4]:
        st.subheader("Download Results")
        col1, col2 = st.columns(2)

        with col1:
            st.caption("**Hierarchy**")
            if result.tier_result:
                _download_btn(pd.DataFrame(result.tier_result.hierarchy_rows),
                              "Tier Hierarchy (T1+T2)", "role_hierarchy_tiers.csv")
            _download_btn(result.unified_hierarchy(),
                          "Full Unified Hierarchy", "role_hierarchy_full.csv")
            _download_btn(result.top_tranids,
                          "Top Tranids", "top_tranids.csv")

        with col2:
            for algo_name, ar in result.algorithm_results.items():
                st.caption(f"**{algo_name.upper()}**")
                _download_btn(ar.profiles,          f"Role Profiles",           f"{algo_name}_profiles.csv")
                _download_btn(ar.biz_hierarchy,     f"Business Role Hierarchy", f"{algo_name}_biz_hierarchy.csv")
                _download_btn(ar.unassigned_users,  f"Unassigned Users",        f"{algo_name}_unassigned_users.csv")
                _download_btn(ar.orphan_grants,     f"Orphan Grants",           f"{algo_name}_orphan_grants.csv")
