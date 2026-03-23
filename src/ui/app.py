"""
IGA Role Miner — Streamlit Web UI

Run with:
    streamlit run src/ui/app.py
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

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
    page_title="IGA: Role Discovery",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""<style>
#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] { visibility: hidden; height: 0 !important; }

/* Narrow icon sidebar */
section[data-testid="stSidebar"] {
    min-width: 72px !important;
    max-width: 72px !important;
    background: #f7f7f7;
    border-right: 1px solid #e5e5e5;
}
section[data-testid="stSidebar"] > div:first-child {
    width: 72px !important;
    padding: 0.8rem 0 !important;
}
[data-testid="stSidebarNav"],
[data-testid="stSidebarCollapseButton"] { display: none !important; }

/* Remove main area excess padding */
.main .block-container {
    padding: 0 1rem 1.5rem !important;
    max-width: 100% !important;
}

/* Algo card inner border */
div[data-testid="stVerticalBlockBorderWrapper"] > div > div[data-testid="stVerticalBlockBorderWrapper"] {
    height: 100%;
}
</style>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def _load_data_cached(ents: str, hr: str, apps: str | None, sample: int | None) -> pd.DataFrame:
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
                label,
                value=int(cur) if isinstance(cur, int) else float(cur),
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
# LEFT NAV SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown(
        "<div style='text-align:center;font-size:1.5rem;padding:0.3rem 0 0.7rem'>🏢</div>"
        "<hr style='margin:0 0.4rem 0.7rem;border-color:#ddd'>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div style='text-align:center;font-size:1.4rem;padding:0.35rem 0;cursor:pointer' "
        "title='Home'>🏠</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div style='text-align:center;font-size:1.4rem;padding:0.35rem 0.4rem;"
        "background:#e8f0fe;border-radius:8px;margin:2px 6px;cursor:pointer' "
        "title='Role Discovery'>🔍</div>",
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:12rem'></div>", unsafe_allow_html=True)
    st.markdown(
        "<div style='text-align:center;font-size:1.3rem;padding:0.35rem 0;cursor:pointer;"
        "color:#888' title='Exit'>↩</div>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# TOP HEADER
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div style="display:flex;align-items:center;border:1.5px solid #ddd;border-radius:8px;
            padding:0.45rem 1rem;margin:0.6rem 0 0.8rem;background:#fff;">
  <span style="border:1.5px solid #aaa;border-radius:5px;padding:2px 10px;
               font-size:0.78rem;font-weight:700;color:#444;margin-right:14px;">Logo</span>
  <span style="font-size:0.95rem;font-weight:600;color:#111;flex:1;">IGA: Role Discovery</span>
  <span style="font-size:1.25rem;color:#555;cursor:pointer" title="User">👤</span>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════

df_raw: pd.DataFrame | None = st.session_state.get("df_raw")


# ══════════════════════════════════════════════════════════════════════════════
# CARD 1 — SELECT STAFF
# ══════════════════════════════════════════════════════════════════════════════

with st.container(border=True):
    st.markdown("**Select Staff**")

    pop_mode = st.radio(
        "Staff selection mode",
        ["All Staff", "Upload CSV", "Filter & Select by HR Attributes", "Sample Data"],
        index=3,  # default: Sample Data (only active option)
        key="pop_mode",
        horizontal=True,
        label_visibility="collapsed",
    )

    # ── Defaults (overridden only in Sample Data mode) ─────────────────────────
    ents_path   = "sample_data/sample_entitlements.csv"
    hr_path     = "sample_data/sample_employees.csv"
    apps_path   = "sample_data/sample_applications.csv"
    tier_defs   = ""
    sample_size = 0
    pop_filter  = PopulationFilter()

    # ── All Staff ──────────────────────────────────────────────────────────────
    if pop_mode == "All Staff":
        st.markdown(
            "<div style='color:#aaa;font-size:0.85rem;padding:0.5rem 0'>"
            "Connects to the enterprise identity store to pull the full staff roster "
            "and their entitlements. No local files required.</div>",
            unsafe_allow_html=True,
        )
        c1, c2 = st.columns(2)
        c1.text_input("Identity Store URL", placeholder="https://iga.corp/api/v1/...",
                      disabled=True, key="all_staff_url")
        c2.text_input("Authentication Token", placeholder="Bearer …",
                      disabled=True, key="all_staff_token")
        st.caption("🚧 Coming soon — only **Sample Data** is active in this release.")

    # ── Upload CSV ─────────────────────────────────────────────────────────────
    elif pop_mode == "Upload CSV":
        st.markdown(
            "<div style='color:#aaa;font-size:0.85rem;padding:0.5rem 0'>"
            "Upload your own entitlement export and HR/employee file to mine roles "
            "against a custom dataset.</div>",
            unsafe_allow_html=True,
        )
        c1, c2 = st.columns(2)
        c1.file_uploader("Entitlements CSV", type=["csv"], disabled=True, key="upl_ents")
        c2.file_uploader("Employees / HR CSV", type=["csv"], disabled=True, key="upl_hr")
        st.caption("🚧 Coming soon — only **Sample Data** is active in this release.")

    # ── Filter & Select by HR Attributes ─────────────────────────────────────
    elif pop_mode == "Filter & Select by HR Attributes":
        st.markdown(
            "<div style='color:#aaa;font-size:0.85rem;padding:0.3rem 0 0.6rem'>"
            "Narrow the population to mine by intersecting org hierarchy, job, "
            "and geography dimensions (AND logic across dimensions).</div>",
            unsafe_allow_html=True,
        )
        r1c1, r1c2, r1c3, r1c4 = st.columns(4)
        r1c1.multiselect("MS Levels",   [], disabled=True, key="hr_ms",
                         help="Managed Segment org hierarchy levels")
        r1c2.multiselect("MG Levels",   [], disabled=True, key="hr_mg",
                         help="Managed Geography org hierarchy levels")
        r1c3.multiselect("Department",  [], disabled=True, key="hr_dept")
        r1c4.multiselect("Manager",     [], disabled=True, key="hr_mgr")
        r2c1, r2c2, r2c3, r2c4 = st.columns(4)
        r2c1.multiselect("Job Function", [], disabled=True, key="hr_jf")
        r2c2.multiselect("Job Family",   [], disabled=True, key="hr_jfam")
        r2c3.multiselect("Region",       [], disabled=True, key="hr_reg")
        r2c4.multiselect("Country",      [], disabled=True, key="hr_cty")
        st.caption("🚧 Coming soon — only **Sample Data** is active in this release.")

    # ── Sample Data (active) ───────────────────────────────────────────────────
    else:  # pop_mode == "Sample Data"
        with st.expander("📁 Data Sources", expanded=False):
            ents_path   = st.text_input("Entitlements CSV",  "sample_data/sample_entitlements.csv", key="p_ents")
            hr_path     = st.text_input("Employees CSV",     "sample_data/sample_employees.csv",    key="p_hr")
            apps_path   = st.text_input("Applications CSV",  "sample_data/sample_applications.csv", key="p_apps")
            tier_defs   = st.text_input(
                "Tier Definitions CSV (optional)", "", key="p_tierdefs",
                help="tier,tranid[,notes] — leave blank for dynamic discovery",
            )
            sample_size = st.number_input(
                "Row limit (0 = all)", min_value=0, value=0,
                step=100, key="p_sample",
                help="Limit the number of employees loaded — useful for quick testing",
            )

    sample_val: int | None = int(sample_size) if sample_size > 0 else None

    # Status + Load button (Load disabled for all modes except Sample Data)
    stat_col, load_col = st.columns([5, 1])
    with stat_col:
        if df_raw is not None:
            st.caption(
                f"✓ **{df_raw['ritsid'].nunique():,}** users · "
                f"**{df_raw['grant_id'].nunique():,}** grants loaded"
            )
    load_disabled = (pop_mode != "Sample Data")
    load_btn = load_col.button(
        "Load Data", key="load_btn",
        disabled=load_disabled,
        use_container_width=True,
    )

    if load_btn and not load_disabled:
        with st.spinner("Loading …"):
            try:
                df_raw = _load_data_cached(ents_path, hr_path, apps_path or None, sample_val)
                st.session_state["df_raw"] = df_raw
                st.session_state.pop("pipeline_result", None)
                st.success(
                    f"✓ {df_raw['ritsid'].nunique():,} users · "
                    f"{df_raw['grant_id'].nunique():,} grants"
                )
            except Exception as exc:
                st.error(f"Load failed: {exc}")
        df_raw = st.session_state.get("df_raw")


# ══════════════════════════════════════════════════════════════════════════════
# CARD 2 — SELECT ALGORITHMS
# ══════════════════════════════════════════════════════════════════════════════

_CFG_MAP = {"louvain": LouvainConfig, "leiden": LeidenConfig, "nmf": NMFConfig}
all_algos     = AlgorithmRegistry.available()
enabled_algos: list[str] = []
algo_configs:  dict      = {}

with st.container(border=True):
    st.markdown("**Select Algorithms**")

    algo_cols = st.columns(max(len(all_algos), 1))
    for col, (algo_name, algo_cls) in zip(algo_cols, all_algos.items()):
        default_cfg = _CFG_MAP.get(algo_name, LouvainConfig)()
        with col:
            with st.container(border=True):
                enabled = st.checkbox(
                    f"**{algo_name.upper()}**",
                    value=True,
                    key=f"algo_en_{algo_name}",
                )
                st.caption(getattr(algo_cls, "description", ""))
                with st.expander(f"⚙ {algo_name} settings", expanded=False):
                    updated_cfg = _algo_widget(algo_name, algo_cls, default_cfg)
            if enabled:
                enabled_algos.append(algo_name)
                algo_configs[algo_name] = updated_cfg

    if not enabled_algos:
        st.warning("Select at least one algorithm.")


# ══════════════════════════════════════════════════════════════════════════════
# CARD 3 — FINE TUNE KNOBS
# ══════════════════════════════════════════════════════════════════════════════

with st.container(border=True):
    knob_col, run_col = st.columns([4, 1])

    with knob_col:
        st.markdown("**Fine Tune Knobs**")

        tier_mode = st.radio(
            "Tier 1/2 discovery",
            ["Dynamic (prevalence thresholds)", "Pre-defined file"],
            key="tier_mode",
            horizontal=True,
        )

        tier_defs_file: str | None = None

        if tier_mode == "Dynamic (prevalence thresholds)":
            k1, k2, k3 = st.columns(3)
            with k1:
                st.caption("**Tier 1 — Staff**")
                t1_prev = st.slider(
                    "Min prevalence", 0.50, 1.0, 0.95, 0.01, key="t1_prev",
                    help="Global prevalence floor for Tier-1 Staff grants",
                )
            with k2:
                st.caption("**Tier 2 — Tech Baseline**")
                t2_min = st.slider("Min prevalence", 0.10, 1.0, 0.50, 0.01, key="t2_min")
                t2_max = st.slider("Max prevalence", 0.10, 1.0, 0.80, 0.01, key="t2_max")
            with k3:
                st.caption("**Business Roles / Orphans**")
                gap          = st.slider("Sub-tier gap",     0.05, 0.5,  0.20, 0.05, key="biz_gap",
                                         help="Prevalence drop that opens a new sub-tier")
                floor        = st.slider("Min prevalence",   0.01, 0.5,  0.10, 0.01, key="biz_floor")
                orphan_thresh= st.slider("Orphan threshold", 0.10, 1.0,  0.50, 0.05, key="orphan_thresh",
                                         help="Grant flagged orphan if max role prevalence < this value")
            hier_cfg = HierarchyConfig().model_copy(update=dict(
                staff_min_prevalence=t1_prev,
                tech_baseline_min_prevalence=t2_min,
                tech_baseline_max_prevalence=t2_max,
                business_gap_threshold=gap,
                business_min_prevalence=floor,
                orphan_grant_max_role_prevalence=orphan_thresh,
            ))
        else:
            tier_file_input = st.text_input(
                "tier_definitions.csv path",
                tier_defs if tier_defs else "",
                key="tier_file_widget",
            )
            tier_defs_file = tier_file_input or None
            k1, k2 = st.columns(2)
            with k1:
                st.caption("**Business Roles**")
                gap   = st.slider("Sub-tier gap",     0.05, 0.5, 0.20, 0.05, key="biz_gap",
                                  help="Prevalence drop that opens a new sub-tier")
                floor = st.slider("Min prevalence",   0.01, 0.5, 0.10, 0.01, key="biz_floor")
            with k2:
                st.caption("**Orphan grants**")
                orphan_thresh = st.slider(
                    "Orphan threshold", 0.10, 1.0, 0.50, 0.05, key="orphan_thresh",
                    help="Grant flagged orphan if max role prevalence < this value",
                )
            hier_cfg = HierarchyConfig().model_copy(update=dict(
                business_gap_threshold=gap,
                business_min_prevalence=floor,
                orphan_grant_max_role_prevalence=orphan_thresh,
            ))

    with run_col:
        st.markdown("<div style='height:2.8rem'></div>", unsafe_allow_html=True)
        run_ready = df_raw is not None and bool(enabled_algos)
        run_btn = st.button(
            "▶ Run Role Discovery",
            disabled=not run_ready,
            type="primary",
            use_container_width=True,
            key="run_btn",
        )
        if not run_ready:
            st.caption("Load data first" if df_raw is None else "Enable at least one algorithm")


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE EXECUTION
# ══════════════════════════════════════════════════════════════════════════════

if run_btn and run_ready:
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
        leiden=algo_configs.get("leiden", LeidenConfig()),
        nmf=algo_configs.get("nmf", NMFConfig()),
    )

    progress_bar = st.progress(0.0, text="Starting …")
    status_text  = st.empty()

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
# RESULTS
# ══════════════════════════════════════════════════════════════════════════════

result: PipelineResult | None = st.session_state.get("pipeline_result")

if result is None and df_raw is None:
    st.markdown(
        "<div style='text-align:center;color:#aaa;padding:2rem 0;font-size:0.9rem'>"
        "Enter data source paths above, load data, then run Role Discovery."
        "</div>",
        unsafe_allow_html=True,
    )

elif result is not None:
    st.divider()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Users",          f"{result.n_users:,}")
    m2.metric("Grants",         f"{result.n_grants:,}")
    m3.metric("Business Roles", f"{result.n_roles:,}")
    m4.metric("Runtime",        f"{result.elapsed_seconds:.1f}s")

    tabs = st.tabs([
        "🏗️ Tier Hierarchy",
        "🏢 Business Roles",
        "👥 User Assignments",
        "📊 Top Tranids",
        "📥 Downloads",
    ])

    # ── Tab 1: Tier Hierarchy ─────────────────────────────────────────────────
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

    # ── Tab 2: Business Roles ─────────────────────────────────────────────────
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

    # ── Tab 3: User Assignments ───────────────────────────────────────────────
    with tabs[2]:
        search = st.text_input("🔍 Search by ritsid", key="assign_search")

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
                primary_df = primary_df[
                    primary_df["ritsid"].str.contains(search, case=False, na=False)]
            st.subheader("Primary Role (1 per user)")
            st.dataframe(primary_df, use_container_width=True, hide_index=True)

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

    # ── Tab 4: Top Tranids ────────────────────────────────────────────────────
    with tabs[3]:
        if result.top_tranids is not None and not result.top_tranids.empty:
            st.caption(
                "Top tranids by distinct-employee coverage. "
                "Label tier 1/2 and save as `tier_definitions.csv`."
            )
            st.dataframe(result.top_tranids, use_container_width=True, hide_index=True)
        else:
            st.info("No top-tranid data.")

    # ── Tab 5: Downloads ──────────────────────────────────────────────────────
    with tabs[4]:
        st.subheader("Download Results")
        col1, col2 = st.columns(2)

        with col1:
            st.caption("**Hierarchy**")
            if result.tier_result:
                _download_btn(
                    pd.DataFrame(result.tier_result.hierarchy_rows),
                    "Tier Hierarchy (T1+T2)", "role_hierarchy_tiers.csv",
                )
            _download_btn(result.unified_hierarchy(), "Full Unified Hierarchy", "role_hierarchy_full.csv")
            _download_btn(result.top_tranids, "Top Tranids", "top_tranids.csv")

        with col2:
            for algo_name, ar in result.algorithm_results.items():
                st.caption(f"**{algo_name.upper()}**")
                _download_btn(ar.profiles,         "Role Profiles",           f"{algo_name}_profiles.csv")
                _download_btn(ar.biz_hierarchy,    "Business Role Hierarchy", f"{algo_name}_biz_hierarchy.csv")
                _download_btn(ar.unassigned_users, "Unassigned Users",        f"{algo_name}_unassigned_users.csv")
                _download_btn(ar.orphan_grants,    "Orphan Grants",           f"{algo_name}_orphan_grants.csv")
