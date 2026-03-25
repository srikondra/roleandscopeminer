"""
Access Delay Analysis — shareable HTML edition
===============================================
Produces a single self-contained .html file with:
  - All charts embedded as base64 (no external dependencies)
  - Full interactive org hierarchy drill-down (div › group › team › dept)
  - Works opened directly from OneDrive, SharePoint, Google Drive, or a USB stick
  - No server, no Python, no Node — recipients just double-click the file

Usage
-----
  pip install pandas numpy matplotlib

  # Basic — auto-detects hierarchy columns
  python access_delay_share.py --input data.csv

  # Explicit hierarchy order (outermost → innermost)
  python access_delay_share.py --input data.csv \
      --hierarchy division,group,team,dept

  # Scope to a subtree (aggregates all children beneath it)
  python access_delay_share.py --input data.csv \
      --filter "division=EMEA"
  python access_delay_share.py --input data.csv \
      --filter "division=EMEA,group=Markets"

  # Custom output filename
  python access_delay_share.py --input data.csv \
      --output "EMEA_access_delay_Q1.html"

  # Override which dept keywords count as revenue-generating
  python access_delay_share.py --input data.csv \
      --rev-depts "Sales,FX,IBD,Trading,Equities"
"""

import argparse, sys, base64, json
from pathlib import Path
from io import BytesIO

try:
    import pandas as pd
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
except ImportError:
    sys.exit("Missing dependencies. Run:  pip install pandas numpy matplotlib")

# ── defaults ──────────────────────────────────────────────────────────────────
DEFAULT_HIER   = ["division", "group", "team", "dept"]
DEFAULT_REV_KW = ["sales", "trader", "traders", "banker", "banking",
                  "revenue", "trading", "brokerage", "fx", "ecm", "dcm", "ibd"]
BUCKETS = [
    ("Day 0",  0,   0),
    ("1–3d",   1,   3),
    ("4–7d",   4,   7),
    ("8–14d",  8,  14),
    ("15–30d", 15, 30),
    ("31–60d", 31, 60),
    ("61d+",   61, 9999),
]
CHART_COLORS = {
    "rev":     "#E24B4A",
    "nonrev":  "#3B8BD4",
    "green":   "#1D9E75",
    "amber":   "#EF9F27",
    "neutral": "#888780",
}

# ── helpers ───────────────────────────────────────────────────────────────────
def nc(s):
    return s.lower().replace("-", "").replace("_", "").replace(" ", "")

def is_rev(s, kw):
    return any(k in s.lower() for k in kw)

def pct(n, t):
    return round(n / t * 100, 1) if t else 0.0

def fig_to_b64(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()

# ── data loading ──────────────────────────────────────────────────────────────
def load(path, hierarchy):
    df = pd.read_csv(path)
    rename = {}
    for c in df.columns:
        k = nc(c)
        if k in ("userid", "user_id"):                  rename[c] = "user_id"
        elif k in ("daysfromstart", "days", "daysstart"): rename[c] = "days"
        elif k in ("hiredate", "hire_date"):              rename[c] = "hire_date"
        elif k in ("requestdate", "request_date"):        rename[c] = "request_date"
        elif k in ("empid", "emp_id"):                    rename[c] = "emp_id"
        for lv in hierarchy:
            if k == nc(lv):
                rename[c] = lv
    df = df.rename(columns=rename)

    if "days" not in df.columns:
        if {"hire_date", "request_date"}.issubset(df.columns):
            df["hire_date"]    = pd.to_datetime(df["hire_date"],    errors="coerce")
            df["request_date"] = pd.to_datetime(df["request_date"], errors="coerce")
            df["days"] = (df["request_date"] - df["hire_date"]).dt.days
        else:
            sys.exit("Cannot find 'daysfromstart' column or date columns to compute it.")

    df["days"] = pd.to_numeric(df["days"], errors="coerce")
    df = df.dropna(subset=["days"])
    df["days"] = df["days"].astype(int)

    for lv in hierarchy:
        if lv not in df.columns:
            print(f"  [warn] column '{lv}' not found in CSV — filling with 'Unknown'")
            df[lv] = "Unknown"
        df[lv] = df[lv].fillna("Unknown").astype(str).str.strip()

    return df


def apply_filter(df, fstr, hierarchy):
    if not fstr:
        return df, "All data"
    parts = [p.strip() for p in fstr.split(",") if "=" in p]
    active = []
    for part in parts:
        col, val = part.split("=", 1)
        col, val = col.strip(), val.strip()
        matched = next((lv for lv in hierarchy if nc(lv) == nc(col)), col)
        df = df[df[matched] == val]
        active.append(val)
    return df, " › ".join(active) if active else "All data"

# ── analysis ──────────────────────────────────────────────────────────────────
def summarise(df, rev_kw):
    d, t = df["days"], len(df)
    hier_cols = [c for c in df.columns
                 if c not in ("days", "user_id", "hire_date", "request_date", "emp_id")]
    rev_mask = df.apply(
        lambda r: any(is_rev(str(r[c]), rev_kw) for c in hier_cols), axis=1
    )
    rev = df[rev_mask]
    non = df[~rev_mask]
    return dict(
        total       = t,
        avg         = round(float(d.mean()), 1),
        median      = round(float(d.median()), 1),
        p90         = int(d.quantile(0.90)),
        day0_pct    = pct((d == 0).sum(), t),
        over30_pct  = pct((d > 30).sum(), t),
        rev_count   = len(rev),
        rev_avg     = round(float(rev["days"].mean()), 1) if len(rev) else None,
        rev_over30  = pct((rev["days"] > 30).sum(), len(rev)) if len(rev) else None,
        non_avg     = round(float(non["days"].mean()), 1) if len(non) else None,
    )


def breakdown(df, col, rev_kw):
    rows = []
    for val, g in df.groupby(col):
        d = g["days"]
        rows.append(dict(
            name        = val,
            count       = len(g),
            avg         = round(float(d.mean()), 1),
            median      = round(float(d.median()), 1),
            p90         = int(d.quantile(0.90)),
            day0_pct    = pct((d == 0).sum(), len(g)),
            over30_pct  = pct((d > 30).sum(), len(g)),
            is_rev      = is_rev(val, rev_kw),
        ))
    return (pd.DataFrame(rows)
              .sort_values("avg", ascending=False)
              .reset_index(drop=True))

# ── chart generators (return base64 PNG strings) ──────────────────────────────
def setup_mpl():
    plt.rcParams.update({
        "font.family":     "sans-serif",
        "font.size":       11,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.grid":       True,
        "axes.grid.axis":  "y",
        "grid.alpha":      0.3,
        "figure.facecolor": "white",
        "axes.facecolor":  "white",
    })


def chart_dist(days):
    total  = len(days)
    counts = [int(((days >= lo) & (days <= hi)).sum()) for _, lo, hi in BUCKETS]
    labels = [b[0] for b in BUCKETS]
    colors = [CHART_COLORS["green"], CHART_COLORS["nonrev"], CHART_COLORS["neutral"],
              CHART_COLORS["amber"], CHART_COLORS["rev"],    CHART_COLORS["rev"],
              CHART_COLORS["rev"]]
    fig, ax = plt.subplots(figsize=(9, 3.8))
    bars = ax.bar(labels, counts, color=colors, width=0.6, zorder=3)
    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                f"{pct(cnt, total)}%", ha="center", va="bottom", fontsize=9, color="#555")
    ax.set_ylabel("Requests")
    ax.set_xlabel("Days from hire date to first access request")
    fig.tight_layout()
    return fig_to_b64(fig)


def chart_dept(bd, col):
    top    = bd.head(14)
    h      = max(4, len(top) * 0.45 + 1)
    fig, ax = plt.subplots(figsize=(9, h))
    colors = [CHART_COLORS["rev"] if r else CHART_COLORS["nonrev"]
              for r in top["is_rev"]]
    ax.barh(top["name"], top["avg"], color=colors, height=0.6, zorder=3)
    for i, (avg, cnt) in enumerate(zip(top["avg"], top["count"])):
        ax.text(avg + 0.15, i, f"{avg}d  (n={cnt})", va="center", fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Average days from hire date")
    rev_p = mpatches.Patch(color=CHART_COLORS["rev"],    label="Revenue dept")
    non_p = mpatches.Patch(color=CHART_COLORS["nonrev"], label="Non-revenue dept")
    ax.legend(handles=[rev_p, non_p], fontsize=9, framealpha=0.5)
    fig.tight_layout()
    return fig_to_b64(fig)


def chart_rev_comp(df, rev_kw):
    hier_cols = [c for c in df.columns
                 if c not in ("days", "user_id", "hire_date", "request_date", "emp_id")]
    rev_mask  = df.apply(lambda r: any(is_rev(str(r[c]), rev_kw) for c in hier_cols), axis=1)
    rev_days  = df[rev_mask]["days"]
    non_days  = df[~rev_mask]["days"]

    labels = [b[0] for b in BUCKETS]
    def pp(s, lo, hi):
        return pct(((s >= lo) & (s <= hi)).sum(), len(s)) if len(s) else 0

    pcts_r = [pp(rev_days, lo, hi) for _, lo, hi in BUCKETS]
    pcts_n = [pp(non_days, lo, hi) for _, lo, hi in BUCKETS]

    x, w = np.arange(len(labels)), 0.35
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.bar(x - w / 2, pcts_r, w, label="Revenue depts",
           color=CHART_COLORS["rev"],    zorder=3, alpha=0.9)
    ax.bar(x + w / 2, pcts_n, w, label="Non-revenue depts",
           color=CHART_COLORS["nonrev"], zorder=3, alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("% of employees in group")
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.legend(fontsize=9, framealpha=0.5)
    fig.tight_layout()
    return fig_to_b64(fig)


def chart_boxplot(df, col, rev_kw):
    top10 = (df.groupby(col)["days"]
               .count()
               .sort_values(ascending=False)
               .head(10)
               .index.tolist())
    sub   = df[df[col].isin(top10)]
    data_by = [sub[sub[col] == v]["days"].values for v in top10]
    h = max(4, len(top10) * 0.42 + 1)
    fig, ax = plt.subplots(figsize=(10, h))
    bp = ax.boxplot(data_by, patch_artist=True,
                    medianprops=dict(color="black", linewidth=1.5),
                    flierprops=dict(marker="o", markersize=3, alpha=0.4))
    for patch, v in zip(bp["boxes"], top10):
        patch.set_facecolor(
            CHART_COLORS["rev"] if is_rev(v, rev_kw) else CHART_COLORS["nonrev"])
        patch.set_alpha(0.7)
    ax.set_xticks(range(1, len(top10) + 1))
    ax.set_xticklabels(top10, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Days from hire date")
    rev_p = mpatches.Patch(color=CHART_COLORS["rev"],    alpha=0.7, label="Revenue dept")
    non_p = mpatches.Patch(color=CHART_COLORS["nonrev"], alpha=0.7, label="Non-revenue dept")
    ax.legend(handles=[rev_p, non_p], fontsize=9, framealpha=0.5)
    fig.tight_layout()
    return fig_to_b64(fig)

# ── HTML assembly ──────────────────────────────────────────────────────────────
def build_html(df, s, hierarchy, breakdowns_list, scope, charts_b64, out_path):

    # ── static summary section ────────────────────────────────────────────────
    def mc(label, value, sub=""):
        sub_html = f'<div class="ms">{sub}</div>' if sub else ""
        return (f'<div class="metric"><div class="ml">{label}</div>'
                f'<div class="mv">{value}</div>{sub_html}</div>')

    rev_avg_s  = f"{s['rev_avg']}d"  if s["rev_avg"]  is not None else "N/A"
    rev_o30_s  = f"{s['rev_over30']}%" if s["rev_over30"] is not None else "N/A"
    non_avg_s  = f"{s['non_avg']}d"  if s["non_avg"]  is not None else "N/A"

    metrics_html = f"""
    <div class="metrics">
      {mc("Total requests",    f"{s['total']:,}")}
      {mc("Average delay",     f"{s['avg']}d", f"median {s['median']}d &nbsp;·&nbsp; p90 {s['p90']}d")}
      {mc("Day-0 access",      f"{s['day0_pct']}%", "received access on hire date")}
      {mc("Delayed 30d+",      f"{s['over30_pct']}%")}
    </div>
    <div class="metrics">
      {mc("Revenue employees", f"{s['rev_count']:,}", "in revenue-generating depts")}
      {mc("Revenue avg delay", rev_avg_s, f"vs overall {s['avg']}d")}
      {mc("Revenue delayed 30d+", rev_o30_s)}
      {mc("Non-revenue avg",   non_avg_s)}
    </div>"""

    # ── static charts section ─────────────────────────────────────────────────
    def chart_card(title, b64, note=""):
        note_html = f'<p class="chart-note">{note}</p>' if note else ""
        return (f'<div class="chart-card"><div class="card-title">{title}</div>'
                f'{note_html}<img src="data:image/png;base64,{b64}" '
                f'style="width:100%;border-radius:6px"></div>')

    static_charts = ""
    idx = 0
    if idx < len(charts_b64):
        static_charts += chart_card("Delay distribution — all requests", charts_b64[idx]); idx += 1
    if idx < len(charts_b64):
        static_charts += chart_card("Revenue vs non-revenue: delay profile", charts_b64[idx]); idx += 1
    for lv, _ in breakdowns_list:
        if idx < len(charts_b64):
            static_charts += chart_card(f"Average delay by {lv}", charts_b64[idx]); idx += 1
        if idx < len(charts_b64):
            static_charts += chart_card(f"Delay spread by {lv} (top 10 by volume)", charts_b64[idx]); idx += 1

    # ── static breakdown tables ───────────────────────────────────────────────
    def bd_table(lv, bd):
        rows = ""
        for _, r in bd.iterrows():
            badge = (f'<span class="badge rev">revenue</span>' if r["is_rev"]
                     else '<span class="badge sup">support</span>')
            avg_style = 'style="color:#E24B4A;font-weight:500"' if r["avg"] > 14 else ""
            rows += (f'<tr><td>{r["name"]}</td><td>{badge}</td>'
                     f'<td class="r">{r["count"]}</td>'
                     f'<td class="r" {avg_style}>{r["avg"]}d</td>'
                     f'<td class="r">{int(r["median"])}d</td>'
                     f'<td class="r">{r["p90"]}d</td>'
                     f'<td class="r">{r["day0_pct"]}%</td>'
                     f'<td class="r">{r["over30_pct"]}%</td></tr>')
        return f"""
        <h2 class="section-h">{lv.title()} breakdown</h2>
        <div class="tw"><table>
          <thead><tr>
            <th>{lv}</th><th>type</th>
            <th class="r">n</th><th class="r">avg</th><th class="r">median</th>
            <th class="r">p90</th><th class="r">day-0 %</th><th class="r">30d+ %</th>
          </tr></thead>
          <tbody>{rows}</tbody>
        </table></div>"""

    all_bd_tables = "\n".join(bd_table(lv, bd) for lv, bd in breakdowns_list)

    # ── top delayed table ─────────────────────────────────────────────────────
    hier_cols = [c for c in hierarchy if c in df.columns]
    top_df    = df.sort_values("days", ascending=False).head(20)
    show_cols = ["user_id"] + hier_cols + ["hire_date", "days"]
    show_cols = [c for c in show_cols if c in top_df.columns]
    top_hdrs  = "".join(f'<th{"" if c != "days" else " class=r"}>{c.replace("_"," ")}</th>'
                        for c in show_cols)
    top_rows  = ""
    for _, r in top_df.iterrows():
        cells = ""
        for c in show_cols:
            val = r.get(c, "—")
            if c == "days":
                hi = ' class="hi"' if int(val) > 30 else ""
                cells += f'<td class="r"{hi}>{val}d</td>'
            else:
                cells += f"<td>{val}</td>"
        top_rows += f"<tr>{cells}</tr>"

    # ── interactive drill-down data ───────────────────────────────────────────
    # Serialise entire dataset as compact JSON for the in-page JS
    cols_needed = hier_cols + ["days"] + (["user_id"] if "user_id" in df.columns else [])
    cols_needed += [c for c in ["hire_date"] if c in df.columns]
    cols_needed = list(dict.fromkeys(cols_needed))  # dedup, preserve order
    records_json = df[cols_needed].to_json(orient="records")
    hierarchy_json = json.dumps(hierarchy)
    rev_kw_json    = json.dumps(DEFAULT_REV_KW)

    date_str = pd.Timestamp.now().strftime("%B %d, %Y")

    # ── full HTML ─────────────────────────────────────────────────────────────
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Access Delay Analysis — {scope}</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif;
  background:#F8F8F6;color:#2c2c2a;font-size:14px;line-height:1.6}}
.wrap{{max-width:1020px;margin:0 auto;padding:2rem 1.5rem}}

/* header */
.page-header{{margin-bottom:2rem}}
.page-header h1{{font-size:22px;font-weight:500;margin-bottom:.3rem}}
.page-header .sub{{font-size:13px;color:#73726c}}
.scope-pill{{display:inline-flex;align-items:center;gap:6px;background:#E6F1FB;
  color:#185FA5;font-size:12px;font-weight:500;padding:3px 12px;border-radius:20px;
  margin-bottom:1.5rem}}

/* tabs */
.tabs{{display:flex;gap:0;border-bottom:1.5px solid #D3D1C7;margin-bottom:1.5rem}}
.tab{{padding:.55rem 1.2rem;font-size:13px;font-weight:500;color:#73726c;
  cursor:pointer;border-bottom:2.5px solid transparent;margin-bottom:-1.5px;
  white-space:nowrap;transition:color .15s}}
.tab:hover{{color:#2c2c2a}}
.tab.active{{color:#185FA5;border-bottom-color:#185FA5}}
.tab-pane{{display:none}}.tab-pane.active{{display:block}}

/* metrics */
.metrics{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;margin-bottom:1rem}}
.metric{{background:#fff;border:.5px solid #D3D1C7;border-radius:10px;padding:.9rem 1rem}}
.ml{{font-size:11px;color:#73726c;text-transform:uppercase;letter-spacing:.06em;margin-bottom:3px}}
.mv{{font-size:22px;font-weight:500}}.ms{{font-size:11px;color:#888;margin-top:2px}}

/* charts */
.chart-card{{background:#fff;border:.5px solid #D3D1C7;border-radius:10px;
  padding:1rem 1.25rem;margin-bottom:1rem}}
.card-title{{font-size:13px;font-weight:500;margin-bottom:.75rem}}
.chart-note{{font-size:11px;color:#888;margin-bottom:.5rem}}

/* tables */
.section-h{{font-size:15px;font-weight:500;margin:1.75rem 0 .75rem;
  border-bottom:1px solid #D3D1C7;padding-bottom:5px}}
.tw{{overflow-x:auto;border-radius:8px;border:.5px solid #D3D1C7;margin-bottom:1.5rem}}
table{{width:100%;border-collapse:collapse;font-size:12px;background:#fff}}
th{{background:#f1efe8;font-weight:500;text-align:left;padding:7px 10px;
  border-bottom:1px solid #D3D1C7;font-size:11px;text-transform:uppercase;letter-spacing:.04em}}
td{{padding:6px 10px;border-bottom:.5px solid #e8e6df;color:#2c2c2a}}
tr:last-child td{{border-bottom:none}}
th.r,td.r{{text-align:right}}
.badge{{display:inline-block;padding:1px 7px;border-radius:4px;font-size:11px;font-weight:500}}
.badge.rev{{background:#FAECE7;color:#993C1D}}.badge.sup{{background:#E6F1FB;color:#185FA5}}
.hi{{color:#E24B4A;font-weight:500}}

/* interactive drill-down */
.selectors{{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));
  gap:10px;margin-bottom:1rem}}
.sel-group{{display:flex;flex-direction:column;gap:4px}}
.sel-lbl{{font-size:11px;font-weight:500;color:#73726c;text-transform:uppercase;letter-spacing:.06em}}
select{{width:100%;font-size:12px;padding:5px 8px;height:32px;
  color:#2c2c2a;background:#fff;border:.5px solid #B4B2A9;
  border-radius:8px;cursor:pointer;appearance:auto}}
select:disabled{{opacity:.4;cursor:not-allowed}}
.breadcrumb{{display:flex;align-items:center;gap:6px;font-size:12px;
  color:#73726c;margin-bottom:.75rem;flex-wrap:wrap;min-height:20px}}
.breadcrumb .bc-link{{color:#185FA5;cursor:pointer;text-decoration:underline}}
.breadcrumb .bc-sep{{color:#B4B2A9}}
.dyn-metrics{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));
  gap:10px;margin-bottom:1rem}}
.dyn-metric{{background:#F4F6FB;border-radius:8px;padding:.75rem 1rem}}
.dyn-metric .dl{{font-size:11px;color:#73726c;margin-bottom:2px}}
.dyn-metric .dv{{font-size:20px;font-weight:500}}
.dyn-metric .ds{{font-size:11px;color:#888;margin-top:1px}}
.children-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(170px,1fr));
  gap:8px;margin-bottom:1rem}}
.child-card{{background:#fff;border:.5px solid #D3D1C7;border-radius:8px;
  padding:.7rem .9rem;cursor:pointer;transition:border-color .15s,box-shadow .15s}}
.child-card:hover{{border-color:#378ADD;box-shadow:0 0 0 2px #E6F1FB}}
.child-name{{font-size:12px;font-weight:500;color:#2c2c2a;margin-bottom:3px;
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
.child-stat{{font-size:11px;color:#73726c}}
.bar-track{{height:4px;background:#E8E6DF;border-radius:2px;margin-top:6px}}
.bar-fill{{height:4px;border-radius:2px}}
.dyn-table-wrap{{overflow-x:auto;border-radius:8px;border:.5px solid #D3D1C7}}
.empty-msg{{font-size:12px;color:#888;padding:.5rem 0}}

/* footer */
.footer{{margin-top:3rem;padding-top:1rem;border-top:1px solid #D3D1C7;
  font-size:11px;color:#888;display:flex;justify-content:space-between}}

@media(max-width:640px){{
  .metrics,.dyn-metrics{{grid-template-columns:1fr 1fr}}
}}
</style>
</head>
<body>
<div class="wrap">

  <div class="page-header">
    <h1>Access delay analysis</h1>
    <div class="sub">Workforce access provisioning report &nbsp;·&nbsp; {date_str}</div>
  </div>
  <div class="scope-pill" id="topScopePill">{scope} &nbsp;·&nbsp; {s['total']:,} employees</div>

  <!-- ── TABS ── -->
  <div class="tabs">
    <div class="tab active" onclick="switchTab('overview',this)">Overview</div>
    <div class="tab" onclick="switchTab('interactive',this)">Interactive drill-down</div>
    <div class="tab" onclick="switchTab('records',this)">Top delayed records</div>
  </div>

  <!-- ══ TAB: OVERVIEW ══ -->
  <div id="tab-overview" class="tab-pane active">
    <h2 class="section-h">Summary metrics</h2>
    {metrics_html}
    <h2 class="section-h">Charts</h2>
    {static_charts}
    {all_bd_tables}
  </div>

  <!-- ══ TAB: INTERACTIVE ══ -->
  <div id="tab-interactive" class="tab-pane">
    <div class="selectors" id="selectors"></div>
    <div class="breadcrumb" id="breadcrumb"></div>
    <div class="scope-pill" id="dynScopePill" style="margin-bottom:1rem"></div>
    <div class="dyn-metrics" id="dynMetrics"></div>
    <div id="childSection">
      <h2 class="section-h" id="childTitle"></h2>
      <div class="children-grid" id="childCards"></div>
    </div>
    <h2 class="section-h">Delay distribution in scope</h2>
    <div class="chart-card">
      <canvas id="distCanvas" height="220"></canvas>
    </div>
    <h2 class="section-h" id="dynTableTitle"></h2>
    <div class="dyn-table-wrap">
      <table><thead id="dynTHead"></thead><tbody id="dynTBody"></tbody></table>
    </div>
  </div>

  <!-- ══ TAB: RECORDS ══ -->
  <div id="tab-records" class="tab-pane">
    <h2 class="section-h">Top 20 most delayed requests</h2>
    <div class="tw"><table>
      <thead><tr>{top_hdrs}</tr></thead>
      <tbody>{top_rows}</tbody>
    </table></div>
  </div>

  <div class="footer">
    <span>Generated by access_delay_share.py</span>
    <span>Scope: {scope} &nbsp;·&nbsp; {s['total']:,} records</span>
  </div>
</div>

<!-- ── embedded data ── -->
<script>
const ALL_DATA   = {records_json};
const HIERARCHY  = {hierarchy_json};
const REV_KW     = {rev_kw_json};
const BUCKETS    = ["Day 0","1-3d","4-7d","8-14d","15-30d","31-60d","61d+"];
const BUCKET_DEF = [[0,0],[1,3],[4,7],[8,14],[15,30],[31,60],[61,9999]];
</script>

<!-- ── Chart.js (CDN, gracefully absent offline) ── -->
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"
  onerror="window.Chart=null"></script>

<script>
/* ── tab switching ──────────────────────────────────────────────── */
function switchTab(id, el) {{
  document.querySelectorAll('.tab-pane').forEach(p=>p.classList.remove('active'));
  document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));
  document.getElementById('tab-'+id).classList.add('active');
  el.classList.add('active');
  if(id==='interactive') renderInteractive();
}}

/* ── helpers ────────────────────────────────────────────────────── */
const isRev = s => REV_KW.some(k=>s.toLowerCase().includes(k));
const pct   = (n,t) => t ? +(n/t*100).toFixed(1) : 0;
const avg   = arr => arr.length ? arr.reduce((a,b)=>a+b,0)/arr.length : 0;
const med   = arr => {{ const s=[...arr].sort((a,b)=>a-b); return s[Math.floor(s.length/2)]||0; }};

function filterData(sel) {{
  let d = ALL_DATA;
  HIERARCHY.forEach((lv,i)=>{{
    if(sel[i]) d=d.filter(r=>r[lv]===sel[i]);
  }});
  return d;
}}

function summariseData(rows) {{
  const days=rows.map(r=>r.days);
  const t=rows.length;
  const revRows=rows.filter(r=>HIERARCHY.some(lv=>isRev(r[lv]||'')));
  return {{
    total:t,
    avg:+(avg(days).toFixed(1)),
    median:+(med(days).toFixed(1)),
    day0pct:pct(days.filter(d=>d===0).length,t),
    over30pct:pct(days.filter(d=>d>30).length,t),
    revAvg:revRows.length ? +(avg(revRows.map(r=>r.days)).toFixed(1)) : null,
  }};
}}

function bucketCounts(rows) {{
  const days=rows.map(r=>r.days);
  return BUCKET_DEF.map(([lo,hi])=>days.filter(d=>d>=lo&&d<=hi).length);
}}

/* ── state ──────────────────────────────────────────────────────── */
let sel = {{}};
let distChartInst = null;

/* ── selector cascade ───────────────────────────────────────────── */
function buildSelectors() {{
  const wrap=document.getElementById('selectors');
  wrap.innerHTML='';
  HIERARCHY.forEach((lv,i)=>{{
    const g=document.createElement('div'); g.className='sel-group';
    g.innerHTML=`<div class="sel-lbl">${{lv}}</div>`+
      `<select id="sel_${{i}}" onchange="onSel(${{i}},this.value)"><option value="">All ${{lv}}s</option></select>`;
    wrap.appendChild(g);
  }});
  refreshSelects();
}}

function getUpstream(upTo) {{
  let d=ALL_DATA;
  for(let i=0;i<upTo;i++) if(sel[i]) d=d.filter(r=>r[HIERARCHY[i]]===sel[i]);
  return d;
}}

function refreshSelects() {{
  HIERARCHY.forEach((lv,i)=>{{
    const el=document.getElementById('sel_'+i); if(!el) return;
    const up=getUpstream(i);
    const opts=[...new Set(up.map(r=>r[lv]).filter(Boolean))].sort();
    el.innerHTML=`<option value="">All ${{lv}}s</option>`+
      opts.map(o=>`<option value="${{o}}"${{o===sel[i]?' selected':''}}>${{o}}</option>`).join('');
    el.disabled = i>0 && !sel[i-1];
  }});
}}

function onSel(level,val) {{
  sel[level]=val||undefined;
  for(let i=level+1;i<HIERARCHY.length;i++) delete sel[i];
  refreshSelects();
  renderInteractive();
}}

/* ── breadcrumb ─────────────────────────────────────────────────── */
function renderBreadcrumb() {{
  const bc=document.getElementById('breadcrumb');
  const parts=[`<span class="bc-link" onclick="resetTo(-1)">All</span>`];
  HIERARCHY.forEach((lv,i)=>{{
    if(!sel[i]) return;
    parts.push(`<span class="bc-sep">›</span>`);
    const isLast=!sel[i+1];
    parts.push(isLast ? `<span>${{sel[i]}}</span>`
      : `<span class="bc-link" onclick="resetTo(${{i}})">${{sel[i]}}</span>`);
  }});
  bc.innerHTML=parts.join('');
}}

function resetTo(level) {{
  if(level===-1) {{ sel={{}}; }}
  else {{ for(let i=level+1;i<HIERARCHY.length;i++) delete sel[i]; }}
  refreshSelects();
  renderInteractive();
}}

/* ── dynamic metrics strip ──────────────────────────────────────── */
function renderMetrics(rows) {{
  const sm=summariseData(rows);
  const dm=document.getElementById('dynMetrics');
  const card=(lbl,val,sub='')=>
    `<div class="dyn-metric"><div class="dl">${{lbl}}</div><div class="dv">${{val}}</div>`+
    (sub?`<div class="ds">${{sub}}</div>`:'')+`</div>`;
  dm.innerHTML=
    card('Employees',sm.total.toLocaleString())+
    card('Avg delay',sm.avg+'d',`median ${{sm.median}}d`)+
    card('Day-0 access',sm.day0pct+'%')+
    card('Delayed 30d+',sm.over30pct+'%');
}}

/* ── child breakdown cards ──────────────────────────────────────── */
function renderChildren(rows) {{
  const deepest=Object.keys(sel).filter(k=>sel[k]).map(Number);
  const nextLevel = deepest.length ? Math.max(...deepest)+1 : 0;
  const title=document.getElementById('childTitle');
  const grid=document.getElementById('childCards');

  if(nextLevel>=HIERARCHY.length) {{
    title.textContent='Leaf level reached';
    grid.innerHTML='<div class="empty-msg">No further breakdown available at this level.</div>';
    return;
  }}
  const lv=HIERARCHY[nextLevel];
  title.textContent=lv.charAt(0).toUpperCase()+lv.slice(1)+' breakdown — click to drill in';

  const groups={{}};
  rows.forEach(r=>{{ const k=r[lv]||'Unknown'; if(!groups[k]) groups[k]=[]; groups[k].push(r.days); }});
  const stats=Object.entries(groups)
    .map(([k,arr])=>({{k,avg:+(avg(arr).toFixed(1)),count:arr.length}}))
    .sort((a,b)=>b.avg-a.avg);
  const maxAvg=Math.max(...stats.map(s=>s.avg),1);

  grid.innerHTML=stats.slice(0,16).map(s=>{{
    const barPct=Math.round(s.avg/maxAvg*100);
    const col=s.avg>30?'#E24B4A':s.avg>14?'#EF9F27':'#1D9E75';
    const revBadge=isRev(s.k)?'<span class="badge rev" style="font-size:10px;margin-left:4px">rev</span>':'';
    return `<div class="child-card" onclick="drillInto(${{nextLevel}},'${{s.k.replace(/'/g,"\\'")}}')">
      <div class="child-name">${{s.k}}${{revBadge}}</div>
      <div class="child-stat">${{s.avg}}d avg &nbsp;·&nbsp; ${{s.count}} people</div>
      <div class="bar-track"><div class="bar-fill" style="width:${{barPct}}%;background:${{col}}"></div></div>
    </div>`;
  }}).join('');
}}

function drillInto(level,val) {{
  sel[level]=val;
  for(let i=level+1;i<HIERARCHY.length;i++) delete sel[i];
  const el=document.getElementById('sel_'+level);
  if(el) el.value=val;
  refreshSelects();
  renderInteractive();
}}

/* ── distribution chart ─────────────────────────────────────────── */
function renderDistChart(rows) {{
  const counts=bucketCounts(rows);
  const total=rows.length;
  const colors=['#1D9E75','#3B8BD4','#888780','#EF9F27','#E24B4A','#E24B4A','#E24B4A'];
  if(!window.Chart) return;
  if(distChartInst) {{ distChartInst.destroy(); distChartInst=null; }}
  distChartInst=new Chart(document.getElementById('distCanvas'),{{
    type:'bar',
    data:{{labels:BUCKETS,datasets:[{{data:counts,backgroundColor:colors,borderRadius:3,borderSkipped:false}}]}},
    options:{{responsive:true,maintainAspectRatio:false,
      plugins:{{legend:{{display:false}},
        tooltip:{{callbacks:{{label:ctx=>`${{ctx.raw}} (${{pct(ctx.raw,total)}}%)`}}}}}},
      scales:{{x:{{ticks:{{font:{{size:11}}}}}},y:{{ticks:{{font:{{size:11}}}},beginAtZero:true}}}}
    }}
  }});
}}

/* ── scoped table ───────────────────────────────────────────────── */
function renderTable(rows) {{
  const title=document.getElementById('dynTableTitle');
  const top=rows.slice().sort((a,b)=>b.days-a.days).slice(0,15);
  title.textContent=`Top delayed in scope (${{rows.length.toLocaleString()}} total)`;

  const cols=['user_id',...HIERARCHY,'hire_date','days'].filter(c=>top[0]&&c in top[0]);
  document.getElementById('dynTHead').innerHTML=
    '<tr>'+cols.map(c=>`<th${{c==='days'?' class=r':''}}>${{c.replace(/_/g,' ')}}</th>`).join('')+'</tr>';
  document.getElementById('dynTBody').innerHTML=top.map(r=>
    '<tr>'+cols.map(c=>{{
      if(c==='days') return `<td class="r${{r.days>30?' hi':''}}"> ${{r.days}}d</td>`;
      const v=r[c]||'—';
      if(HIERARCHY.includes(c)&&isRev(v)) return `<td><span class="badge rev">${{v}}</span></td>`;
      return `<td>${{v}}</td>`;
    }}).join('')+'</tr>'
  ).join('');
}}

/* ── main render ────────────────────────────────────────────────── */
function renderInteractive() {{
  const rows=filterData(sel);
  renderBreadcrumb();
  const deepest=Object.entries(sel).filter(([,v])=>v);
  const scopeLabel=deepest.length
    ? deepest.map(([,v])=>v).join(' › ')
    : 'All data';
  document.getElementById('dynScopePill').textContent=
    `${{scopeLabel}} · ${{rows.length.toLocaleString()}} employees`;
  renderMetrics(rows);
  renderChildren(rows);
  renderDistChart(rows);
  renderTable(rows);
}}

/* ── boot ───────────────────────────────────────────────────────── */
buildSelectors();
renderInteractive();
</script>
</body>
</html>"""

    out_path.write_text(html, encoding="utf-8")
    kb = out_path.stat().st_size // 1024
    print(f"  Output → {out_path}  ({kb} KB)")


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Access delay analysis — single self-contained HTML output",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--input",     "-i", required=True, help="Path to input CSV")
    ap.add_argument("--output",    "-o", default="access_delay_report.html",
                    help="Output HTML filename (default: access_delay_report.html)")
    ap.add_argument("--hierarchy", default="",
                    help="Comma-separated hierarchy columns outermost→innermost "
                         "(default: division,group,team,dept)")
    ap.add_argument("--filter",    default="",
                    help='Scope filter e.g. "division=EMEA" or "division=EMEA,group=Markets"')
    ap.add_argument("--rev-depts", default="",
                    help="Comma-separated revenue dept keywords (overrides defaults)")
    args = ap.parse_args()

    hierarchy = (
        [c.strip() for c in args.hierarchy.split(",") if c.strip()]
        if args.hierarchy else DEFAULT_HIER
    )
    rev_kw = (
        [k.strip().lower() for k in args.rev_depts.split(",") if k.strip()]
        if args.rev_depts else DEFAULT_REV_KW
    )

    print(f"Loading {args.input} ...")
    df = load(args.input, hierarchy)
    df, scope = apply_filter(df, args.filter, hierarchy)
    print(f"  {len(df):,} records · {df[hierarchy[0]].nunique() if hierarchy[0] in df else '?'} "
          f"top-level orgs · scope: {scope}")

    s = summarise(df, rev_kw)
    print(f"\n  Avg delay    : {s['avg']}d  (median {s['median']}d, p90 {s['p90']}d)")
    print(f"  Day-0 access : {s['day0_pct']}%")
    print(f"  Delayed 30d+ : {s['over30_pct']}%")
    if s["rev_avg"]:
        print(f"  Revenue avg  : {s['rev_avg']}d  ({s['rev_count']} employees)")

    print("\nGenerating charts ...")
    setup_mpl()
    charts_b64     = []
    breakdowns_list = []

    charts_b64.append(chart_dist(df["days"]))
    charts_b64.append(chart_rev_comp(df, rev_kw))

    for lv in hierarchy:
        if df[lv].nunique() < 2:
            continue
        bd = breakdown(df, lv, rev_kw)
        breakdowns_list.append((lv, bd))
        charts_b64.append(chart_dept(bd, lv))
        if df[lv].nunique() <= 25:
            charts_b64.append(chart_boxplot(df, lv, rev_kw))

    print(f"  {len(charts_b64)} charts embedded")

    out_path = Path(args.output)
    print(f"\nBuilding HTML report ...")
    build_html(df, s, hierarchy, breakdowns_list, scope, charts_b64, out_path)
    print("\nDone. Upload this single file to OneDrive/SharePoint/Google Drive and share the link.")


if __name__ == "__main__":
    main()
