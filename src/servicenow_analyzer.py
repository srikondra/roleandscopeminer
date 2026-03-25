import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import date, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
import requests

# ── Apigee config (replace with real values or load from env/secrets) ─────────
APIGEE_GRAPHQL_URL   = "https://api.dummy-corp.com/servicenow/v1/graphql"
APIGEE_CLIENT_ID     = "dummyClientId_abc123"
APIGEE_CLIENT_SECRET = "dummyClientSecret_xyz789"


def graphql_request(query: str) -> dict:
    """Execute a GraphQL query against the Apigee-proxied endpoint.

    Apigee authenticates via client_id / client_secret passed as query parameters.
    """
    response = requests.post(
        APIGEE_GRAPHQL_URL,
        json={"query": query},
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
            "X-IBM-Client-Id": APIGEE_CLIENT_ID,
            "X-IBM-Client-Secret": APIGEE_CLIENT_SECRET,
        },
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    if "errors" in payload:
        raise RuntimeError(f"GraphQL errors: {payload['errors']}")
    return payload["data"]


st.set_page_config(page_title="ServiceNow Self-Assigned Analyzer", layout="wide")
st.title("ServiceNow Incident Analyzer")
st.caption("Find incidents where creator and assignee are the same person")

# ── Sidebar filters ──────────────────────────────────────────────────────────
st.sidebar.header("Filters")
today = date.today()
start_date = st.sidebar.date_input("Start date", today - timedelta(days=30))
end_date = st.sidebar.date_input("End date", today)
run = st.sidebar.button("Analyze", type="primary", use_container_width=True)


# ── GraphQL query display ─────────────────────────────────────────────────────
PAGE_SIZE   = 100
MAX_WORKERS = 20  # max concurrent daily-slot requests

REST_USER_GEID   = "Rest_User"
ROUTINE_PATTERN  = re.compile(r"daily|weekly\s+checkouts?", re.IGNORECASE)


def build_query(start: date, end: date, page: int = 1, size: int = PAGE_SIZE) -> str:
    # Cross-field equality (createdByGeid == assignedToGeid) is not supported in the
    # custom GraphQL schema — fetch by date range, then post-filter client-side.
    return f"""query GetSelfAssignedIncidents {{
  incidents(
    filter: {{
      createdTimestamp: {{
        greaterThenEqualTo: "{start.isoformat()}T00:00:00Z"
        lessthanEqualTo: "{end.isoformat()}T23:59:59Z"
      }}
    }}
    page: {page}
    size: {size}
  ) {{
    id
    number
    createdByGeid
    assignedToGeid
    callerGeid
    shortDescription
    categoryType
    originatingGroupName
    assignmentGroupName
    createdTimestamp
  }}
}}"""


def fetch_day(day: date) -> list[dict]:
    """Fetch all records for a single day, paging within the day if needed."""
    records = graphql_request(build_query(day, day, page=1))["incidents"]
    page = 2
    while len(records) % PAGE_SIZE == 0 and records:
        more = graphql_request(build_query(day, day, page=page))["incidents"]
        records.extend(more)
        if len(more) < PAGE_SIZE:
            break
        page += 1
    return records


def classify_buckets(df: pd.DataFrame) -> pd.DataFrame:
    """Assign each incident to exactly one bucket (priority: B2 > B1 > B3)."""
    df = df.copy()
    df["bucket"] = "Other"
    # Apply lowest priority first; higher priority overwrites
    df.loc[df["shortDescription"].str.contains(ROUTINE_PATTERN, na=False), "bucket"] = "B3: Routine"
    df.loc[df["createdByGeid"] == df["assignedToGeid"],                     "bucket"] = "B1: Self-Assigned"
    df.loc[df["createdByGeid"] == REST_USER_GEID,                           "bucket"] = "B2: Rest User"
    return df


def fetch_all_incidents(start: date, end: date) -> tuple[pd.DataFrame, int]:
    """Fan out one request per day in parallel, classify into buckets."""
    days = [start + timedelta(days=i) for i in range((end - start).days + 1)]

    all_records: list[dict] = []
    with ThreadPoolExecutor(max_workers=min(len(days), MAX_WORKERS)) as ex:
        futures = {ex.submit(fetch_day, day): day for day in days}
        for fut in as_completed(futures):
            all_records.extend(fut.result())

    total = len(all_records)
    if not all_records:
        return pd.DataFrame(), total

    df = pd.DataFrame(all_records)
    df["createdTimestamp"] = pd.to_datetime(df["createdTimestamp"])
    df = classify_buckets(df)
    return df, total



# ── Session state ─────────────────────────────────────────────────────────────
if "df" not in st.session_state:
    st.session_state.df = None
    st.session_state.total = 0

if run:
    with st.spinner("Fetching incidents..."):
        df, total = fetch_all_incidents(start_date, end_date)
        st.session_state.df    = df
        st.session_state.total = total

# ── Show GraphQL query ────────────────────────────────────────────────────────
with st.expander("GraphQL query", expanded=False):
    st.code(build_query(start_date, end_date), language="graphql")

# ── Results ───────────────────────────────────────────────────────────────────
df = st.session_state.df
total = st.session_state.total

if df is not None and not df.empty:
    self_count  = len(df)
    pct         = self_count / total * 100 if total else 0
    unique_users = df["createdByGeid"].nunique()
    top_group   = df["assignmentGroupName"].value_counts().idxmax()

    # Metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total incidents", f"{total:,}", help="All incidents in period")
    c2.metric("Self-assigned",   f"{self_count:,}", f"{pct:.1f}% of total")
    c3.metric("Unique users",    unique_users,   help="Users who self-assigned")
    c4.metric("Top group",       top_group,      help="Most self-assignments")

    st.divider()

    # Charts
    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("Weekly trend")
        df["week"] = df["createdTimestamp"].dt.to_period("W").dt.start_time
        weekly = df.groupby("week").size().reset_index(name="count")
        fig_trend = px.bar(
            weekly, x="week", y="count",
            labels={"week": "Week", "count": "Self-assigned"},
            color_discrete_sequence=["#378ADD"],
        )
        fig_trend.update_layout(
            margin=dict(l=0, r=0, t=8, b=0),
            height=240,
            showlegend=False,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        fig_trend.update_xaxes(showgrid=False)
        fig_trend.update_yaxes(gridcolor="rgba(0,0,0,0.06)")
        st.plotly_chart(fig_trend, use_container_width=True)

    with col_r:
        st.subheader("By assignment group")
        grp = df["assignmentGroupName"].value_counts().reset_index()
        grp.columns = ["group", "count"]
        fig_grp = px.bar(
            grp, x="count", y="group", orientation="h",
            labels={"group": "", "count": "Count"},
            color_discrete_sequence=["#1D9E75"],
        )
        fig_grp.update_layout(
            margin=dict(l=0, r=0, t=8, b=0),
            height=240,
            showlegend=False,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            yaxis={"categoryorder": "total ascending"},
        )
        fig_grp.update_xaxes(gridcolor="rgba(0,0,0,0.06)")
        fig_grp.update_yaxes(showgrid=False)
        st.plotly_chart(fig_grp, use_container_width=True)

    st.divider()

    # Tabs
    tab_inc, tab_usr, tab_cat = st.tabs(["Incidents", "By user", "By category"])

    with tab_inc:
        display_df = df[[
            "number", "createdByGeid", "categoryType",
            "assignmentGroupName", "shortDescription", "createdTimestamp"
        ]].copy()
        display_df.columns = [
            "Number", "GEID", "Category",
            "Assignment group", "Short description", "Created"
        ]
        display_df["Created"] = display_df["Created"].dt.strftime("%Y-%m-%d %H:%M")
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    with tab_usr:
        user_df = (
            df.groupby("createdByGeid")
            .agg(
                self_assigned=("id", "count"),
                groups=("assignmentGroupName", lambda x: ", ".join(sorted(set(x)))),
                categories=("categoryType", lambda x: ", ".join(sorted(set(x)))),
            )
            .reset_index()
            .sort_values("self_assigned", ascending=False)
        )
        user_df.columns = ["GEID", "Self-assigned count", "Groups", "Categories"]
        st.dataframe(user_df, use_container_width=True, hide_index=True)

    with tab_cat:
        cat_df = df["categoryType"].value_counts().reset_index()
        cat_df.columns = ["Category", "Count"]
        cat_df["% of self-assigned"] = (cat_df["Count"] / self_count * 100).round(1)
        fig_cat = px.pie(
            cat_df, names="Category", values="Count",
            color_discrete_sequence=px.colors.qualitative.Pastel,
        )
        fig_cat.update_layout(margin=dict(l=0, r=0, t=8, b=0), height=320)
        st.plotly_chart(fig_cat, use_container_width=True)


elif df is not None and df.empty:
    st.info("No self-assigned incidents found for the selected period.")
else:
    st.info("Set a date range and click **Analyze** to get started.")
