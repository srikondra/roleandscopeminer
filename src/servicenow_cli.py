"""
CLI runner for ServiceNow incident analysis.
Fetches incidents via Apigee, classifies into buckets, writes CSVs to output/.

Usage:
    python src/servicenow_cli.py
    python src/servicenow_cli.py --start 2025-01-01 --end 2025-01-31
    python src/servicenow_cli.py --start 2025-01-01 --end 2025-01-31 --output results/
"""

import argparse
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def _make_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=3,
        connect=3,
        backoff_factor=0.5,
        status_forcelist=[500, 502, 503, 504],
        allowed_methods=["POST"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.verify = SSL_CA_BUNDLE
    session.cert   = (SSL_CLIENT_CERT, SSL_CLIENT_KEY)
    return session


SESSION = _make_session()

# ── Apigee config ─────────────────────────────────────────────────────────────
APIGEE_GRAPHQL_URL = "https://api.dummy-corp.com/servicenow/v1/graphql"
APIGEE_CLIENT_ID     = "dummyClientId_abc123"
APIGEE_CLIENT_SECRET = "dummyClientSecret_xyz789"

# Server verification: path to corporate CA bundle (.pem)
SSL_CA_BUNDLE = "certs/corporate-ca-bundle.pem"

# Client certificate for mutual TLS (mTLS) — required by this Apigee instance.
# Ask your infra/security team for these files.
SSL_CLIENT_CERT = "certs/client.crt"   # client certificate
SSL_CLIENT_KEY  = "certs/client.key"   # private key (no passphrase)

PAGE_SIZE       = 100
MAX_WORKERS     = 20
REST_USER_GEID  = "Rest_User"
ROUTINE_PATTERN = re.compile(r"daily|weekly\s+checkouts?", re.IGNORECASE)


# ── API ───────────────────────────────────────────────────────────────────────
def graphql_request(query: str) -> dict:
    response = SESSION.post(
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


def build_query(start: date, end: date, page: int = 1, size: int = PAGE_SIZE) -> str:
    return f"""query GetIncidents {{
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
    interactionRelations {{
      id
      interaction {{
        number
        type
        workNotes
      }}
    }}
  }}
}}"""


def _flatten_interactions(df: pd.DataFrame) -> pd.DataFrame:
    def extract(relations):
        if not relations:
            return None, None, None
        interaction = (relations[0] or {}).get("interaction") or {}
        return interaction.get("number"), interaction.get("type"), interaction.get("workNotes")

    df = df.copy()
    df[["interaction_number", "interaction_type", "interaction_workNotes"]] = df["interactionRelations"].apply(
        lambda r: pd.Series(extract(r))
    )
    return df.drop(columns=["interactionRelations"])


def fetch_day(day: date) -> list[dict]:
    records = graphql_request(build_query(day, day, page=1))["incidents"]
    page = 2
    while len(records) % PAGE_SIZE == 0 and records:
        more = graphql_request(build_query(day, day, page=page))["incidents"]
        records.extend(more)
        if len(more) < PAGE_SIZE:
            break
        page += 1
    return records


def fetch_all_incidents(start: date, end: date) -> tuple[pd.DataFrame, int]:
    days = [start + timedelta(days=i) for i in range((end - start).days + 1)]
    print(f"Fetching {len(days)} days ({start} → {end}) with up to {MAX_WORKERS} parallel workers...")

    all_records: list[dict] = []
    with ThreadPoolExecutor(max_workers=min(len(days), MAX_WORKERS)) as ex:
        futures = {ex.submit(fetch_day, day): day for day in days}
        for i, fut in enumerate(as_completed(futures), 1):
            day = futures[fut]
            try:
                records = fut.result()
                all_records.extend(records)
                print(f"  [{i}/{len(days)}] {day}: {len(records)} records")
            except Exception as e:
                print(f"  [{i}/{len(days)}] {day}: ERROR — {e}", file=sys.stderr)

    total = len(all_records)
    if not all_records:
        return pd.DataFrame(), total

    df = pd.DataFrame(all_records)
    df["createdTimestamp"] = pd.to_datetime(df["createdTimestamp"])
    df = df.sort_values("createdTimestamp").reset_index(drop=True)
    df = _flatten_interactions(df)
    return df, total


def classify_buckets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["bucket"] = "Other"
    df.loc[df["shortDescription"].str.contains(ROUTINE_PATTERN, na=False), "bucket"] = "B3: Routine"
    df.loc[df["createdByGeid"] == df["assignedToGeid"],                     "bucket"] = "B1: Self-Assigned"
    df.loc[df["createdByGeid"] == REST_USER_GEID,                           "bucket"] = "B2: Rest User"
    return df


# ── Output ────────────────────────────────────────────────────────────────────
def write_outputs(df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for bucket, filename in [
        ("B2: Rest User",     "b2_rest_user.csv"),
        ("B1: Self-Assigned", "b1_self_assigned.csv"),
        ("B3: Routine",       "b3_routine.csv"),
    ]:
        path = output_dir / filename
        df[df["bucket"] == bucket].to_csv(path, index=False)
        print(f"  {bucket:<20} → {path}")


# ── Entry point ───────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="ServiceNow incident bucket analysis")
    parser.add_argument("--start",  default=str(date.today() - timedelta(days=30)),
                        help="Start date YYYY-MM-DD (default: 30 days ago)")
    parser.add_argument("--end",    default=str(date.today()),
                        help="End date YYYY-MM-DD (default: today)")
    parser.add_argument("--output", default="output",
                        help="Output directory (default: output/)")
    args = parser.parse_args()

    start = date.fromisoformat(args.start)
    end   = date.fromisoformat(args.end)
    if end < start:
        print("Error: --end must be >= --start", file=sys.stderr)
        sys.exit(1)

    df, _ = fetch_all_incidents(start, end)
    if df.empty:
        print("No incidents found for the given date range.")
        return

    df = classify_buckets(df)
    write_outputs(df, Path(args.output))


if __name__ == "__main__":
    main()
