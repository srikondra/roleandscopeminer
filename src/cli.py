"""
IGA Role Miner — command-line entry point.

Usage
-----
# Full pipeline
python -m src.cli --ents data/ents.csv --hr data/hr.csv

# Top-N tranid report only (no HR join needed)
python -m src.cli --ents data/ents.csv --hr data/hr.csv --top-tranids 50

# With pre-defined tier definitions
python -m src.cli --ents ... --hr ... --tier-defs tier_definitions.csv

# Launch the web UI
python -m src.cli --ui
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("role_miner.cli")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="role_miner",
        description="IGA Role Miner — discovers enterprise role candidates from entitlement data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--ents",       metavar="PATH", help="Entitlements CSV path")
    p.add_argument("--hr",         metavar="PATH", help="Employees CSV path")
    p.add_argument("--apps",       metavar="PATH", help="Applications CSV path (optional)")
    p.add_argument("--out",        metavar="DIR",  default="output", help="Output directory")
    p.add_argument("--sample",     type=int,       default=0,        metavar="N",
                   help="Limit to N employees (0 = all)")
    p.add_argument("--tier-defs",  metavar="PATH",
                   help="Pre-defined tier CSV (tier, tranid[, notes])")
    p.add_argument("--top-tranids", type=int,      default=None,     metavar="N",
                   help="Produce top-N tranid report and exit (no full pipeline)")
    p.add_argument("--no-louvain", action="store_true", help="Disable Louvain algorithm")
    p.add_argument("--no-nmf",     action="store_true", help="Disable NMF algorithm")
    p.add_argument("--ui",         action="store_true", help="Launch the Streamlit web UI")
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    # ── Launch UI ──────────────────────────────────────────────────────────────
    if args.ui:
        import subprocess
        ui_path = Path(__file__).parent / "ui" / "app.py"
        log.info("Launching Streamlit UI: %s", ui_path)
        subprocess.run(["streamlit", "run", str(ui_path)], check=True)
        return

    # ── Build config ───────────────────────────────────────────────────────────
    from src.config import PipelineConfig

    # Use CONFIG defaults when paths not supplied
    cfg_kwargs: dict = {"output_dir": args.out}
    if args.ents:
        cfg_kwargs["csv_entitlements"] = args.ents
    if args.hr:
        cfg_kwargs["csv_employees"] = args.hr
    if args.apps:
        cfg_kwargs["csv_applications"] = args.apps
    if args.sample:
        cfg_kwargs["sample_size"] = args.sample
    if args.tier_defs:
        cfg_kwargs["tier_definitions_file"] = args.tier_defs

    enabled: list[str] = []
    if not args.no_louvain:
        enabled.append("louvain")
    if not args.no_nmf:
        enabled.append("nmf")
    cfg_kwargs["enabled_algorithms"] = enabled

    cfg = PipelineConfig(**cfg_kwargs)

    # ── Top-tranid report only ─────────────────────────────────────────────────
    if args.top_tranids is not None:
        from src.data import DataLoader, top_tranid_by_population

        log.info("Running top-tranid report (N=%d) …",
                 args.top_tranids if args.top_tranids > 0 else 50)
        df     = DataLoader(cfg).load()
        n      = args.top_tranids if args.top_tranids > 0 else 50
        result = top_tranid_by_population(df, n=n,
                                          ignore_csiid=cfg.top_tranid_ignore_csiid)

        out = Path(cfg.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        p  = out / f"top_tranids_{ts}.csv"
        result.to_csv(p, index=False)
        log.info("Saved %s  (%d rows)", p, len(result))
        return

    # ── Full pipeline ──────────────────────────────────────────────────────────
    from src.pipeline import PipelineRunner
    PipelineRunner(cfg).run()


if __name__ == "__main__":
    main()
