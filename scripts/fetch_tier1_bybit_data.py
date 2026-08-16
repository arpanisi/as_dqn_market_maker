#!/usr/bin/env python
"""Fetch real Bybit public data for the documented Tier 1 window."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data_pipeline.bybit_fetch import FUNDING_PROVIDER_BYBIT, FUNDING_PROVIDER_COINALYZE, fetch_tier1_bybit_data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--category", default="linear")
    parser.add_argument("--start", default="2026-07-01T00:00:00Z")
    parser.add_argument("--end", default="2026-07-02T00:00:00Z")
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--cadence", default="5s")
    parser.add_argument("--funding-source-csv", default=None)
    parser.add_argument(
        "--funding-provider",
        default=FUNDING_PROVIDER_COINALYZE,
        choices=(FUNDING_PROVIDER_COINALYZE, FUNDING_PROVIDER_BYBIT),
        help=(
            "Funding-rate source: 'coinalyze' (default, Tier 1/Tier 2) or "
            "'bybit' (Bybit's own V5 endpoint, Tier 3 source of record)."
        ),
    )
    parser.add_argument("--bybit-api-base", default="https://api.bybit.com")
    args = parser.parse_args()

    result = fetch_tier1_bybit_data(
        symbol=args.symbol,
        category=args.category,
        start=args.start,
        end=args.end,
        output_dir=ROOT / args.output_dir,
        cadence=args.cadence,
        funding_source_csv=args.funding_source_csv,
        funding_provider=args.funding_provider,
        bybit_api_base=args.bybit_api_base,
    )
    print(
        json.dumps(
            {
                "symbol": result.symbol,
                "start": result.start.isoformat().replace("+00:00", "Z"),
                "end": result.end.isoformat().replace("+00:00", "Z"),
                "trades": result.trades,
                "mid_prices": result.mid_prices,
                "book_snapshots": result.book_snapshots,
                "funding_rates": result.funding_rates,
                "raw_files": [str(path.relative_to(ROOT)) for path in result.raw_files],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
