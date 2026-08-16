#!/usr/bin/env python
"""Tier 1 smoke test: one day, fixed derived gamma/skew=0."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from backtest.engine import FixedPolicy, run_fixed_policy_backtest
from data_pipeline.loaders import load_book_snapshots_csv, load_funding_csv, load_mid_prices_csv, load_trades_csv
from settings import DEFAULT_SETTINGS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mid-prices", required=True)
    parser.add_argument("--books", required=True)
    parser.add_argument("--trades", required=True)
    parser.add_argument("--funding", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--output", default="outputs/tier1/tier1_summary.json")
    parser.add_argument("--pnl-output", default="outputs/tier1/tier1_cycle_pnl.csv")
    args = parser.parse_args()

    funding_list = load_funding_csv(args.funding)
    tier1_gamma = DEFAULT_SETTINGS.market.tier1_risk_aversion
    result = run_fixed_policy_backtest(
        mid_prices=load_mid_prices_csv(args.mid_prices),
        books=load_book_snapshots_csv(args.books),
        trades=load_trades_csv(args.trades),
        funding_rates=funding_list,
        policy=FixedPolicy(tier1_gamma, 0.0),
        start=args.start,
        end=args.end,
    )
    
    # Step 12 Funding Arbitrage check
    from strategy.funding_arb import FundingArbitrageStrategy
    mid_df = load_mid_prices_csv(args.mid_prices)
    price_map = mid_df["mid_price"].to_dict()
    arb_strat = FundingArbitrageStrategy()
    arb_df = arb_strat.run_backtest(funding_list, price_map, price_map)

    non_zero_rates = [rate.rate for rate in funding_list if rate.rate != 0.0]
    funding_data_quality = "ok" if non_zero_rates else "degenerate"

    summary = {
        "cycles": len(result.cycles),
        "total_fills": result.total_fills,
        "max_spread_fraction": result.max_spread_fraction,
        "step12_arb_settlements": len(arb_df),
        "tier1_gamma": tier1_gamma,
        "tier1_gamma_derivation": DEFAULT_SETTINGS.market.tier1_gamma_derivation,
        "funding_data_quality": funding_data_quality,
        "passed": (
            result.total_fills > 0
            and result.max_spread_fraction < 0.01
            and len(arb_df) == 3
            and bool(non_zero_rates)
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if args.pnl_output:
        pnl_output = Path(args.pnl_output)
        pnl_output.parent.mkdir(parents=True, exist_ok=True)
        result.pnl_series.rename("pnl_delta").to_csv(pnl_output, index_label="timestamp")
    print(json.dumps(summary, indent=2))
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
