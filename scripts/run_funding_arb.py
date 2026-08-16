#!/usr/bin/env python
"""Run Step 12 standalone funding-rate arbitrage backtest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data_pipeline.loaders import load_funding_csv, load_mid_prices_csv, load_spot_trades_csv
from settings import DEFAULT_SETTINGS
from strategy.funding_arb import FundingArbitrageStrategy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--funding", default="data/funding.csv")
    parser.add_argument("--spot-trades", required=True)
    parser.add_argument("--perp-marks", default="data/mid_prices.csv")
    parser.add_argument("--settlements-output", default="outputs/funding_arb/settlements.csv")
    parser.add_argument("--daily-returns-output", default="outputs/funding_arb/daily_returns.csv")
    parser.add_argument("--summary-output", default="outputs/funding_arb/summary.json")
    args = parser.parse_args()

    funding = load_funding_csv(args.funding)
    spot_trades = load_spot_trades_csv(args.spot_trades)
    perp_marks = load_mid_prices_csv(args.perp_marks)
    settings = DEFAULT_SETTINGS.funding_arbitrage
    strategy = FundingArbitrageStrategy(n_arb=settings.n_arb, f_min=settings.f_min)
    settlements = strategy.run_backtest(funding, spot_trades, perp_marks)
    report = strategy.evaluate(settlements)

    settlements_output = Path(args.settlements_output)
    daily_output = Path(args.daily_returns_output)
    summary_output = Path(args.summary_output)
    settlements_output.parent.mkdir(parents=True, exist_ok=True)
    daily_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.parent.mkdir(parents=True, exist_ok=True)

    settlements.to_csv(settlements_output, index=False)
    report.daily_returns.rename("daily_return").to_frame().to_csv(daily_output, index_label="date")
    payload = {
        "sharpe": report.sharpe,
        "total_funding_pnl": report.total_funding_pnl,
        "total_basis_pnl": report.total_basis_pnl,
        "total_net_pnl": report.total_net_pnl,
        "n_arb": settings.n_arb,
        "f_min": settings.f_min,
        "settlements": int(len(settlements)),
    }
    summary_output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
