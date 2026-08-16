#!/usr/bin/env python
"""Run Step 5 CMA-ES static baseline optimization."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from backtest.engine import FixedPolicy, run_fixed_policy_backtest
from data_pipeline.loaders import load_book_snapshots_csv, load_funding_csv, load_mid_prices_csv, load_trades_csv
from evaluation.protocol import sharpe_ratio_from_pnl
from optimization.static_baseline import search_static_baseline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mid-prices", required=True)
    parser.add_argument("--books", required=True)
    parser.add_argument("--trades", required=True)
    parser.add_argument("--funding", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--population-size", type=int, default=None)
    parser.add_argument("--generations", type=int, default=None)
    parser.add_argument("--output", default="outputs/baseline/static_baseline.json")
    parser.add_argument("--pnl-output", default="outputs/baseline/static_cycle_pnl.csv")
    args = parser.parse_args()

    mid_prices = load_mid_prices_csv(args.mid_prices)
    books = load_book_snapshots_csv(args.books)
    trades = load_trades_csv(args.trades)
    funding = load_funding_csv(args.funding)
    best_pnl = None

    def score(gamma: float, skew: float) -> float:
        nonlocal best_pnl
        result = run_fixed_policy_backtest(
            mid_prices=mid_prices,
            books=books,
            trades=trades,
            funding_rates=funding,
            policy=FixedPolicy(gamma, skew),
            start=args.start,
            end=args.end,
        )
        sharpe = sharpe_ratio_from_pnl(result.pnl_series)
        if best_pnl is None or sharpe > best_pnl[0]:
            best_pnl = (sharpe, result.pnl_series)
        return sharpe

    result = search_static_baseline(
        score,
        seed=args.seed,
        population_size=args.population_size,
        generations=args.generations,
    )
    payload = {
        "risk_aversion": result.best.risk_aversion,
        "skew": result.best.skew,
        "training_sharpe": result.best.sharpe,
        "evaluations": len(result.history),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if best_pnl is not None:
        pnl_output = Path(args.pnl_output)
        pnl_output.parent.mkdir(parents=True, exist_ok=True)
        best_pnl[1].rename("pnl_delta").to_csv(pnl_output, index_label="timestamp")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
