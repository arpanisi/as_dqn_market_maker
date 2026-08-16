#!/usr/bin/env python
"""Run frozen Step 6 agent over a held-out window."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agent.double_dqn import DoubleDQNAgent
from backtest.adaptive import evaluate_adaptive_agent
from data_pipeline.loaders import load_book_snapshots_csv, load_funding_csv, load_mid_prices_csv, load_trades_csv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mid-prices", required=True)
    parser.add_argument("--books", required=True)
    parser.add_argument("--trades", required=True)
    parser.add_argument("--funding", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--pnl-output", default="outputs/evaluation/adaptive_cycle_pnl.csv")
    args = parser.parse_args()

    agent = DoubleDQNAgent(device=args.device)
    agent.load(args.weights)
    result = evaluate_adaptive_agent(
        agent=agent,
        mid_prices=load_mid_prices_csv(args.mid_prices),
        books=load_book_snapshots_csv(args.books),
        trades=load_trades_csv(args.trades),
        funding_rates=load_funding_csv(args.funding),
        start=args.start,
        end=args.end,
    )
    output = Path(args.pnl_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.backtest.pnl_series.rename("pnl_delta").to_csv(output, index_label="timestamp")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
