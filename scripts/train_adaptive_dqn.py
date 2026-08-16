#!/usr/bin/env python
"""Run Step 6 Double-DQN training and export adaptive per-cycle P&L."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agent.double_dqn import DoubleDQNAgent
from backtest.adaptive import funding_skew_correlation, train_adaptive_agent
from data_pipeline.loaders import load_book_snapshots_csv, load_funding_csv, load_mid_prices_csv, load_trades_csv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mid-prices", required=True)
    parser.add_argument("--books", required=True)
    parser.add_argument("--trades", required=True)
    parser.add_argument("--funding", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--device", default=None)
    parser.add_argument("--weights-output", default="outputs/models/dqn_weights.pt")
    parser.add_argument("--pnl-output", default="outputs/training/adaptive_training_cycle_pnl.csv")
    parser.add_argument("--summary-output", default="outputs/training/dqn_training_summary.json")
    args = parser.parse_args()

    agent = DoubleDQNAgent(device=args.device)
    result = train_adaptive_agent(
        agent=agent,
        mid_prices=load_mid_prices_csv(args.mid_prices),
        books=load_book_snapshots_csv(args.books),
        trades=load_trades_csv(args.trades),
        funding_rates=load_funding_csv(args.funding),
        start=args.start,
        end=args.end,
        epochs=args.epochs,
    )
    agent.save(args.weights_output)
    pnl_output = Path(args.pnl_output)
    pnl_output.parent.mkdir(parents=True, exist_ok=True)
    result.backtest.pnl_series.rename("pnl_delta").to_csv(pnl_output, index_label="timestamp")
    action_gammas = [agent.actions[i].risk_aversion for i in result.action_indices]
    action_skews = [agent.actions[i].skew for i in result.action_indices]
    summary = {
        "cycles": len(result.backtest.cycles),
        "loss_observations": len(result.losses),
        "first_loss": result.losses[0] if result.losses else None,
        "last_loss": result.losses[-1] if result.losses else None,
        "mean_reward": sum(result.rewards) / len(result.rewards) if result.rewards else 0.0,
        "unique_gamma_count": len(set(action_gammas)),
        "unique_skew_count": len(set(action_skews)),
        "funding_skew_correlation": funding_skew_correlation(result.funding_values, result.skew_values),
        "weights_output": args.weights_output,
        "pnl_output": args.pnl_output,
    }
    summary_output = Path(args.summary_output)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
