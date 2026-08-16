#!/usr/bin/env python
"""Generate Step 7 evaluation from two per-cycle P&L CSV files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluation.protocol import evaluate_static_vs_adaptive, time_of_day_breakdown


def load_pnl(path: str) -> pd.Series:
    frame = pd.read_csv(path)
    if not {"timestamp", "pnl_delta"} <= set(frame.columns):
        raise ValueError(f"{path} must contain timestamp,pnl_delta")
    return pd.Series(frame["pnl_delta"].astype(float).to_numpy(), index=pd.to_datetime(frame["timestamp"], utc=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--static-pnl", required=True)
    parser.add_argument("--adaptive-pnl", required=True)
    parser.add_argument("--output", default="outputs/evaluation/evaluation_report.json")
    args = parser.parse_args()

    static = load_pnl(args.static_pnl)
    adaptive = load_pnl(args.adaptive_pnl)
    report = evaluate_static_vs_adaptive(static, adaptive)
    tod = time_of_day_breakdown(static, adaptive)
    payload = {
        "static_baseline_sharpe": report.static_baseline_sharpe,
        "adaptive_agent_sharpe": report.adaptive_agent_sharpe,
        "static_baseline_sortino": report.static_baseline_sortino,
        "adaptive_agent_sortino": report.adaptive_agent_sortino,
        "adaptive_win_count": report.adaptive_win_count,
        "total_days": report.total_days,
        "daily_pnl": [item.__dict__ | {"date": item.date.isoformat()} for item in report.daily_pnl],
        "time_of_day": [item.__dict__ for item in tod],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
