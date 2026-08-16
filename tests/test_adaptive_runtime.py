import json

import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from agent.double_dqn import DoubleDQNAgent
from backtest.adaptive import evaluate_adaptive_agent, train_adaptive_agent
from data_pipeline.loaders import load_book_snapshots_csv
from data_pipeline.replay import Trade
from data_pipeline.series import mid_price_frame


def test_adaptive_training_runtime_produces_pnl_and_actions(tmp_path):
    books_path = tmp_path / "books.csv"
    books_path.write_text(
        "timestamp,bids,asks\n"
        f"2024-06-01T00:00:00Z,\"{json.dumps([[99.0, 0.0]])}\",\"{json.dumps([[101.0, 0.0]])}\"\n",
        encoding="utf-8",
    )
    mids = mid_price_frame(
        [
            ("2024-06-01T00:00:00Z", 100.0),
            ("2024-06-01T00:00:05Z", 100.0),
            ("2024-06-01T00:00:10Z", 100.0),
        ]
    )
    trades = [Trade(pd.Timestamp("2024-06-01T00:00:01Z"), 99.5, 0.01, "sell")]
    agent = DoubleDQNAgent(seed=7, warmup_transitions=2, batch_size=2, device="cpu")

    result = train_adaptive_agent(
        agent=agent,
        mid_prices=mids,
        books=load_book_snapshots_csv(books_path),
        trades=trades,
        funding_rates=[],
        start="2024-06-01T00:00:00Z",
        end="2024-06-01T00:00:10Z",
        epochs=1,
    )
    frozen = evaluate_adaptive_agent(
        agent=agent,
        mid_prices=mids,
        books=load_book_snapshots_csv(books_path),
        trades=trades,
        funding_rates=[],
        start="2024-06-01T00:00:00Z",
        end="2024-06-01T00:00:10Z",
    )

    assert len(result.backtest.cycles) == 2
    assert len(result.action_indices) == 2
    assert len(frozen.backtest.cycles) == 2


def test_adaptive_pnl_marks_pre_fill_inventory_not_post_fill_inventory(tmp_path):
    """Regression for the shared P&L bug: a cycle with a fill while the mark moves must
    mark only the pre-fill inventory (q_old) against the price move. Here the book's best
    bid sits at 200.0, above every possible ask quote in the 4x5 action grid, so the agent's
    ask always fills immediately at 200.0 regardless of the chosen action: realized = +2.0,
    q_old = 0, mark moves 100 -> 105, so correct pnl_delta = +2.0 (not 2.0 + (-0.01)*5)."""
    books_path = tmp_path / "books.csv"
    books_path.write_text(
        "timestamp,bids,asks\n"
        f"2024-06-01T00:00:00Z,\"{json.dumps([[200.0, 0.5]])}\",\"{json.dumps([[201.0, 0.5]])}\"\n",
        encoding="utf-8",
    )
    mids = mid_price_frame(
        [
            ("2024-06-01T00:00:00Z", 100.0),
            ("2024-06-01T00:00:05Z", 105.0),
        ]
    )
    agent = DoubleDQNAgent(seed=7, warmup_transitions=2, batch_size=2, device="cpu")

    result = evaluate_adaptive_agent(
        agent=agent,
        mid_prices=mids,
        books=load_book_snapshots_csv(books_path),
        trades=[],
        funding_rates=[],
        start="2024-06-01T00:00:00Z",
        end="2024-06-01T00:00:05Z",
    )

    assert len(result.backtest.cycles) == 1
    cycle = result.backtest.cycles[0]
    assert cycle.fills == 1
    assert cycle.pnl_delta == pytest.approx(2.0)
    assert cycle.pnl_delta != pytest.approx(1.95)
