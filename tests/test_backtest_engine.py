import json

import pandas as pd
import pytest

from backtest.engine import FixedPolicy, run_fixed_policy_backtest
from data_pipeline.loaders import load_book_snapshots_csv
from data_pipeline.replay import Trade
from data_pipeline.series import mid_price_frame
from model.quoting import avellaneda_stoikov_quote


def test_fixed_policy_backtest_runs_5_second_cycles_and_records_fills(tmp_path):
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

    result = run_fixed_policy_backtest(
        mid_prices=mids,
        books=load_book_snapshots_csv(books_path),
        trades=trades,
        funding_rates=[],
        policy=FixedPolicy(1.0, 0.0),
        start="2024-06-01T00:00:00Z",
        end="2024-06-01T00:00:10Z",
    )

    assert len(result.cycles) == 2
    assert result.pnl_series.index[0] == pd.Timestamp("2024-06-01T00:00:00Z")


def test_fixed_policy_pnl_marks_pre_fill_inventory_not_post_fill_inventory(tmp_path):
    """A fill in a cycle where the mark also moves must not mark the newly-bought
    inventory against the cycle's whole price move (that inventory did not exist at
    the cycle start). pnl_delta = realized + q_old*(new_mark - last_mark) + funding."""
    books_path = tmp_path / "books.csv"
    books_path.write_text(
        "timestamp,bids,asks\n"
        f"2024-06-01T00:00:00Z,\"{json.dumps([[95.0, 0.5]])}\",\"{json.dumps([[96.0, 0.5]])}\"\n",
        encoding="utf-8",
    )
    # Mark moves 100 -> 105 within the single cycle.
    mids = mid_price_frame(
        [
            ("2024-06-01T00:00:00Z", 100.0),
            ("2024-06-01T00:00:05Z", 105.0),
        ]
    )
    # One sell trade that hits our passive bid quote during the cycle.
    trades = [Trade(pd.Timestamp("2024-06-01T00:00:01Z"), 95.0, 0.01, "sell")]

    result = run_fixed_policy_backtest(
        mid_prices=mids,
        books=load_book_snapshots_csv(books_path),
        trades=trades,
        funding_rates=[],
        policy=FixedPolicy(1.0, 0.0),
        start="2024-06-01T00:00:00Z",
        end="2024-06-01T00:00:05Z",
    )

    assert len(result.cycles) == 1
    assert result.total_fills == 1
    cycle = result.cycles[0]

    # Deterministically reproduce the quote the engine posted (kappa floor applies:
    # the fill trade lies after the 24h intensity window, so the fit is degenerate).
    quote = avellaneda_stoikov_quote(
        mid_price=100.0,
        inventory_q=0.0,
        sigma=0.0,
        kappa=0.01,
        risk_aversion=1.0,
        skew=0.0,
        timestamp=pd.Timestamp("2024-06-01T00:00:00Z"),
    )
    realized = -quote.size_btc * quote.bid  # 0.01 BTC bought at quote.bid
    q_old = 0.0  # flat before this cycle's fill
    last_mark = 100.0
    new_mark = 105.0
    expected_pnl = realized + q_old * (new_mark - last_mark)  # funding == 0
    buggy_pnl = realized + (q_old + quote.size_btc) * (new_mark - last_mark)

    assert cycle.pnl_delta == pytest.approx(expected_pnl)
    assert cycle.pnl_delta != pytest.approx(buggy_pnl)
