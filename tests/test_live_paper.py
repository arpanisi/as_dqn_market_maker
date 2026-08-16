import pandas as pd
import pytest

from backtest.fills import Fill
from data_pipeline.replay import BookDelta, OrderBookReplay
from live.paper import (
    MarketState,
    PaperExecutionState,
    apply_book_message,
    quote_from_policy,
    recorded_live_quotes_match_backtest,
    subscription_message,
)


class FixedActionProvider:
    def select_action(self, state, epsilon=0.0):
        return type("Action", (), {"risk_aversion": 0.1, "skew": 0.0})()


def test_live_and_backtest_quote_paths_match_for_recorded_state():
    provider = FixedActionProvider()
    state = MarketState(
        timestamp="2024-06-01T12:30:00Z",
        mid_price=60000.0,
        inventory_q=0.0,
        sigma=0.5,
        kappa=2.0,
        state_vector=[0.0],
    )

    assert recorded_live_quotes_match_backtest(
        [state],
        live_quote_fn=lambda item: quote_from_policy(item, provider),
        backtest_quote_fn=lambda item: quote_from_policy(item, provider),
    )


def test_paper_execution_tracks_positions_and_aggregate_exposure():
    state = PaperExecutionState()
    fills = [
        Fill("bid", 100.0, 0.2, pd.Timestamp("2024-06-01T00:00:00Z"), "maker"),
        Fill("ask", 101.0, 0.05, pd.Timestamp("2024-06-01T00:00:01Z"), "maker"),
    ]

    position = state.apply_fills("BTCUSDT", fills, mark_price=102.0)
    exposure = state.aggregate_exposure()

    assert position.q_btc == pytest.approx(0.15)
    assert exposure.total_notional_exposure == pytest.approx(0.15 * 102.0)
    assert exposure.positions[0].symbol == "BTCUSDT"


def test_live_book_messages_use_replay_snapshot_delta_transform():
    book = OrderBookReplay()
    apply_book_message(
        book,
        {
            "type": "snapshot",
            "timestamp": "2024-06-01T00:00:00Z",
            "bids": [(100.0, 1.0)],
            "asks": [(100.2, 1.0)],
        },
    )
    apply_book_message(
        book,
        {
            "type": "delta",
            "deltas": [BookDelta(pd.Timestamp("2024-06-01T00:00:01Z"), "bid", 100.1, 2.0)],
        },
    )

    assert book.best_bid() == 100.1
    assert book.best_ask() == 100.2


def test_bybit_subscription_targets_linear_btcusdt_feeds():
    message = subscription_message()

    assert message["op"] == "subscribe"
    assert "orderbook.200.BTCUSDT" in message["args"]
    assert "publicTrade.BTCUSDT" in message["args"]
