import pandas as pd
import pytest

from backtest.fills import PostedQuote, apply_fills_to_inventory, funding_cash_flows, simulate_quote_fill
from data_pipeline.replay import FundingRate, OrderBookReplay, Trade
from data_pipeline.series import mid_price_frame


def test_passive_bid_uses_same_side_queue_and_opposite_sell_trades_only():
    book = OrderBookReplay.from_snapshot(
        "2024-06-01T00:00:00Z",
        bids=[(100.0, 1.0)],
        asks=[(100.2, 1.0)],
    )
    quote = PostedQuote("bid", 100.0, 0.5, pd.Timestamp("2024-06-01T00:00:00Z"))
    trades = [
        Trade(pd.Timestamp("2024-06-01T00:00:01Z"), 100.0, 5.0, "buy"),
        Trade(pd.Timestamp("2024-06-01T00:00:02Z"), 100.1, 5.0, "sell"),
        Trade(pd.Timestamp("2024-06-01T00:00:03Z"), 100.0, 1.2, "sell"),
    ]

    fills = simulate_quote_fill(quote, book, trades, "2024-06-01T00:00:05Z")

    assert len(fills) == 1
    assert fills[0].side == "bid"
    assert fills[0].size_btc == pytest.approx(0.2)
    assert apply_fills_to_inventory(0.0, fills) == pytest.approx(0.2)


def test_marketable_quote_fills_immediately_at_opposite_book_price():
    book = OrderBookReplay.from_snapshot(
        "2024-06-01T00:00:00Z",
        bids=[(100.0, 1.0)],
        asks=[(100.2, 0.3), (100.3, 0.4)],
    )
    quote = PostedQuote("bid", 100.3, 0.5, pd.Timestamp("2024-06-01T00:00:00Z"))

    fills = simulate_quote_fill(quote, book, [], "2024-06-01T00:00:05Z")

    assert [fill.price for fill in fills] == [100.2, 100.3]
    assert sum(fill.size_btc for fill in fills) == pytest.approx(0.5)
    assert all(fill.liquidity == "taker" for fill in fills)


def test_funding_cash_flow_uses_actual_btc_inventory_not_unit_inventory():
    marks = mid_price_frame(
        [
            ("2024-06-01T07:59:00Z", 60000.0),
            ("2024-06-01T08:00:00Z", 60100.0),
        ]
    )

    flows = funding_cash_flows(
        q_btc=0.02,
        funding_rates=[FundingRate(pd.Timestamp("2024-06-01T08:00:00Z"), 0.0001)],
        mark_prices=marks,
        start_time="2024-06-01T07:00:00Z",
        end_time="2024-06-01T09:00:00Z",
    )

    assert flows[0].cash_flow == pytest.approx(-0.02 * 0.0001 * 60100.0)

