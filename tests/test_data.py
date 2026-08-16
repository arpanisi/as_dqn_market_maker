import pandas as pd
import pytest

from data_pipeline.replay import BookDelta, FundingRate, OrderBookReplay
from data_pipeline.series import (
    funding_series,
    mid_price_frame,
    nearest_mid_price,
    trailing_realized_volatility,
)


def test_order_book_replay_snapshot_delta_and_mid_price():
    replay = OrderBookReplay.from_snapshot(
        1717200000000,
        bids=[(100.0, 2.0), (99.9, 1.0)],
        asks=[(100.2, 1.5), (100.3, 2.0)],
    )

    assert replay.best_bid() == 100.0
    assert replay.best_ask() == 100.2
    assert replay.mid_price() == pytest.approx(100.1)

    replay.apply_delta(BookDelta(pd.Timestamp("2024-06-01T00:00:01Z"), "bid", 100.1, 3.0))
    replay.apply_delta(BookDelta(pd.Timestamp("2024-06-01T00:00:02Z"), "ask", 100.2, 0.0))

    assert replay.best_bid() == 100.1
    assert replay.best_ask() == 100.3
    assert replay.depth_at("ask", 100.2) == 0.0


def test_volatility_and_funding_alignment():
    mids = mid_price_frame(
        [
            ("2024-06-01T00:00:00Z", 100.0),
            ("2024-06-01T00:01:00Z", 101.0),
            ("2024-06-01T00:02:00Z", 100.5),
        ]
    )
    vol = trailing_realized_volatility(mids, window="5min", periods_per_year=365)

    assert vol.iloc[0] != vol.iloc[0]
    assert vol.iloc[-1] > 0
    assert nearest_mid_price(mids, "2024-06-01T00:01:30Z") == 101.0

    target = pd.DatetimeIndex(
        pd.to_datetime(
            [
                "2024-06-01T00:00:00Z",
                "2024-06-01T08:00:00Z",
                "2024-06-01T09:00:00Z",
            ],
            utc=True,
        )
    )
    aligned = funding_series([FundingRate(pd.Timestamp("2024-06-01T08:00:00Z"), 0.0001)], target)

    assert aligned.iloc[0] == 0.0
    assert aligned.iloc[1] == pytest.approx(0.0001)
    assert aligned.iloc[2] == pytest.approx(0.0001)
