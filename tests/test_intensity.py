import pandas as pd

from data_pipeline.replay import Trade
from data_pipeline.series import mid_price_frame
from model.intensity import estimate_arrival_intensity


def test_estimate_arrival_intensity_positive_kappa_for_decaying_activity():
    mids = mid_price_frame((("2024-06-01T00:00:00Z", 100.0), ("2024-06-02T00:00:00Z", 100.0)))
    trades = []
    for i in range(50):
        trades.append(Trade(pd.Timestamp("2024-06-01T01:00:00Z") + pd.Timedelta(seconds=i), 100.01, 1.0, "buy"))
    for i in range(20):
        trades.append(Trade(pd.Timestamp("2024-06-01T02:00:00Z") + pd.Timedelta(seconds=i), 100.10, 1.0, "buy"))
    for i in range(5):
        trades.append(Trade(pd.Timestamp("2024-06-01T03:00:00Z") + pd.Timedelta(seconds=i), 100.30, 1.0, "buy"))

    estimate = estimate_arrival_intensity(trades, mids, "2024-06-02T00:00:00Z", bins=3)

    assert estimate.A > 0
    assert estimate.kappa > 0
    assert not estimate.degenerate
    assert estimate.observations == 75


def test_estimate_arrival_intensity_clamps_degenerate_kappa():
    mids = mid_price_frame((("2024-06-01T00:00:00Z", 100.0), ("2024-06-02T00:00:00Z", 100.0)))
    trades = []
    for i in range(5):
        trades.append(Trade(pd.Timestamp("2024-06-01T01:00:00Z") + pd.Timedelta(seconds=i), 100.01, 1.0, "sell"))
    for i in range(20):
        trades.append(Trade(pd.Timestamp("2024-06-01T02:00:00Z") + pd.Timedelta(seconds=i), 100.10, 1.0, "sell"))
    for i in range(50):
        trades.append(Trade(pd.Timestamp("2024-06-01T03:00:00Z") + pd.Timedelta(seconds=i), 100.30, 1.0, "sell"))

    estimate = estimate_arrival_intensity(trades, mids, "2024-06-02T00:00:00Z", bins=3)

    assert estimate.kappa == 0.01
    assert estimate.degenerate
