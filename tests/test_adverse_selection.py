import pandas as pd
import pytest

from data_pipeline.replay import Trade
from data_pipeline.series import mid_price_frame
from signals.adverse_selection import adverse_selection_signal, signal_at


def test_adverse_selection_signal_is_timestamped_after_impact_horizon():
    mids = mid_price_frame(
        [
            ("2024-06-01T00:00:00Z", 100.0),
            ("2024-06-01T00:00:10Z", 101.0),
            ("2024-06-01T00:00:20Z", 100.5),
        ]
    )
    trades = [
        Trade(pd.Timestamp("2024-06-01T00:00:00Z"), 100.0, 1.0, "buy"),
        Trade(pd.Timestamp("2024-06-01T00:00:10Z"), 101.0, 1.0, "sell"),
    ]

    signal = adverse_selection_signal(trades, mids, impact_horizon="10s")

    assert signal.index[0] == pd.Timestamp("2024-06-01T00:00:10Z")
    assert signal.iloc[0]["signed_price_impact"] == pytest.approx(1.0)
    assert signal.iloc[1]["signed_price_impact"] == pytest.approx(0.5)
    assert signal_at(signal, "2024-06-01T00:00:09Z") == 0.0
    assert signal_at(signal, "2024-06-01T00:00:10Z") == pytest.approx(1.0)


def test_adverse_selection_empty_input_returns_empty_signal():
    mids = mid_price_frame([("2024-06-01T00:00:00Z", 100.0)])

    signal = adverse_selection_signal([], mids)

    assert signal.empty
    assert signal_at(signal, "2024-06-01T00:00:00Z") == 0.0
