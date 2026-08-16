"""Step 9 adverse-selection proxy."""

from __future__ import annotations

import numpy as np
import pandas as pd

from data_pipeline.replay import Trade, to_utc_timestamp
from data_pipeline.series import nearest_mid_price


def adverse_selection_signal(
    trades: list[Trade],
    mid_prices: pd.DataFrame,
    *,
    impact_horizon: str = "10s",
    average_window: str = "5min",
    threshold_window: str = "1h",
    threshold_std_multiplier: float = 1.5,
) -> pd.DataFrame:
    """Compute trailing signed price-impact signal without leaking current-decision information."""
    rows: list[tuple[pd.Timestamp, float]] = []
    horizon = pd.Timedelta(impact_horizon)
    for trade in sorted(trades, key=lambda item: to_utc_timestamp(item.timestamp)):
        trade_ts = to_utc_timestamp(trade.timestamp)
        future_ts = trade_ts + horizon
        signed_direction = 1.0 if trade.aggressor_side == "buy" else -1.0
        start_mid = nearest_mid_price(mid_prices, trade_ts)
        future_mid = nearest_mid_price(mid_prices, future_ts)
        signed_impact = signed_direction * (future_mid - start_mid)
        signal_timestamp = future_ts
        rows.append((signal_timestamp, float(signed_impact)))

    if not rows:
        empty_index = pd.DatetimeIndex([], name="timestamp", tz="UTC")
        return pd.DataFrame(
            {
                "signed_price_impact": pd.Series(dtype=float),
                "trailing_impact": pd.Series(dtype=float),
                "threshold_mean": pd.Series(dtype=float),
                "threshold_std": pd.Series(dtype=float),
                "adverse_selection": pd.Series(dtype=bool),
            },
            index=empty_index,
        )

    impacts = (
        pd.DataFrame(rows, columns=["timestamp", "signed_price_impact"])
        .sort_values("timestamp")
        .set_index("timestamp")["signed_price_impact"]
    )
    trailing = impacts.rolling(average_window, min_periods=1).mean()
    threshold_mean = trailing.rolling(threshold_window, min_periods=2).mean()
    threshold_std = trailing.rolling(threshold_window, min_periods=2).std(ddof=1).fillna(0.0)
    threshold = threshold_mean + threshold_std_multiplier * threshold_std
    adverse = (trailing > threshold).fillna(False)

    return pd.DataFrame(
        {
            "signed_price_impact": impacts,
            "trailing_impact": trailing,
            "threshold_mean": threshold_mean,
            "threshold_std": threshold_std,
            "adverse_selection": adverse.astype(bool),
        }
    )


def signal_at(signal: pd.DataFrame, timestamp: object, column: str = "trailing_impact") -> float:
    """Return the latest available signal value at or before a decision timestamp."""
    if signal.empty:
        return 0.0
    ts = to_utc_timestamp(timestamp)
    values = signal[column].sort_index().loc[:ts]
    if values.empty:
        return 0.0
    value = values.iloc[-1]
    if isinstance(value, (bool, np.bool_)):
        return float(value)
    if pd.isna(value):
        return 0.0
    return float(value)
