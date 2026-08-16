"""Derived Step 1 time series: mid-price, volatility, and funding."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .replay import FundingRate, to_utc_timestamp


def mid_price_frame(records: object) -> pd.DataFrame:
    """Create a UTC-indexed mid-price frame from `(timestamp, mid)` records."""
    rows = [(to_utc_timestamp(ts), float(mid)) for ts, mid in records]
    frame = pd.DataFrame(rows, columns=["timestamp", "mid_price"])
    if frame.empty:
        return frame.set_index(pd.DatetimeIndex([], name="timestamp"))
    return frame.drop_duplicates("timestamp").sort_values("timestamp").set_index("timestamp")


def trailing_realized_volatility(
    mid_prices: pd.DataFrame,
    window: str = "5min",
    periods_per_year: int = 365 * 24 * 60 * 12,
) -> pd.Series:
    """Annualized trailing volatility of log mid-price returns."""
    if "mid_price" not in mid_prices:
        raise ValueError("mid_prices must contain a 'mid_price' column")
    mids = mid_prices["mid_price"].astype(float).sort_index()
    log_returns = np.log(mids / mids.shift(1))
    return log_returns.rolling(window, min_periods=2).std(ddof=1) * np.sqrt(periods_per_year)


def funding_series(
    funding_rates: object,
    target_index: pd.DatetimeIndex,
) -> pd.Series:
    """Forward-fill realized funding rates onto a target timeline."""
    target_index = pd.DatetimeIndex(target_index).tz_convert("UTC")
    rows = [(to_utc_timestamp(rate.timestamp), float(rate.rate)) for rate in funding_rates]
    if not rows:
        return pd.Series(0.0, index=target_index, name="funding_rate")
    funding = (
        pd.DataFrame(rows, columns=["timestamp", "funding_rate"])
        .drop_duplicates("timestamp", keep="last")
        .sort_values("timestamp")
        .set_index("timestamp")["funding_rate"]
    )
    aligned = funding.reindex(funding.index.union(target_index)).sort_index().ffill().reindex(target_index)
    return aligned.fillna(0.0).rename("funding_rate")


def nearest_mid_price(mid_prices: pd.DataFrame, timestamp: object) -> float:
    """Return the last known mid-price at or before `timestamp`."""
    ts = to_utc_timestamp(timestamp)
    mids = mid_prices["mid_price"].sort_index()
    eligible = mids.loc[:ts]
    if eligible.empty:
        raise ValueError(f"no mid-price available at or before {ts}")
    return float(eligible.iloc[-1])

