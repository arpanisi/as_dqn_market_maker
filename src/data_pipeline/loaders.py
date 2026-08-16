"""File loaders for Vast/runtime jobs.

Expected CSV columns:
- book snapshots: timestamp,bids,asks where bids/asks are JSON lists [[price,size], ...]
- trades: timestamp,price,size,aggressor_side
- funding: timestamp,rate
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from data_pipeline.replay import FundingRate, Trade, to_utc_timestamp


def load_mid_prices_csv(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    _require_columns(frame, {"timestamp", "mid_price"}, path)
    frame["timestamp"] = frame["timestamp"].map(to_utc_timestamp)
    frame["mid_price"] = frame["mid_price"].astype(float)
    return frame.drop_duplicates("timestamp").sort_values("timestamp").set_index("timestamp")


def load_trades_csv(path: str | Path) -> list[Trade]:
    frame = pd.read_csv(path)
    _require_columns(frame, {"timestamp", "price", "size", "aggressor_side"}, path)
    return [
        Trade(to_utc_timestamp(row.timestamp), float(row.price), float(row.size), str(row.aggressor_side))
        for row in frame.itertuples(index=False)
    ]


def load_funding_csv(path: str | Path) -> list[FundingRate]:
    frame = pd.read_csv(path)
    _require_columns(frame, {"timestamp", "rate"}, path)
    return [FundingRate(to_utc_timestamp(row.timestamp), float(row.rate)) for row in frame.itertuples(index=False)]


def load_spot_trades_csv(path: str | Path) -> pd.DataFrame:
    """Loads Bybit spot BTCUSDT trade prints dataframe indexed by UTC timestamp."""
    frame = pd.read_csv(path)
    _require_columns(frame, {"timestamp", "price"}, path)
    frame["timestamp"] = frame["timestamp"].map(to_utc_timestamp)
    frame["price"] = frame["price"].astype(float)
    return frame.sort_values("timestamp")


def get_spot_asof_price(spot_df: pd.DataFrame, timestamp: pd.Timestamp) -> float:
    """Returns the last spot trade price at or before the given timestamp (as-of join)."""
    if spot_df.empty:
        raise ValueError("Spot trades dataframe is empty.")
    ts_utc = to_utc_timestamp(timestamp)
    past_trades = spot_df[spot_df["timestamp"] <= ts_utc]
    if past_trades.empty:
        # Fallback to first trade if timestamp precedes first trade
        return float(spot_df.iloc[0]["price"])
    return float(past_trades.iloc[-1]["price"])


def load_book_snapshots_csv(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    _require_columns(frame, {"timestamp", "bids", "asks"}, path)
    frame["timestamp"] = frame["timestamp"].map(to_utc_timestamp)
    frame["bids"] = frame["bids"].map(json.loads)
    frame["asks"] = frame["asks"].map(json.loads)
    return frame.sort_values("timestamp")


def verify_data_range(index: pd.DatetimeIndex, start: str, end: str) -> None:
    if index.empty:
        raise ValueError("data is empty")
    actual_start = index.min()
    actual_end = index.max()
    required_start = to_utc_timestamp(start)
    required_end = to_utc_timestamp(end)
    if actual_start > required_start or actual_end < required_end:
        raise ValueError(
            f"data range {actual_start} to {actual_end} does not cover required {required_start} to {required_end}"
        )


def _require_columns(frame: pd.DataFrame, columns: set[str], path: str | Path) -> None:
    missing = columns - set(frame.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {', '.join(sorted(missing))}")
