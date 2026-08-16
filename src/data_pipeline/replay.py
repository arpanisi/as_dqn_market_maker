"""Order-book replay primitives for Step 1."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timezone
from typing import Iterable, Literal

import numpy as np
import pandas as pd

BookSide = Literal["bid", "ask"]
AggressorSide = Literal["buy", "sell"]


@dataclass(frozen=True)
class BookDelta:
    timestamp: pd.Timestamp
    side: BookSide
    price: float
    size: float


@dataclass(frozen=True)
class Trade:
    timestamp: pd.Timestamp
    price: float
    size: float
    aggressor_side: AggressorSide


@dataclass(frozen=True)
class FundingRate:
    timestamp: pd.Timestamp
    rate: float


def to_utc_timestamp(value: object) -> pd.Timestamp:
    """Convert milliseconds-since-epoch or timestamp-like values to UTC."""
    if isinstance(value, (int, float, np.integer, np.floating)):
        return pd.to_datetime(value, unit="ms", utc=True)
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize(timezone.utc)
    return ts.tz_convert(timezone.utc)


class OrderBookReplay:
    """Cumulative price-level book replay from snapshot and delta records."""

    def __init__(self) -> None:
        self._bids: dict[float, float] = {}
        self._asks: dict[float, float] = {}
        self._last_timestamp: pd.Timestamp | None = None

    @classmethod
    def from_snapshot(
        cls,
        timestamp: object,
        bids: Iterable[tuple[float, float]],
        asks: Iterable[tuple[float, float]],
    ) -> "OrderBookReplay":
        replay = cls()
        replay.apply_snapshot(timestamp, bids, asks)
        return replay

    @property
    def last_timestamp(self) -> pd.Timestamp | None:
        return self._last_timestamp

    def apply_snapshot(
        self,
        timestamp: object,
        bids: Iterable[tuple[float, float]],
        asks: Iterable[tuple[float, float]],
    ) -> None:
        self._bids = _clean_levels(bids)
        self._asks = _clean_levels(asks)
        self._last_timestamp = to_utc_timestamp(timestamp)

    def apply_delta(self, delta: BookDelta) -> None:
        side_book = self._bids if delta.side == "bid" else self._asks
        if delta.size <= 0:
            side_book.pop(float(delta.price), None)
        else:
            side_book[float(delta.price)] = float(delta.size)
        self._last_timestamp = to_utc_timestamp(delta.timestamp)

    def apply_deltas(self, deltas: Iterable[BookDelta]) -> None:
        for delta in deltas:
            self.apply_delta(delta)

    def best_bid(self) -> float:
        if not self._bids:
            raise ValueError("book has no bid levels")
        return max(self._bids)

    def best_ask(self) -> float:
        if not self._asks:
            raise ValueError("book has no ask levels")
        return min(self._asks)

    def mid_price(self) -> float:
        return (self.best_bid() + self.best_ask()) / 2.0

    def depth_at(self, side: BookSide, price: float) -> float:
        book = self._bids if side == "bid" else self._asks
        return float(book.get(float(price), 0.0))

    def snapshot(self) -> dict[str, list[tuple[float, float]]]:
        return {
            "bids": sorted(self._bids.items(), reverse=True),
            "asks": sorted(self._asks.items()),
        }


def _clean_levels(levels: Iterable[tuple[float, float]]) -> dict[float, float]:
    return {float(price): float(size) for price, size in levels if float(size) > 0}

