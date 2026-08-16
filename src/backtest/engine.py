"""Executable 5-second backtest loop."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from backtest.fills import PostedQuote, funding_cash_flows, simulate_quote_fill
from data_pipeline.replay import FundingRate, OrderBookReplay, Trade, to_utc_timestamp
from data_pipeline.series import nearest_mid_price, trailing_realized_volatility
from model.intensity import IntensityEstimate
from model.quoting import avellaneda_stoikov_quote
from settings import DEFAULT_SETTINGS


@dataclass(frozen=True)
class FixedPolicy:
    risk_aversion: float
    skew: float


@dataclass(frozen=True)
class CycleResult:
    timestamp: pd.Timestamp
    bid: float
    ask: float
    spread_fraction: float
    fills: int
    q_btc: float
    pnl_delta: float


@dataclass(frozen=True)
class BacktestResult:
    cycles: tuple[CycleResult, ...]

    @property
    def pnl_series(self) -> pd.Series:
        return pd.Series([c.pnl_delta for c in self.cycles], index=pd.DatetimeIndex([c.timestamp for c in self.cycles]))

    @property
    def total_fills(self) -> int:
        return sum(c.fills for c in self.cycles)

    @property
    def max_spread_fraction(self) -> float:
        return max((c.spread_fraction for c in self.cycles), default=0.0)


def run_fixed_policy_backtest(
    *,
    mid_prices: pd.DataFrame,
    books: pd.DataFrame,
    trades: list[Trade],
    funding_rates: list[FundingRate],
    policy: FixedPolicy,
    start: object,
    end: object,
    cadence: str = "5s",
) -> BacktestResult:
    start_ts = to_utc_timestamp(start)
    end_ts = to_utc_timestamp(end)
    decision_times = pd.date_range(start_ts, end_ts, freq=cadence, inclusive="left")
    vol = trailing_realized_volatility(mid_prices).ffill().fillna(0.0)
    trade_index = _build_trade_index(trades, mid_prices)
    book_index = _build_book_index(books)
    q_btc = 0.0
    last_mark = nearest_mid_price(mid_prices, start_ts)
    cycles: list[CycleResult] = []

    for ts in decision_times:
        next_ts = ts + pd.Timedelta(cadence)
        book = _book_at_indexed(book_index, ts)
        mid = nearest_mid_price(mid_prices, ts)
        sigma = float(vol.loc[:ts].iloc[-1]) if not vol.loc[:ts].empty else 0.0
        intensity = _estimate_arrival_intensity_from_index(trade_index, ts)
        inventory_q = q_btc / DEFAULT_SETTINGS.market.fixed_quote_size_btc
        quote = avellaneda_stoikov_quote(
            mid_price=mid,
            inventory_q=inventory_q,
            sigma=sigma,
            kappa=intensity.kappa,
            risk_aversion=policy.risk_aversion,
            skew=policy.skew,
            timestamp=ts,
        )
        window_trades = _trades_between(trade_index, ts, next_ts)
        bid_fills = simulate_quote_fill(PostedQuote("bid", quote.bid, quote.size_btc, ts), book, window_trades, next_ts)
        ask_fills = simulate_quote_fill(PostedQuote("ask", quote.ask, quote.size_btc, ts), book, window_trades, next_ts)
        fills = bid_fills + ask_fills
        realized = sum((-f.size_btc * f.price if f.side == "bid" else f.size_btc * f.price) for f in fills)
        q_old = q_btc
        q_btc += sum((f.size_btc if f.side == "bid" else -f.size_btc) for f in fills)
        funding = sum(flow.cash_flow for flow in funding_cash_flows(q_btc=q_btc, funding_rates=funding_rates, mark_prices=mid_prices, start_time=ts, end_time=next_ts))
        new_mark = nearest_mid_price(mid_prices, next_ts)
        unrealized_delta = q_old * (new_mark - last_mark)
        last_mark = new_mark
        pnl_delta = float(realized + unrealized_delta + funding)
        cycles.append(
            CycleResult(
                timestamp=ts,
                bid=quote.bid,
                ask=quote.ask,
                spread_fraction=float((quote.ask - quote.bid) / mid),
                fills=len(fills),
                q_btc=float(q_btc),
                pnl_delta=pnl_delta,
            )
        )
    return BacktestResult(tuple(cycles))


def _book_at(books: pd.DataFrame, timestamp: pd.Timestamp) -> OrderBookReplay:
    eligible = books[books["timestamp"] <= timestamp]
    if eligible.empty:
        raise ValueError(f"no book snapshot available at or before {timestamp}")
    row = eligible.iloc[-1]
    return OrderBookReplay.from_snapshot(row.timestamp, row.bids, row.asks)


def _build_book_index(books: pd.DataFrame) -> dict[str, object]:
    ordered = books.sort_values("timestamp").reset_index(drop=True)
    timestamps = pd.DatetimeIndex(ordered["timestamp"].map(to_utc_timestamp))
    return {
        "times_ns": np.asarray([ts.value for ts in timestamps], dtype=np.int64),
        "books": ordered,
    }


def _book_at_indexed(book_index: dict[str, object], timestamp: pd.Timestamp) -> OrderBookReplay:
    times_ns = book_index["times_ns"]
    books = book_index["books"]
    idx = int(np.searchsorted(times_ns, to_utc_timestamp(timestamp).value, side="right")) - 1
    if idx < 0:
        raise ValueError(f"no book snapshot available at or before {timestamp}")
    row = books.iloc[idx]
    return OrderBookReplay.from_snapshot(row.timestamp, row.bids, row.asks)


def _build_trade_index(trades: list[Trade], mid_prices: pd.DataFrame) -> dict[str, object]:
    ordered = sorted(trades, key=lambda item: to_utc_timestamp(item.timestamp))
    timestamps = pd.DatetimeIndex([to_utc_timestamp(t.timestamp) for t in ordered])
    times_ns = np.asarray([ts.value for ts in timestamps], dtype=np.int64)
    prices = np.asarray([float(t.price) for t in ordered], dtype=float)
    mid_frame = (
        mid_prices["mid_price"]
        .sort_index()
        .reset_index()
        .rename(columns={"timestamp": "timestamp", "mid_price": "mid_price"})
    )
    trade_frame = pd.DataFrame({"timestamp": timestamps, "price": prices})
    aligned = pd.merge_asof(trade_frame, mid_frame, on="timestamp", direction="backward")
    aligned["mid_price"] = aligned["mid_price"].bfill()
    distances = np.abs(prices - aligned["mid_price"].to_numpy(dtype=float))
    return {"trades": ordered, "times_ns": times_ns, "distances": distances}


def _trades_between(trade_index: dict[str, object], start: pd.Timestamp, end: pd.Timestamp) -> list[Trade]:
    times_ns = trade_index["times_ns"]
    trades = trade_index["trades"]
    left = int(np.searchsorted(times_ns, to_utc_timestamp(start).value, side="right"))
    right = int(np.searchsorted(times_ns, to_utc_timestamp(end).value, side="right"))
    return trades[left:right]


def _estimate_arrival_intensity_from_index(
    trade_index: dict[str, object],
    window_end: object,
    window: str = "24h",
    bins: int = 20,
    kappa_floor: float | None = None,
) -> IntensityEstimate:
    if kappa_floor is None:
        kappa_floor = DEFAULT_SETTINGS.intensity.kappa_floor
    end = to_utc_timestamp(window_end)
    start = end - pd.Timedelta(window)
    times_ns = trade_index["times_ns"]
    distances_all = trade_index["distances"]
    left = int(np.searchsorted(times_ns, start.value, side="right"))
    right = int(np.searchsorted(times_ns, end.value, side="right"))
    distances = distances_all[left:right]
    observations = int(len(distances))
    if observations < 2:
        return IntensityEstimate(end, 0.0, kappa_floor, 0.0, True, observations)
    if np.allclose(distances.max(), distances.min()):
        rate = observations / pd.Timedelta(window).total_seconds()
        return IntensityEstimate(end, rate, kappa_floor, 0.0, True, observations)

    counts, edges = np.histogram(distances, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2.0
    nonzero = counts > 0
    if nonzero.sum() < 2:
        rate = observations / pd.Timedelta(window).total_seconds()
        return IntensityEstimate(end, rate, kappa_floor, 0.0, True, observations)

    bin_width = max((edges[-1] - edges[0]) / bins, np.finfo(float).eps)
    rates = counts[nonzero] / (pd.Timedelta(window).total_seconds() * bin_width)
    x = centers[nonzero]
    y = np.log(rates)
    slope, intercept = np.polyfit(x, y, 1)
    predicted = slope * x + intercept
    ss_res = float(np.sum((y - predicted) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    kappa = float(-slope)
    degenerate = kappa <= kappa_floor
    if degenerate:
        kappa = kappa_floor
    return IntensityEstimate(end, float(np.exp(intercept)), kappa, float(r_squared), degenerate, observations)
