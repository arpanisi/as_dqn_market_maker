"""Step 4 shared fill simulation and funding accounting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import pandas as pd

from data_pipeline.replay import AggressorSide, BookSide, FundingRate, OrderBookReplay, Trade, to_utc_timestamp

QuoteSide = Literal["bid", "ask"]


@dataclass(frozen=True)
class PostedQuote:
    side: QuoteSide
    price: float
    size_btc: float
    timestamp: pd.Timestamp


@dataclass(frozen=True)
class Fill:
    side: QuoteSide
    price: float
    size_btc: float
    timestamp: pd.Timestamp
    liquidity: Literal["maker", "taker"]


@dataclass(frozen=True)
class FundingCashFlow:
    timestamp: pd.Timestamp
    q_btc: float
    funding_rate: float
    mark_price: float
    cash_flow: float


def simulate_quote_fill(
    quote: PostedQuote,
    book: OrderBookReplay,
    trades: Iterable[Trade],
    cancel_time: object,
) -> list[Fill]:
    """Fill one posted quote until cancellation using side-specific queue matching."""
    posted_at = to_utc_timestamp(quote.timestamp)
    cancel_at = to_utc_timestamp(cancel_time)
    immediate = _marketable_fill(quote, book, posted_at)
    if immediate:
        return immediate

    own_side: BookSide = "bid" if quote.side == "bid" else "ask"
    queue_ahead = book.depth_at(own_side, quote.price)
    consumed = 0.0
    remaining = float(quote.size_btc)
    fills: list[Fill] = []

    for trade in sorted(trades, key=lambda item: to_utc_timestamp(item.timestamp)):
        trade_ts = to_utc_timestamp(trade.timestamp)
        if trade_ts <= posted_at or trade_ts > cancel_at:
            continue
        if not _qualifies_for_passive_fill(quote, trade):
            continue

        trade_size = float(trade.size)
        if consumed + trade_size <= queue_ahead:
            consumed += trade_size
            continue

        fillable = min(remaining, consumed + trade_size - queue_ahead)
        if fillable > 0:
            fills.append(Fill(quote.side, float(quote.price), float(fillable), trade_ts, "maker"))
            remaining -= fillable
        consumed += trade_size
        if remaining <= 0:
            break

    return fills


def apply_fills_to_inventory(q_btc: float, fills: Iterable[Fill]) -> float:
    """Update signed BTC inventory: bid fills buy, ask fills sell."""
    inventory = float(q_btc)
    for fill in fills:
        inventory += fill.size_btc if fill.side == "bid" else -fill.size_btc
    return inventory


def funding_cash_flows(
    *,
    q_btc: float,
    funding_rates: Iterable[FundingRate],
    mark_prices: pd.DataFrame,
    start_time: object,
    end_time: object,
) -> list[FundingCashFlow]:
    """Apply exact -q_BTC * f * mark funding cash flow for settlements in a window."""
    start = to_utc_timestamp(start_time)
    end = to_utc_timestamp(end_time)
    flows: list[FundingCashFlow] = []
    for funding in sorted(funding_rates, key=lambda item: to_utc_timestamp(item.timestamp)):
        ts = to_utc_timestamp(funding.timestamp)
        if not (start < ts <= end):
            continue
        mark = _mark_price_at(mark_prices, ts)
        cash_flow = -float(q_btc) * float(funding.rate) * mark
        flows.append(FundingCashFlow(ts, float(q_btc), float(funding.rate), mark, float(cash_flow)))
    return flows


def _marketable_fill(quote: PostedQuote, book: OrderBookReplay, timestamp: pd.Timestamp) -> list[Fill]:
    remaining = float(quote.size_btc)
    fills: list[Fill] = []
    snapshot = book.snapshot()
    levels = snapshot["asks"] if quote.side == "bid" else snapshot["bids"]
    for price, size in levels:
        marketable = price <= quote.price if quote.side == "bid" else price >= quote.price
        if not marketable:
            break
        fill_size = min(remaining, float(size))
        fills.append(Fill(quote.side, float(price), float(fill_size), timestamp, "taker"))
        remaining -= fill_size
        if remaining <= 0:
            break
    return fills


def _qualifies_for_passive_fill(quote: PostedQuote, trade: Trade) -> bool:
    expected_side: AggressorSide = "sell" if quote.side == "bid" else "buy"
    if trade.aggressor_side != expected_side:
        return False
    return trade.price <= quote.price if quote.side == "bid" else trade.price >= quote.price


def _mark_price_at(mark_prices: pd.DataFrame, timestamp: pd.Timestamp) -> float:
    mids = mark_prices["mid_price"].sort_index()
    eligible = mids.loc[:timestamp]
    if eligible.empty:
        raise ValueError(f"no mark price available at or before {timestamp}")
    return float(eligible.iloc[-1])
