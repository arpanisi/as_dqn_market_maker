"""Step 3 Avellaneda-Stoikov quoting engine."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from data_pipeline.replay import to_utc_timestamp
from settings import DEFAULT_SETTINGS


@dataclass(frozen=True)
class Quote:
    bid: float
    ask: float
    size_btc: float
    reservation_price: float
    total_spread: float
    sigma_price: float
    time_remaining_years: float


def time_remaining_to_next_utc_hour(timestamp: object) -> float:
    """Hours remaining until the next top-of-UTC-hour boundary."""
    ts = to_utc_timestamp(timestamp)
    next_hour = ts.ceil("h")
    if next_hour == ts:
        next_hour = ts + pd.Timedelta(hours=1)
    return (next_hour - ts).total_seconds() / 3600.0


def round_to_tick(price: float, tick_size: float | None = None) -> float:
    if tick_size is None:
        tick_size = DEFAULT_SETTINGS.market.tick_size
    return float(np.round(price / tick_size) * tick_size)


def avellaneda_stoikov_quote(
    *,
    mid_price: float,
    inventory_q: float,
    sigma: float,
    kappa: float,
    risk_aversion: float,
    skew: float,
    timestamp: object,
    tick_size: float | None = None,
    fixed_quote_size_btc: float | None = None,
) -> Quote:
    """Compute one bid/ask quote pair for the current 5-second cycle."""
    if tick_size is None:
        tick_size = DEFAULT_SETTINGS.market.tick_size
    if fixed_quote_size_btc is None:
        fixed_quote_size_btc = DEFAULT_SETTINGS.market.fixed_quote_size_btc

    if mid_price <= 0:
        raise ValueError("mid_price must be positive")
    if kappa <= 0:
        raise ValueError("kappa must be positive")
    if risk_aversion <= 0:
        raise ValueError("risk_aversion must be positive")
    if 1.0 + skew <= 0:
        raise ValueError("1 + skew must be positive")

    sigma_price = float(sigma) * float(mid_price)
    remaining_hours = time_remaining_to_next_utc_hour(timestamp)
    remaining_years = remaining_hours / DEFAULT_SETTINGS.time.hours_per_year

    gamma = float(risk_aversion)
    r = float(mid_price) - float(inventory_q) * gamma * sigma_price**2 * remaining_years
    total_spread = gamma * sigma_price**2 * remaining_years + (2.0 / gamma) * np.log1p(gamma / float(kappa))

    ask = (r + total_spread / 2.0) * (1.0 + float(skew))
    bid = (r - total_spread / 2.0) * (1.0 + float(skew))

    rounded_ask = round_to_tick(ask, tick_size)
    rounded_bid = round_to_tick(bid, tick_size)
    while rounded_ask <= rounded_bid:
        rounded_ask += tick_size
        rounded_bid -= tick_size

    return Quote(
        bid=round(rounded_bid, 10),
        ask=round(rounded_ask, 10),
        size_btc=float(fixed_quote_size_btc),
        reservation_price=float(r),
        total_spread=float(total_spread),
        sigma_price=float(sigma_price),
        time_remaining_years=float(remaining_years),
    )
