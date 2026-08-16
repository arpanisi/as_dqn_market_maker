"""Step 2 order-arrival intensity estimation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from data_pipeline.replay import Trade, to_utc_timestamp
from data_pipeline.series import nearest_mid_price
from settings import DEFAULT_SETTINGS


@dataclass(frozen=True)
class IntensityEstimate:
    window_end: pd.Timestamp
    A: float
    kappa: float
    r_squared: float
    degenerate: bool
    observations: int


def estimate_arrival_intensity(
    trades: list[Trade],
    mid_prices: pd.DataFrame,
    window_end: object,
    window: str = "24h",
    bins: int = 20,
    kappa_floor: float | None = None,
) -> IntensityEstimate:
    """Fit lambda(delta) = A * exp(-kappa * delta) in a trailing window."""
    if kappa_floor is None:
        kappa_floor = DEFAULT_SETTINGS.intensity.kappa_floor

    end = to_utc_timestamp(window_end)
    start = end - pd.Timedelta(window)
    window_trades = [t for t in trades if start < to_utc_timestamp(t.timestamp) <= end]
    if len(window_trades) < 2:
        return IntensityEstimate(end, 0.0, kappa_floor, 0.0, True, len(window_trades))

    distances = np.array(
        [abs(float(t.price) - nearest_mid_price(mid_prices, t.timestamp)) for t in window_trades],
        dtype=float,
    )
    if np.allclose(distances.max(), distances.min()):
        rate = len(window_trades) / pd.Timedelta(window).total_seconds()
        return IntensityEstimate(end, rate, kappa_floor, 0.0, True, len(window_trades))

    counts, edges = np.histogram(distances, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2.0
    nonzero = counts > 0
    if nonzero.sum() < 2:
        rate = len(window_trades) / pd.Timedelta(window).total_seconds()
        return IntensityEstimate(end, rate, kappa_floor, 0.0, True, len(window_trades))

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
    return IntensityEstimate(end, float(np.exp(intercept)), kappa, float(r_squared), degenerate, len(window_trades))
