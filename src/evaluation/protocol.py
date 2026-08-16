"""Step 7 baseline-vs-adaptive evaluation report."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from data_pipeline.replay import to_utc_timestamp
from settings import DEFAULT_SETTINGS


@dataclass(frozen=True)
class DailyPnL:
    date: pd.Timestamp
    static_baseline_pnl: float
    adaptive_agent_pnl: float
    adaptive_won: bool


@dataclass(frozen=True)
class EvaluationReport:
    static_baseline_sharpe: float
    adaptive_agent_sharpe: float
    static_baseline_sortino: float
    adaptive_agent_sortino: float
    adaptive_win_count: int
    total_days: int
    daily_pnl: tuple[DailyPnL, ...]


@dataclass(frozen=True)
class HourOfDayPnL:
    utc_hour: int
    static_baseline_sharpe: float
    adaptive_agent_sharpe: float
    sharpe_difference: float
    static_baseline_pnl_by_day: tuple[float, ...]
    adaptive_agent_pnl_by_day: tuple[float, ...]


def sharpe_ratio_from_pnl(pnl: Iterable[float], reference_capital: float | None = None) -> float:
    returns = _returns(pnl, reference_capital)
    if returns.size < 2:
        return 0.0
    std = float(returns.std(ddof=1))
    if std == 0.0:
        return 0.0
    return float(returns.mean() / std)


def sortino_ratio_from_pnl(pnl: Iterable[float], reference_capital: float | None = None) -> float:
    returns = _returns(pnl, reference_capital)
    downside = returns[returns < 0.0]
    if returns.size < 2 or downside.size < 2:
        return 0.0
    downside_std = float(downside.std(ddof=1))
    if downside_std == 0.0:
        return 0.0
    return float(returns.mean() / downside_std)


def evaluate_static_vs_adaptive(
    static_pnl: pd.Series,
    adaptive_pnl: pd.Series,
    *,
    reference_capital: float | None = None,
) -> EvaluationReport:
    """Evaluate full-window daily Sharpe and independently checkable day wins."""
    static_daily = _daily_pnl(static_pnl)
    adaptive_daily = _daily_pnl(adaptive_pnl)
    joined = pd.concat(
        [static_daily.rename("static_baseline_pnl"), adaptive_daily.rename("adaptive_agent_pnl")],
        axis=1,
    ).fillna(0.0)
    joined = joined.sort_index()

    daily = tuple(
        DailyPnL(
            date=pd.Timestamp(index),
            static_baseline_pnl=float(row.static_baseline_pnl),
            adaptive_agent_pnl=float(row.adaptive_agent_pnl),
            adaptive_won=bool(row.adaptive_agent_pnl > row.static_baseline_pnl),
        )
        for index, row in joined.iterrows()
    )

    static_values = joined["static_baseline_pnl"].to_numpy(dtype=float)
    adaptive_values = joined["adaptive_agent_pnl"].to_numpy(dtype=float)
    win_count = int((joined["adaptive_agent_pnl"] > joined["static_baseline_pnl"]).sum())

    return EvaluationReport(
        static_baseline_sharpe=sharpe_ratio_from_pnl(static_values, reference_capital),
        adaptive_agent_sharpe=sharpe_ratio_from_pnl(adaptive_values, reference_capital),
        static_baseline_sortino=sortino_ratio_from_pnl(static_values, reference_capital),
        adaptive_agent_sortino=sortino_ratio_from_pnl(adaptive_values, reference_capital),
        adaptive_win_count=win_count,
        total_days=len(daily),
        daily_pnl=daily,
    )


def time_of_day_breakdown(
    static_cycle_pnl: pd.Series,
    adaptive_cycle_pnl: pd.Series,
    *,
    reference_capital: float | None = None,
) -> tuple[HourOfDayPnL, ...]:
    """Re-aggregate per-cycle P&L into 24 UTC hour buckets with one value per day."""
    static_grid = _hour_day_grid(static_cycle_pnl)
    adaptive_grid = _hour_day_grid(adaptive_cycle_pnl)
    all_days = static_grid.columns.union(adaptive_grid.columns).sort_values()
    static_grid = static_grid.reindex(index=range(24), columns=all_days, fill_value=0.0)
    adaptive_grid = adaptive_grid.reindex(index=range(24), columns=all_days, fill_value=0.0)

    rows: list[HourOfDayPnL] = []
    for hour in range(24):
        static_values = static_grid.loc[hour].to_numpy(dtype=float)
        adaptive_values = adaptive_grid.loc[hour].to_numpy(dtype=float)
        static_sharpe = sharpe_ratio_from_pnl(static_values, reference_capital)
        adaptive_sharpe = sharpe_ratio_from_pnl(adaptive_values, reference_capital)
        rows.append(
            HourOfDayPnL(
                utc_hour=hour,
                static_baseline_sharpe=static_sharpe,
                adaptive_agent_sharpe=adaptive_sharpe,
                sharpe_difference=adaptive_sharpe - static_sharpe,
                static_baseline_pnl_by_day=tuple(float(v) for v in static_values),
                adaptive_agent_pnl_by_day=tuple(float(v) for v in adaptive_values),
            )
        )
    return tuple(rows)


def _daily_pnl(pnl: pd.Series) -> pd.Series:
    if not isinstance(pnl.index, pd.DatetimeIndex):
        raise ValueError("P&L series must use a DatetimeIndex")
    index = pd.DatetimeIndex([to_utc_timestamp(ts) for ts in pnl.index])
    series = pd.Series(pnl.to_numpy(dtype=float), index=index)
    return series.groupby(series.index.floor("D")).sum()


def _hour_day_grid(pnl: pd.Series) -> pd.DataFrame:
    if not isinstance(pnl.index, pd.DatetimeIndex):
        raise ValueError("P&L series must use a DatetimeIndex")
    index = pd.DatetimeIndex([to_utc_timestamp(ts) for ts in pnl.index])
    series = pd.Series(pnl.to_numpy(dtype=float), index=index)
    frame = pd.DataFrame({"pnl": series})
    frame["day"] = frame.index.floor("D")
    frame["hour"] = frame.index.hour
    return frame.pivot_table(index="hour", columns="day", values="pnl", aggfunc="sum", fill_value=0.0)


def _returns(pnl: Iterable[float], reference_capital: float | None) -> np.ndarray:
    capital = float(reference_capital or DEFAULT_SETTINGS.evaluation.reference_capital)
    return np.asarray(tuple(pnl), dtype=float) / capital
