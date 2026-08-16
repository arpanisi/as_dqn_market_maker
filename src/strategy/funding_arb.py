"""Step 12 standalone funding-rate arbitrage strategy."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from data_pipeline.replay import FundingRate, to_utc_timestamp
from evaluation.protocol import sharpe_ratio_from_pnl

DEFAULT_N_ARB = 50_000.0
DEFAULT_F_MIN = 0.0001


@dataclass(frozen=True)
class ArbSettlementRecord:
    timestamp: pd.Timestamp
    funding_rate: float
    spot_price: float
    perp_mark_price: float
    held_position_type: str
    held_spot_units: float
    held_perp_units: float
    entry_spot_price: float | None
    entry_perp_mark_price: float | None
    funding_pnl: float
    basis_pnl: float
    net_pnl: float
    next_position_type: str
    next_spot_units: float
    next_perp_units: float


@dataclass(frozen=True)
class FundingArbitrageReport:
    sharpe: float
    total_funding_pnl: float
    total_basis_pnl: float
    total_net_pnl: float
    daily_returns: pd.Series
    settlements: pd.DataFrame


class FundingArbitrageStrategy:
    """Delta-neutral carry strategy, fully separate from market-making state."""

    def __init__(self, n_arb: float = DEFAULT_N_ARB, f_min: float = DEFAULT_F_MIN):
        self.n_arb = float(n_arb)
        self.f_min = float(f_min)

    def run_backtest(
        self,
        funding_rates: list[FundingRate],
        spot_marks: pd.DataFrame | pd.Series | dict[pd.Timestamp, float],
        perp_marks: pd.DataFrame | pd.Series | dict[pd.Timestamp, float],
    ) -> pd.DataFrame:
        if not funding_rates:
            return pd.DataFrame([record.__dict__ for record in []])

        sorted_rates = sorted(funding_rates, key=lambda item: to_utc_timestamp(item.timestamp))
        spot_series = _marks_to_series(spot_marks, "spot")
        perp_series = _marks_to_series(perp_marks, "perp")

        position_type = "FLAT"
        spot_units = 0.0
        perp_units = 0.0
        entry_spot_price: float | None = None
        entry_perp_mark_price: float | None = None
        records: list[ArbSettlementRecord] = []

        for funding in sorted_rates:
            timestamp = to_utc_timestamp(funding.timestamp)
            funding_rate = float(funding.rate)
            spot_price = _asof_price(spot_series, timestamp, "spot")
            perp_mark_price = _asof_price(perp_series, timestamp, "perp")

            held_position_type = position_type
            held_spot_units = spot_units
            held_perp_units = perp_units
            held_entry_spot = entry_spot_price
            held_entry_perp = entry_perp_mark_price

            if position_type == "FLAT":
                funding_pnl = 0.0
                basis_pnl = 0.0
            else:
                if entry_spot_price is None or entry_perp_mark_price is None:
                    raise ValueError("open funding-arb position is missing entry marks")
                funding_pnl = -perp_units * funding_rate * perp_mark_price
                basis_pnl = spot_units * (spot_price - entry_spot_price) + perp_units * (
                    perp_mark_price - entry_perp_mark_price
                )
            net_pnl = funding_pnl + basis_pnl

            position_type, spot_units, perp_units, entry_spot_price, entry_perp_mark_price = self._next_position(
                funding_rate,
                spot_price,
                perp_mark_price,
            )

            records.append(
                ArbSettlementRecord(
                    timestamp=timestamp,
                    funding_rate=funding_rate,
                    spot_price=spot_price,
                    perp_mark_price=perp_mark_price,
                    held_position_type=held_position_type,
                    held_spot_units=held_spot_units,
                    held_perp_units=held_perp_units,
                    entry_spot_price=held_entry_spot,
                    entry_perp_mark_price=held_entry_perp,
                    funding_pnl=funding_pnl,
                    basis_pnl=basis_pnl,
                    net_pnl=net_pnl,
                    next_position_type=position_type,
                    next_spot_units=spot_units,
                    next_perp_units=perp_units,
                )
            )

        return pd.DataFrame([record.__dict__ for record in records])

    def compute_daily_returns(self, settlement_df: pd.DataFrame) -> pd.Series:
        if settlement_df.empty or "net_pnl" not in settlement_df.columns:
            return pd.Series(dtype=float)
        timestamps = pd.DatetimeIndex([to_utc_timestamp(ts) for ts in settlement_df["timestamp"]])
        series = pd.Series(settlement_df["net_pnl"].astype(float).to_numpy(), index=timestamps)
        daily_pnl = series.groupby(series.index.floor("D")).sum()
        return daily_pnl / self.n_arb

    def evaluate(self, settlement_df: pd.DataFrame) -> FundingArbitrageReport:
        daily_returns = self.compute_daily_returns(settlement_df)
        daily_pnl = daily_returns * self.n_arb
        return FundingArbitrageReport(
            sharpe=sharpe_ratio_from_pnl(daily_pnl, reference_capital=self.n_arb),
            total_funding_pnl=float(settlement_df["funding_pnl"].sum()) if not settlement_df.empty else 0.0,
            total_basis_pnl=float(settlement_df["basis_pnl"].sum()) if not settlement_df.empty else 0.0,
            total_net_pnl=float(settlement_df["net_pnl"].sum()) if not settlement_df.empty else 0.0,
            daily_returns=daily_returns,
            settlements=settlement_df,
        )

    def _next_position(
        self,
        funding_rate: float,
        spot_price: float,
        perp_mark_price: float,
    ) -> tuple[str, float, float, float | None, float | None]:
        if funding_rate > self.f_min:
            return (
                "SHORT_PERP_LONG_SPOT",
                self.n_arb / spot_price,
                -self.n_arb / perp_mark_price,
                spot_price,
                perp_mark_price,
            )
        if funding_rate < -self.f_min:
            return (
                "LONG_PERP_SHORT_SPOT",
                -self.n_arb / spot_price,
                self.n_arb / perp_mark_price,
                spot_price,
                perp_mark_price,
            )
        return "FLAT", 0.0, 0.0, None, None


def _marks_to_series(marks: pd.DataFrame | pd.Series | dict[pd.Timestamp, float], label: str) -> pd.Series:
    if isinstance(marks, dict):
        series = pd.Series({to_utc_timestamp(ts): float(price) for ts, price in marks.items()})
    elif isinstance(marks, pd.Series):
        series = marks.copy()
        series.index = pd.DatetimeIndex([to_utc_timestamp(ts) for ts in series.index])
        series = series.astype(float)
    else:
        frame = marks.copy()
        if "timestamp" in frame.columns:
            price_column = _price_column(frame, label)
            series = pd.Series(
                frame[price_column].astype(float).to_numpy(),
                index=pd.DatetimeIndex([to_utc_timestamp(ts) for ts in frame["timestamp"]]),
            )
        else:
            price_column = _price_column(frame, label)
            series = pd.Series(
                frame[price_column].astype(float).to_numpy(),
                index=pd.DatetimeIndex([to_utc_timestamp(ts) for ts in frame.index]),
            )
    series = series.sort_index()
    if series.empty:
        raise ValueError(f"{label} mark series is empty")
    return series[~series.index.duplicated(keep="last")]


def _price_column(frame: pd.DataFrame, label: str) -> str:
    candidates = ("price", "mid_price", "mark_price", "perp_mark_price", f"{label}_price")
    for column in candidates:
        if column in frame.columns:
            return column
    raise ValueError(f"{label} marks must include one of: {', '.join(candidates)}")


def _asof_price(series: pd.Series, timestamp: pd.Timestamp, label: str) -> float:
    if timestamp < series.index[0]:
        raise ValueError(f"no {label} mark exists at or before {timestamp}")
    position = series.index.searchsorted(timestamp, side="right") - 1
    return float(series.iloc[position])
