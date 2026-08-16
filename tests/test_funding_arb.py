from __future__ import annotations

import pandas as pd
import pytest

from data_pipeline.replay import FundingRate
from strategy.funding_arb import FundingArbitrageStrategy


@pytest.fixture
def funding_rates() -> list[FundingRate]:
    return [
        FundingRate(pd.Timestamp("2025-05-01 00:00:00", tz="UTC"), 0.0005),
        FundingRate(pd.Timestamp("2025-05-01 08:00:00", tz="UTC"), -0.0003),
        FundingRate(pd.Timestamp("2025-05-01 16:00:00", tz="UTC"), 0.00005),
        FundingRate(pd.Timestamp("2025-05-02 00:00:00", tz="UTC"), 0.0002),
    ]


@pytest.fixture
def spot_trades() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2025-04-30 23:59:00Z",
                    "2025-05-01 07:59:59Z",
                    "2025-05-01 15:59:59Z",
                    "2025-05-01 23:59:59Z",
                ]
            ),
            "price": [10000.0, 10100.0, 10050.0, 10200.0],
        }
    )


@pytest.fixture
def perp_marks() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2025-05-01 00:00:00Z",
                    "2025-05-01 08:00:00Z",
                    "2025-05-01 16:00:00Z",
                    "2025-05-02 00:00:00Z",
                ]
            ),
            "mid_price": [10005.0, 10105.0, 10055.0, 10205.0],
        }
    )


def test_funding_arb_position_triggers_flip_and_flatten(funding_rates, spot_trades, perp_marks) -> None:
    strategy = FundingArbitrageStrategy(n_arb=50_000.0, f_min=0.0001)
    result = strategy.run_backtest(funding_rates, spot_trades, perp_marks)

    assert result.iloc[0]["next_position_type"] == "SHORT_PERP_LONG_SPOT"
    assert result.iloc[0]["next_spot_units"] > 0
    assert result.iloc[0]["next_perp_units"] < 0
    assert result.iloc[1]["held_position_type"] == "SHORT_PERP_LONG_SPOT"
    assert result.iloc[1]["next_position_type"] == "LONG_PERP_SHORT_SPOT"
    assert result.iloc[1]["next_spot_units"] < 0
    assert result.iloc[1]["next_perp_units"] > 0
    assert result.iloc[2]["next_position_type"] == "FLAT"
    assert result.iloc[2]["next_spot_units"] == 0.0
    assert result.iloc[2]["next_perp_units"] == 0.0


def test_funding_arb_uses_spot_trade_asof_mark(funding_rates, spot_trades, perp_marks) -> None:
    strategy = FundingArbitrageStrategy(n_arb=50_000.0, f_min=0.0001)
    result = strategy.run_backtest(funding_rates, spot_trades, perp_marks)

    assert result.iloc[0]["spot_price"] == 10000.0
    assert result.iloc[1]["spot_price"] == 10100.0
    assert result.iloc[2]["spot_price"] == 10050.0
    assert result.iloc[3]["spot_price"] == 10200.0


def test_funding_arb_pnl_components_match_locked_formula(funding_rates, spot_trades, perp_marks) -> None:
    strategy = FundingArbitrageStrategy(n_arb=50_000.0, f_min=0.0001)
    result = strategy.run_backtest(funding_rates, spot_trades, perp_marks)

    first = result.iloc[0]
    second = result.iloc[1]
    expected_spot_units = 50_000.0 / 10000.0
    expected_perp_units = -50_000.0 / 10005.0
    assert first["funding_pnl"] == 0.0
    assert first["basis_pnl"] == 0.0
    assert second["held_spot_units"] == pytest.approx(expected_spot_units)
    assert second["held_perp_units"] == pytest.approx(expected_perp_units)
    assert second["funding_pnl"] == pytest.approx(-expected_perp_units * -0.0003 * 10105.0)
    assert second["basis_pnl"] == pytest.approx(
        expected_spot_units * (10100.0 - 10000.0) + expected_perp_units * (10105.0 - 10005.0)
    )
    assert second["net_pnl"] == pytest.approx(second["funding_pnl"] + second["basis_pnl"])


def test_funding_arb_daily_returns_and_report(funding_rates, spot_trades, perp_marks) -> None:
    strategy = FundingArbitrageStrategy(n_arb=50_000.0, f_min=0.0001)
    result = strategy.run_backtest(funding_rates, spot_trades, perp_marks)
    daily_returns = strategy.compute_daily_returns(result)
    report = strategy.evaluate(result)

    assert len(daily_returns) == 2
    first_day_pnl = result.iloc[:3]["net_pnl"].sum()
    assert daily_returns.iloc[0] == pytest.approx(first_day_pnl / 50_000.0)
    assert report.total_net_pnl == pytest.approx(result["net_pnl"].sum())
    assert report.total_funding_pnl == pytest.approx(result["funding_pnl"].sum())
    assert report.total_basis_pnl == pytest.approx(result["basis_pnl"].sum())


def test_funding_arb_requires_marks_at_or_before_first_settlement(funding_rates, perp_marks) -> None:
    spot_trades = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2025-05-01 00:00:01Z"]),
            "price": [10000.0],
        }
    )
    strategy = FundingArbitrageStrategy()
    with pytest.raises(ValueError, match="no spot mark exists"):
        strategy.run_backtest(funding_rates, spot_trades, perp_marks)
