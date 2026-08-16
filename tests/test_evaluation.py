import pandas as pd
import pytest

from evaluation.protocol import (
    evaluate_static_vs_adaptive,
    sharpe_ratio_from_pnl,
    sortino_ratio_from_pnl,
    time_of_day_breakdown,
)


def test_step7_report_contains_sharpe_win_count_and_per_day_pnl():
    index = pd.to_datetime(
        [
            "2026-07-01T00:00:00Z",
            "2026-07-01T12:00:00Z",
            "2026-07-02T00:00:00Z",
            "2026-07-03T00:00:00Z",
        ],
        utc=True,
    )
    static = pd.Series([100.0, 50.0, -20.0, 40.0], index=index)
    adaptive = pd.Series([120.0, 60.0, -30.0, 45.0], index=index)

    report = evaluate_static_vs_adaptive(static, adaptive, reference_capital=100_000.0)

    assert report.total_days == 3
    assert report.adaptive_win_count == 2
    assert report.daily_pnl[0].static_baseline_pnl == pytest.approx(150.0)
    assert report.daily_pnl[0].adaptive_agent_pnl == pytest.approx(180.0)
    assert report.static_baseline_sharpe == pytest.approx(sharpe_ratio_from_pnl([150.0, -20.0, 40.0], 100_000.0))
    assert report.adaptive_agent_sortino == pytest.approx(sortino_ratio_from_pnl([180.0, -30.0, 45.0], 100_000.0))


def test_evaluation_rejects_non_datetime_pnl_series():
    with pytest.raises(ValueError):
        evaluate_static_vs_adaptive(pd.Series([1.0]), pd.Series([2.0]))


def test_time_of_day_breakdown_reaggregates_per_cycle_pnl_to_hour_day_grid():
    index = pd.to_datetime(
        [
            "2026-07-01T00:00:05Z",
            "2026-07-01T00:00:10Z",
            "2026-07-01T01:00:05Z",
            "2026-07-02T00:00:05Z",
            "2026-07-02T01:00:05Z",
        ],
        utc=True,
    )
    static = pd.Series([1.0, 2.0, 10.0, 4.0, 20.0], index=index)
    adaptive = pd.Series([2.0, 3.0, 8.0, 5.0, 25.0], index=index)

    breakdown = time_of_day_breakdown(static, adaptive, reference_capital=100_000.0)

    assert len(breakdown) == 24
    assert breakdown[0].static_baseline_pnl_by_day == pytest.approx((3.0, 4.0))
    assert breakdown[0].adaptive_agent_pnl_by_day == pytest.approx((5.0, 5.0))
    assert breakdown[1].static_baseline_pnl_by_day == pytest.approx((10.0, 20.0))
    assert breakdown[1].adaptive_agent_pnl_by_day == pytest.approx((8.0, 25.0))
