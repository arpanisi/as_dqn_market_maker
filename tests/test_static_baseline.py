import pytest

cma = pytest.importorskip("cma")

from optimization.static_baseline import search_static_baseline, sharpe_ratio


def test_sharpe_ratio_uses_reference_capital_returns():
    assert sharpe_ratio([100.0, 200.0, -50.0], reference_capital=100_000.0) == pytest.approx(
        sharpe_ratio([0.001, 0.002, -0.0005], reference_capital=1.0)
    )


def test_static_baseline_search_converges_on_synthetic_optimum():
    def score(gamma: float, skew: float) -> float:
        return -((gamma - 0.2) ** 2) - ((skew - 0.05) ** 2)

    first = search_static_baseline(score, seed=1, population_size=16, generations=25).best
    second = search_static_baseline(score, seed=2, population_size=16, generations=25).best

    assert first.risk_aversion == pytest.approx(0.2, rel=0.25)
    assert first.skew == pytest.approx(0.05, abs=0.02)
    assert abs(first.sharpe - second.sharpe) < 0.05
