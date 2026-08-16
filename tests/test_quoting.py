import pytest

from model.quoting import avellaneda_stoikov_quote, time_remaining_to_next_utc_hour
from settings import DEFAULT_SETTINGS


def test_time_remaining_rolls_to_next_utc_hour():
    assert time_remaining_to_next_utc_hour("2024-06-01T12:00:00Z") == pytest.approx(1.0)
    assert time_remaining_to_next_utc_hour("2024-06-01T12:30:00Z") == pytest.approx(0.5)


def test_quote_is_deterministic_tick_aligned_and_ordered():
    kwargs = dict(
        mid_price=60000.0,
        inventory_q=0.0,
        sigma=0.5,
        kappa=2.0,
        risk_aversion=0.1,
        skew=0.0,
        timestamp="2024-06-01T12:30:00Z",
    )

    quote1 = avellaneda_stoikov_quote(**kwargs)
    quote2 = avellaneda_stoikov_quote(**kwargs)

    assert quote1 == quote2
    assert quote1.ask > quote1.bid
    assert round(quote1.ask * 10) == pytest.approx(quote1.ask * 10)
    assert round(quote1.bid * 10) == pytest.approx(quote1.bid * 10)
    assert quote1.size_btc == pytest.approx(0.01)


def test_inventory_moves_reservation_price_to_discourage_accumulation():
    flat = avellaneda_stoikov_quote(
        mid_price=60000.0,
        inventory_q=0.0,
        sigma=0.5,
        kappa=2.0,
        risk_aversion=0.1,
        skew=0.0,
        timestamp="2024-06-01T12:30:00Z",
    )
    long_inventory = avellaneda_stoikov_quote(
        mid_price=60000.0,
        inventory_q=3.0,
        sigma=0.5,
        kappa=2.0,
        risk_aversion=0.1,
        skew=0.0,
        timestamp="2024-06-01T12:30:00Z",
    )

    assert long_inventory.reservation_price < flat.reservation_price


def test_spread_widens_with_time_and_risk_aversion():
    short_horizon = avellaneda_stoikov_quote(
        mid_price=60000.0,
        inventory_q=0.0,
        sigma=0.5,
        kappa=2.0,
        risk_aversion=0.1,
        skew=0.0,
        timestamp="2024-06-01T12:59:00Z",
    )
    long_horizon = avellaneda_stoikov_quote(
        mid_price=60000.0,
        inventory_q=0.0,
        sigma=0.5,
        kappa=2.0,
        risk_aversion=0.1,
        skew=0.0,
        timestamp="2024-06-01T12:00:00Z",
    )
    high_gamma = avellaneda_stoikov_quote(
        mid_price=60000.0,
        inventory_q=0.0,
        sigma=0.5,
        kappa=2.0,
        risk_aversion=0.2,
        skew=0.0,
        timestamp="2024-06-01T12:00:00Z",
    )

    assert long_horizon.total_spread > short_horizon.total_spread
    assert high_gamma.total_spread > long_horizon.total_spread


def test_tier1_gamma_is_real_data_scaled_below_one_percent():
    derivation = DEFAULT_SETTINGS.market.tier1_gamma_derivation
    leading_order_gamma = derivation["target_spread_fraction"] / (
        derivation["observed_annualized_sigma"] ** 2
        * derivation["observed_mid_price"]
        * derivation["observed_time_remaining_years"]
    )
    quote = avellaneda_stoikov_quote(
        mid_price=derivation["observed_mid_price"],
        inventory_q=0.0,
        sigma=derivation["observed_annualized_sigma"],
        kappa=0.01,
        risk_aversion=DEFAULT_SETTINGS.market.tier1_risk_aversion,
        skew=0.0,
        timestamp="2026-07-01T22:26:15Z",
    )

    assert derivation["leading_order_gamma_for_50bps"] == pytest.approx(leading_order_gamma)
    assert DEFAULT_SETTINGS.market.tier1_risk_aversion == pytest.approx(1e-5)
    assert (quote.ask - quote.bid) / derivation["observed_mid_price"] < 0.01
