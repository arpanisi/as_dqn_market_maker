"""YAML-backed project settings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SETTINGS_PATH = PROJECT_ROOT / "config" / "market_maker.yaml"


@dataclass(frozen=True)
class InstrumentSettings:
    symbol: str
    category: str


@dataclass(frozen=True)
class MarketSettings:
    tick_size: float
    fixed_quote_size_btc: float
    tier1_risk_aversion: float
    tier1_gamma_derivation: dict[str, Any]


@dataclass(frozen=True)
class TimeSettings:
    hours_per_year: int


@dataclass(frozen=True)
class IntensitySettings:
    kappa_floor: float


@dataclass(frozen=True)
class BaselineSettings:
    gamma_min: float
    gamma_max: float
    skew_min: float
    skew_max: float
    population_size: int
    generations: int


@dataclass(frozen=True)
class AgentSettings:
    risk_aversions: tuple[float, ...]
    skews: tuple[float, ...]
    hidden_units: int
    learning_rate: float
    batch_size: int
    discount_beta: float
    replay_capacity: int
    warmup_transitions: int
    target_sync_steps: int
    epsilon_start: float
    epsilon_end: float
    epsilon_decay_fraction: float


@dataclass(frozen=True)
class EvaluationSettings:
    reference_capital: float


@dataclass(frozen=True)
class FundingArbitrageSettings:
    n_arb: float
    f_min: float


@dataclass(frozen=True)
class AppSettings:
    instrument: InstrumentSettings
    market: MarketSettings
    time: TimeSettings
    intensity: IntensitySettings
    baseline: BaselineSettings
    agent: AgentSettings
    evaluation: EvaluationSettings
    funding_arbitrage: FundingArbitrageSettings


def load_settings(path: str | Path | None = None) -> AppSettings:
    settings_path = Path(path) if path is not None else DEFAULT_SETTINGS_PATH
    with settings_path.open("r", encoding="utf-8") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh)

    return AppSettings(
        instrument=InstrumentSettings(**raw["instrument"]),
        market=MarketSettings(**raw["market"]),
        time=TimeSettings(**raw["time"]),
        intensity=IntensitySettings(**raw["intensity"]),
        baseline=BaselineSettings(**raw["baseline"]),
        agent=AgentSettings(
            **{
                **raw["agent"],
                "risk_aversions": tuple(float(v) for v in raw["agent"]["risk_aversions"]),
                "skews": tuple(float(v) for v in raw["agent"]["skews"]),
            }
        ),
        evaluation=EvaluationSettings(**raw["evaluation"]),
        funding_arbitrage=FundingArbitrageSettings(**raw["funding_arbitrage"]),
    )


DEFAULT_SETTINGS = load_settings()
