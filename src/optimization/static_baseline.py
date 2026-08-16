"""Step 5 fixed-parameter baseline search."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import cma
import numpy as np

from settings import DEFAULT_SETTINGS


@dataclass(frozen=True)
class BaselineCandidate:
    risk_aversion: float
    skew: float
    sharpe: float


@dataclass(frozen=True)
class BaselineSearchResult:
    best: BaselineCandidate
    history: tuple[BaselineCandidate, ...]


def sharpe_ratio(pnl: np.ndarray, reference_capital: float = 100_000.0) -> float:
    returns = np.asarray(pnl, dtype=float) / float(reference_capital)
    if returns.size < 2:
        return 0.0
    std = float(returns.std(ddof=1))
    if std == 0.0:
        return 0.0
    return float(returns.mean() / std)


def search_static_baseline(
    score_fn: Callable[[float, float], float],
    *,
    seed: int = 0,
    population_size: int | None = None,
    generations: int | None = None,
) -> BaselineSearchResult:
    """CMA-ES search over log-gamma and linear skew."""
    cfg = DEFAULT_SETTINGS.baseline
    population_size = population_size or cfg.population_size
    generations = generations or cfg.generations

    lower_bounds = [np.log(cfg.gamma_min), cfg.skew_min]
    upper_bounds = [np.log(cfg.gamma_max), cfg.skew_max]
    start = [(lower_bounds[0] + upper_bounds[0]) / 2.0, 0.0]
    sigma0 = 0.5
    strategy = cma.CMAEvolutionStrategy(
        start,
        sigma0,
        {
            "bounds": [lower_bounds, upper_bounds],
            "popsize": population_size,
            "seed": seed,
            "verbose": -9,
        },
    )
    history: list[BaselineCandidate] = []

    for _ in range(generations):
        raw_candidates = strategy.ask()
        losses: list[float] = []
        for raw in raw_candidates:
            log_gamma, skew = raw
            gamma = float(np.exp(log_gamma))
            score = float(score_fn(gamma, float(skew)))
            losses.append(-score)
            history.append(BaselineCandidate(gamma, float(skew), score))
        strategy.tell(raw_candidates, losses)

    best = max(history, key=lambda item: item.sharpe)
    return BaselineSearchResult(best=best, history=tuple(history))
