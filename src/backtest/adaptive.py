"""Adaptive Double-DQN training/evaluation runtime."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from agent.double_dqn import DoubleDQNAgent, Transition, build_state
from backtest.engine import BacktestResult, CycleResult, _book_at
from backtest.fills import PostedQuote, funding_cash_flows, simulate_quote_fill
from data_pipeline.replay import FundingRate, Trade, to_utc_timestamp
from data_pipeline.series import funding_series, nearest_mid_price, trailing_realized_volatility
from model.intensity import estimate_arrival_intensity
from model.quoting import avellaneda_stoikov_quote, time_remaining_to_next_utc_hour
from settings import DEFAULT_SETTINGS
from signals.adverse_selection import adverse_selection_signal, signal_at


@dataclass(frozen=True)
class AdaptiveRunResult:
    backtest: BacktestResult
    losses: tuple[float, ...]
    rewards: tuple[float, ...]
    action_indices: tuple[int, ...]
    funding_values: tuple[float, ...]
    skew_values: tuple[float, ...]


def train_adaptive_agent(
    *,
    agent: DoubleDQNAgent,
    mid_prices: pd.DataFrame,
    books: pd.DataFrame,
    trades: list[Trade],
    funding_rates: list[FundingRate],
    start: object,
    end: object,
    epochs: int = 5,
    cadence: str = "5s",
) -> AdaptiveRunResult:
    result = _run_adaptive(
        agent=agent,
        mid_prices=mid_prices,
        books=books,
        trades=trades,
        funding_rates=funding_rates,
        start=start,
        end=end,
        epochs=epochs,
        train=True,
        cadence=cadence,
    )
    return result


def evaluate_adaptive_agent(
    *,
    agent: DoubleDQNAgent,
    mid_prices: pd.DataFrame,
    books: pd.DataFrame,
    trades: list[Trade],
    funding_rates: list[FundingRate],
    start: object,
    end: object,
    cadence: str = "5s",
) -> AdaptiveRunResult:
    return _run_adaptive(
        agent=agent,
        mid_prices=mid_prices,
        books=books,
        trades=trades,
        funding_rates=funding_rates,
        start=start,
        end=end,
        epochs=1,
        train=False,
        cadence=cadence,
    )


def funding_skew_correlation(funding_values: tuple[float, ...], skew_values: tuple[float, ...]) -> float:
    if len(funding_values) < 2 or len(skew_values) < 2:
        return 0.0
    funding = np.asarray(funding_values, dtype=float)
    skews = np.asarray(skew_values, dtype=float)
    if np.allclose(funding.std(), 0.0) or np.allclose(skews.std(), 0.0):
        return 0.0
    return float(np.corrcoef(np.sign(funding), skews)[0, 1])


def _run_adaptive(
    *,
    agent: DoubleDQNAgent,
    mid_prices: pd.DataFrame,
    books: pd.DataFrame,
    trades: list[Trade],
    funding_rates: list[FundingRate],
    start: object,
    end: object,
    epochs: int,
    train: bool,
    cadence: str,
) -> AdaptiveRunResult:
    start_ts = to_utc_timestamp(start)
    end_ts = to_utc_timestamp(end)
    decision_times = pd.date_range(start_ts, end_ts, freq=cadence, inclusive="left")
    total_training_steps = max(1, len(decision_times) * max(1, epochs))
    vol = trailing_realized_volatility(mid_prices).ffill().fillna(0.0)
    funding = funding_series(funding_rates, pd.DatetimeIndex(mid_prices.index))
    adverse = adverse_selection_signal(trades, mid_prices)
    cycles: list[CycleResult] = []
    losses: list[float] = []
    rewards: list[float] = []
    action_indices: list[int] = []
    funding_values: list[float] = []
    skew_values: list[float] = []
    global_step = 0

    for _ in range(epochs):
        q_btc = 0.0
        last_mark = nearest_mid_price(mid_prices, start_ts)
        pnl_history: list[float] = []
        previous_state: np.ndarray | None = None
        previous_action: int | None = None
        previous_reward: float | None = None
        previous_done = False

        for ts in decision_times:
            next_ts = ts + pd.Timedelta(cadence)
            mid = nearest_mid_price(mid_prices, ts)
            sigma = _last_at(vol, ts)
            intensity = estimate_arrival_intensity(trades, mid_prices, ts)
            funding_rate = _last_at(funding, ts)
            state = build_state(
                inventory_q=q_btc / DEFAULT_SETTINGS.market.fixed_quote_size_btc,
                realized_volatility=sigma,
                arrival_A=intensity.A,
                arrival_kappa=intensity.kappa,
                time_remaining_fraction=time_remaining_to_next_utc_hour(ts),
                adverse_selection=signal_at(adverse, ts),
                funding_rate=funding_rate,
            )

            if train and previous_state is not None and previous_action is not None and previous_reward is not None:
                loss = agent.learn_from_transition(
                    Transition(previous_state, previous_action, previous_reward, state, previous_done)
                )
                if loss is not None:
                    losses.append(loss)

            epsilon = agent.epsilon_for_step(global_step, total_training_steps) if train else 0.0
            action = agent.select_action(state, epsilon=epsilon)
            quote = avellaneda_stoikov_quote(
                mid_price=mid,
                inventory_q=q_btc / DEFAULT_SETTINGS.market.fixed_quote_size_btc,
                sigma=sigma,
                kappa=intensity.kappa,
                risk_aversion=action.risk_aversion,
                skew=action.skew,
                timestamp=ts,
            )
            book = _book_at(books, ts)
            window_trades = [t for t in trades if ts < to_utc_timestamp(t.timestamp) <= next_ts]
            fills = simulate_quote_fill(PostedQuote("bid", quote.bid, quote.size_btc, ts), book, window_trades, next_ts)
            fills += simulate_quote_fill(PostedQuote("ask", quote.ask, quote.size_btc, ts), book, window_trades, next_ts)
            realized = sum((-f.size_btc * f.price if f.side == "bid" else f.size_btc * f.price) for f in fills)
            q_old = q_btc
            q_btc += sum((f.size_btc if f.side == "bid" else -f.size_btc) for f in fills)
            cash_flow = sum(
                flow.cash_flow
                for flow in funding_cash_flows(
                    q_btc=q_btc,
                    funding_rates=funding_rates,
                    mark_prices=mid_prices,
                    start_time=ts,
                    end_time=next_ts,
                )
            )
            new_mark = nearest_mid_price(mid_prices, next_ts)
            pnl_delta = float(realized + q_old * (new_mark - last_mark) + cash_flow)
            last_mark = new_mark
            reward = _sharpe_shaped_reward(pnl_delta, pnl_history)
            pnl_history.append(pnl_delta)
            rewards.append(reward)
            action_indices.append(action.index)
            funding_values.append(funding_rate)
            skew_values.append(action.skew)
            cycles.append(
                CycleResult(
                    timestamp=ts,
                    bid=quote.bid,
                    ask=quote.ask,
                    spread_fraction=float((quote.ask - quote.bid) / mid),
                    fills=len(fills),
                    q_btc=float(q_btc),
                    pnl_delta=pnl_delta,
                )
            )
            previous_state = state
            previous_action = action.index
            previous_reward = reward
            previous_done = False
            global_step += 1

        if train and previous_state is not None and previous_action is not None and previous_reward is not None:
            terminal_loss = agent.learn_from_transition(
                Transition(previous_state, previous_action, previous_reward, previous_state, True)
            )
            if terminal_loss is not None:
                losses.append(terminal_loss)

    return AdaptiveRunResult(
        backtest=BacktestResult(tuple(cycles)),
        losses=tuple(losses),
        rewards=tuple(rewards),
        action_indices=tuple(action_indices),
        funding_values=tuple(funding_values),
        skew_values=tuple(skew_values),
    )


def _sharpe_shaped_reward(pnl_delta: float, pnl_history: list[float]) -> float:
    if len(pnl_history) < 60:
        return float(pnl_delta)
    trailing = np.asarray(pnl_history[-60:], dtype=float)
    vol = float(trailing.std(ddof=1))
    if vol == 0.0:
        return float(pnl_delta)
    return float(pnl_delta / vol)


def _last_at(series: pd.Series, timestamp: pd.Timestamp) -> float:
    values = series.loc[:timestamp]
    if values.empty:
        return 0.0
    value = values.iloc[-1]
    if pd.isna(value):
        return 0.0
    return float(value)
