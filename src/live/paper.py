"""Step 8 live paper execution primitives."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Protocol

from backtest.fills import Fill, PostedQuote, apply_fills_to_inventory
from data_pipeline.replay import OrderBookReplay, Trade, to_utc_timestamp
from model.quoting import Quote, avellaneda_stoikov_quote
from settings import DEFAULT_SETTINGS


class ActionProvider(Protocol):
    def select_action(self, state: object, epsilon: float = 0.0) -> object:
        ...


@dataclass(frozen=True)
class MarketState:
    timestamp: object
    mid_price: float
    inventory_q: float
    sigma: float
    kappa: float
    state_vector: object


@dataclass(frozen=True)
class QuotePair:
    bid: PostedQuote
    ask: PostedQuote
    model_quote: Quote
    risk_aversion: float
    skew: float


@dataclass
class InstrumentPosition:
    symbol: str
    q_btc: float = 0.0
    realized_pnl: float = 0.0
    mark_price: float = 0.0

    @property
    def notional_exposure(self) -> float:
        return self.q_btc * self.mark_price


@dataclass(frozen=True)
class AggregateExposure:
    total_notional_exposure: float
    positions: tuple[InstrumentPosition, ...]


@dataclass
class PaperExecutionState:
    positions: dict[str, InstrumentPosition] = field(default_factory=dict)

    def position_for(self, symbol: str) -> InstrumentPosition:
        if symbol not in self.positions:
            self.positions[symbol] = InstrumentPosition(symbol=symbol)
        return self.positions[symbol]

    def apply_fills(self, symbol: str, fills: Iterable[Fill], mark_price: float) -> InstrumentPosition:
        position = self.position_for(symbol)
        old_inventory = position.q_btc
        position.q_btc = apply_fills_to_inventory(position.q_btc, fills)
        position.mark_price = float(mark_price)
        for fill in fills:
            signed_size = fill.size_btc if fill.side == "ask" else -fill.size_btc
            position.realized_pnl += signed_size * fill.price
        if old_inventory != position.q_btc:
            self.positions[symbol] = position
        return position

    def mark(self, symbol: str, mark_price: float) -> InstrumentPosition:
        position = self.position_for(symbol)
        position.mark_price = float(mark_price)
        return position

    def aggregate_exposure(self) -> AggregateExposure:
        positions = tuple(self.positions.values())
        return AggregateExposure(
            total_notional_exposure=sum(position.notional_exposure for position in positions),
            positions=positions,
        )


def quote_from_policy(market_state: MarketState, action_provider: ActionProvider) -> QuotePair:
    """Use the same Step 3 quote formula for live and replayed backtest states."""
    action = action_provider.select_action(market_state.state_vector, epsilon=0.0)
    risk_aversion = float(action.risk_aversion)
    skew = float(action.skew)
    quote = avellaneda_stoikov_quote(
        mid_price=market_state.mid_price,
        inventory_q=market_state.inventory_q,
        sigma=market_state.sigma,
        kappa=market_state.kappa,
        risk_aversion=risk_aversion,
        skew=skew,
        timestamp=market_state.timestamp,
    )
    timestamp = to_utc_timestamp(market_state.timestamp)
    return QuotePair(
        bid=PostedQuote("bid", quote.bid, quote.size_btc, timestamp),
        ask=PostedQuote("ask", quote.ask, quote.size_btc, timestamp),
        model_quote=quote,
        risk_aversion=risk_aversion,
        skew=skew,
    )


def recorded_live_quotes_match_backtest(
    states: Iterable[MarketState],
    live_quote_fn: Callable[[MarketState], QuotePair],
    backtest_quote_fn: Callable[[MarketState], QuotePair],
) -> bool:
    """Acceptance helper: recorded live states must reproduce identical backtest quotes."""
    for state in states:
        live_quote = live_quote_fn(state)
        backtest_quote = backtest_quote_fn(state)
        if live_quote.bid != backtest_quote.bid or live_quote.ask != backtest_quote.ask:
            return False
    return True


def apply_book_message(book: OrderBookReplay, message: object) -> None:
    """Route live messages through the same snapshot/delta book transformation as replay."""
    if isinstance(message, dict) and message.get("type") == "snapshot":
        book.apply_snapshot(message["timestamp"], message["bids"], message["asks"])
        return
    if isinstance(message, dict) and message.get("type") == "delta":
        book.apply_deltas(message["deltas"])
        return
    raise ValueError("unsupported book message")


class BybitWebSocketConfig:
    url = "wss://stream.bybit.com/v5/public/linear"
    orderbook_topic = f"orderbook.200.{DEFAULT_SETTINGS.instrument.symbol}"
    trade_topic = f"publicTrade.{DEFAULT_SETTINGS.instrument.symbol}"


def subscription_message() -> dict[str, object]:
    return {
        "op": "subscribe",
        "args": [BybitWebSocketConfig.orderbook_topic, BybitWebSocketConfig.trade_topic],
    }
