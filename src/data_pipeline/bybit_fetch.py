"""Bybit public-data acquisition for Tier 1 runtime CSVs."""

from __future__ import annotations

import csv
import gzip
import json
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

from data_pipeline import coinalyze_fetch
from data_pipeline.replay import to_utc_timestamp

TRADING_ARCHIVE_BASE = "https://public.bybit.com/trading"
BYBIT_API_BASE = "https://api.bybit.com"

FUNDING_PROVIDER_COINALYZE = "coinalyze"
FUNDING_PROVIDER_BYBIT = "bybit"


@dataclass(frozen=True)
class BybitFetchResult:
    symbol: str
    start: pd.Timestamp
    end: pd.Timestamp
    trades: int
    mid_prices: int
    book_snapshots: int
    funding_rates: int
    raw_files: tuple[Path, ...]


def fetch_tier1_bybit_data(
    *,
    symbol: str = "BTCUSDT",
    category: str = "linear",
    start: str = "2026-07-01T00:00:00Z",
    end: str = "2026-07-02T00:00:00Z",
    output_dir: str | Path = "data",
    cadence: str = "5s",
    funding_source_csv: str | Path | None = None,
    funding_provider: str = FUNDING_PROVIDER_COINALYZE,
    bybit_api_base: str = BYBIT_API_BASE,
) -> BybitFetchResult:
    """Fetch real Bybit public trades/funding and write the project runtime CSVs.

    Bybit's public trading archive supplies historical public trade prints. The runtime
    book snapshot file is built as a deterministic top-of-book replay frame from those
    prints so the simulator has positive queue depth at every 5-second decision time.

    Funding source selection:
    - `funding_source_csv` (explicit manual override) always wins when provided.
    - Otherwise `funding_provider="coinalyze"` (the default) pulls real, non-geo-blocked
      funding rates from Coinalyze, which is the documented Tier 1/Tier 2 source.
    - `funding_provider="bybit"` uses Bybit's own geo-blocked public V5 funding-history
      endpoint — kept as the explicit, documented Tier 3 source of record.
    """
    start_ts = to_utc_timestamp(start)
    end_ts = to_utc_timestamp(end)
    output = Path(output_dir)
    raw_dir = output / "raw"
    output.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    trade_frame, raw_files = _download_trade_archive(symbol, start_ts, end_ts, raw_dir)
    if trade_frame.empty:
        raise ValueError(f"no Bybit public trades found for {symbol} over {start_ts} to {end_ts}")

    funding_frame = (
        _load_funding_source_csv(funding_source_csv, start_ts, end_ts)
        if funding_source_csv is not None
        else _fetch_funding_by_provider(funding_provider, symbol, category, start_ts, end_ts, bybit_api_base)
    )
    if funding_frame.empty:
        raise ValueError(f"no funding rates found for {symbol} over {start_ts} to {end_ts}")

    timeline = pd.date_range(start_ts, end_ts, freq=cadence, inclusive="left")
    mid_frame = _build_mid_prices(trade_frame, timeline)
    book_frame = _build_book_snapshots(mid_frame)
    runtime_trades = _sample_runtime_trades(trade_frame, timeline, cadence)

    _write_mid_prices(mid_frame, output / "mid_prices.csv")
    _write_book_snapshots(book_frame, output / "book_snapshots.csv")
    _write_trades(runtime_trades, output / "trades.csv")
    _write_funding(funding_frame, output / "funding.csv")

    return BybitFetchResult(
        symbol=symbol,
        start=start_ts,
        end=end_ts,
        trades=len(runtime_trades),
        mid_prices=len(mid_frame),
        book_snapshots=len(book_frame),
        funding_rates=len(funding_frame),
        raw_files=tuple(raw_files),
    )


def _download_trade_archive(
    symbol: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    raw_dir: Path,
) -> tuple[pd.DataFrame, list[Path]]:
    frames: list[pd.DataFrame] = []
    raw_files: list[Path] = []
    for day in _days_covering(start, end):
        filename = f"{symbol}{day.isoformat()}.csv.gz"
        url = f"{TRADING_ARCHIVE_BASE}/{symbol}/{filename}"
        raw_path = raw_dir / filename
        if not raw_path.exists():
            raw_path.write_bytes(_http_get(url))
        raw_files.append(raw_path)
        frames.append(_parse_trade_archive(raw_path, start=start, end=end))

    frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if frame.empty:
        return frame, raw_files
    frame = frame[(frame["timestamp"] >= start) & (frame["timestamp"] < end)]
    frame = frame.drop_duplicates(["timestamp", "price", "size", "aggressor_side"]).sort_values("timestamp")
    return frame.reset_index(drop=True), raw_files


def _parse_trade_archive(path: Path, start: pd.Timestamp | None = None, end: pd.Timestamp | None = None) -> pd.DataFrame:
    start_ts = to_utc_timestamp(start) if start is not None else None
    end_ts = to_utc_timestamp(end) if end is not None else None
    parsed = []
    with gzip.open(path, "rt", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            timestamp = _first_present(row, ("timestamp", "time", "Time", "execTime", "T"))
            price = _first_present(row, ("price", "Price", "execPrice", "p"))
            size = _first_present(row, ("size", "Size", "qty", "execQty", "v"))
            side = _first_present(row, ("side", "Side", "S"))
            if timestamp is None or price is None or size is None or side is None:
                continue
            ts = _parse_archive_timestamp(timestamp)
            if start_ts is not None and ts < start_ts:
                continue
            if end_ts is not None and ts >= end_ts:
                continue
            parsed.append(
                {
                    "timestamp": ts,
                    "price": float(price),
                    "size": float(size),
                    "aggressor_side": _normalize_side(str(side)),
                }
            )
    return pd.DataFrame(parsed, columns=["timestamp", "price", "size", "aggressor_side"])


def _fetch_funding_by_provider(
    provider: str,
    symbol: str,
    category: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    bybit_api_base: str,
) -> pd.DataFrame:
    """Dispatch to the configured funding provider.

    Coinalyze is the default (Tier 1/Tier 2); Bybit's own V5 endpoint remains the
    explicit Tier 3 source of record.
    """
    if provider == FUNDING_PROVIDER_COINALYZE:
        hourly = coinalyze_fetch.fetch_funding_rate_history(start=start, end=end)
        return coinalyze_fetch.reduce_to_settlement_rows(hourly, start, end)
    if provider == FUNDING_PROVIDER_BYBIT:
        return _fetch_funding_history(symbol, category, start, end, api_base=bybit_api_base)
    raise ValueError(f"unsupported funding_provider {provider!r} (expected {FUNDING_PROVIDER_COINALYZE!r} or {FUNDING_PROVIDER_BYBIT!r})")


def _fetch_funding_history(
    symbol: str,
    category: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    api_base: str = BYBIT_API_BASE,
) -> pd.DataFrame:
    """Fetch no-auth Bybit funding history using the public V5 pagination pattern.

    Mirrors the working approach from jooray/funding-rate-tools: ask for a page ending
    at `endTime`, then walk backward by setting `endTime` to oldest_timestamp - 1.
    """
    rows = []
    end_time_ms = int(end.timestamp() * 1000)
    start_time_ms = int(start.timestamp() * 1000)

    while True:
        query = urllib.parse.urlencode(
            {
                "category": category,
                "symbol": symbol.upper(),
                "limit": 200,
                "endTime": end_time_ms,
            }
        )
        payload = json.loads(_http_get(f"{api_base.rstrip('/')}/v5/market/funding/history?{query}").decode("utf-8"))
        if payload.get("retCode") != 0:
            raise ValueError(f"Bybit funding-history request failed: {payload}")
        batch = payload.get("result", {}).get("list", [])
        if not batch:
            break

        oldest_ts = None
        for item in batch:
            timestamp_ms = int(item["fundingRateTimestamp"])
            oldest_ts = timestamp_ms if oldest_ts is None else min(oldest_ts, timestamp_ms)
            if start_time_ms < timestamp_ms <= int(end.timestamp() * 1000):
                rows.append({"timestamp": to_utc_timestamp(timestamp_ms), "rate": float(item["fundingRate"])})

        if oldest_ts is None or oldest_ts <= start_time_ms or len(batch) < 200:
            break
        end_time_ms = oldest_ts - 1
        time.sleep(0.2)

    return pd.DataFrame(rows, columns=["timestamp", "rate"]).drop_duplicates("timestamp", keep="last").sort_values("timestamp").reset_index(drop=True)


def _load_funding_source_csv(path: str | Path, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = {"timestamp", "rate"} - set(frame.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {', '.join(sorted(missing))}")
    frame = frame.copy()
    frame["timestamp"] = frame["timestamp"].map(to_utc_timestamp)
    frame["rate"] = frame["rate"].astype(float)
    frame = frame[(frame["timestamp"] > start) & (frame["timestamp"] <= end)]
    return frame[["timestamp", "rate"]].drop_duplicates("timestamp", keep="last").sort_values("timestamp").reset_index(drop=True)


def _build_mid_prices(trades: pd.DataFrame, timeline: pd.DatetimeIndex) -> pd.DataFrame:
    trade_prices = trades.drop_duplicates("timestamp", keep="last").set_index("timestamp")["price"].sort_index()
    mids = trade_prices.reindex(trade_prices.index.union(timeline)).sort_index().ffill().reindex(timeline)
    mids = mids.bfill()
    if mids.isna().any():
        raise ValueError("could not build continuous mid-price series from Bybit trades")
    return pd.DataFrame({"timestamp": timeline, "mid_price": mids.astype(float).to_numpy()})


def _build_book_snapshots(mid_prices: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in mid_prices.itertuples(index=False):
        mid = float(row.mid_price)
        bid = round(mid - 0.1, 1)
        ask = round(mid + 0.1, 1)
        rows.append(
            {
                "timestamp": row.timestamp,
                "bids": [[bid, 100.0], [round(bid - 5.0, 1), 100.0], [round(bid - 10.0, 1), 100.0]],
                "asks": [[ask, 100.0], [round(ask + 5.0, 1), 100.0], [round(ask + 10.0, 1), 100.0]],
            }
        )
    return pd.DataFrame(rows, columns=["timestamp", "bids", "asks"])


def _sample_runtime_trades(trades: pd.DataFrame, timeline: pd.DatetimeIndex, cadence: str) -> pd.DataFrame:
    if trades.empty:
        return trades
    frame = trades.copy()
    start = timeline[0]
    cadence_delta = pd.Timedelta(cadence)
    bucket = ((frame["timestamp"] - start) // cadence_delta).astype(int)
    frame = frame.assign(_bucket=bucket)
    frame = frame[(frame["_bucket"] >= 0) & (frame["_bucket"] < len(timeline))]
    sampled = (
        frame.sort_values("timestamp")
        .groupby(["_bucket", "aggressor_side"], as_index=False)
        .tail(1)
        .drop(columns=["_bucket"])
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    return sampled[["timestamp", "price", "size", "aggressor_side"]]


def _write_mid_prices(frame: pd.DataFrame, path: Path) -> None:
    out = frame.copy()
    out["timestamp"] = out["timestamp"].map(_format_ts)
    out.to_csv(path, index=False)


def _write_book_snapshots(frame: pd.DataFrame, path: Path) -> None:
    out = frame.copy()
    out["timestamp"] = out["timestamp"].map(_format_ts)
    out["bids"] = out["bids"].map(json.dumps)
    out["asks"] = out["asks"].map(json.dumps)
    out.to_csv(path, index=False)


def _write_trades(frame: pd.DataFrame, path: Path) -> None:
    out = frame.copy()
    out["timestamp"] = out["timestamp"].map(_format_ts)
    out.to_csv(path, index=False)


def _write_funding(frame: pd.DataFrame, path: Path) -> None:
    out = frame.copy()
    out["timestamp"] = out["timestamp"].map(_format_ts)
    out.to_csv(path, index=False)


def _days_covering(start: pd.Timestamp, end: pd.Timestamp) -> Iterable[date]:
    current = start.date()
    last = (end - pd.Timedelta(microseconds=1)).date()
    while current <= last:
        yield current
        current += timedelta(days=1)


def _http_get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "as-dqn-market-maker/1.0"})
    with urllib.request.urlopen(req, timeout=60) as response:
        return response.read()


def _first_present(row: dict[str, str], names: tuple[str, ...]) -> str | None:
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            return value
    return None


def _parse_archive_timestamp(value: str) -> pd.Timestamp:
    numeric = float(value)
    seconds = numeric / 1000.0 if numeric > 10_000_000_000 else numeric
    return pd.Timestamp(datetime.fromtimestamp(seconds, tz=timezone.utc))


def _normalize_side(value: str) -> str:
    side = value.strip().lower()
    if side in {"buy", "b"}:
        return "buy"
    if side in {"sell", "s"}:
        return "sell"
    raise ValueError(f"unsupported Bybit trade side {value!r}")


def _format_ts(value: object) -> str:
    return to_utc_timestamp(value).isoformat().replace("+00:00", "Z")
