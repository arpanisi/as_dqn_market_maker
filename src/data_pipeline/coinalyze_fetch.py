"""Coinalyze funding-rate acquisition — the default Tier 1/Tier 2 funding source.

Coinalyze is a free, key-based (not geo-gated) third-party aggregator. Its
`/v1/funding-rate-history` endpoint returns funding rate in *percent* units, so
every returned value is divided by 100 before being treated as the fractional
rate `f` used elsewhere in this project (e.g. against `f_min = 0.0001`). The
series is piecewise-constant and changes exactly on the 00:00/08:00/16:00 UTC
8-hour settlement boundaries, so the raw hourly rows are collapsed to one row
per genuine settlement boundary, matching the shape `data/funding.csv` uses.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Mapping

import pandas as pd

from data_pipeline.replay import to_utc_timestamp

COINALYZE_API_BASE = "https://api.coinalyze.net/v1"
COINALYZE_FUNDING_SYMBOL = "BTCUSDT.6"
COINALYZE_INTERVAL = "1hour"
SETTLEMENT_FREQ = "8h"
MAX_HTTP_RETRIES = 3

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_coinalyze_api_key(
    env: Mapping[str, str] | None = None,
    env_file: str | Path | None = None,
) -> str:
    """Return COINALYZE_API_KEY from the process environment or the project .env file."""
    values = dict(os.environ if env is None else env)
    if "COINALYZE_API_KEY" not in values or not values.get("COINALYZE_API_KEY", "").strip():
        values.update(_load_env_file(Path(env_file) if env_file is not None else PROJECT_ROOT / ".env"))
    key = values.get("COINALYZE_API_KEY", "").strip()
    if not key:
        raise ValueError("COINALYZE_API_KEY is not set (expected in .env or the process environment)")
    return key


def _load_env_file(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    if not path.exists():
        return result
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        result[key.strip()] = value.strip().strip('"').strip("'")
    return result


def fetch_funding_rate_history(
    *,
    symbol: str = COINALYZE_FUNDING_SYMBOL,
    interval: str = COINALYZE_INTERVAL,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    api_key: str | None = None,
) -> pd.DataFrame:
    """Fetch Coinalyze hourly funding candles and convert percent to fraction.

    Returns a DataFrame with columns `timestamp` (UTC) and `rate` (fraction).
    """
    key = api_key or load_coinalyze_api_key()
    start_ts = to_utc_timestamp(start)
    end_ts = to_utc_timestamp(end)
    query = urllib.parse.urlencode(
        {
            "symbols": symbol,
            "interval": interval,
            "from": int(start_ts.timestamp()),
            "to": int(end_ts.timestamp()),
        }
    )
    url = f"{COINALYZE_API_BASE}/funding-rate-history?{query}"
    payload = json.loads(_http_get(url, key).decode("utf-8"))
    rows: list[dict[str, object]] = []
    if not isinstance(payload, list):
        raise ValueError(f"Coinalyze funding-rate-history returned an unexpected payload: {payload!r}")
    for entry in payload:
        for candle in entry.get("history", []):
            rows.append(
                {
                    "timestamp": to_utc_timestamp(int(candle["t"]) * 1000),
                    "rate": float(candle["c"]) / 100.0,
                }
            )
    if not rows:
        raise ValueError("Coinalyze funding-rate-history returned no candles")
    frame = pd.DataFrame(rows, columns=["timestamp", "rate"])
    return frame.drop_duplicates("timestamp", keep="last").sort_values("timestamp").reset_index(drop=True)


def reduce_to_settlement_rows(
    hourly_frame: pd.DataFrame,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
) -> pd.DataFrame:
    """Collapse the piecewise-constant hourly series to one row per real 8-hour settlement boundary.

    Funding settles at 00:00/08:00/16:00 UTC. The rate in effect at each boundary is the last
    hourly value at or before that boundary; consecutive boundaries inside the same settlement
    period naturally carry the same rate. Boundaries are generated within `(start, end]`,
    matching the window convention used by `data/funding.csv`.
    """
    start_ts = to_utc_timestamp(start)
    end_ts = to_utc_timestamp(end)
    series = hourly_frame.sort_values("timestamp")
    rows: list[dict[str, object]] = []
    for boundary in _settlement_boundaries(start_ts, end_ts):
        prior = series[series["timestamp"] <= boundary]
        if prior.empty:
            raise ValueError(f"no Coinalyze funding rate at or before settlement boundary {boundary}")
        rows.append({"timestamp": boundary, "rate": float(prior.iloc[-1]["rate"])})
    return pd.DataFrame(rows, columns=["timestamp", "rate"])


def _settlement_boundaries(start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    start_utc = to_utc_timestamp(start)
    end_utc = to_utc_timestamp(end)
    boundaries: list[pd.Timestamp] = []
    cursor = start_utc.replace(hour=0, minute=0, second=0, microsecond=0)
    while cursor <= end_utc:
        if cursor > start_utc:
            boundaries.append(cursor)
        cursor += pd.Timedelta(hours=8)
    return boundaries


def _http_get(url: str, api_key: str) -> bytes:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "as-dqn-market-maker/1.0", "api_key": api_key},
    )
    for attempt in range(MAX_HTTP_RETRIES + 1):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            if exc.code == 429 and attempt < MAX_HTTP_RETRIES:
                retry_after = exc.headers.get("Retry-After")
                time.sleep(float(retry_after) if retry_after else 60.0)
                continue
            raise
