import json

import pandas as pd
import pytest

import data_pipeline.coinalyze_fetch as coinalyze_fetch
from data_pipeline.bybit_fetch import fetch_tier1_bybit_data
from data_pipeline.loaders import load_funding_csv


def _hourly_payload() -> list[dict]:
    hours = list(range(24))
    return [
        {
            "symbol": "BTCUSDT.6",
            "history": [
                {
                    "t": 1782864000 + 3600 * hour,
                    "o": rate,
                    "h": rate,
                    "l": rate,
                    "c": rate,
                }
                for hour, rate in zip(
                    hours,
                    [0.004036] * 8 + [0.009078] * 8 + [0.008186] * 8,
                )
            ]
            + [{"t": 1782950400, "o": 0.008594, "h": 0.008594, "l": 0.008594, "c": 0.008594}],
        }
    ]


def test_fetch_funding_rate_history_converts_percent_to_fraction(monkeypatch):
    def fake_http_get(url, api_key):
        assert "api.coinalyze.net/v1/funding-rate-history" in url
        return json.dumps(_hourly_payload()).encode("utf-8")

    monkeypatch.setattr(coinalyze_fetch, "_http_get", fake_http_get)

    frame = coinalyze_fetch.fetch_funding_rate_history(
        start="2026-07-01T00:00:00Z",
        end="2026-07-02T00:00:00Z",
        api_key="test-key",
    )

    assert len(frame) == 25
    assert frame.iloc[0]["rate"] == pytest.approx(0.00004036)
    assert frame.iloc[8]["rate"] == pytest.approx(0.00009078)
    assert frame.iloc[24]["rate"] == pytest.approx(0.00008594)


def test_reduce_to_settlement_rows_collapses_hourly_to_one_per_boundary():
    hourly = pd.DataFrame(
        {
            "timestamp": [
                pd.Timestamp("2026-07-01T00:00:00Z"),
                pd.Timestamp("2026-07-01T01:00:00Z"),
                pd.Timestamp("2026-07-01T08:00:00Z"),
                pd.Timestamp("2026-07-01T09:00:00Z"),
                pd.Timestamp("2026-07-01T16:00:00Z"),
                pd.Timestamp("2026-07-02T00:00:00Z"),
            ],
            "rate": [0.00004036, 0.00004036, 0.00009078, 0.00009078, 0.00008186, 0.00008594],
        }
    )

    reduced = coinalyze_fetch.reduce_to_settlement_rows(
        hourly,
        "2026-07-01T00:00:00Z",
        "2026-07-02T00:00:00Z",
    )

    assert reduced["timestamp"].tolist() == [
        pd.Timestamp("2026-07-01T08:00:00Z"),
        pd.Timestamp("2026-07-01T16:00:00Z"),
        pd.Timestamp("2026-07-02T00:00:00Z"),
    ]
    assert reduced["rate"].tolist() == [0.00009078, 0.00008186, 0.00008594]


def test_load_coinalyze_api_key_from_env_file(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("COINALYZE_API_KEY=abc-123\n", encoding="utf-8")

    assert coinalyze_fetch.load_coinalyze_api_key(env={}, env_file=env_file) == "abc-123"


def test_load_coinalyze_api_key_missing_raises(tmp_path):
    with pytest.raises(ValueError, match="COINALYZE_API_KEY is not set"):
        coinalyze_fetch.load_coinalyze_api_key(env={}, env_file=tmp_path / "missing.env")


def test_fetch_tier1_bybit_data_defaults_to_coinalyze_funding(monkeypatch, tmp_path):
    trades = pd.DataFrame(
        {
            "timestamp": [
                pd.Timestamp("2026-07-01T00:00:00.100Z"),
                pd.Timestamp("2026-07-01T00:00:06.000Z"),
            ],
            "price": [100000.0, 100000.2],
            "size": [0.01, 0.03],
            "aggressor_side": ["buy", "buy"],
        }
    )

    def fake_download(symbol, start, end, raw_dir):
        return trades, []

    def fake_http_get(url, api_key=None):
        assert "api.coinalyze.net/v1/funding-rate-history" in url
        return json.dumps(_hourly_payload()).encode("utf-8")

    monkeypatch.setattr("data_pipeline.bybit_fetch._download_trade_archive", fake_download)
    monkeypatch.setattr(coinalyze_fetch, "_http_get", fake_http_get)
    monkeypatch.setenv("COINALYZE_API_KEY", "test-key")

    result = fetch_tier1_bybit_data(
        start="2026-07-01T00:00:00Z",
        end="2026-07-02T00:00:00Z",
        output_dir=tmp_path,
        cadence="8h",
    )

    assert result.funding_rates == 3
    funding = load_funding_csv(tmp_path / "funding.csv")
    assert [ts.isoformat().replace("+00:00", "Z") for ts in [item.timestamp for item in funding]] == [
        "2026-07-01T08:00:00Z",
        "2026-07-01T16:00:00Z",
        "2026-07-02T00:00:00Z",
    ]
    assert [item.rate for item in funding] == [0.00009078, 0.00008186, 0.00008594]
