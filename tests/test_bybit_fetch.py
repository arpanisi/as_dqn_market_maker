import gzip
import json

import pandas as pd

import data_pipeline.bybit_fetch as bybit_fetch
from data_pipeline.bybit_fetch import FUNDING_PROVIDER_BYBIT
from data_pipeline.loaders import (
    load_book_snapshots_csv,
    load_funding_csv,
    load_mid_prices_csv,
    load_trades_csv,
    verify_data_range,
)


def test_fetch_tier1_bybit_data_writes_runtime_csvs(monkeypatch, tmp_path):
    archive = gzip.compress(
        (
            "timestamp,symbol,side,size,price\n"
            "1782864000.100,BTCUSDT,Buy,0.010,100000.0\n"
            "1782864001.200,BTCUSDT,Sell,0.020,99999.8\n"
            "1782864006.000,BTCUSDT,Buy,0.030,100000.2\n"
            "1782864010.000,BTCUSDT,Sell,0.040,99999.9\n"
        ).encode("utf-8")
    )
    funding_payload = {
        "retCode": 0,
        "result": {
            "list": [
                {"fundingRateTimestamp": "1782950400000", "fundingRate": "0.0003"},
                {"fundingRateTimestamp": "1782921600000", "fundingRate": "-0.0001"},
                {"fundingRateTimestamp": "1782892800000", "fundingRate": "0.0002"},
                {"fundingRateTimestamp": "1782864000000", "fundingRate": "0.0001"},
            ]
        },
    }
    seen_funding_urls = []

    def fake_http_get(url):
        if "/v5/market/funding/history" in url:
            seen_funding_urls.append(url)
            return json.dumps(funding_payload).encode("utf-8")
        return archive

    monkeypatch.setattr(bybit_fetch, "_http_get", fake_http_get)

    result = bybit_fetch.fetch_tier1_bybit_data(
        start="2026-07-01T00:00:00Z",
        end="2026-07-02T00:00:00Z",
        output_dir=tmp_path,
        cadence="8h",
        funding_provider=FUNDING_PROVIDER_BYBIT,
    )

    mids = load_mid_prices_csv(tmp_path / "mid_prices.csv")
    books = load_book_snapshots_csv(tmp_path / "book_snapshots.csv")
    trades = load_trades_csv(tmp_path / "trades.csv")
    funding = load_funding_csv(tmp_path / "funding.csv")

    assert result.trades == 2
    assert result.mid_prices == 3
    assert result.book_snapshots == 3
    assert result.funding_rates == 3
    assert seen_funding_urls
    assert all("endTime=" in url for url in seen_funding_urls)
    verify_data_range(mids.index, "2026-07-01T00:00:00Z", "2026-07-01T16:00:00Z")
    assert len(trades) == 2
    assert [rate.rate for rate in funding] == [0.0002, -0.0001, 0.0003]
    assert all(level[1] > 0 for levels in books["bids"] for level in levels)
    assert all(level[1] > 0 for levels in books["asks"] for level in levels)


def test_parse_trade_archive_accepts_millisecond_timestamps(tmp_path):
    raw_path = tmp_path / "BTCUSDT2026-07-01.csv.gz"
    raw_path.write_bytes(
        gzip.compress(
            "timestamp,side,size,price\n1782864000150,Sell,0.125,100001.5\n".encode("utf-8")
        )
    )

    frame = bybit_fetch._parse_trade_archive(raw_path)

    assert frame.iloc[0]["timestamp"] == pd.Timestamp("2026-07-01T00:00:00.150Z")
    assert frame.iloc[0]["aggressor_side"] == "sell"
    assert frame.iloc[0]["price"] == 100001.5


def test_load_funding_source_csv_filters_to_tier_window(tmp_path):
    funding_csv = tmp_path / "funding.csv"
    funding_csv.write_text(
        "timestamp,rate\n"
        "2026-07-01T00:00:00Z,0.1\n"
        "2026-07-01T08:00:00Z,0.2\n"
        "2026-07-01T16:00:00Z,-0.1\n"
        "2026-07-02T00:00:00Z,0.3\n"
        "2026-07-02T08:00:00Z,0.4\n",
        encoding="utf-8",
    )

    frame = bybit_fetch._load_funding_source_csv(
        funding_csv,
        pd.Timestamp("2026-07-01T00:00:00Z"),
        pd.Timestamp("2026-07-02T00:00:00Z"),
    )

    assert frame["rate"].tolist() == [0.2, -0.1, 0.3]
