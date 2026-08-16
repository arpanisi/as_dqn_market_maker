import json

import pandas as pd
import pytest

from data_pipeline.loaders import (
    load_book_snapshots_csv,
    load_funding_csv,
    load_mid_prices_csv,
    load_trades_csv,
    verify_data_range,
)


def test_csv_loaders_parse_runtime_inputs(tmp_path):
    mids = tmp_path / "mids.csv"
    mids.write_text("timestamp,mid_price\n2024-06-01T00:00:00Z,100.1\n", encoding="utf-8")
    trades = tmp_path / "trades.csv"
    trades.write_text("timestamp,price,size,aggressor_side\n2024-06-01T00:00:01Z,100.0,0.2,sell\n", encoding="utf-8")
    funding = tmp_path / "funding.csv"
    funding.write_text("timestamp,rate\n2024-06-01T08:00:00Z,0.0001\n", encoding="utf-8")
    books = tmp_path / "books.csv"
    books.write_text(
        "timestamp,bids,asks\n"
        f"2024-06-01T00:00:00Z,\"{json.dumps([[100.0, 1.0]])}\",\"{json.dumps([[100.2, 1.0]])}\"\n",
        encoding="utf-8",
    )

    assert load_mid_prices_csv(mids).iloc[0]["mid_price"] == 100.1
    assert load_trades_csv(trades)[0].aggressor_side == "sell"
    assert load_funding_csv(funding)[0].rate == pytest.approx(0.0001)
    assert load_book_snapshots_csv(books).iloc[0]["bids"] == [[100.0, 1.0]]


def test_verify_data_range_rejects_missing_required_coverage():
    index = pd.to_datetime(["2024-06-02T00:00:00Z", "2024-06-03T00:00:00Z"], utc=True)

    with pytest.raises(ValueError):
        verify_data_range(index, "2024-06-01T00:00:00Z", "2024-06-03T00:00:00Z")
