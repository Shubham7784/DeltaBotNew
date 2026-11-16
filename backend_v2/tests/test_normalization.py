from datetime import UTC, datetime
from decimal import Decimal

import pytest
from pydantic import ValidationError

from app.market_data.normalization import (
    epoch_to_datetime,
    normalize_candle,
    normalize_websocket_candle,
)


def test_epoch_units_normalize_to_utc() -> None:
    assert epoch_to_datetime(1_700_000_000, "seconds") == datetime(
        2023, 11, 14, 22, 13, 20, tzinfo=UTC
    )
    assert epoch_to_datetime(1_700_000_000_000_000, "microseconds") == datetime(
        2023, 11, 14, 22, 13, 20, tzinfo=UTC
    )


def test_normalize_candle_preserves_decimal_precision() -> None:
    candle = normalize_candle(
        {
            "time": 1_700_000_000,
            "open": "35000.12345678",
            "high": "35002",
            "low": "34999",
            "close": "35001.25",
            "volume": "0.001",
        }
    )

    assert candle.open == Decimal("35000.12345678")
    assert candle.volume == Decimal("0.001")
    assert candle.open_time.tzinfo == UTC


def test_invalid_ohlc_is_rejected() -> None:
    with pytest.raises(ValidationError, match="high"):
        normalize_candle(
            {
                "time": 1_700_000_000,
                "open": 10,
                "high": 8,
                "low": 9,
                "close": 10,
                "volume": 1,
            }
        )


def test_non_finite_ohlc_is_rejected() -> None:
    with pytest.raises(ValidationError, match="finite"):
        normalize_candle(
            {
                "time": 1_700_000_000,
                "open": "NaN",
                "high": 12,
                "low": 9,
                "close": 10,
                "volume": 1,
            }
        )


def test_normalize_websocket_candle_uses_candle_start_microseconds() -> None:
    candle = normalize_websocket_candle(
        {
            "cst": 1_700_000_000_000_000,
            "o": "35000.12345678",
            "h": "35002",
            "l": "34999",
            "c": "35001.25",
            "v": "0.001",
        }
    )

    assert candle.open_time == datetime(2023, 11, 14, 22, 13, 20, tzinfo=UTC)
    assert candle.open == Decimal("35000.12345678")
    assert candle.volume == Decimal("0.001")


def test_normalize_websocket_candle_rejects_missing_or_invalid_values() -> None:
    with pytest.raises(ValueError, match="missing the 'cst'"):
        normalize_websocket_candle({})
    with pytest.raises(ValueError, match="invalid decimal"):
        normalize_websocket_candle(
            {
                "cst": 1_700_000_000_000_000,
                "o": "invalid",
                "h": "35002",
                "l": "34999",
                "c": "35001",
                "v": "1",
            }
        )
