from datetime import UTC, datetime, timedelta

import pytest

from app.market_data.coverage import candle_gap_between


def test_candle_gap_reports_missing_bounds_and_count() -> None:
    previous = datetime(2024, 1, 1, tzinfo=UTC)
    following = previous + timedelta(minutes=20)

    gap = candle_gap_between(previous, following, "5m")

    assert gap is not None
    assert gap.missing_start == previous + timedelta(minutes=5)
    assert gap.missing_end == previous + timedelta(minutes=15)
    assert gap.missing_count == 3


def test_candle_gap_returns_none_for_adjacent_candles() -> None:
    previous = datetime(2024, 1, 1, tzinfo=UTC)
    following = previous + timedelta(minutes=15)

    assert candle_gap_between(previous, following, "15m") is None


def test_candle_gap_rejects_non_increasing_timestamps() -> None:
    timestamp = datetime(2024, 1, 1, tzinfo=UTC)
    with pytest.raises(ValueError, match="after previous"):
        candle_gap_between(timestamp, timestamp, "5m")
