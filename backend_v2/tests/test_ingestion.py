from datetime import UTC, datetime, timedelta

import pytest

import app.market_data.ingestion as ingestion
from app.market_data.ingestion import backfill_recent_completed_hourly_candles


@pytest.mark.asyncio
async def test_recent_hourly_backfill_targets_only_completed_candles(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    async def fake_backfill(_client, _session_factory, **kwargs) -> int:
        _ = (_client, _session_factory)
        calls.append(kwargs)
        return 2

    monkeypatch.setattr(ingestion, "run_audited_backfill", fake_backfill)
    now = datetime(2026, 11, 13, 17, 42, tzinfo=UTC)

    count = await backfill_recent_completed_hourly_candles(
        object(), object(), now=now
    )

    assert count == 4
    assert [call["symbol"] for call in calls] == ["BTCUSD", "ETHUSD"]
    assert all(call["resolution"] == "1h" for call in calls)
    assert all(call["end"] == datetime(2026, 11, 13, 17, tzinfo=UTC) for call in calls)
    assert all(
        call["start"] == datetime(2026, 11, 5, 17, tzinfo=UTC) for call in calls
    )
    assert all(
        call["end"] == now.replace(minute=0, second=0, microsecond=0)
        for call in calls
    )
    assert all(call["start"] == call["end"] - timedelta(days=8) for call in calls)


@pytest.mark.asyncio
async def test_recent_hourly_backfill_rejects_naive_time() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        await backfill_recent_completed_hourly_candles(
            object(),
            object(),
            now=datetime(2026, 11, 13, 17),
        )
