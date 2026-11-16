import asyncio
import time

import pytest

from app.api.routes import system
from app.core.config import Settings


class FakeRedis:
    def __init__(self, values: dict[str, str]) -> None:
        self._values = values

    async def get(self, key: str) -> str | None:
        return self._values.get(key)


class FreshOnReadRedis:
    async def get(self, _key: str) -> str:
        await asyncio.sleep(0.01)
        return str(time.time())


@pytest.mark.asyncio
async def test_market_stream_requires_fresh_data_for_both_assets(monkeypatch) -> None:
    monkeypatch.setattr(system, "redis_client", FakeRedis({}))
    status = await system._check_market_stream(Settings())
    assert status.status == "unavailable"
    assert "BTCUSD" in (status.detail or "")

    now = str(time.time())
    monkeypatch.setattr(
        system,
        "redis_client",
        FakeRedis(
            {
                "market-data:last-received:BTCUSD": now,
                "market-data:last-received:ETHUSD": now,
            }
        ),
    )
    status = await system._check_market_stream(Settings())
    assert status.status == "available"


@pytest.mark.asyncio
async def test_market_stream_allows_events_received_during_readiness_check(
    monkeypatch,
) -> None:
    monkeypatch.setattr(system, "redis_client", FreshOnReadRedis())

    status = await system._check_market_stream(Settings())

    assert status.status == "available"


@pytest.mark.asyncio
async def test_market_stream_rejects_stale_and_invalid_timestamps(monkeypatch) -> None:
    monkeypatch.setattr(
        system,
        "redis_client",
        FakeRedis(
            {
                "market-data:last-received:BTCUSD": "1",
                "market-data:last-received:ETHUSD": "1",
            }
        ),
    )
    status = await system._check_market_stream(Settings())
    assert status.status == "unavailable"
    assert "stale" in (status.detail or "")

    monkeypatch.setattr(
        system,
        "redis_client",
        FakeRedis(
            {
                "market-data:last-received:BTCUSD": "NaN",
                "market-data:last-received:ETHUSD": "1",
            }
        ),
    )
    status = await system._check_market_stream(Settings())
    assert status.status == "unavailable"
    assert "Invalid" in (status.detail or "")
