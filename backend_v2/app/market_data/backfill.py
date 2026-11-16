from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.dialects.postgresql.dml import Insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.models import Candle
from app.market_data.delta_client import DeltaRestClient
from app.market_data.normalization import normalize_candle

RESOLUTION_SECONDS: dict[str, int] = {
    "1m": 60,
    "3m": 180,
    "5m": 300,
    "15m": 900,
    "30m": 1_800,
    "1h": 3_600,
    "2h": 7_200,
    "4h": 14_400,
    "6h": 21_600,
    "1d": 86_400,
    "1w": 604_800,
}
MAX_CANDLES_PER_REQUEST = 2_000


def build_candle_upsert_statement(values: list[dict[str, Any]]) -> Insert:
    if not values:
        raise ValueError("At least one candle is required for an upsert")
    statement = insert(Candle).values(values)
    return statement.on_conflict_do_update(
        index_elements=["venue", "symbol", "resolution", "open_time"],
        set_={
            "open": statement.excluded.open,
            "high": statement.excluded.high,
            "low": statement.excluded.low,
            "close": statement.excluded.close,
            "volume": statement.excluded.volume,
            "source": statement.excluded.source,
        },
    )


async def upsert_candles(
    session_factory: async_sessionmaker[AsyncSession],
    *,
    venue: str,
    symbol: str,
    resolution: str,
    source: str,
    candles: list[dict[str, Any]],
) -> int:
    normalized = [normalize_candle(candle) for candle in candles]
    if not normalized:
        return 0

    values = [
        {
            "venue": venue,
            "symbol": symbol,
            "resolution": resolution,
            "open_time": candle.open_time,
            "open": candle.open,
            "high": candle.high,
            "low": candle.low,
            "close": candle.close,
            "volume": candle.volume,
            "source": source,
        }
        for candle in normalized
    ]
    statement = build_candle_upsert_statement(values)
    async with session_factory() as session:
        async with session.begin():
            await session.execute(statement)
    return len(values)


async def backfill_candles(
    client: DeltaRestClient,
    session_factory: async_sessionmaker[AsyncSession],
    *,
    symbol: str,
    resolution: str,
    start: datetime,
    end: datetime,
    venue: str = "delta",
    source: str = "delta_rest",
    progress: Callable[[int], None] | None = None,
) -> int:
    if resolution not in RESOLUTION_SECONDS:
        raise ValueError(f"Unsupported candle resolution: {resolution}")
    if start.tzinfo is None or end.tzinfo is None:
        raise ValueError("Backfill bounds must be timezone-aware")
    start_utc, end_utc = start.astimezone(UTC), end.astimezone(UTC)
    if end_utc <= start_utc:
        raise ValueError("Backfill end must be after start")

    chunk_duration = timedelta(
        seconds=RESOLUTION_SECONDS[resolution] * (MAX_CANDLES_PER_REQUEST - 1)
    )
    cursor = start_utc
    total = 0
    while cursor < end_utc:
        chunk_end = min(cursor + chunk_duration, end_utc)
        rows = await client.get_candles(
            symbol=symbol,
            resolution=resolution,
            start=cursor,
            end=chunk_end,
        )
        count = await upsert_candles(
            session_factory,
            venue=venue,
            symbol=symbol,
            resolution=resolution,
            source=source,
            candles=rows,
        )
        total += count
        if progress is not None:
            progress(count)
        cursor = chunk_end
    return total
