import uuid
from datetime import UTC, datetime, timedelta

from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.market_data.backfill import backfill_candles
from app.market_data.delta_client import DeltaRestClient
from app.models import IngestRun

RECENT_HOURLY_CANDLE_SYMBOLS = ("BTCUSD", "ETHUSD")
RECENT_HOURLY_CANDLE_LOOKBACK = timedelta(days=8)


async def run_audited_backfill(
    client: DeltaRestClient,
    session_factory: async_sessionmaker[AsyncSession],
    *,
    symbol: str,
    resolution: str,
    start: datetime,
    end: datetime,
    venue: str = "delta",
) -> int:
    run_id = uuid.uuid4()
    async with session_factory() as session:
        async with session.begin():
            session.add(
                IngestRun(
                    id=run_id,
                    venue=venue,
                    symbol=symbol,
                    resolution=resolution,
                    start_time=start,
                    end_time=end,
                    status="running",
                )
            )

    try:
        count = await backfill_candles(
            client,
            session_factory,
            symbol=symbol,
            resolution=resolution,
            start=start,
            end=end,
            venue=venue,
        )
    except Exception as error:
        async with session_factory() as session:
            async with session.begin():
                await session.execute(
                    update(IngestRun)
                    .where(IngestRun.id == run_id)
                    .values(
                        status="failed",
                        error=f"{type(error).__name__}: {error}",
                        finished_at=datetime.now(UTC),
                    )
                )
        raise

    async with session_factory() as session:
        async with session.begin():
            await session.execute(
                update(IngestRun)
                .where(IngestRun.id == run_id)
                .values(
                    status="completed",
                    record_count=count,
                    finished_at=datetime.now(UTC),
                )
            )
    return count


async def backfill_recent_completed_hourly_candles(
    client: DeltaRestClient,
    session_factory: async_sessionmaker[AsyncSession],
    *,
    now: datetime | None = None,
) -> int:
    timestamp = now or datetime.now(UTC)
    if timestamp.tzinfo is None:
        raise ValueError("Recent candle backfill time must be timezone-aware")
    current_hour = timestamp.astimezone(UTC).replace(
        minute=0, second=0, microsecond=0
    )
    end = current_hour
    start = end - RECENT_HOURLY_CANDLE_LOOKBACK

    total = 0
    for symbol in RECENT_HOURLY_CANDLE_SYMBOLS:
        total += await run_audited_backfill(
            client,
            session_factory,
            symbol=symbol,
            resolution="1h",
            start=start,
            end=end,
        )
    return total
