from dataclasses import dataclass
from datetime import datetime, timedelta

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.market_data.backfill import RESOLUTION_SECONDS
from app.models import Candle


@dataclass(frozen=True)
class CoverageReport:
    venue: str
    symbol: str
    resolution: str
    first_candle: datetime | None
    last_candle: datetime | None
    count: int
    expected_count: int
    missing_count: int


@dataclass(frozen=True)
class CandleGap:
    previous_time: datetime
    next_time: datetime
    missing_start: datetime
    missing_end: datetime
    missing_count: int


def candle_gap_between(
    previous_time: datetime,
    next_time: datetime,
    resolution: str,
) -> CandleGap | None:
    if resolution not in RESOLUTION_SECONDS:
        raise ValueError(f"Unsupported candle resolution: {resolution}")
    if next_time <= previous_time:
        raise ValueError("Next candle time must be after previous candle time")

    interval_seconds = RESOLUTION_SECONDS[resolution]
    missing_count = int((next_time - previous_time).total_seconds() // interval_seconds) - 1
    if missing_count <= 0:
        return None

    interval = timedelta(seconds=interval_seconds)
    return CandleGap(
        previous_time=previous_time,
        next_time=next_time,
        missing_start=previous_time + interval,
        missing_end=next_time - interval,
        missing_count=missing_count,
    )


async def get_coverage_report(
    session_factory: async_sessionmaker[AsyncSession],
    *,
    venue: str,
    symbol: str,
    resolution: str,
) -> CoverageReport:
    if resolution not in RESOLUTION_SECONDS:
        raise ValueError(f"Unsupported candle resolution: {resolution}")

    async with session_factory() as session:
        result = await session.execute(
            select(
                func.min(Candle.open_time),
                func.max(Candle.open_time),
                func.count(),
            ).where(
                Candle.venue == venue,
                Candle.symbol == symbol,
                Candle.resolution == resolution,
            )
        )
    first, last, count = result.one()
    expected_count = (
        int((last - first).total_seconds() // RESOLUTION_SECONDS[resolution]) + 1
        if first is not None and last is not None
        else 0
    )
    return CoverageReport(
        venue=venue,
        symbol=symbol,
        resolution=resolution,
        first_candle=first,
        last_candle=last,
        count=count,
        expected_count=expected_count,
        missing_count=max(expected_count - count, 0),
    )


async def get_candle_gaps(
    session_factory: async_sessionmaker[AsyncSession],
    *,
    venue: str,
    symbol: str,
    resolution: str,
) -> list[CandleGap]:
    if resolution not in RESOLUTION_SECONDS:
        raise ValueError(f"Unsupported candle resolution: {resolution}")

    previous_time = func.lag(Candle.open_time).over(
        order_by=Candle.open_time
    ).label("previous_time")
    candle_times = (
        select(Candle.open_time.label("next_time"), previous_time)
        .where(
            Candle.venue == venue,
            Candle.symbol == symbol,
            Candle.resolution == resolution,
        )
        .subquery()
    )
    interval = timedelta(seconds=RESOLUTION_SECONDS[resolution])
    statement = (
        select(candle_times.c.previous_time, candle_times.c.next_time)
        .where(
            candle_times.c.previous_time.is_not(None),
            candle_times.c.next_time
            > candle_times.c.previous_time + interval,
        )
        .order_by(candle_times.c.next_time)
    )
    async with session_factory() as session:
        result = await session.execute(statement)

    gaps = []
    for previous, following in result:
        gap = candle_gap_between(previous, following, resolution)
        if gap is not None:
            gaps.append(gap)
    return gaps
