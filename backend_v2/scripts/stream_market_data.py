import asyncio
import logging
import signal

import httpx
from sqlalchemy.exc import SQLAlchemyError

from app.core.config import get_settings
from app.database import SessionFactory, engine
from app.market_data.delta_client import DeltaApiError, DeltaRestClient
from app.market_data.ingestion import backfill_recent_completed_hourly_candles
from app.market_data.websocket import (
    DeltaWebSocketCollector,
    persist_market_event,
)

logging.basicConfig(level=get_settings().log_level)
logger = logging.getLogger(__name__)
RECENT_CANDLE_BACKFILL_INTERVAL_SECONDS = 900


async def maintain_recent_hourly_candles(
    client: DeltaRestClient,
    stop_event: asyncio.Event,
) -> None:
    while not stop_event.is_set():
        try:
            count = await backfill_recent_completed_hourly_candles(
                client, SessionFactory
            )
            logger.info("Recent hourly candle backfill updated %d records", count)
        except (DeltaApiError, httpx.HTTPError, SQLAlchemyError, ValueError):
            logger.exception("Recent hourly candle backfill failed")

        try:
            await asyncio.wait_for(
                stop_event.wait(),
                timeout=RECENT_CANDLE_BACKFILL_INTERVAL_SECONDS,
            )
        except asyncio.TimeoutError:
            pass


async def main() -> None:
    settings = get_settings()
    client = DeltaRestClient(settings)
    collector = DeltaWebSocketCollector(
        settings,
        ["BTCUSD", "ETHUSD"],
        on_event=lambda event: persist_market_event(SessionFactory, event),
    )
    stop_event = asyncio.Event()

    def stop() -> None:
        collector.stop()
        stop_event.set()

    loop = asyncio.get_running_loop()
    for stop_signal in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(stop_signal, stop)
        except NotImplementedError:
            signal.signal(
                stop_signal,
                lambda *_: loop.call_soon_threadsafe(stop),
            )
    backfill_task = asyncio.create_task(
        maintain_recent_hourly_candles(client, stop_event),
        name="recent-hourly-candle-backfill",
    )
    try:
        await collector.run_forever()
    finally:
        stop()
        try:
            await backfill_task
        finally:
            await client.close()
            await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
