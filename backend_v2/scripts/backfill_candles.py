import argparse
import asyncio
from datetime import datetime

from app.core.config import get_settings
from app.database import SessionFactory, engine
from app.market_data.delta_client import DeltaRestClient
from app.market_data.ingestion import run_audited_backfill


def parse_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise argparse.ArgumentTypeError("Timestamp must include a UTC offset or Z")
    return parsed


async def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill Delta Exchange candles")
    parser.add_argument("--symbol", required=True, choices=["BTCUSD", "ETHUSD"])
    parser.add_argument("--resolution", required=True)
    parser.add_argument("--start", required=True, type=parse_datetime)
    parser.add_argument("--end", required=True, type=parse_datetime)
    args = parser.parse_args()

    client = DeltaRestClient(get_settings())
    try:
        count = await run_audited_backfill(
            client,
            SessionFactory,
            symbol=args.symbol,
            resolution=args.resolution,
            start=args.start,
            end=args.end,
        )
        print(f"stored_or_updated_candles={count}")
    finally:
        await client.close()
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
