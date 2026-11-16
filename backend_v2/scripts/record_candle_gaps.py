import argparse
import asyncio
import uuid
from datetime import UTC, datetime

from sqlalchemy.dialects.postgresql import insert

from app.database import SessionFactory, engine
from app.market_data.coverage import RESOLUTION_SECONDS, get_candle_gaps
from app.models import DataQualityEvent


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Record missing Delta candle intervals as data-quality events"
    )
    parser.add_argument("--symbols", nargs="+", default=["BTCUSD", "ETHUSD"])
    parser.add_argument(
        "--resolutions",
        nargs="+",
        default=["5m", "15m"],
        choices=sorted(RESOLUTION_SECONDS),
    )
    args = parser.parse_args()

    detected_at = datetime.now(UTC)
    values = []
    for symbol in args.symbols:
        for resolution in args.resolutions:
            gaps = await get_candle_gaps(
                SessionFactory,
                venue="delta",
                symbol=symbol,
                resolution=resolution,
            )
            for gap in gaps:
                event_id = uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    (
                        f"deltabot:candle-gap:delta:{symbol}:{resolution}:"
                        f"{gap.missing_start.isoformat()}:{gap.missing_end.isoformat()}"
                    ),
                )
                values.append(
                    {
                        "id": event_id,
                        "venue": "delta",
                        "symbol": symbol,
                        "event_type": "candle_gap",
                        "occurred_at": detected_at,
                        "details": {
                            "status": "gap_in_stored_history",
                            "resolution": resolution,
                            "missing_start": gap.missing_start.isoformat(),
                            "missing_end": gap.missing_end.isoformat(),
                            "missing_count": gap.missing_count,
                            "previous_available_candle": gap.previous_time.isoformat(),
                            "next_available_candle": gap.next_time.isoformat(),
                        },
                    }
                )

    if values:
        statement = insert(DataQualityEvent).values(values)
        statement = statement.on_conflict_do_update(
            index_elements=["id"],
            set_={
                "occurred_at": statement.excluded.occurred_at,
                "details": statement.excluded.details,
            },
        )
        async with SessionFactory() as session:
            async with session.begin():
                await session.execute(statement)

    print(f"Recorded {len(values)} Delta candle gaps.")
    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
