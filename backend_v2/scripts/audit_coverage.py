import argparse
import asyncio

from app.database import SessionFactory, engine
from app.market_data.backfill import RESOLUTION_SECONDS
from app.market_data.coverage import get_coverage_report


async def main() -> None:
    parser = argparse.ArgumentParser(description="Report stored candle coverage")
    parser.add_argument("--symbols", nargs="+", default=["BTCUSD", "ETHUSD"])
    parser.add_argument(
        "--resolutions",
        nargs="+",
        default=["5m", "15m", "1h"],
        choices=sorted(RESOLUTION_SECONDS),
    )
    args = parser.parse_args()
    print("symbol,resolution,first,last,count,expected,missing")
    for symbol in args.symbols:
        for resolution in args.resolutions:
            report = await get_coverage_report(
                SessionFactory,
                venue="delta",
                symbol=symbol,
                resolution=resolution,
            )
            print(
                f"{symbol},{resolution},{report.first_candle or ''},"
                f"{report.last_candle or ''},{report.count},"
                f"{report.expected_count},{report.missing_count}"
            )
    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
