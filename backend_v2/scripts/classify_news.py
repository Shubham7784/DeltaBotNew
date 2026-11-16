import argparse
import asyncio
import logging

from app.core.config import get_settings
from app.database import SessionFactory, engine
from app.news.sentiment import classify_news_forever


async def main() -> None:
    parser = argparse.ArgumentParser(description="Classify unscored news headlines")
    parser.add_argument(
        "--once",
        action="store_true",
        help="Classify one batch instead of polling continuously",
    )
    args = parser.parse_args()
    settings = get_settings()
    logging.basicConfig(level=settings.log_level)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    try:
        await classify_news_forever(
            settings,
            SessionFactory,
            once=args.once,
        )
    finally:
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
