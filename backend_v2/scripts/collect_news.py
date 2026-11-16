import argparse
import asyncio
import logging

from app.core.config import get_settings
from app.database import SessionFactory, engine
from app.news.collector import poll_news_feeds


async def main() -> None:
    parser = argparse.ArgumentParser(description="Collect public RSS/Atom news feeds")
    parser.add_argument(
        "--once",
        action="store_true",
        help="Fetch each configured feed once instead of polling continuously",
    )
    args = parser.parse_args()
    settings = get_settings()
    logging.basicConfig(level=settings.log_level)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    try:
        await poll_news_feeds(settings, SessionFactory, once=args.once)
    finally:
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
