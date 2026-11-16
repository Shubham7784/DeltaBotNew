from collections.abc import AsyncIterator

from redis.asyncio import Redis, from_url
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.config import get_settings

settings = get_settings()
engine = create_async_engine(
    settings.database_url.get_secret_value(),
    pool_pre_ping=True,
)
SessionFactory = async_sessionmaker(engine, expire_on_commit=False)
redis_client: Redis = from_url(str(settings.redis_url), decode_responses=True)


async def get_db_session() -> AsyncIterator[AsyncSession]:
    async with SessionFactory() as session:
        yield session


async def close_database_resources() -> None:
    await redis_client.aclose()
    await engine.dispose()
