import asyncio

from app.core.config import get_settings
from app.database import SessionFactory, engine
from app.market_data.catalog import refresh_supported_products
from app.market_data.delta_client import DeltaRestClient


async def main() -> None:
    client = DeltaRestClient(get_settings())
    try:
        products = await refresh_supported_products(client, SessionFactory)
        for symbol, product_id in products:
            print(f"{symbol},{product_id}")
    finally:
        await client.close()
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
