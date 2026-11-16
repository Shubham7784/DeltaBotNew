from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.market_data.delta_client import DeltaRestClient
from app.models import Asset, MarketProduct

SUPPORTED_ASSETS = {
    "BTC": "Bitcoin",
    "ETH": "Ethereum",
}


def _asset_symbol(product: dict[str, Any], key: str) -> str:
    asset = product.get(key)
    if not isinstance(asset, dict) or not isinstance(asset.get("symbol"), str):
        raise ValueError(f"Delta product has no {key}.symbol")
    return asset["symbol"]


async def refresh_supported_products(
    client: DeltaRestClient,
    session_factory: async_sessionmaker[AsyncSession],
    *,
    venue: str = "delta",
) -> list[tuple[str, int]]:
    products = await client.get_products()
    supported = [
        product
        for product in products
        if product.get("contract_type") == "perpetual_futures"
        and product.get("state") == "live"
        and product.get("symbol") in {"BTCUSD", "ETHUSD"}
    ]
    assets_found = {
        _asset_symbol(product, "underlying_asset") for product in supported
    }
    if assets_found != set(SUPPORTED_ASSETS):
        missing = sorted(set(SUPPORTED_ASSETS) - assets_found)
        raise ValueError(f"Delta live product catalog is missing assets: {missing}")

    async with session_factory() as session:
        async with session.begin():
            for symbol, name in SUPPORTED_ASSETS.items():
                statement = insert(Asset).values(symbol=symbol, name=name)
                await session.execute(
                    statement.on_conflict_do_update(
                        index_elements=["symbol"],
                        set_={"name": statement.excluded.name},
                    )
                )

            for product in supported:
                underlying = _asset_symbol(product, "underlying_asset")
                quote = _asset_symbol(product, "quoting_asset")
                settle = _asset_symbol(product, "settling_asset")
                launch_time = product.get("launch_time")
                launch_at = (
                    datetime.fromisoformat(launch_time.replace("Z", "+00:00"))
                    if isinstance(launch_time, str)
                    else None
                )
                statement = insert(MarketProduct).values(
                    venue=venue,
                    product_id=int(product["id"]),
                    symbol=str(product["symbol"]),
                    underlying_asset=underlying,
                    quoting_asset=quote,
                    settling_asset=settle,
                    contract_type=str(product["contract_type"]),
                    state=str(product["state"]),
                    tick_size=Decimal(str(product["tick_size"])),
                    contract_value=(
                        Decimal(str(product["contract_value"]))
                        if product.get("contract_value") is not None
                        else None
                    ),
                    launch_time=launch_at,
                    raw_payload=product,
                )
                await session.execute(
                    statement.on_conflict_do_update(
                        index_elements=["venue", "product_id"],
                        set_={
                            "symbol": statement.excluded.symbol,
                            "underlying_asset": statement.excluded.underlying_asset,
                            "quoting_asset": statement.excluded.quoting_asset,
                            "settling_asset": statement.excluded.settling_asset,
                            "contract_type": statement.excluded.contract_type,
                            "state": statement.excluded.state,
                            "tick_size": statement.excluded.tick_size,
                            "contract_value": statement.excluded.contract_value,
                            "launch_time": statement.excluded.launch_time,
                            "observed_at": datetime.now(UTC),
                            "raw_payload": statement.excluded.raw_payload,
                        },
                    )
                )

    return sorted(
        (str(product["symbol"]), int(product["id"])) for product in supported
    )
