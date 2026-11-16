from datetime import UTC, datetime

import httpx
import pytest

from app.core.config import Settings
from app.market_data.delta_client import DeltaRestClient


@pytest.mark.asyncio
async def test_products_follow_cursor_pagination() -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.params.get("after") == "next":
            return httpx.Response(
                200,
                json={"success": True, "result": [{"symbol": "ETHUSD"}], "meta": {}},
            )
        return httpx.Response(
            200,
            json={
                "success": True,
                "result": [{"symbol": "BTCUSD"}],
                "meta": {"after": "next"},
            },
        )

    http = httpx.AsyncClient(
        base_url="https://api.india.delta.exchange",
        transport=httpx.MockTransport(handler),
    )
    client = DeltaRestClient(Settings(), http)
    try:
        products = await client.get_products(page_size=1)
    finally:
        await http.aclose()

    assert [product["symbol"] for product in products] == ["BTCUSD", "ETHUSD"]
    assert len(requests) == 2
    assert requests[1].url.params["after"] == "next"


@pytest.mark.asyncio
async def test_candles_use_epoch_seconds_and_reject_naive_bounds() -> None:
    requested: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requested.append(request)
        return httpx.Response(200, json={"success": True, "result": []})

    http = httpx.AsyncClient(
        base_url="https://api.india.delta.exchange",
        transport=httpx.MockTransport(handler),
    )
    client = DeltaRestClient(Settings(), http)
    try:
        await client.get_candles(
            symbol="BTCUSD",
            resolution="1h",
            start=datetime(2024, 1, 1, tzinfo=UTC),
            end=datetime(2024, 1, 2, tzinfo=UTC),
        )
        with pytest.raises(ValueError, match="timezone-aware"):
            await client.get_candles(
                symbol="BTCUSD",
                resolution="1h",
                start=datetime(2024, 1, 1),
                end=datetime(2024, 1, 2, tzinfo=UTC),
            )
    finally:
        await http.aclose()

    assert requested[0].url.params["start"] == "1704067200"
    assert requested[0].url.params["end"] == "1704153600"
