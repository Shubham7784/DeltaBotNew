import asyncio
from datetime import UTC, datetime
from typing import Any

import httpx

from app.core.config import Settings


class DeltaApiError(RuntimeError):
    pass


class DeltaRestClient:
    def __init__(
        self,
        settings: Settings,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._settings = settings
        self._client = client or httpx.AsyncClient(
            base_url=str(settings.delta_rest_base_url).rstrip("/"),
            timeout=settings.delta_request_timeout_seconds,
            headers={"Accept": "application/json"},
        )
        self._owns_client = client is None

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def _get_json(
        self,
        path: str,
        params: dict[str, str | int] | None = None,
    ) -> dict[str, Any]:
        for attempt in range(self._settings.delta_max_retries + 1):
            try:
                response = await self._client.get(path, params=params)
                if response.status_code == 429 or response.status_code >= 500:
                    response.raise_for_status()
                response.raise_for_status()
                try:
                    body = response.json()
                except ValueError as error:
                    raise DeltaApiError(
                        f"Delta API returned invalid JSON for {path}"
                    ) from error
                if not isinstance(body, dict) or body.get("success") is not True:
                    raise DeltaApiError(
                        f"Delta API returned an unsuccessful response for {path}"
                    )
                return body
            except (
                httpx.TimeoutException,
                httpx.NetworkError,
                httpx.HTTPStatusError,
            ) as error:
                status_code = (
                    error.response.status_code
                    if isinstance(error, httpx.HTTPStatusError)
                    else None
                )
                if status_code is not None and status_code != 429 and status_code < 500:
                    raise
                if attempt >= self._settings.delta_max_retries:
                    raise
                await asyncio.sleep(min(2**attempt, 8))
        raise AssertionError("Retry loop must return or raise")

    async def check_connectivity(self) -> None:
        body = await self._get_json("/v2/products", {"page_size": 1})
        if not isinstance(body.get("result"), list):
            raise DeltaApiError("Delta products response has no result list")

    async def get_products(self, page_size: int = 100) -> list[dict[str, Any]]:
        products: list[dict[str, Any]] = []
        cursor: str | None = None

        while True:
            params: dict[str, str | int] = {"page_size": page_size}
            if cursor is not None:
                params["after"] = cursor
            body = await self._get_json("/v2/products", params)
            result = body.get("result")
            if not isinstance(result, list):
                raise DeltaApiError("Delta products response has no result list")
            products.extend(item for item in result if isinstance(item, dict))
            meta = body.get("meta")
            next_cursor = meta.get("after") if isinstance(meta, dict) else None
            if not next_cursor or next_cursor == cursor:
                break
            cursor = str(next_cursor)

        return products

    async def get_candles(
        self,
        *,
        symbol: str,
        resolution: str,
        start: datetime,
        end: datetime,
    ) -> list[dict[str, Any]]:
        if start.tzinfo is None or end.tzinfo is None:
            raise ValueError("Candle query bounds must be timezone-aware")
        start_utc = start.astimezone(UTC)
        end_utc = end.astimezone(UTC)
        if end_utc <= start_utc:
            raise ValueError("Candle query end must be after start")

        body = await self._get_json(
            "/v2/history/candles",
            {
                "symbol": symbol,
                "resolution": resolution,
                "start": int(start_utc.timestamp()),
                "end": int(end_utc.timestamp()),
            },
        )
        result = body.get("result")
        if not isinstance(result, list):
            raise DeltaApiError("Delta candles response has no result list")
        if len(result) > 2_000:
            raise DeltaApiError("Delta candles response exceeded the documented limit")
        return [item for item in result if isinstance(item, dict)]
