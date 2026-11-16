from datetime import UTC, datetime, timedelta

import httpx
import pytest
from fastapi import FastAPI
from sqlalchemy.ext.asyncio import AsyncEngine

from app.api.routes import market
from app.database import engine
from app.market_data.features import (
    FEATURE_DEFINITIONS,
    FEATURE_ORDER,
    FEATURE_SCHEMA_VERSION,
    FeatureResolution,
    FeatureValue,
    MarketFeatureVector,
)
from app.market_data.regimes import ASSET_SYMBOLS, Asset

CHECKED_AT = datetime(2026, 11, 15, 8, 4, tzinfo=UTC)
AS_OF = datetime(2026, 11, 15, 8, 0, tzinfo=UTC)


@pytest.mark.parametrize(
    ("latest_at", "expected_status"),
    [
        (None, "missing"),
        (CHECKED_AT + timedelta(seconds=1), "future_timestamp"),
        (CHECKED_AT - timedelta(minutes=2), "fresh"),
        (CHECKED_AT - timedelta(minutes=2, seconds=1), "stale"),
    ],
)
def test_source_freshness_classifies_timestamp_edges(
    latest_at: datetime | None,
    expected_status: str,
) -> None:
    source = market._source_freshness(
        source="trades",
        latest_at=latest_at,
        checked_at=CHECKED_AT,
        stale_after=timedelta(minutes=2),
    )

    assert source.status == expected_status


def make_vector(asset: Asset, *, incomplete: bool) -> MarketFeatureVector:
    unavailable_name = "trade_count_15m" if incomplete else None
    features = [
        FeatureValue(
            name=definition.name,
            family=definition.family,
            value=None if definition.name == unavailable_name else 1.0,
            unit=definition.unit,
            source=definition.source,
            source_as_of=AS_OF - timedelta(seconds=1),
            window_start=AS_OF - timedelta(minutes=15),
            window_end=AS_OF,
            sample_count=0 if definition.name == unavailable_name else 1,
            status="insufficient_data"
            if definition.name == unavailable_name
            else "available",
            missing_reason="A complete event window is unavailable."
            if definition.name == unavailable_name
            else None,
        )
        for definition in FEATURE_DEFINITIONS
    ]
    return MarketFeatureVector(
        schema_version=FEATURE_SCHEMA_VERSION,
        asset=asset,
        symbol=ASSET_SYMBOLS[asset][0],
        resolution="1h",
        as_of=AS_OF,
        computed_at=CHECKED_AT,
        feature_order=list(FEATURE_ORDER),
        features=features,
        interpretation="Test data",
    )


@pytest.mark.asyncio
async def test_data_continuity_reports_per_asset_and_source_coverage(monkeypatch) -> None:
    calls: list[tuple[Asset, FeatureResolution, datetime, datetime]] = []

    async def fake_get_feature_vector(
        database_engine: AsyncEngine,
        *,
        asset: Asset,
        resolution: FeatureResolution,
        as_of: datetime,
        now: datetime,
    ) -> MarketFeatureVector:
        assert database_engine is engine
        calls.append((asset, resolution, as_of, now))
        return make_vector(asset, incomplete=asset == "ETH")

    monkeypatch.setattr(market, "get_market_feature_vector", fake_get_feature_vector)

    event_timestamps = {
        (ASSET_SYMBOLS[asset][0], event_type): CHECKED_AT - timedelta(seconds=30)
        for asset in ("BTC", "ETH")
        for event_type in ("ob_l1", "trades")
    }
    metric_timestamps: dict[tuple[str, str], datetime] = {}
    for asset in ("BTC", "ETH"):
        perpetual_symbol, index_symbol = ASSET_SYMBOLS[asset]
        for symbol, metric in (
            (perpetual_symbol, "funding_rate"),
            (perpetual_symbol, "open_interest"),
            (perpetual_symbol, "mark_price"),
            (index_symbol, "index_price"),
        ):
            age = (
                timedelta(minutes=3)
                if asset == "ETH" and metric == "open_interest"
                else timedelta(seconds=30)
            )
            metric_timestamps[(symbol, metric)] = CHECKED_AT - age

    async def fake_latest_source_timestamps(
        database_engine: AsyncEngine,
    ) -> tuple[dict[tuple[str, str], datetime], dict[tuple[str, str], datetime]]:
        assert database_engine is engine
        return event_timestamps, metric_timestamps

    monkeypatch.setattr(
        market,
        "_latest_source_timestamps",
        fake_latest_source_timestamps,
    )

    report = await market.collect_market_data_continuity(
        engine,
        checked_at=CHECKED_AT,
    )

    assert report.as_of == AS_OF
    assert report.resolution == "1h"
    assert report.feature_vectors_complete is False
    assert report.live_sources_fresh is False
    assert calls == [
        ("BTC", "1h", AS_OF, CHECKED_AT),
        ("ETH", "1h", AS_OF, CHECKED_AT),
    ]

    btc, eth = report.assets
    assert btc.feature_vector_complete is True
    assert btc.live_sources_fresh is True
    assert btc.available_features == btc.total_features == len(FEATURE_ORDER)
    assert eth.feature_vector_complete is False
    assert eth.live_sources_fresh is False
    assert eth.available_features == len(FEATURE_ORDER) - 1
    assert [feature.name for feature in eth.unavailable_features] == [
        "trade_count_15m"
    ]
    trade_source = next(
        source
        for source in eth.feature_sources
        if source.source == "market_events:trades"
    )
    assert trade_source.complete is False
    assert trade_source.available_features == 1
    assert trade_source.total_features == 2
    assert trade_source.unavailable_features == ["trade_count_15m"]
    open_interest = next(
        source for source in eth.live_sources if source.source == "open_interest"
    )
    assert open_interest.status == "stale"
    assert open_interest.age_seconds == 180

    async def fake_collect_data_continuity(
        database_engine: AsyncEngine,
    ) -> market.MarketDataContinuityReport:
        assert database_engine is engine
        return report

    monkeypatch.setattr(
        market,
        "collect_market_data_continuity",
        fake_collect_data_continuity,
    )
    test_app = FastAPI()
    test_app.include_router(market.router)
    test_app.state.database_engine = engine
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=test_app),
        base_url="http://test",
    ) as client:
        response = await client.get("/api/v1/market/data-continuity")

    assert response.status_code == 200
    assert response.json()["feature_vectors_complete"] is False
    assert response.json()["live_sources_fresh"] is False
