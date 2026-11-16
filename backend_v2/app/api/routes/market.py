import asyncio
from datetime import UTC, datetime, timedelta
from typing import Literal

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncEngine

from app.market_data.bias_evaluation import (
    BiasEvaluationReport,
    evaluate_session_biases,
)
from app.market_data.features import (
    FUNDING_FRESHNESS,
    METRIC_FRESHNESS,
    ORDER_FLOW_FRESHNESS,
    FeatureResolution,
    FeatureValue,
    MarketFeatureVector,
    aligned_as_of,
    get_market_feature_vector,
)
from app.market_data.regimes import (
    ASSET_SYMBOLS,
    Asset,
    MarketRegimeSnapshot,
    classify_current_regimes,
)
from app.market_data.session_bias import (
    SessionBiasSnapshot,
    assess_session_biases,
    get_current_session_biases,
)
from app.models import MarketEvent, MarketMetric

router = APIRouter(prefix="/api/v1/market", tags=["market"])


class FeatureSourceContinuity(BaseModel):
    source: str
    available_features: int
    total_features: int
    complete: bool
    unavailable_features: list[str]


class LiveSourceFreshness(BaseModel):
    source: str
    latest_at: datetime | None
    age_seconds: float | None
    stale_after_seconds: float
    status: Literal["fresh", "stale", "missing", "future_timestamp"]


class AssetDataContinuity(BaseModel):
    asset: Asset
    symbol: str
    as_of: datetime
    available_features: int
    total_features: int
    feature_vector_complete: bool
    live_sources_fresh: bool
    feature_sources: list[FeatureSourceContinuity]
    live_sources: list[LiveSourceFreshness]
    unavailable_features: list[FeatureValue]


class MarketDataContinuityReport(BaseModel):
    checked_at: datetime
    as_of: datetime
    resolution: Literal["1h"]
    schema_version: str
    feature_vectors_complete: bool
    live_sources_fresh: bool
    assets: list[AssetDataContinuity]


async def _latest_source_timestamps(
    database_engine: AsyncEngine,
) -> tuple[dict[tuple[str, str], datetime], dict[tuple[str, str], datetime]]:
    event_symbols = [symbols[0] for symbols in ASSET_SYMBOLS.values()]
    metric_symbols = [
        symbol for symbols in ASSET_SYMBOLS.values() for symbol in symbols
    ]
    async with database_engine.connect() as connection:
        events_result = await connection.execute(
            select(
                MarketEvent.symbol,
                MarketEvent.event_type,
                func.max(MarketEvent.event_time),
            )
            .where(
                MarketEvent.venue == "delta",
                MarketEvent.symbol.in_(event_symbols),
                MarketEvent.event_type.in_(("ob_l1", "trades")),
            )
            .group_by(MarketEvent.symbol, MarketEvent.event_type)
        )
        metrics_result = await connection.execute(
            select(
                MarketMetric.symbol,
                MarketMetric.metric,
                func.max(MarketMetric.event_time),
            )
            .where(
                MarketMetric.venue == "delta",
                MarketMetric.symbol.in_(metric_symbols),
                MarketMetric.metric.in_(
                    ("funding_rate", "open_interest", "mark_price", "index_price")
                ),
            )
            .group_by(MarketMetric.symbol, MarketMetric.metric)
        )
        event_timestamps = {
            (row[0], row[1]): row[2]
            for row in events_result.all()
            if row[2] is not None
        }
        metric_timestamps = {
            (row[0], row[1]): row[2]
            for row in metrics_result.all()
            if row[2] is not None
        }
    return event_timestamps, metric_timestamps


def _source_freshness(
    *,
    source: str,
    latest_at: datetime | None,
    checked_at: datetime,
    stale_after: timedelta,
) -> LiveSourceFreshness:
    if latest_at is None:
        return LiveSourceFreshness(
            source=source,
            latest_at=None,
            age_seconds=None,
            stale_after_seconds=stale_after.total_seconds(),
            status="missing",
        )
    age_seconds = (
        checked_at - latest_at.astimezone(UTC)
    ).total_seconds()
    if age_seconds < 0:
        status: Literal["fresh", "stale", "missing", "future_timestamp"] = (
            "future_timestamp"
        )
    elif age_seconds > stale_after.total_seconds():
        status = "stale"
    else:
        status = "fresh"
    return LiveSourceFreshness(
        source=source,
        latest_at=latest_at,
        age_seconds=age_seconds,
        stale_after_seconds=stale_after.total_seconds(),
        status=status,
    )


async def collect_market_data_continuity(
    database_engine: AsyncEngine,
    *,
    checked_at: datetime | None = None,
) -> MarketDataContinuityReport:
    feature_time = checked_at or datetime.now(UTC)
    as_of = aligned_as_of(resolution="1h", now=feature_time)
    vectors = await asyncio.gather(
        *(
            get_market_feature_vector(
                database_engine,
                asset=asset,
                resolution="1h",
                as_of=as_of,
                now=feature_time,
            )
            for asset in ("BTC", "ETH")
        )
    )
    event_timestamps, metric_timestamps = await _latest_source_timestamps(
        database_engine
    )
    report_time = checked_at or datetime.now(UTC)

    assets: list[AssetDataContinuity] = []
    for vector in vectors:
        perpetual_symbol, index_symbol = ASSET_SYMBOLS[vector.asset]
        live_sources = [
            _source_freshness(
                source="order_book",
                latest_at=event_timestamps.get((perpetual_symbol, "ob_l1")),
                checked_at=report_time,
                stale_after=ORDER_FLOW_FRESHNESS,
            ),
            _source_freshness(
                source="trades",
                latest_at=event_timestamps.get((perpetual_symbol, "trades")),
                checked_at=report_time,
                stale_after=ORDER_FLOW_FRESHNESS,
            ),
            _source_freshness(
                source="funding_rate",
                latest_at=metric_timestamps.get((perpetual_symbol, "funding_rate")),
                checked_at=report_time,
                stale_after=FUNDING_FRESHNESS,
            ),
            _source_freshness(
                source="open_interest",
                latest_at=metric_timestamps.get((perpetual_symbol, "open_interest")),
                checked_at=report_time,
                stale_after=METRIC_FRESHNESS,
            ),
            _source_freshness(
                source="mark_price",
                latest_at=metric_timestamps.get((perpetual_symbol, "mark_price")),
                checked_at=report_time,
                stale_after=METRIC_FRESHNESS,
            ),
            _source_freshness(
                source="index_price",
                latest_at=metric_timestamps.get((index_symbol, "index_price")),
                checked_at=report_time,
                stale_after=METRIC_FRESHNESS,
            ),
        ]
        features_by_source: dict[str, list[FeatureValue]] = {}
        for feature in vector.features:
            features_by_source.setdefault(feature.source, []).append(feature)
        sources = [
            FeatureSourceContinuity(
                source=source,
                available_features=sum(
                    feature.status == "available" for feature in features
                ),
                total_features=len(features),
                complete=all(
                    feature.status == "available" for feature in features
                ),
                unavailable_features=[
                    feature.name
                    for feature in features
                    if feature.status != "available"
                ],
            )
            for source, features in sorted(features_by_source.items())
        ]
        unavailable_features = [
            feature
            for feature in vector.features
            if feature.status != "available"
        ]
        assets.append(
            AssetDataContinuity(
                asset=vector.asset,
                symbol=vector.symbol,
                as_of=vector.as_of,
                available_features=len(vector.features) - len(unavailable_features),
                total_features=len(vector.features),
                feature_vector_complete=not unavailable_features,
                live_sources_fresh=all(
                    source.status == "fresh" for source in live_sources
                ),
                feature_sources=sources,
                live_sources=live_sources,
                unavailable_features=unavailable_features,
            )
        )

    return MarketDataContinuityReport(
        checked_at=report_time,
        as_of=as_of,
        resolution="1h",
        schema_version=vectors[0].schema_version,
        feature_vectors_complete=all(
            asset.feature_vector_complete for asset in assets
        ),
        live_sources_fresh=all(asset.live_sources_fresh for asset in assets),
        assets=assets,
    )


@router.get(
    "/data-continuity",
    response_model=MarketDataContinuityReport,
    summary="Check current Delta feature-source continuity",
)
async def market_data_continuity(request: Request) -> MarketDataContinuityReport:
    return await collect_market_data_continuity(request.app.state.database_engine)


@router.get(
    "/features",
    response_model=MarketFeatureVector,
    summary="Get ordered, as-of market research features",
)
async def current_market_features(
    request: Request,
    asset: Asset = "BTC",
    resolution: FeatureResolution = Query(default="15m"),
    as_of: datetime | None = Query(default=None),
) -> MarketFeatureVector:
    now = datetime.now(UTC)
    try:
        feature_as_of = aligned_as_of(
            resolution=resolution,
            now=now,
            requested=as_of,
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return await get_market_feature_vector(
        request.app.state.database_engine,
        asset=asset,
        resolution=resolution,
        as_of=feature_as_of,
        now=now,
    )


@router.get(
    "/regimes",
    response_model=dict[str, MarketRegimeSnapshot],
    summary="Get current research-only market regimes",
)
async def current_market_regimes(
    request: Request,
) -> dict[str, MarketRegimeSnapshot]:
    return await classify_current_regimes(request.app.state.database_engine)


@router.get(
    "/bias",
    response_model=list[SessionBiasSnapshot],
    summary="Read locked bias decisions for the current UTC session",
)
async def current_session_biases(request: Request) -> list[SessionBiasSnapshot]:
    return await get_current_session_biases(request.app.state.database_engine)


@router.post(
    "/bias/assess",
    response_model=list[SessionBiasSnapshot],
    summary="Assess and lock BTC/ETH bias for the current UTC session",
)
async def assess_current_session_biases(
    request: Request,
) -> list[SessionBiasSnapshot]:
    return await assess_session_biases(request.app.state.database_engine)


@router.get(
    "/bias/evaluation",
    response_model=BiasEvaluationReport,
    summary="Replay historical session-bias decisions",
)
async def session_bias_evaluation(
    request: Request,
    days: int = Query(default=30, ge=7, le=90),
) -> BiasEvaluationReport:
    return await evaluate_session_biases(
        request.app.state.database_engine,
        days=days,
    )
