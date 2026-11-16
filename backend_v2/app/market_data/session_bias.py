import uuid
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from typing import Literal, cast

from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

from app.market_data.regimes import (
    ASSET_SYMBOLS,
    Asset,
    MarketRegimeSnapshot,
    classify_current_regimes,
)
from app.models import Candle, NewsArticle, NewsSentiment, SessionBias
from app.news.sentiment import MODEL_REVISION

SESSION_BIAS_ALGORITHM_VERSION = "session-bias-v1"
NEWS_WINDOW = timedelta(hours=24)
MINIMUM_NEWS_ARTICLES = 3
MINIMUM_NEWS_MEAN_CONFIDENCE = 0.55
NEWS_DIRECTION_THRESHOLD = 0.10
HIGHER_TIMEFRAME_TREND_THRESHOLD_PERCENT = 0.15
HIGHER_TIMEFRAME_CONSISTENCY_THRESHOLD = 2 / 3
MINIMUM_FOUR_HOUR_CANDLES = 12
MAXIMUM_FOUR_HOUR_AGE = timedelta(hours=8)
ONE_HOUR = timedelta(hours=1)
FOUR_HOURS = timedelta(hours=4)

BiasDirection = Literal["LONG_ONLY", "SHORT_ONLY", "NO_TRADE"]
BiasStatus = Literal["active", "suspended", "no_trade"]
NewsDirection = Literal["bullish", "bearish", "neutral", "insufficient"]
TrendDirection = Literal["trend_up", "trend_down", "range", "unavailable"]


class NewsBiasContext(BaseModel):
    asset: Asset
    direction: NewsDirection
    article_count: int
    average_weighted_sentiment: float | None
    mean_confidence: float | None
    as_of: datetime | None
    window_start: datetime
    window_end: datetime
    model_revision: str
    evidence: list[str]


class HigherTimeframeContext(BaseModel):
    timeframe: Literal["4h"] = "4h"
    trend: TrendDirection
    ema_6: float | None
    ema_12: float | None
    ema_spread_percent: float | None
    directional_consistency_3_bars: float | None
    confidence: float
    data_as_of: datetime | None
    evidence: list[str]


class SessionBiasContext(BaseModel):
    assessed_at: datetime
    regime: MarketRegimeSnapshot
    higher_timeframe: HigherTimeframeContext
    news: NewsBiasContext


class SessionBiasSnapshot(BaseModel):
    id: uuid.UUID
    asset: Asset
    session_date: date
    direction: BiasDirection
    status: BiasStatus
    entry_allowed: bool
    version: int
    confidence: float
    algorithm_version: str
    decided_at: datetime
    last_checked_at: datetime
    updated_at: datetime
    suspended_at: datetime | None
    suspension_reason: str | None
    evidence: list[str]
    invalidation_conditions: list[str]
    decision_context: SessionBiasContext
    latest_context: SessionBiasContext


@dataclass(frozen=True)
class BiasCandidate:
    direction: BiasDirection
    status: BiasStatus
    confidence: float
    evidence: list[str]
    invalidation_conditions: list[str]


@dataclass(frozen=True)
class BiasState:
    direction: BiasDirection
    status: BiasStatus
    suspension_reason: str | None


def _ema(values: list[float], period: int) -> float:
    if len(values) < period:
        raise ValueError(f"At least {period} values are required for EMA")
    current = sum(values[:period]) / period
    alpha = 2 / (period + 1)
    for value in values[period:]:
        current = alpha * value + (1 - alpha) * current
    return current


def build_higher_timeframe_context(
    asset: Asset,
    candles: list[tuple[datetime, Decimal]],
    *,
    computed_at: datetime,
) -> HigherTimeframeContext:
    if computed_at.tzinfo is None:
        raise ValueError("Higher-timeframe computation time must be timezone-aware")
    timestamp = computed_at.astimezone(UTC)
    grouped: dict[datetime, dict[datetime, Decimal]] = defaultdict(dict)
    for open_time, close in candles:
        if open_time.tzinfo is None:
            return _unavailable_higher_timeframe(
                "Hourly candles contain a timezone-naive open time."
            )
        open_utc = open_time.astimezone(UTC).replace(minute=0, second=0, microsecond=0)
        if open_utc + ONE_HOUR > timestamp:
            continue
        bucket = open_utc.replace(hour=open_utc.hour - open_utc.hour % 4)
        if bucket + FOUR_HOURS <= timestamp:
            grouped[bucket][open_utc] = close

    completed_buckets: list[tuple[datetime, Decimal]] = []
    for bucket, hourly_closes in sorted(grouped.items()):
        expected = [bucket + timedelta(hours=index) for index in range(4)]
        if all(open_time in hourly_closes for open_time in expected):
            completed_buckets.append((bucket, hourly_closes[expected[-1]]))

    if len(completed_buckets) < MINIMUM_FOUR_HOUR_CANDLES:
        return _unavailable_higher_timeframe(
            f"Only {len(completed_buckets)} complete 4h candles are available."
        )

    data_as_of = completed_buckets[-1][0] + FOUR_HOURS
    if timestamp - data_as_of > MAXIMUM_FOUR_HOUR_AGE:
        return _unavailable_higher_timeframe(
            "The latest completed 4h candle is stale.",
            data_as_of=data_as_of,
        )

    sample = completed_buckets[-48:]
    recent = sample[-MINIMUM_FOUR_HOUR_CANDLES:]
    if any(
        later[0] - earlier[0] != FOUR_HOURS
        for earlier, later in zip(recent, recent[1:])
    ):
        return _unavailable_higher_timeframe(
            "A gap exists in the 12-candle 4h trend window.",
            data_as_of=data_as_of,
        )
    if any(close <= 0 for _, close in sample):
        return _unavailable_higher_timeframe(
            "The 4h history contains a nonpositive close.",
            data_as_of=data_as_of,
        )

    closes = [float(close) for _, close in sample]
    ema_6 = _ema(closes, 6)
    ema_12 = _ema(closes, 12)
    spread = (ema_6 / ema_12 - 1) * 100
    direction = 1 if spread > 0 else -1 if spread < 0 else 0
    recent_returns = [
        closes[index] / closes[index - 1] - 1
        for index in range(len(closes) - 3, len(closes))
    ]
    consistency = (
        sum(value * direction > 0 for value in recent_returns) / len(recent_returns)
        if direction
        else 0.5
    )

    if (
        spread >= HIGHER_TIMEFRAME_TREND_THRESHOLD_PERCENT
        and consistency >= HIGHER_TIMEFRAME_CONSISTENCY_THRESHOLD
    ):
        trend: TrendDirection = "trend_up"
        confidence = (
            0.5 + 0.5 * min(abs(spread) / 0.5, 1.0)
        ) * consistency
    elif (
        spread <= -HIGHER_TIMEFRAME_TREND_THRESHOLD_PERCENT
        and consistency >= HIGHER_TIMEFRAME_CONSISTENCY_THRESHOLD
    ):
        trend = "trend_down"
        confidence = (
            0.5 + 0.5 * min(abs(spread) / 0.5, 1.0)
        ) * consistency
    else:
        trend = "range"
        confidence = (
            0.5
            * (
                1
                - min(
                    abs(spread) / HIGHER_TIMEFRAME_TREND_THRESHOLD_PERCENT,
                    1.0,
                )
            )
            + 0.5 * (1 - consistency)
        )
    return HigherTimeframeContext(
        trend=trend,
        ema_6=ema_6,
        ema_12=ema_12,
        ema_spread_percent=spread,
        directional_consistency_3_bars=consistency,
        confidence=max(0.0, min(confidence, 1.0)),
        data_as_of=data_as_of,
        evidence=[
            f"{asset} 4h EMA6/EMA12 spread is {spread:.3f}%.",
            f"Last 3 completed 4h bars have {consistency:.3f} directional consistency.",
            f"Completed 4h data through {data_as_of.isoformat()}.",
        ],
    )


def _unavailable_higher_timeframe(
    reason: str,
    *,
    data_as_of: datetime | None = None,
) -> HigherTimeframeContext:
    return HigherTimeframeContext(
        trend="unavailable",
        ema_6=None,
        ema_12=None,
        ema_spread_percent=None,
        directional_consistency_3_bars=None,
        confidence=0.0,
        data_as_of=data_as_of,
        evidence=[reason],
    )


def build_news_bias_context(
    asset: Asset,
    *,
    article_count: int,
    average_weighted_sentiment: float | None,
    mean_confidence: float | None,
    as_of: datetime | None,
    window_start: datetime,
    window_end: datetime,
) -> NewsBiasContext:
    if article_count < MINIMUM_NEWS_ARTICLES:
        direction: NewsDirection = "insufficient"
        reason = (
            f"Only {article_count} relevant scored headlines; "
            f"{MINIMUM_NEWS_ARTICLES} are required."
        )
    elif mean_confidence is None or mean_confidence < MINIMUM_NEWS_MEAN_CONFIDENCE:
        direction = "insufficient"
        reason = (
            "Mean headline confidence is missing or below "
            f"{MINIMUM_NEWS_MEAN_CONFIDENCE:.2f}."
        )
    elif average_weighted_sentiment is None:
        direction = "insufficient"
        reason = "The recent headline sentiment aggregate is missing."
    elif average_weighted_sentiment >= NEWS_DIRECTION_THRESHOLD:
        direction = "bullish"
        reason = "Recent model-scored headlines are directionally bullish."
    elif average_weighted_sentiment <= -NEWS_DIRECTION_THRESHOLD:
        direction = "bearish"
        reason = "Recent model-scored headlines are directionally bearish."
    else:
        direction = "neutral"
        reason = "Recent model-scored headlines are directionally neutral."

    return NewsBiasContext(
        asset=asset,
        direction=direction,
        article_count=article_count,
        average_weighted_sentiment=average_weighted_sentiment,
        mean_confidence=mean_confidence,
        as_of=as_of,
        window_start=window_start,
        window_end=window_end,
        model_revision=MODEL_REVISION,
        evidence=[
            f"{article_count} relevant scored headlines in the preceding 24h.",
            (
                f"Mean weighted sentiment: {average_weighted_sentiment:.4f}."
                if average_weighted_sentiment is not None
                else "Mean weighted sentiment is unavailable."
            ),
            (
                f"Mean model confidence: {mean_confidence:.3f}."
                if mean_confidence is not None
                else "Mean model confidence is unavailable."
            ),
            reason,
        ],
    )


def build_bias_candidate(context: SessionBiasContext) -> BiasCandidate:
    regime = context.regime.regime
    if regime == "trend_up":
        required_trend: Literal["trend_up", "trend_down"] = "trend_up"
        direction: BiasDirection = "LONG_ONLY"
        required_news: NewsDirection = "bullish"
    elif regime == "trend_down":
        required_trend = "trend_down"
        direction = "SHORT_ONLY"
        required_news = "bearish"
    else:
        required_trend = "trend_up"
        direction = "NO_TRADE"
        required_news = "bullish"

    blockers: list[str] = []
    if regime not in {"trend_up", "trend_down"}:
        blockers.append(f"1h regime is {regime}, not a directional trend.")
    if (
        regime in {"trend_up", "trend_down"}
        and context.higher_timeframe.trend != required_trend
    ):
        blockers.append(
            f"4h context {context.higher_timeframe.trend} does not confirm {required_trend}."
        )
    if (
        regime in {"trend_up", "trend_down"}
        and context.news.direction != required_news
    ):
        blockers.append(
            f"24h news direction {context.news.direction} does not confirm {required_news}."
        )

    evidence = [
        f"1h regime: {regime} (heuristic confidence {context.regime.confidence:.3f}).",
        f"4h context: {context.higher_timeframe.trend}.",
        f"24h news direction: {context.news.direction}.",
        *blockers,
    ]
    if blockers:
        return BiasCandidate(
            direction="NO_TRADE",
            status="no_trade",
            confidence=0.0,
            evidence=evidence,
            invalidation_conditions=[
                "This NO_TRADE decision remains locked for the UTC session; reassess at the next UTC day."
            ],
        )

    sentiment_strength = min(
        abs(context.news.average_weighted_sentiment or 0.0)
        / max(NEWS_DIRECTION_THRESHOLD * 5, 0.5),
        1.0,
    )
    news_confidence = (context.news.mean_confidence or 0.0) * sentiment_strength
    confidence = min(
        context.regime.confidence,
        context.higher_timeframe.confidence,
        news_confidence,
    )
    return BiasCandidate(
        direction=direction,
        status="active",
        confidence=max(0.0, min(confidence, 1.0)),
        evidence=evidence,
        invalidation_conditions=[
            f"Suspend if the 1h regime is no longer {required_trend}.",
            f"Suspend if the 4h context no longer confirms {required_trend}.",
            "Suspend if recent scored-news sentiment no longer confirms the locked direction.",
        ],
    )


def reconcile_locked_bias(
    direction: BiasDirection,
    status: BiasStatus,
    current_context: SessionBiasContext,
    *,
    existing_reason: str | None = None,
) -> BiasState:
    if status != "active":
        return BiasState(direction, status, existing_reason)

    expected_regime = "trend_up" if direction == "LONG_ONLY" else "trend_down"
    expected_news = "bullish" if direction == "LONG_ONLY" else "bearish"
    reasons = []
    if current_context.regime.regime != expected_regime:
        reasons.append(f"1h regime invalidated: {current_context.regime.regime}.")
    if current_context.higher_timeframe.trend != expected_regime:
        reasons.append(
            f"4h confirmation invalidated: {current_context.higher_timeframe.trend}."
        )
    if current_context.news.direction != expected_news:
        reasons.append(
            f"24h news confirmation invalidated: {current_context.news.direction}."
        )
    if reasons:
        return BiasState(direction, "suspended", " ".join(reasons))
    return BiasState(direction, status, existing_reason)


def _session_bias_snapshot(row: SessionBias) -> SessionBiasSnapshot:
    direction = cast(BiasDirection, row.direction)
    status = cast(BiasStatus, row.status)
    return SessionBiasSnapshot(
        id=row.id,
        asset=cast(Asset, row.asset),
        session_date=row.session_date,
        direction=direction,
        status=status,
        entry_allowed=status == "active" and direction != "NO_TRADE",
        version=row.version,
        confidence=row.confidence,
        algorithm_version=row.algorithm_version,
        decided_at=row.decided_at,
        last_checked_at=row.last_checked_at,
        updated_at=row.updated_at,
        suspended_at=row.suspended_at,
        suspension_reason=row.suspension_reason,
        evidence=row.evidence,
        invalidation_conditions=row.invalidation_conditions,
        decision_context=SessionBiasContext.model_validate(row.decision_context),
        latest_context=SessionBiasContext.model_validate(row.latest_context),
    )


async def _load_higher_timeframe_contexts(
    database_engine: AsyncEngine,
    *,
    computed_at: datetime,
) -> dict[Asset, HigherTimeframeContext]:
    candles_by_asset: dict[Asset, list[tuple[datetime, Decimal]]] = {
        asset: [] for asset in ASSET_SYMBOLS
    }
    symbol_to_asset = {
        symbols[0]: asset for asset, symbols in ASSET_SYMBOLS.items()
    }
    async with database_engine.connect() as connection:
        result = await connection.execute(
            select(Candle.symbol, Candle.open_time, Candle.close)
            .where(
                Candle.venue == "delta",
                Candle.symbol.in_(tuple(symbol_to_asset)),
                Candle.resolution == "1h",
                Candle.open_time >= computed_at - timedelta(days=8),
            )
            .order_by(Candle.symbol, Candle.open_time)
        )
        for row in result.all():
            asset = symbol_to_asset[row[0]]
            candles_by_asset[asset].append((row[1], row[2]))
    return {
        asset: build_higher_timeframe_context(
            asset,
            candles_by_asset[asset],
            computed_at=computed_at,
        )
        for asset in ASSET_SYMBOLS
    }


async def _load_recent_news_contexts(
    database_engine: AsyncEngine,
    *,
    computed_at: datetime,
) -> dict[Asset, NewsBiasContext]:
    window_start = computed_at - NEWS_WINDOW
    observed_at = func.coalesce(NewsArticle.published_at, NewsArticle.first_seen_at)
    contexts: dict[Asset, NewsBiasContext] = {}
    async with database_engine.connect() as connection:
        result = await connection.execute(
            select(
                NewsSentiment.asset,
                func.count(func.distinct(NewsSentiment.article_id)),
                func.avg(NewsSentiment.weighted_sentiment),
                func.avg(NewsSentiment.confidence),
                func.max(observed_at),
            )
            .select_from(NewsArticle)
            .join(NewsSentiment, NewsSentiment.article_id == NewsArticle.id)
            .where(
                NewsSentiment.asset.in_(tuple(ASSET_SYMBOLS)),
                NewsSentiment.model_revision == MODEL_REVISION,
                NewsSentiment.relevance_weight > 0,
                NewsArticle.first_seen_at <= computed_at,
                NewsSentiment.scored_at <= computed_at,
                observed_at >= window_start,
                observed_at <= computed_at,
            )
            .group_by(NewsSentiment.asset)
        )
        rows = {row[0]: row for row in result.all()}

    for asset in ASSET_SYMBOLS:
        row = rows.get(asset)
        contexts[asset] = build_news_bias_context(
            asset,
            article_count=int(row[1]) if row else 0,
            average_weighted_sentiment=float(row[2]) if row and row[2] is not None else None,
            mean_confidence=float(row[3]) if row and row[3] is not None else None,
            as_of=row[4] if row else None,
            window_start=window_start,
            window_end=computed_at,
        )
    return contexts


async def _load_bias_contexts(
    database_engine: AsyncEngine,
    *,
    computed_at: datetime,
) -> dict[Asset, SessionBiasContext]:
    regimes = await classify_current_regimes(
        database_engine,
        computed_at=computed_at,
    )
    higher_timeframes = await _load_higher_timeframe_contexts(
        database_engine,
        computed_at=computed_at,
    )
    news = await _load_recent_news_contexts(
        database_engine,
        computed_at=computed_at,
    )
    return {
        asset: SessionBiasContext(
            assessed_at=computed_at,
            regime=regimes[asset],
            higher_timeframe=higher_timeframes[asset],
            news=news[asset],
        )
        for asset in ASSET_SYMBOLS
    }


async def assess_session_biases(
    database_engine: AsyncEngine,
    *,
    assessed_at: datetime | None = None,
) -> list[SessionBiasSnapshot]:
    timestamp = assessed_at or datetime.now(UTC)
    if timestamp.tzinfo is None:
        raise ValueError("Session-bias assessment time must be timezone-aware")
    timestamp = timestamp.astimezone(UTC)
    contexts = await _load_bias_contexts(database_engine, computed_at=timestamp)
    values = []
    for asset, context in contexts.items():
        candidate = build_bias_candidate(context)
        serialized_context = context.model_dump(mode="json")
        values.append(
            {
                "id": uuid.uuid4(),
                "asset": asset,
                "session_date": timestamp.date(),
                "direction": candidate.direction,
                "status": candidate.status,
                "version": 1,
                "confidence": candidate.confidence,
                "algorithm_version": SESSION_BIAS_ALGORITHM_VERSION,
                "decided_at": timestamp,
                "last_checked_at": timestamp,
                "updated_at": timestamp,
                "suspended_at": None,
                "suspension_reason": None,
                "evidence": candidate.evidence,
                "invalidation_conditions": candidate.invalidation_conditions,
                "decision_context": serialized_context,
                "latest_context": serialized_context,
            }
        )

    session_factory = async_sessionmaker(database_engine, expire_on_commit=False)
    async with session_factory() as session:
        async with session.begin():
            statement = insert(SessionBias).values(values).on_conflict_do_nothing(
                index_elements=["asset", "session_date"]
            )
            await session.execute(statement)
            result = await session.execute(
                select(SessionBias)
                .where(SessionBias.session_date == timestamp.date())
                .order_by(SessionBias.asset)
                .with_for_update()
            )
            rows = list(result.scalars())
            for row in rows:
                context = contexts[cast(Asset, row.asset)]
                serialized_context = context.model_dump(mode="json")
                row.latest_context = serialized_context
                row.last_checked_at = timestamp
                row.updated_at = timestamp
                state = reconcile_locked_bias(
                    cast(BiasDirection, row.direction),
                    cast(BiasStatus, row.status),
                    context,
                    existing_reason=row.suspension_reason,
                )
                if state.status != row.status:
                    row.status = state.status
                    row.suspended_at = timestamp
                    row.suspension_reason = state.suspension_reason
                if state.direction != row.direction:
                    row.direction = state.direction
            snapshots = [_session_bias_snapshot(row) for row in rows]
    return snapshots


async def get_current_session_biases(
    database_engine: AsyncEngine,
    *,
    now: datetime | None = None,
) -> list[SessionBiasSnapshot]:
    timestamp = now or datetime.now(UTC)
    if timestamp.tzinfo is None:
        raise ValueError("Session-bias read time must be timezone-aware")
    session_factory = async_sessionmaker(database_engine, expire_on_commit=False)
    async with session_factory() as session:
        result = await session.execute(
            select(SessionBias)
            .where(SessionBias.session_date == timestamp.astimezone(UTC).date())
            .order_by(SessionBias.asset)
        )
        return [_session_bias_snapshot(row) for row in result.scalars()]
