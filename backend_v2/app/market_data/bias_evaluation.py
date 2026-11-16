import statistics
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal

from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncEngine

from app.market_data.regimes import (
    ASSET_SYMBOLS,
    Asset,
    HourlyCandle,
    MarketRegimeSnapshot,
    RegimeLabel,
    classify_hourly_regime,
)
from app.market_data.session_bias import (
    BiasDirection,
    HigherTimeframeContext,
    NEWS_WINDOW,
    NewsBiasContext,
    SessionBiasContext,
    build_bias_candidate,
    build_higher_timeframe_context,
    build_news_bias_context,
    reconcile_locked_bias,
)
from app.models import Candle, NewsArticle, NewsSentiment
from app.news.sentiment import MODEL_REVISION

DEFAULT_EVALUATION_DAYS = 30
MINIMUM_EVALUATION_DAYS = 7
MAXIMUM_EVALUATION_DAYS = 90
INVALIDATION_CHECK_INTERVAL = timedelta(hours=4)
INVALIDATION_CHECK_HOURS = (4, 8, 12, 16, 20)
ONE_HOUR = timedelta(hours=1)


@dataclass(frozen=True)
class ReplayCandle:
    open_time: datetime
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal


@dataclass(frozen=True)
class SentimentObservation:
    article_id: uuid.UUID
    published_at: datetime | None
    first_seen_at: datetime
    scored_at: datetime
    weighted_sentiment: Decimal
    confidence: Decimal
    relevance_weight: Decimal

    @property
    def observed_at(self) -> datetime:
        return self.published_at or self.first_seen_at


@dataclass(frozen=True)
class ReplayedSession:
    asset: Asset
    session_start: datetime
    direction: BiasDirection
    initial_regime: RegimeLabel
    no_trade_reasons: tuple[str, ...]
    invalidation_at: datetime | None
    market_return_24h_percent: float | None
    effective_directional_return_percent: float | None
    full_day_directional_return_percent: float | None
    post_invalidation_directional_return_percent: float | None


class AssetBiasEvaluation(BaseModel):
    sessions: int
    long_only_sessions: int
    short_only_sessions: int
    no_trade_sessions: int
    no_trade_reasons: dict[str, int]
    directional_sessions: int
    invalidated_sessions: int
    invalidation_rate_percent: float | None
    outcome_sessions: int
    missing_outcome_sessions: int
    mean_effective_directional_return_percent: float | None
    mean_full_day_directional_return_percent: float | None
    mean_market_return_24h_percent: float | None
    effective_directional_win_rate_percent: float | None
    post_invalidation_outcomes: int
    post_invalidation_adverse_percent: float | None
    mean_post_invalidation_directional_return_percent: float | None
    adjacent_session_comparisons: int
    unchanged_decision_pairs: int
    decision_stability_percent: float | None
    adjacent_directional_comparisons: int
    unchanged_directional_pairs: int
    directional_stability_percent: float | None


class AssetRegimeOutcome(BaseModel):
    asset: Asset
    initial_regime: RegimeLabel
    direction: BiasDirection
    sessions: int
    directional_sessions: int
    invalidated_sessions: int
    outcome_sessions: int
    mean_effective_directional_return_percent: float | None
    mean_full_day_directional_return_percent: float | None
    mean_market_return_24h_percent: float | None
    effective_directional_win_rate_percent: float | None


class BiasEvaluationReport(BaseModel):
    algorithm_version: str
    checked_at: datetime
    window_start: datetime
    window_end: datetime
    evaluation_days: int
    assessment_time_utc: str
    invalidation_check_interval_hours: int
    total_sessions: int
    assets: dict[Asset, AssetBiasEvaluation]
    by_asset_regime: list[AssetRegimeOutcome]
    evaluation_note: str


@dataclass(frozen=True)
class _EvaluationOutcome:
    market_return: float | None
    effective_directional_return: float | None
    full_day_directional_return: float | None
    post_invalidation_directional_return: float | None


def _session_context_at(
    asset: Asset,
    candles: list[ReplayCandle],
    observations: list[SentimentObservation],
    *,
    timestamp: datetime,
) -> SessionBiasContext:
    hourly = [
        HourlyCandle(
            open_time=candle.open_time,
            close=candle.close,
            volume=candle.volume,
        )
        for candle in candles
    ]
    regime: MarketRegimeSnapshot = classify_hourly_regime(
        asset,
        hourly,
        computed_at=timestamp,
    )
    higher_timeframe: HigherTimeframeContext = build_higher_timeframe_context(
        asset,
        [(candle.open_time, candle.close) for candle in candles],
        computed_at=timestamp,
    )
    window_start = timestamp - NEWS_WINDOW
    recent = [
        observation
        for observation in observations
        if observation.observed_at <= timestamp
        and observation.observed_at >= window_start
        and observation.first_seen_at <= timestamp
        and observation.scored_at <= timestamp
        and observation.relevance_weight > 0
    ]
    distinct: dict[uuid.UUID, SentimentObservation] = {
        observation.article_id: observation for observation in recent
    }
    news: NewsBiasContext = build_news_bias_context(
        asset,
        article_count=len(distinct),
        average_weighted_sentiment=(
            statistics.fmean(
                float(observation.weighted_sentiment)
                for observation in distinct.values()
            )
            if distinct
            else None
        ),
        mean_confidence=(
            statistics.fmean(
                float(observation.confidence) for observation in distinct.values()
            )
            if distinct
            else None
        ),
        as_of=max(
            (observation.observed_at for observation in distinct.values()),
            default=None,
        ),
        window_start=window_start,
        window_end=timestamp,
    )
    return SessionBiasContext(
        assessed_at=timestamp,
        regime=regime,
        higher_timeframe=higher_timeframe,
        news=news,
    )


def _exact_close_at(
    candles: list[ReplayCandle],
    timestamp: datetime,
) -> Decimal | None:
    target_open_time = timestamp - ONE_HOUR
    for candle in candles:
        if candle.open_time == target_open_time:
            return candle.close
    return None


def _session_outcome(
    direction: BiasDirection,
    candles: list[ReplayCandle],
    *,
    session_start: datetime,
    invalidation_at: datetime | None,
) -> _EvaluationOutcome:
    session_end = session_start + timedelta(days=1)
    entry_price = _exact_close_at(candles, session_start)
    exit_price = _exact_close_at(candles, session_end)
    session_bars = [
        candle
        for candle in candles
        if session_start <= candle.open_time < session_end
    ]
    expected_times = [
        session_start + timedelta(hours=index) for index in range(24)
    ]
    has_complete_path = (
        len(session_bars) == len(expected_times)
        and [candle.open_time for candle in session_bars] == expected_times
    )
    if (
        entry_price is None
        or exit_price is None
        or not has_complete_path
        or entry_price <= 0
    ):
        return _EvaluationOutcome(None, None, None, None)

    market_return = float((exit_price / entry_price - 1) * 100)
    if direction == "NO_TRADE":
        return _EvaluationOutcome(market_return, None, None, None)

    sign = 1 if direction == "LONG_ONLY" else -1
    full_day_directional = sign * market_return
    exit_time = invalidation_at or session_end
    effective_exit_price = _exact_close_at(candles, exit_time)
    effective_directional = (
        sign * float((effective_exit_price / entry_price - 1) * 100)
        if effective_exit_price is not None
        else None
    )
    post_invalidation = None
    if invalidation_at is not None and effective_exit_price is not None:
        post_invalidation = sign * float(
            (exit_price / effective_exit_price - 1) * 100
        )
    return _EvaluationOutcome(
        market_return=market_return,
        effective_directional_return=effective_directional,
        full_day_directional_return=full_day_directional,
        post_invalidation_directional_return=post_invalidation,
    )


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _percentage(numerator: int, denominator: int) -> float | None:
    return numerator / denominator * 100 if denominator else None


def _summarize_asset(sessions: list[ReplayedSession]) -> AssetBiasEvaluation:
    directional = [session for session in sessions if session.direction != "NO_TRADE"]
    invalidated = [session for session in directional if session.invalidation_at]
    effective_returns = [
        session.effective_directional_return_percent
        for session in directional
        if session.effective_directional_return_percent is not None
    ]
    full_day_returns = [
        session.full_day_directional_return_percent
        for session in directional
        if session.full_day_directional_return_percent is not None
    ]
    market_returns = [
        session.market_return_24h_percent
        for session in sessions
        if session.market_return_24h_percent is not None
    ]
    post_returns = [
        session.post_invalidation_directional_return_percent
        for session in invalidated
        if session.post_invalidation_directional_return_percent is not None
    ]
    adjacent_pairs = list(zip(sessions, sessions[1:]))
    unchanged_pairs = sum(left.direction == right.direction for left, right in adjacent_pairs)
    directional_pairs = [
        (left, right)
        for left, right in adjacent_pairs
        if left.direction != "NO_TRADE" and right.direction != "NO_TRADE"
    ]
    unchanged_directional = sum(
        left.direction == right.direction for left, right in directional_pairs
    )
    wins = sum(value > 0 for value in effective_returns)
    no_trade_reasons = Counter(
        reason
        for session in sessions
        for reason in session.no_trade_reasons
    )
    return AssetBiasEvaluation(
        sessions=len(sessions),
        long_only_sessions=sum(session.direction == "LONG_ONLY" for session in sessions),
        short_only_sessions=sum(session.direction == "SHORT_ONLY" for session in sessions),
        no_trade_sessions=sum(session.direction == "NO_TRADE" for session in sessions),
        no_trade_reasons=dict(sorted(no_trade_reasons.items())),
        directional_sessions=len(directional),
        invalidated_sessions=len(invalidated),
        invalidation_rate_percent=_percentage(len(invalidated), len(directional)),
        outcome_sessions=len(effective_returns),
        missing_outcome_sessions=len(directional) - len(effective_returns),
        mean_effective_directional_return_percent=_mean(effective_returns),
        mean_full_day_directional_return_percent=_mean(full_day_returns),
        mean_market_return_24h_percent=_mean(market_returns),
        effective_directional_win_rate_percent=_percentage(wins, len(effective_returns)),
        post_invalidation_outcomes=len(post_returns),
        post_invalidation_adverse_percent=_percentage(
            sum(value <= 0 for value in post_returns),
            len(post_returns),
        ),
        mean_post_invalidation_directional_return_percent=_mean(post_returns),
        adjacent_session_comparisons=len(adjacent_pairs),
        unchanged_decision_pairs=unchanged_pairs,
        decision_stability_percent=_percentage(unchanged_pairs, len(adjacent_pairs)),
        adjacent_directional_comparisons=len(directional_pairs),
        unchanged_directional_pairs=unchanged_directional,
        directional_stability_percent=_percentage(
            unchanged_directional,
            len(directional_pairs),
        ),
    )


def run_chronological_replay(
    candles_by_asset: dict[Asset, list[ReplayCandle]],
    sentiments_by_asset: dict[Asset, list[SentimentObservation]],
    *,
    window_start: datetime,
    window_end: datetime,
    checked_at: datetime | None = None,
) -> BiasEvaluationReport:
    if (
        window_start.tzinfo is None
        or window_end.tzinfo is None
        or (checked_at is not None and checked_at.tzinfo is None)
    ):
        raise ValueError("Bias evaluation times must be timezone-aware")
    start = window_start.astimezone(UTC)
    end = window_end.astimezone(UTC)
    if start.time() != datetime.min.time() or end.time() != datetime.min.time():
        raise ValueError("Replay window bounds must be UTC midnight")
    days = (end - start).days
    if (
        days < MINIMUM_EVALUATION_DAYS
        or days > MAXIMUM_EVALUATION_DAYS
        or end - start != timedelta(days=days)
    ):
        raise ValueError(
            f"evaluation window must be {MINIMUM_EVALUATION_DAYS} to "
            f"{MAXIMUM_EVALUATION_DAYS} whole days"
        )

    results: dict[Asset, list[ReplayedSession]] = {
        asset: [] for asset in ASSET_SYMBOLS
    }
    session_start = start
    while session_start < end:
        for asset in ASSET_SYMBOLS:
            candles = sorted(candles_by_asset.get(asset, []), key=lambda bar: bar.open_time)
            observations = sentiments_by_asset.get(asset, [])
            initial_context = _session_context_at(
                asset,
                candles,
                observations,
                timestamp=session_start,
            )
            candidate = build_bias_candidate(initial_context)
            status = candidate.status
            invalidation_at = None
            if status == "active":
                for check_hour in INVALIDATION_CHECK_HOURS:
                    check_time = session_start + timedelta(hours=check_hour)
                    current_context = _session_context_at(
                        asset,
                        candles,
                        observations,
                        timestamp=check_time,
                    )
                    state = reconcile_locked_bias(
                        candidate.direction,
                        status,
                        current_context,
                    )
                    if state.status == "suspended":
                        status = state.status
                        invalidation_at = check_time
                        break
            outcome = _session_outcome(
                candidate.direction,
                candles,
                session_start=session_start,
                invalidation_at=invalidation_at,
            )
            results[asset].append(
                ReplayedSession(
                    asset=asset,
                    session_start=session_start,
                    direction=candidate.direction,
                    initial_regime=initial_context.regime.regime,
                    no_trade_reasons=(
                        tuple(candidate.evidence[3:])
                        if candidate.direction == "NO_TRADE"
                        else ()
                    ),
                    invalidation_at=invalidation_at,
                    market_return_24h_percent=outcome.market_return,
                    effective_directional_return_percent=(
                        outcome.effective_directional_return
                    ),
                    full_day_directional_return_percent=(
                        outcome.full_day_directional_return
                    ),
                    post_invalidation_directional_return_percent=(
                        outcome.post_invalidation_directional_return
                    ),
                )
            )
        session_start += timedelta(days=1)

    all_sessions = [session for asset in ASSET_SYMBOLS for session in results[asset]]
    regime_groups: dict[tuple[Asset, RegimeLabel, BiasDirection], list[ReplayedSession]] = (
        defaultdict(list)
    )
    for session in all_sessions:
        regime_groups[
            (session.asset, session.initial_regime, session.direction)
        ].append(session)

    by_asset_regime = []
    for (asset, regime, direction), group in sorted(
        regime_groups.items(),
        key=lambda item: (item[0][0], item[0][1], item[0][2]),
    ):
        directional = [row for row in group if row.direction != "NO_TRADE"]
        invalidated = [row for row in directional if row.invalidation_at is not None]
        effective_returns = [
            row.effective_directional_return_percent
            for row in directional
            if row.effective_directional_return_percent is not None
        ]
        full_day_returns = [
            row.full_day_directional_return_percent
            for row in directional
            if row.full_day_directional_return_percent is not None
        ]
        market_returns = [
            row.market_return_24h_percent
            for row in group
            if row.market_return_24h_percent is not None
        ]
        by_asset_regime.append(
            AssetRegimeOutcome(
                asset=asset,
                initial_regime=regime,
                direction=direction,
                sessions=len(group),
                directional_sessions=len(directional),
                invalidated_sessions=len(invalidated),
                outcome_sessions=len(effective_returns),
                mean_effective_directional_return_percent=_mean(effective_returns),
                mean_full_day_directional_return_percent=_mean(full_day_returns),
                mean_market_return_24h_percent=_mean(market_returns),
                effective_directional_win_rate_percent=_percentage(
                    sum(value > 0 for value in effective_returns),
                    len(effective_returns),
                ),
            )
        )

    return BiasEvaluationReport(
        algorithm_version="session-bias-v1",
        checked_at=(checked_at or datetime.now(UTC)).astimezone(UTC),
        window_start=start,
        window_end=end,
        evaluation_days=days,
        assessment_time_utc="00:00",
        invalidation_check_interval_hours=int(
            INVALIDATION_CHECK_INTERVAL.total_seconds() // 3600
        ),
        total_sessions=len(all_sessions),
        assets={
            asset: _summarize_asset(results[asset]) for asset in ASSET_SYMBOLS
        },
        by_asset_regime=by_asset_regime,
        evaluation_note=(
            "Chronological research replay only: one decision at 00:00 UTC and "
            "invalidation checks every 4 hours. Scored headlines are excluded until "
            "their observed, first-seen, and scoring times; outcomes use complete "
            "hourly candles. Decision stability includes NO_TRADE labels; directional "
            "stability excludes them. Returns are descriptive, omit fees/slippage/fills, "
            "and are not evidence of predictive accuracy or trading performance."
        ),
    )


async def evaluate_session_biases(
    database_engine: AsyncEngine,
    *,
    days: int = DEFAULT_EVALUATION_DAYS,
    now: datetime | None = None,
) -> BiasEvaluationReport:
    if days < MINIMUM_EVALUATION_DAYS or days > MAXIMUM_EVALUATION_DAYS:
        raise ValueError(
            f"days must be between {MINIMUM_EVALUATION_DAYS} and "
            f"{MAXIMUM_EVALUATION_DAYS}"
        )
    checked_at = now or datetime.now(UTC)
    if checked_at.tzinfo is None:
        raise ValueError("Bias evaluation time must be timezone-aware")
    checked_at = checked_at.astimezone(UTC)
    window_end = checked_at.replace(hour=0, minute=0, second=0, microsecond=0)
    window_start = window_end - timedelta(days=days)
    history_start = window_start - timedelta(days=8)
    symbols = [symbol_pair[0] for symbol_pair in ASSET_SYMBOLS.values()]
    symbol_to_asset = {
        symbol_pair[0]: asset for asset, symbol_pair in ASSET_SYMBOLS.items()
    }
    candles_by_asset: dict[Asset, list[ReplayCandle]] = {
        asset: [] for asset in ASSET_SYMBOLS
    }
    sentiments_by_asset: dict[Asset, list[SentimentObservation]] = {
        asset: [] for asset in ASSET_SYMBOLS
    }
    article_time = func.coalesce(NewsArticle.published_at, NewsArticle.first_seen_at)

    async with database_engine.connect() as connection:
        candle_result = await connection.execute(
            select(
                Candle.symbol,
                Candle.open_time,
                Candle.high,
                Candle.low,
                Candle.close,
                Candle.volume,
            )
            .where(
                Candle.venue == "delta",
                Candle.symbol.in_(symbols),
                Candle.resolution == "1h",
                Candle.open_time >= history_start,
                Candle.open_time < window_end,
            )
            .order_by(Candle.symbol, Candle.open_time)
        )
        for row in candle_result.all():
            candles_by_asset[symbol_to_asset[row[0]]].append(
                ReplayCandle(
                    open_time=row[1],
                    high=row[2],
                    low=row[3],
                    close=row[4],
                    volume=row[5],
                )
            )

        sentiment_result = await connection.execute(
            select(
                NewsSentiment.asset,
                NewsSentiment.article_id,
                NewsArticle.published_at,
                NewsArticle.first_seen_at,
                NewsSentiment.scored_at,
                NewsSentiment.weighted_sentiment,
                NewsSentiment.confidence,
                NewsSentiment.relevance_weight,
            )
            .select_from(NewsArticle)
            .join(NewsSentiment, NewsSentiment.article_id == NewsArticle.id)
            .where(
                NewsSentiment.asset.in_(tuple(ASSET_SYMBOLS)),
                NewsSentiment.model_revision == MODEL_REVISION,
                NewsSentiment.relevance_weight > 0,
                NewsArticle.first_seen_at < window_end,
                NewsSentiment.scored_at < window_end,
                article_time >= history_start,
                article_time < window_end,
            )
            .order_by(NewsSentiment.asset, article_time)
        )
        for row in sentiment_result.all():
            sentiments_by_asset[row[0]].append(
                SentimentObservation(
                    article_id=row[1],
                    published_at=row[2],
                    first_seen_at=row[3],
                    scored_at=row[4],
                    weighted_sentiment=row[5],
                    confidence=row[6],
                    relevance_weight=row[7],
                )
            )

    return run_chronological_replay(
        candles_by_asset,
        sentiments_by_asset,
        window_start=window_start,
        window_end=window_end,
        checked_at=checked_at,
    )
