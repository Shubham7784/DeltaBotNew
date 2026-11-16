from datetime import UTC, datetime, timedelta
from decimal import Decimal

from app.market_data.regimes import MarketRegimeSnapshot, RegimeFeatures
from app.market_data.session_bias import (
    HigherTimeframeContext,
    SessionBiasContext,
    build_bias_candidate,
    build_higher_timeframe_context,
    build_news_bias_context,
    reconcile_locked_bias,
)

START = datetime(2026, 1, 1, tzinfo=UTC)


def hourly_closes(count: int) -> list[tuple[datetime, Decimal]]:
    return [
        (
            START + timedelta(hours=index),
            Decimal(str(100 * 1.005**index)),
        )
        for index in range(count)
    ]


def make_context(
    *,
    regime: str = "trend_up",
    higher_trend: str = "trend_up",
    news_direction: str = "bullish",
) -> SessionBiasContext:
    now = START + timedelta(days=3)
    news = build_news_bias_context(
        "BTC",
        article_count=4,
        average_weighted_sentiment=0.25 if news_direction == "bullish" else -0.25,
        mean_confidence=0.8,
        as_of=now,
        window_start=now - timedelta(hours=24),
        window_end=now,
    )
    if news_direction != "bullish":
        news = news.model_copy(update={"direction": news_direction})
    higher_timeframe = HigherTimeframeContext(
        trend=higher_trend,
        ema_6=105.0,
        ema_12=100.0,
        ema_spread_percent=0.5,
        directional_consistency_3_bars=1.0,
        confidence=0.9,
        data_as_of=now,
        evidence=[],
    )
    regime_snapshot = MarketRegimeSnapshot(
        asset="BTC",
        regime=regime,
        confidence=0.8,
        algorithm_version="ohlcv-regime-v1",
        computed_at=now,
        data_as_of=now,
        features=RegimeFeatures(),
        evidence=[],
        invalidation_conditions=[],
        interpretation="test fixture",
    )
    return SessionBiasContext(
        assessed_at=now,
        regime=regime_snapshot,
        higher_timeframe=higher_timeframe,
        news=news,
    )


def test_higher_timeframe_requires_contiguous_completed_four_hour_bars() -> None:
    candles = hourly_closes(64)
    now = START + timedelta(hours=64, minutes=30)
    context = build_higher_timeframe_context("BTC", candles, computed_at=now)
    gap_context = build_higher_timeframe_context(
        "BTC",
        [candle for index, candle in enumerate(candles) if index != 55],
        computed_at=now,
    )
    stale_context = build_higher_timeframe_context(
        "BTC",
        candles,
        computed_at=now + timedelta(hours=9),
    )

    assert context.trend == "trend_up"
    assert context.data_as_of == START + timedelta(hours=64)
    assert gap_context.trend == "unavailable"
    assert "gap" in gap_context.evidence[0]
    assert stale_context.trend == "unavailable"
    assert "stale" in stale_context.evidence[0]


def test_news_direction_fails_closed_on_small_or_low_confidence_samples() -> None:
    window_end = START + timedelta(days=1)
    common = {
        "average_weighted_sentiment": 0.3,
        "as_of": window_end,
        "window_start": window_end - timedelta(hours=24),
        "window_end": window_end,
    }

    bullish = build_news_bias_context(
        "BTC", article_count=3, mean_confidence=0.7, **common
    )
    too_few = build_news_bias_context(
        "BTC", article_count=2, mean_confidence=0.9, **common
    )
    low_confidence = build_news_bias_context(
        "BTC", article_count=4, mean_confidence=0.5, **common
    )

    assert bullish.direction == "bullish"
    assert too_few.direction == "insufficient"
    assert low_confidence.direction == "insufficient"


def test_bias_requires_three_way_alignment_and_locks_no_trade_on_conflict() -> None:
    aligned = build_bias_candidate(make_context())
    conflicting = build_bias_candidate(
        make_context(higher_trend="trend_down")
    )
    bearish = build_bias_candidate(
        make_context(
            regime="trend_down",
            higher_trend="trend_down",
            news_direction="bearish",
        )
    )

    assert aligned.direction == "LONG_ONLY"
    assert aligned.status == "active"
    assert bearish.direction == "SHORT_ONLY"
    assert conflicting.direction == "NO_TRADE"
    assert conflicting.status == "no_trade"


def test_reassessment_suspends_without_reversing_or_reactivating_direction() -> None:
    invalidated = reconcile_locked_bias(
        "LONG_ONLY",
        "active",
        make_context(
            regime="trend_down",
            higher_trend="trend_down",
            news_direction="bearish",
        ),
    )
    later_recovery = reconcile_locked_bias(
        invalidated.direction,
        invalidated.status,
        make_context(),
        existing_reason=invalidated.suspension_reason,
    )

    assert invalidated.direction == "LONG_ONLY"
    assert invalidated.status == "suspended"
    assert invalidated.suspension_reason is not None
    assert later_recovery.direction == "LONG_ONLY"
    assert later_recovery.status == "suspended"
    assert later_recovery.suspension_reason == invalidated.suspension_reason
