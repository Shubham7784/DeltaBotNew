import uuid
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from app.market_data.bias_evaluation import (
    ReplayCandle,
    SentimentObservation,
    run_chronological_replay,
)

START = datetime(2026, 1, 1, tzinfo=UTC)


def make_hourly_history(
    start: datetime,
    end: datetime,
) -> list[ReplayCandle]:
    candles = []
    timestamp = start
    index = 0
    while timestamp < end:
        close = Decimal(str(100 * 1.001**index))
        candles.append(
            ReplayCandle(
                open_time=timestamp,
                high=close * Decimal("1.005"),
                low=close * Decimal("0.995"),
                close=close,
                volume=Decimal("100"),
            )
        )
        timestamp += timedelta(hours=1)
        index += 1
    return candles


def sentiment_item(
    timestamp: datetime,
    weighted_sentiment: str,
    *,
    scored_at: datetime | None = None,
) -> SentimentObservation:
    return SentimentObservation(
        article_id=uuid.uuid4(),
        published_at=timestamp,
        first_seen_at=timestamp,
        scored_at=scored_at or timestamp,
        weighted_sentiment=Decimal(weighted_sentiment),
        confidence=Decimal("0.8"),
        relevance_weight=Decimal("1"),
    )


def test_replay_is_chronological_and_reports_stability_and_invalidation() -> None:
    window_start = START + timedelta(days=9)
    window_end = window_start + timedelta(days=7)
    history = make_hourly_history(
        window_start - timedelta(days=8),
        window_end,
    )
    btc_news = []
    eth_news = []
    for day in range(7):
        session_start = window_start + timedelta(days=day)
        for offset in (1, 2, 3):
            btc_news.append(
                sentiment_item(
                    session_start - timedelta(hours=offset),
                    "0.25",
                )
            )
            eth_news.append(
                sentiment_item(
                    session_start - timedelta(hours=offset),
                    "0.25",
                    scored_at=(
                        session_start + timedelta(hours=1)
                        if day == 0
                        else session_start - timedelta(hours=offset)
                    ),
                )
            )
        if day == 0:
            for minute in (0, 5, 10):
                btc_news.append(
                    sentiment_item(
                        session_start + timedelta(hours=7, minutes=minute),
                        "-0.25",
                    )
                )

    report = run_chronological_replay(
        {"BTC": history, "ETH": history},
        {"BTC": btc_news, "ETH": eth_news},
        window_start=window_start,
        window_end=window_end,
        checked_at=window_end + timedelta(hours=1),
    )

    btc = report.assets["BTC"]
    eth = report.assets["ETH"]
    assert report.total_sessions == 14
    assert btc.directional_sessions == 6
    assert btc.invalidated_sessions == 1
    assert btc.post_invalidation_outcomes == 1
    assert btc.directional_stability_percent == 100
    assert eth.no_trade_sessions == 1
    assert eth.long_only_sessions == 6
    assert btc.long_only_sessions == 6
    assert eth.invalidated_sessions == 0
    assert any(
        "news direction insufficient" in reason
        for reason in eth.no_trade_reasons
    )
    assert btc.mean_effective_directional_return_percent is not None
    assert btc.mean_market_return_24h_percent is not None
    assert any(
        bucket.asset == "BTC" and bucket.initial_regime == "trend_up"
        for bucket in report.by_asset_regime
    )


def test_replay_requires_utc_midnight_and_bounded_whole_day_window() -> None:
    with pytest.raises(ValueError, match="UTC midnight"):
        run_chronological_replay(
            {"BTC": [], "ETH": []},
            {"BTC": [], "ETH": []},
            window_start=START + timedelta(hours=1),
            window_end=START + timedelta(days=7),
        )
