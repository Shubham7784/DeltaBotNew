from datetime import UTC, datetime, timedelta
from decimal import Decimal

from app.market_data.regimes import (
    DerivativeContext,
    HourlyCandle,
    classify_hourly_regime,
)


START = datetime(2026, 1, 1, tzinfo=UTC)


def make_candles(closes: list[float]) -> list[HourlyCandle]:
    return [
        HourlyCandle(
            open_time=START + timedelta(hours=index),
            close=Decimal(str(close)),
            volume=Decimal("10"),
        )
        for index, close in enumerate(closes)
    ]


def calculation_time(candle_count: int) -> datetime:
    return START + timedelta(hours=candle_count, minutes=30)


def test_classifies_consistent_uptrend_and_includes_context_only_derivatives() -> None:
    candles = make_candles([100 * 1.005**index for index in range(50)])
    result = classify_hourly_regime(
        "BTC",
        candles,
        derivatives=DerivativeContext(funding_rate=0.0001),
        computed_at=calculation_time(len(candles)),
    )

    assert result.regime == "trend_up"
    assert result.confidence > 0
    assert result.features.directional_consistency_12h == 1
    assert result.features.volume_ratio_24h_vs_prior_6d is None
    assert result.features.funding_rate == 0.0001
    assert "do not alter the regime label" in result.interpretation


def test_classifies_range_and_high_volatility() -> None:
    now = calculation_time(50)
    range_result = classify_hourly_regime(
        "ETH", make_candles([100.0] * 50), computed_at=now
    )
    alternating_closes = [100.0]
    for index in range(1, 50):
        alternating_closes.append(
            alternating_closes[-1] * (1.03 if index % 2 else 0.97)
        )
    volatility_result = classify_hourly_regime(
        "ETH", make_candles(alternating_closes), computed_at=now
    )

    assert range_result.regime == "range"
    assert volatility_result.regime == "high_volatility"
    assert (
        volatility_result.features.realized_volatility_24h_percent
        is not None
        and volatility_result.features.realized_volatility_24h_percent >= 5
    )


def test_unavailable_for_stale_or_discontinuous_completed_candles() -> None:
    candles = make_candles([100 * 1.001**index for index in range(51)])
    stale_result = classify_hourly_regime(
        "BTC",
        candles[:50],
        computed_at=calculation_time(50) + timedelta(hours=3),
    )
    gap_result = classify_hourly_regime(
        "BTC",
        candles[:20] + candles[21:],
        computed_at=calculation_time(51),
    )
    incomplete_latest_result = classify_hourly_regime(
        "BTC",
        candles,
        computed_at=calculation_time(50),
    )

    assert stale_result.regime == "unavailable"
    assert "stale" in stale_result.evidence[0]
    assert gap_result.regime == "unavailable"
    assert "gap" in gap_result.evidence[0]
    assert incomplete_latest_result.data_as_of == START + timedelta(hours=50)



def test_classifies_directional_breakdown_momentum_with_high_volatility() -> None:
    now = calculation_time(50)
    # Steep steady decline: -1% every hour
    steep_decline = [100.0 * (0.99**index) for index in range(50)]
    result = classify_hourly_regime("BTC", make_candles(steep_decline), computed_at=now)

    assert result.regime == "trend_down"
    assert result.features.ema_spread_percent is not None and result.features.ema_spread_percent < 0
    assert result.features.return_24h_percent is not None and result.features.return_24h_percent < 0
