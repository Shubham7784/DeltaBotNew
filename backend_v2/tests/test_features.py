from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from app.market_data.features import (
    CANDLE_FEATURE_ORDER,
    CANDLE_FEATURE_SCHEMA_VERSION,
    FEATURE_ORDER,
    EventWindow,
    FeatureCandle,
    FeatureInputs,
    MetricSample,
    aligned_as_of,
    build_market_feature_vector,
)

AS_OF = datetime(2026, 1, 1, 6, 0, tzinfo=UTC)
INTERVAL = timedelta(minutes=15)


def make_candle(open_time: datetime, close: int, volume: int) -> FeatureCandle:
    return FeatureCandle(
        open_time=open_time,
        open=Decimal(close - 1),
        high=Decimal(close + 1),
        low=Decimal(close - 2),
        close=Decimal(close),
        volume=Decimal(volume),
    )


def make_inputs() -> FeatureInputs:
    candles = [
        make_candle(AS_OF - INTERVAL * index, 101 + 30 - index, 10 + index)
        for index in range(30, 0, -1)
    ]
    hourly = {
        "BTC": [
            make_candle(AS_OF - timedelta(hours=2), 100, 100),
            make_candle(AS_OF - timedelta(hours=1), 105, 110),
        ],
        "ETH": [
            make_candle(AS_OF - timedelta(hours=2), 200, 200),
            make_candle(AS_OF - timedelta(hours=1), 204, 210),
        ],
    }
    latest_metrics = {
        ("BTCUSD", "funding_rate"): MetricSample(
            Decimal("0.0001"), AS_OF - timedelta(minutes=1)
        ),
        ("BTCUSD", "open_interest"): MetricSample(
            Decimal("100"), AS_OF - timedelta(seconds=10)
        ),
        ("BTCUSD", "mark_price"): MetricSample(
            Decimal("102"), AS_OF - timedelta(seconds=5)
        ),
        (".DEXBTUSD", "index_price"): MetricSample(
            Decimal("101"), AS_OF - timedelta(seconds=5)
        ),
    }
    baseline_metrics = {
        ("BTCUSD", "open_interest"): MetricSample(
            Decimal("95"), AS_OF - INTERVAL - timedelta(seconds=30)
        ),
        (".DEXBTUSD", "index_price"): MetricSample(
            Decimal("100"), AS_OF - INTERVAL - timedelta(seconds=30)
        ),
    }
    window_start = AS_OF - ORDER_FLOW_WINDOW
    order_book = EventWindow(
        sample_count=100,
        first_at=window_start + timedelta(seconds=1),
        last_at=AS_OF - timedelta(seconds=1),
        first_payload={"bp": "100", "ap": "101", "bs": "10", "as": "5"},
        last_payload={"bp": "101", "ap": "102", "bs": "12", "as": "4"},
    )
    trades = EventWindow(
        sample_count=2,
        first_at=window_start + timedelta(seconds=1),
        last_at=AS_OF - timedelta(seconds=1),
        samples=(
            (window_start + timedelta(seconds=1), {"s": "2.5", "r": "m"}),
            (AS_OF - timedelta(seconds=1), {"s": "1.5", "r": "t"}),
        ),
    )
    return FeatureInputs(
        asset="BTC",
        resolution="15m",
        as_of=AS_OF,
        computed_at=AS_OF + timedelta(minutes=1),
        candles=candles,
        hourly_candles=hourly,
        latest_metrics=latest_metrics,
        baseline_metrics=baseline_metrics,
        order_book=order_book,
        trades=trades,
    )


ORDER_FLOW_WINDOW = timedelta(minutes=15)


def test_feature_vector_has_fixed_order_provenance_and_expected_values() -> None:
    result = build_market_feature_vector(make_inputs())
    features = {feature.name: feature for feature in result.features}

    assert result.schema_version == "shared-market-features-v1"
    assert result.feature_order == list(FEATURE_ORDER)
    assert [feature.name for feature in result.features] == list(FEATURE_ORDER)
    assert features["close_price"].value == 130
    assert features["return_1_period_pct"].value == pytest.approx(100 / 129)
    assert features["top_book_imbalance"].value == pytest.approx(0.5)
    assert features["trade_volume_15m_contracts"].value == 4
    assert features["open_interest_change_15m_pct"].value == pytest.approx(
        (100 / 95 - 1) * 100
    )
    assert features["mark_index_basis_pct"].value == pytest.approx(
        (102 / 101 - 1) * 100
    )
    assert features["btc_minus_eth_return_1h_pct"].value == pytest.approx(3)
    assert all(
        feature.source_as_of is None or feature.source_as_of <= result.as_of
        for feature in result.features
    )
    assert "unsigned" in result.interpretation


def test_incomplete_future_candle_is_ignored_and_a_gap_is_not_filled() -> None:
    inputs = make_inputs()
    future_candle = make_candle(AS_OF, 10_000, 99_999)
    with_future = build_market_feature_vector(
        replace(inputs, candles=[*inputs.candles, future_candle])
    )
    without_future = build_market_feature_vector(inputs)
    assert with_future.features[0].value == without_future.features[0].value

    candles_with_gap = inputs.candles[:20] + inputs.candles[21:]
    with_gap = build_market_feature_vector(replace(inputs, candles=candles_with_gap))
    features = {feature.name: feature for feature in with_gap.features}
    assert features["close_price"].status == "available"
    assert features["return_12_period_pct"].status == "insufficient_data"
    assert features["return_12_period_pct"].value is None
    assert features["return_12_period_pct"].sample_count == 13


def test_partial_or_malformed_order_flow_is_not_filled_with_zero() -> None:
    inputs = make_inputs()
    partial_book = replace(
        inputs.order_book,
        first_at=AS_OF - timedelta(minutes=10),
    )
    partial_trades = replace(
        inputs.trades,
        samples=((AS_OF - timedelta(seconds=1), {"s": "not-a-number"}),),
    )
    partial = build_market_feature_vector(
        replace(inputs, order_book=partial_book, trades=partial_trades)
    )
    features = {feature.name: feature for feature in partial.features}

    assert features["bid_size_change_15m_pct"].status == "insufficient_data"
    assert features["trade_count_15m"].status == "available"
    assert features["trade_volume_15m_contracts"].status == "invalid_data"
    assert features["trade_volume_15m_contracts"].value is None


def test_as_of_requires_timezone_utc_boundary_and_not_future_time() -> None:
    with pytest.raises(ValueError, match="timezone"):
        aligned_as_of(resolution="15m", now=AS_OF, requested=AS_OF.replace(tzinfo=None))
    with pytest.raises(ValueError, match="boundary"):
        aligned_as_of(
            resolution="15m",
            now=AS_OF + timedelta(hours=1),
            requested=AS_OF + timedelta(minutes=5),
        )
    with pytest.raises(ValueError, match="future"):
        aligned_as_of(
            resolution="15m",
            now=AS_OF,
            requested=AS_OF + timedelta(minutes=15),
        )


def test_candle_feature_schema_vector() -> None:
    inputs = replace(make_inputs(), schema_version=CANDLE_FEATURE_SCHEMA_VERSION)
    result = build_market_feature_vector(inputs)
    assert result.schema_version == CANDLE_FEATURE_SCHEMA_VERSION
    assert result.feature_order == list(CANDLE_FEATURE_ORDER)
    assert len(result.features) == len(CANDLE_FEATURE_ORDER) == 18
    features = {f.name: f for f in result.features}
    assert features["close_price"].value == 130
    assert features["btc_minus_eth_return_1h_pct"].value == pytest.approx(3)
    assert all(f.status == "available" for f in result.features)
