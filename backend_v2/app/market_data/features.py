import math
import statistics
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation
from typing import AsyncGenerator, Literal

from pydantic import BaseModel, field_validator, model_validator
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine

from app.market_data.backfill import RESOLUTION_SECONDS
from app.market_data.regimes import ASSET_SYMBOLS, Asset
from app.models import Candle, MarketEvent, MarketMetric

FeatureFamily = Literal[
    "price",
    "trend",
    "momentum",
    "volatility",
    "volume",
    "order_flow",
    "derivatives",
    "market_context",
]
FeatureStatus = Literal[
    "available",
    "insufficient_data",
    "missing",
    "stale",
    "invalid_data",
]
FeatureResolution = Literal["5m", "15m", "1h"]

FEATURE_SCHEMA_VERSION = "shared-market-features-v1"
MAX_CANDLE_HISTORY = 200
ORDER_FLOW_WINDOW = timedelta(minutes=15)
ORDER_FLOW_FRESHNESS = timedelta(minutes=2)
METRIC_FRESHNESS = timedelta(minutes=2)
MAX_METRIC_SKEW = timedelta(minutes=1)
FUNDING_FRESHNESS = timedelta(hours=12)
MAX_EVENT_WINDOW_EDGE_AGE = timedelta(minutes=1)


@dataclass(frozen=True)
class FeatureDefinition:
    name: str
    family: FeatureFamily
    unit: str
    source: str


FEATURE_DEFINITIONS = (
    FeatureDefinition("close_price", "price", "quote_price", "candles:delta"),
    FeatureDefinition("return_1_period_pct", "price", "percent", "candles:delta"),
    FeatureDefinition(
        "ema_8_21_spread_pct", "trend", "percent", "candles:delta"
    ),
    FeatureDefinition(
        "close_vs_ema_21_pct", "trend", "percent", "candles:delta"
    ),
    FeatureDefinition(
        "trend_consistency_12", "trend", "fraction", "candles:delta"
    ),
    FeatureDefinition("return_4_period_pct", "momentum", "percent", "candles:delta"),
    FeatureDefinition("return_12_period_pct", "momentum", "percent", "candles:delta"),
    FeatureDefinition("rsi_14", "momentum", "index_0_100", "candles:delta"),
    FeatureDefinition(
        "realized_volatility_12_pct",
        "volatility",
        "percent",
        "candles:delta",
    ),
    FeatureDefinition(
        "realized_volatility_24_pct",
        "volatility",
        "percent",
        "candles:delta",
    ),
    FeatureDefinition("atr_14_pct", "volatility", "percent", "candles:delta"),
    FeatureDefinition("current_volume", "volume", "contracts", "candles:delta"),
    FeatureDefinition("volume_sma_20", "volume", "contracts", "candles:delta"),
    FeatureDefinition("volume_ratio_20", "volume", "ratio", "candles:delta"),
    FeatureDefinition("volume_zscore_20", "volume", "zscore", "candles:delta"),
    FeatureDefinition(
        "top_book_imbalance",
        "order_flow",
        "fraction",
        "market_events:ob_l1",
    ),
    FeatureDefinition(
        "top_book_spread_bps",
        "order_flow",
        "basis_points",
        "market_events:ob_l1",
    ),
    FeatureDefinition(
        "bid_size_change_15m_pct",
        "order_flow",
        "percent",
        "market_events:ob_l1",
    ),
    FeatureDefinition(
        "ask_size_change_15m_pct",
        "order_flow",
        "percent",
        "market_events:ob_l1",
    ),
    FeatureDefinition(
        "trade_count_15m",
        "order_flow",
        "events",
        "market_events:trades",
    ),
    FeatureDefinition(
        "trade_volume_15m_contracts",
        "order_flow",
        "contracts",
        "market_events:trades",
    ),
    FeatureDefinition(
        "funding_rate",
        "derivatives",
        "exchange_rate",
        "market_metrics:funding_rate",
    ),
    FeatureDefinition(
        "open_interest",
        "derivatives",
        "contracts",
        "market_metrics:open_interest",
    ),
    FeatureDefinition(
        "open_interest_change_15m_pct",
        "derivatives",
        "percent",
        "market_metrics:open_interest",
    ),
    FeatureDefinition(
        "mark_index_basis_pct",
        "derivatives",
        "percent",
        "market_metrics:mark_price+index_price",
    ),
    FeatureDefinition(
        "index_return_15m_pct",
        "market_context",
        "percent",
        "market_metrics:index_price",
    ),
    FeatureDefinition(
        "btc_return_1h_pct",
        "market_context",
        "percent",
        "candles:delta:BTCUSD:1h",
    ),
    FeatureDefinition(
        "eth_return_1h_pct",
        "market_context",
        "percent",
        "candles:delta:ETHUSD:1h",
    ),
    FeatureDefinition(
        "btc_minus_eth_return_1h_pct",
        "market_context",
        "percentage_points",
        "candles:delta:BTCUSD+ETHUSD:1h",
    ),
)
FEATURE_ORDER = tuple(definition.name for definition in FEATURE_DEFINITIONS)
FEATURE_DEFINITIONS_BY_NAME = {
    definition.name: definition for definition in FEATURE_DEFINITIONS
}
CANDLE_FEATURE_SCHEMA_VERSION = "candle-market-features-v1"
CANDLE_FEATURE_DEFINITIONS = tuple(
    definition for definition in FEATURE_DEFINITIONS if definition.source.startswith("candles:")
)
CANDLE_FEATURE_ORDER = tuple(definition.name for definition in CANDLE_FEATURE_DEFINITIONS)
FEATURE_SCHEMAS: dict[str, tuple[str, ...]] = {
    FEATURE_SCHEMA_VERSION: FEATURE_ORDER,
    CANDLE_FEATURE_SCHEMA_VERSION: CANDLE_FEATURE_ORDER,
}


class FeatureValue(BaseModel):
    name: str
    family: FeatureFamily
    value: float | None
    unit: str
    source: str
    source_as_of: datetime | None
    window_start: datetime | None
    window_end: datetime
    sample_count: int
    status: FeatureStatus
    missing_reason: str | None

    @field_validator("value")
    @classmethod
    def value_must_be_finite(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("Feature values must be finite")
        return value

    @model_validator(mode="after")
    def status_matches_value(self) -> "FeatureValue":
        if self.status == "available" and self.value is None:
            raise ValueError("Available features must have a value")
        if self.status != "available" and self.value is not None:
            raise ValueError("Unavailable features must not have a fallback value")
        if self.status == "available" and self.missing_reason is not None:
            raise ValueError("Available features must not have a missing reason")
        if self.status != "available" and not self.missing_reason:
            raise ValueError("Unavailable features require a reason")
        return self


class MarketFeatureVector(BaseModel):
    schema_version: str
    asset: Asset
    symbol: str
    resolution: FeatureResolution
    as_of: datetime
    computed_at: datetime
    feature_order: list[str]
    features: list[FeatureValue]
    interpretation: str

    @model_validator(mode="after")
    def features_follow_schema_order(self) -> "MarketFeatureVector":
        expected_order = FEATURE_SCHEMAS.get(self.schema_version, FEATURE_ORDER)
        actual_order = [feature.name for feature in self.features]
        if self.feature_order != list(expected_order) or actual_order != self.feature_order:
            raise ValueError(f"Feature vector does not match the declared feature order for {self.schema_version}")
        if any(
            feature.source_as_of is not None
            and feature.source_as_of > self.as_of
            for feature in self.features
        ):
            raise ValueError("Feature provenance cannot be later than as_of")
        return self


@dataclass(frozen=True)
class FeatureCandle:
    open_time: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal


@dataclass(frozen=True)
class MetricSample:
    value: Decimal
    event_time: datetime


@dataclass(frozen=True)
class EventWindow:
    sample_count: int
    first_at: datetime | None
    last_at: datetime | None
    first_payload: dict[str, object] | None = None
    last_payload: dict[str, object] | None = None
    samples: tuple[tuple[datetime, dict[str, object]], ...] = ()


@dataclass(frozen=True)
class FeatureInputs:
    asset: Asset
    resolution: FeatureResolution
    as_of: datetime
    computed_at: datetime
    candles: list[FeatureCandle]
    hourly_candles: dict[Asset, list[FeatureCandle]]
    latest_metrics: dict[tuple[str, str], MetricSample] = field(default_factory=dict)
    baseline_metrics: dict[tuple[str, str], MetricSample] = field(default_factory=dict)
    order_book: EventWindow = field(
        default_factory=lambda: EventWindow(sample_count=0, first_at=None, last_at=None)
    )
    trades: EventWindow = field(
        default_factory=lambda: EventWindow(sample_count=0, first_at=None, last_at=None)
    )
    schema_version: str = FEATURE_SCHEMA_VERSION


def aligned_as_of(
    *,
    resolution: FeatureResolution,
    now: datetime | None = None,
    requested: datetime | None = None,
) -> datetime:
    timestamp = now or datetime.now(UTC)
    if timestamp.tzinfo is None:
        raise ValueError("Feature computation time must be timezone-aware")
    timestamp = timestamp.astimezone(UTC)
    duration = RESOLUTION_SECONDS[resolution]
    if requested is None:
        epoch = int(timestamp.timestamp())
        return datetime.fromtimestamp(epoch - epoch % duration, UTC)
    if requested.tzinfo is None:
        raise ValueError("Feature as_of must include a timezone")
    as_of = requested.astimezone(UTC)
    if as_of > timestamp:
        raise ValueError("Feature as_of cannot be in the future")
    if int(as_of.timestamp()) % duration != 0 or as_of.microsecond:
        raise ValueError(f"Feature as_of must align to the {resolution} UTC boundary")
    return as_of


def _ema(values: list[float], period: int) -> float:
    initial = sum(values[:period]) / period
    alpha = 2 / (period + 1)
    for value in values[period:]:
        initial = alpha * value + (1 - alpha) * initial
    return initial


def _feature(
    name: str,
    value: float | None,
    *,
    as_of: datetime,
    source_as_of: datetime | None,
    window_start: datetime | None,
    sample_count: int,
    status: FeatureStatus = "available",
    reason: str | None = None,
) -> FeatureValue:
    definition = FEATURE_DEFINITIONS_BY_NAME[name]
    return FeatureValue(
        name=name,
        family=definition.family,
        value=value,
        unit=definition.unit,
        source=definition.source,
        source_as_of=source_as_of,
        window_start=window_start,
        window_end=as_of,
        sample_count=sample_count,
        status=status,
        missing_reason=reason,
    )


def _unavailable_feature(
    name: str,
    *,
    as_of: datetime,
    reason: str,
    status: FeatureStatus = "insufficient_data",
    source_as_of: datetime | None = None,
    window_start: datetime | None = None,
    sample_count: int = 0,
) -> FeatureValue:
    return _feature(
        name,
        None,
        as_of=as_of,
        source_as_of=source_as_of,
        window_start=window_start,
        sample_count=sample_count,
        status=status,
        reason=reason,
    )


def _complete_tail(
    candles: list[FeatureCandle],
    *,
    count: int,
    resolution_seconds: int,
    as_of: datetime,
) -> tuple[list[FeatureCandle] | None, FeatureStatus, str | None]:
    if not candles:
        return None, "missing", "No candles are stored for this resolution."
    interval = timedelta(seconds=resolution_seconds)
    latest_close_time = candles[-1].open_time + interval
    if latest_close_time != as_of:
        return (
            None,
            "stale",
            "The latest stored candle does not close exactly at as_of.",
        )
    if len(candles) < count:
        return (
            None,
            "insufficient_data",
            f"{count} completed candles are required; only {len(candles)} are available.",
        )
    sample = candles[-count:]
    expected_open_times = [
        as_of - interval * offset for offset in range(count, 0, -1)
    ]
    if [candle.open_time for candle in sample] != expected_open_times:
        return (
            None,
            "insufficient_data",
            f"The required {count}-candle window is not contiguous.",
        )
    return sample, "available", None


def _add_price_features(inputs: FeatureInputs, values: dict[str, FeatureValue]) -> None:
    duration = RESOLUTION_SECONDS[inputs.resolution]
    as_of = inputs.as_of
    candles = sorted(
        (
            candle
            for candle in inputs.candles
            if candle.open_time + timedelta(seconds=duration) <= as_of
        ),
        key=lambda candle: candle.open_time,
    )
    latest_source_time = (
        candles[-1].open_time + timedelta(seconds=duration)
        if candles
        else None
    )
    interval = timedelta(seconds=duration)

    def sample(
        name: str,
        count: int,
    ) -> list[FeatureCandle] | None:
        candidate_tail = candles[-count:]
        result, status, reason = _complete_tail(
            candles,
            count=count,
            resolution_seconds=duration,
            as_of=as_of,
        )
        if result is None:
            values[name] = _unavailable_feature(
                name,
                as_of=as_of,
                reason=reason or "Price feature is unavailable.",
                status=status,
                source_as_of=latest_source_time,
                window_start=(
                    candidate_tail[0].open_time
                    if candidate_tail
                    else as_of - interval * count
                ),
                sample_count=len(candidate_tail),
            )
        return result

    def put(
        name: str,
        value: float,
        candles: list[FeatureCandle],
    ) -> None:
        if not math.isfinite(value):
            values[name] = _unavailable_feature(
                name,
                as_of=as_of,
                reason="Calculation produced a non-finite value.",
                status="invalid_data",
                source_as_of=latest_source_time,
                window_start=candles[0].open_time,
                sample_count=len(candles),
            )
            return
        values[name] = _feature(
            name,
            value,
            as_of=as_of,
            source_as_of=latest_source_time,
            window_start=candles[0].open_time,
            sample_count=len(candles),
        )

    close_bars = sample("close_price", 1)
    if close_bars is not None:
        close = float(close_bars[-1].close)
        if close > 0:
            put("close_price", close, close_bars)
        else:
            values["close_price"] = _unavailable_feature(
                "close_price",
                as_of=as_of,
                reason="Close price must be positive.",
                status="invalid_data",
                source_as_of=latest_source_time,
                window_start=close_bars[0].open_time,
                sample_count=1,
            )

    for feature_name, period in (
        ("return_1_period_pct", 1),
        ("return_4_period_pct", 4),
        ("return_12_period_pct", 12),
    ):
        bars = sample(feature_name, period + 1)
        if bars is not None:
            start_close, end_close = float(bars[0].close), float(bars[-1].close)
            if start_close > 0:
                put(feature_name, (end_close / start_close - 1) * 100, bars)
            else:
                values[feature_name] = _unavailable_feature(
                    feature_name,
                    as_of=as_of,
                    reason="Return denominator must be positive.",
                    status="invalid_data",
                    source_as_of=latest_source_time,
                    window_start=bars[0].open_time,
                    sample_count=len(bars),
                )

    trend_bars = sample("ema_8_21_spread_pct", 21)
    if trend_bars is not None:
        closes = [float(candle.close) for candle in trend_bars]
        if min(closes) > 0:
            ema_8 = _ema(closes, 8)
            ema_21 = _ema(closes, 21)
            put("ema_8_21_spread_pct", (ema_8 / ema_21 - 1) * 100, trend_bars)
        else:
            values["ema_8_21_spread_pct"] = _unavailable_feature(
                "ema_8_21_spread_pct",
                as_of=as_of,
                reason="EMA inputs must be positive.",
                status="invalid_data",
                source_as_of=latest_source_time,
                window_start=trend_bars[0].open_time,
                sample_count=len(trend_bars),
            )

    close_ema_bars = sample("close_vs_ema_21_pct", 21)
    if close_ema_bars is not None:
        closes = [float(candle.close) for candle in close_ema_bars]
        ema_21 = _ema(closes, 21)
        if ema_21 > 0:
            put("close_vs_ema_21_pct", (closes[-1] / ema_21 - 1) * 100, close_ema_bars)
        else:
            values["close_vs_ema_21_pct"] = _unavailable_feature(
                "close_vs_ema_21_pct",
                as_of=as_of,
                reason="EMA denominator must be positive.",
                status="invalid_data",
                source_as_of=latest_source_time,
                window_start=close_ema_bars[0].open_time,
                sample_count=len(close_ema_bars),
            )

    consistency_bars = sample("trend_consistency_12", 13)
    if consistency_bars is not None:
        closes = [float(candle.close) for candle in consistency_bars]
        full_trend = sample("ema_8_21_spread_pct", 21)
        if full_trend is None:
            values["trend_consistency_12"] = _unavailable_feature(
                "trend_consistency_12",
                as_of=as_of,
                reason="A contiguous 21-candle EMA trend is required.",
                source_as_of=latest_source_time,
                window_start=consistency_bars[0].open_time,
                sample_count=len(consistency_bars),
            )
        else:
            ema_closes = [float(candle.close) for candle in full_trend]
            spread = _ema(ema_closes, 8) / _ema(ema_closes, 21) - 1
            direction = 1 if spread > 0 else -1 if spread < 0 else 0
            returns = [
                closes[index] / closes[index - 1] - 1
                for index in range(1, len(closes))
            ]
            consistency = (
                sum(value * direction > 0 for value in returns) / len(returns)
                if direction
                else 0.5
            )
            put("trend_consistency_12", consistency, consistency_bars)

    rsi_bars = sample("rsi_14", 15)
    if rsi_bars is not None:
        closes = [float(candle.close) for candle in rsi_bars]
        changes = [later - earlier for earlier, later in zip(closes, closes[1:])]
        gains = [max(change, 0.0) for change in changes]
        losses = [max(-change, 0.0) for change in changes]
        average_gain = statistics.fmean(gains)
        average_loss = statistics.fmean(losses)
        rsi = (
            50.0
            if average_gain == 0 and average_loss == 0
            else 100.0
            if average_loss == 0
            else 100 - 100 / (1 + average_gain / average_loss)
        )
        put("rsi_14", rsi, rsi_bars)

    for feature_name, periods in (
        ("realized_volatility_12_pct", 12),
        ("realized_volatility_24_pct", 24),
    ):
        bars = sample(feature_name, periods + 1)
        if bars is not None:
            closes = [float(candle.close) for candle in bars]
            if min(closes) > 0:
                log_returns = [
                    math.log(closes[index] / closes[index - 1])
                    for index in range(1, len(closes))
                ]
                volatility = (
                    statistics.stdev(log_returns) * math.sqrt(periods) * 100
                    if len(log_returns) > 1
                    else 0.0
                )
                put(feature_name, volatility, bars)
            else:
                values[feature_name] = _unavailable_feature(
                    feature_name,
                    as_of=as_of,
                    reason="Volatility inputs must be positive.",
                    status="invalid_data",
                    source_as_of=latest_source_time,
                    window_start=bars[0].open_time,
                    sample_count=len(bars),
                )

    atr_bars = sample("atr_14_pct", 15)
    if atr_bars is not None:
        true_ranges = []
        for previous, current in zip(atr_bars, atr_bars[1:]):
            prior_close = float(previous.close)
            true_ranges.append(
                max(
                    float(current.high) - float(current.low),
                    abs(float(current.high) - prior_close),
                    abs(float(current.low) - prior_close),
                )
            )
        last_close = float(atr_bars[-1].close)
        if last_close > 0:
            put(
                "atr_14_pct",
                statistics.fmean(true_ranges) / last_close * 100,
                atr_bars,
            )
        else:
            values["atr_14_pct"] = _unavailable_feature(
                "atr_14_pct",
                as_of=as_of,
                reason="ATR denominator must be positive.",
                status="invalid_data",
                source_as_of=latest_source_time,
                window_start=atr_bars[0].open_time,
                sample_count=len(atr_bars),
            )

    for feature_name, count in (
        ("current_volume", 1),
        ("volume_sma_20", 20),
        ("volume_ratio_20", 21),
        ("volume_zscore_20", 21),
    ):
        bars = sample(feature_name, count)
        if bars is None:
            continue
        volumes = [float(candle.volume) for candle in bars]
        if any(volume < 0 or not math.isfinite(volume) for volume in volumes):
            values[feature_name] = _unavailable_feature(
                feature_name,
                as_of=as_of,
                reason="Volume inputs must be finite and nonnegative.",
                status="invalid_data",
                source_as_of=latest_source_time,
                window_start=bars[0].open_time,
                sample_count=len(bars),
            )
            continue
        if feature_name == "current_volume":
            value = volumes[-1]
        elif feature_name == "volume_sma_20":
            value = statistics.fmean(volumes)
        else:
            baseline = volumes[:-1]
            baseline_mean = statistics.fmean(baseline)
            if feature_name == "volume_ratio_20":
                if baseline_mean <= 0:
                    values[feature_name] = _unavailable_feature(
                        feature_name,
                        as_of=as_of,
                        reason="Prior 20-candle mean volume must be positive.",
                        status="invalid_data",
                        source_as_of=latest_source_time,
                        window_start=bars[0].open_time,
                        sample_count=len(bars),
                    )
                    continue
                value = volumes[-1] / baseline_mean
            else:
                baseline_std = statistics.stdev(baseline)
                if baseline_std == 0:
                    values[feature_name] = _unavailable_feature(
                        feature_name,
                        as_of=as_of,
                        reason="Prior 20-candle volume standard deviation is zero.",
                        status="invalid_data",
                        source_as_of=latest_source_time,
                        window_start=bars[0].open_time,
                        sample_count=len(bars),
                    )
                    continue
                value = (volumes[-1] - baseline_mean) / baseline_std
        put(feature_name, value, bars)


def _decimal_field(payload: dict[str, object], key: str) -> tuple[Decimal | None, str | None]:
    raw_value = payload.get(key)
    if raw_value is None:
        return None, f"Payload is missing {key!r}."
    try:
        value = Decimal(str(raw_value))
    except (InvalidOperation, ValueError):
        return None, f"Payload field {key!r} is not a decimal."
    if not value.is_finite():
        return None, f"Payload field {key!r} is non-finite."
    return value, None


def _event_window_complete(window: EventWindow, *, as_of: datetime) -> bool:
    start = as_of - ORDER_FLOW_WINDOW
    return (
        window.sample_count > 0
        and window.first_at is not None
        and window.last_at is not None
        and window.first_at <= as_of
        and window.last_at <= as_of
        and window.first_at <= start + MAX_EVENT_WINDOW_EDGE_AGE
        and window.last_at >= as_of - MAX_EVENT_WINDOW_EDGE_AGE
    )


def _add_order_flow_features(
    inputs: FeatureInputs,
    values: dict[str, FeatureValue],
) -> None:
    as_of = inputs.as_of
    book = inputs.order_book
    book_fresh = (
        book.last_at is not None
        and book.last_at <= as_of
        and as_of - book.last_at <= ORDER_FLOW_FRESHNESS
    )
    if book.last_payload is None or book.last_at is None:
        for name in ("top_book_imbalance", "top_book_spread_bps"):
            values[name] = _unavailable_feature(
                name,
                as_of=as_of,
                reason="No top-of-book snapshot is stored in the event window.",
                status="missing",
                window_start=as_of - ORDER_FLOW_WINDOW,
                sample_count=book.sample_count,
            )
    elif not book_fresh:
        for name in ("top_book_imbalance", "top_book_spread_bps"):
            values[name] = _unavailable_feature(
                name,
                as_of=as_of,
                reason="Latest top-of-book snapshot is more than 2 minutes old.",
                status="stale",
                source_as_of=book.last_at,
                window_start=book.last_at,
                sample_count=book.sample_count,
            )
    else:
        bid_price, bid_price_error = _decimal_field(book.last_payload, "bp")
        ask_price, ask_price_error = _decimal_field(book.last_payload, "ap")
        bid_size, bid_size_error = _decimal_field(book.last_payload, "bs")
        ask_size, ask_size_error = _decimal_field(book.last_payload, "as")
        errors = [
            error
            for error in (
                bid_price_error,
                ask_price_error,
                bid_size_error,
                ask_size_error,
            )
            if error is not None
        ]
        if errors:
            for name in ("top_book_imbalance", "top_book_spread_bps"):
                values[name] = _unavailable_feature(
                    name,
                    as_of=as_of,
                    reason=" ".join(errors),
                    status="invalid_data",
                    source_as_of=book.last_at,
                    window_start=book.last_at,
                    sample_count=book.sample_count,
                )
        elif (
            bid_price is None
            or ask_price is None
            or bid_size is None
            or ask_size is None
            or bid_price <= 0
            or ask_price <= 0
            or ask_price < bid_price
            or bid_size < 0
            or ask_size < 0
            or bid_size + ask_size <= 0
        ):
            for name in ("top_book_imbalance", "top_book_spread_bps"):
                values[name] = _unavailable_feature(
                    name,
                    as_of=as_of,
                    reason="Top-of-book prices or sizes violate expected bounds.",
                    status="invalid_data",
                    source_as_of=book.last_at,
                    window_start=book.last_at,
                    sample_count=book.sample_count,
                )
        else:
            values["top_book_imbalance"] = _feature(
                "top_book_imbalance",
                float((bid_size - ask_size) / (bid_size + ask_size)),
                as_of=as_of,
                source_as_of=book.last_at,
                window_start=book.last_at,
                sample_count=book.sample_count,
            )
            midpoint = (bid_price + ask_price) / 2
            values["top_book_spread_bps"] = _feature(
                "top_book_spread_bps",
                float((ask_price - bid_price) / midpoint * Decimal(10_000)),
                as_of=as_of,
                source_as_of=book.last_at,
                window_start=book.last_at,
                sample_count=book.sample_count,
            )

    complete_book_window = _event_window_complete(book, as_of=as_of)
    for name, payload_key in (
        ("bid_size_change_15m_pct", "bs"),
        ("ask_size_change_15m_pct", "as"),
    ):
        if not complete_book_window:
            values[name] = _unavailable_feature(
                name,
                as_of=as_of,
                reason="A complete 15-minute top-of-book event window is unavailable.",
                status="insufficient_data",
                source_as_of=book.last_at,
                window_start=as_of - ORDER_FLOW_WINDOW,
                sample_count=book.sample_count,
            )
            continue
        first_payload = book.first_payload or {}
        last_payload = book.last_payload or {}
        first_size, first_error = _decimal_field(first_payload, payload_key)
        last_size, last_error = _decimal_field(last_payload, payload_key)
        if first_error or last_error or first_size is None or last_size is None:
            values[name] = _unavailable_feature(
                name,
                as_of=as_of,
                reason=first_error or last_error or "Book sizes are missing.",
                status="invalid_data",
                source_as_of=book.last_at,
                window_start=book.first_at,
                sample_count=book.sample_count,
            )
        elif first_size <= 0 or last_size < 0:
            values[name] = _unavailable_feature(
                name,
                as_of=as_of,
                reason="Initial book size must be positive and latest size nonnegative.",
                status="invalid_data",
                source_as_of=book.last_at,
                window_start=book.first_at,
                sample_count=book.sample_count,
            )
        else:
            values[name] = _feature(
                name,
                float((last_size / first_size - 1) * 100),
                as_of=as_of,
                source_as_of=book.last_at,
                window_start=book.first_at,
                sample_count=book.sample_count,
            )

    trades = inputs.trades
    if not _event_window_complete(trades, as_of=as_of):
        reason = "A complete 15-minute trade-event window is unavailable."
        for name in ("trade_count_15m", "trade_volume_15m_contracts"):
            values[name] = _unavailable_feature(
                name,
                as_of=as_of,
                reason=reason,
                status="insufficient_data",
                source_as_of=trades.last_at,
                window_start=as_of - ORDER_FLOW_WINDOW,
                sample_count=trades.sample_count,
            )
        return

    values["trade_count_15m"] = _feature(
        "trade_count_15m",
        float(trades.sample_count),
        as_of=as_of,
        source_as_of=trades.last_at,
        window_start=trades.first_at,
        sample_count=trades.sample_count,
    )
    total_size = Decimal(0)
    invalid_trade_count = 0
    for _, payload in trades.samples:
        size, error = _decimal_field(payload, "s")
        if error is not None or size is None or size <= 0:
            invalid_trade_count += 1
        else:
            total_size += size
    if invalid_trade_count:
        values["trade_volume_15m_contracts"] = _unavailable_feature(
            "trade_volume_15m_contracts",
            as_of=as_of,
            reason=f"{invalid_trade_count} trade size payloads were invalid.",
            status="invalid_data",
            source_as_of=trades.last_at,
            window_start=trades.first_at,
            sample_count=trades.sample_count,
        )
    else:
        values["trade_volume_15m_contracts"] = _feature(
            "trade_volume_15m_contracts",
            float(total_size),
            as_of=as_of,
            source_as_of=trades.last_at,
            window_start=trades.first_at,
            sample_count=trades.sample_count,
        )


def _metric_feature(
    inputs: FeatureInputs,
    values: dict[str, FeatureValue],
    name: str,
    sample: MetricSample | None,
    *,
    freshness: timedelta,
    window_start: datetime | None = None,
) -> bool:
    as_of = inputs.as_of
    if sample is None:
        values[name] = _unavailable_feature(
            name,
            as_of=as_of,
            reason="No source metric is stored at or before as_of.",
            status="missing",
            window_start=window_start,
        )
        return False
    if sample.event_time > as_of:
        values[name] = _unavailable_feature(
            name,
            as_of=as_of,
            reason="Source metric timestamp is later than as_of.",
            status="invalid_data",
            source_as_of=sample.event_time,
            window_start=window_start,
            sample_count=1,
        )
        return False
    if as_of - sample.event_time > freshness:
        values[name] = _unavailable_feature(
            name,
            as_of=as_of,
            reason=f"Source metric is older than {freshness}.",
            status="stale",
            source_as_of=sample.event_time,
            window_start=window_start,
            sample_count=1,
        )
        return False
    numeric_value = float(sample.value)
    if not sample.value.is_finite() or not math.isfinite(numeric_value):
        values[name] = _unavailable_feature(
            name,
            as_of=as_of,
            reason="Source metric is not a finite number.",
            status="invalid_data",
            source_as_of=sample.event_time,
            window_start=window_start,
            sample_count=1,
        )
        return False
    if name == "open_interest" and sample.value <= 0:
        values[name] = _unavailable_feature(
            name,
            as_of=as_of,
            reason="Open interest must be positive.",
            status="invalid_data",
            source_as_of=sample.event_time,
            window_start=window_start,
            sample_count=1,
        )
        return False
    values[name] = _feature(
        name,
        numeric_value,
        as_of=as_of,
        source_as_of=sample.event_time,
        window_start=window_start or sample.event_time,
        sample_count=1,
    )
    return True


def _add_derivative_features(
    inputs: FeatureInputs,
    values: dict[str, FeatureValue],
) -> None:
    asset_symbols = ASSET_SYMBOLS[inputs.asset]
    perpetual_symbol, index_symbol = asset_symbols
    funding = inputs.latest_metrics.get((perpetual_symbol, "funding_rate"))
    open_interest = inputs.latest_metrics.get((perpetual_symbol, "open_interest"))
    mark = inputs.latest_metrics.get((perpetual_symbol, "mark_price"))
    index = inputs.latest_metrics.get((index_symbol, "index_price"))
    _metric_feature(
        inputs,
        values,
        "funding_rate",
        funding,
        freshness=FUNDING_FRESHNESS,
    )
    _metric_feature(
        inputs,
        values,
        "open_interest",
        open_interest,
        freshness=METRIC_FRESHNESS,
    )

    cutoff = inputs.as_of - ORDER_FLOW_WINDOW
    interest_baseline = inputs.baseline_metrics.get(
        (perpetual_symbol, "open_interest")
    )
    interest_current_valid = (
        open_interest is not None
        and open_interest.event_time <= inputs.as_of
        and inputs.as_of - open_interest.event_time <= METRIC_FRESHNESS
    )
    baseline_valid = (
        interest_baseline is not None
        and interest_baseline.event_time <= cutoff
        and cutoff - interest_baseline.event_time <= MAX_EVENT_WINDOW_EDGE_AGE
    )
    if not interest_current_valid or not baseline_valid:
        values["open_interest_change_15m_pct"] = _unavailable_feature(
            "open_interest_change_15m_pct",
            as_of=inputs.as_of,
            reason="Fresh open-interest values bracketing the 15-minute window are required.",
            status="insufficient_data",
            source_as_of=open_interest.event_time if open_interest else None,
            window_start=cutoff,
            sample_count=int(interest_baseline is not None)
            + int(open_interest is not None),
        )
    elif (
        interest_baseline is None
        or open_interest is None
        or not interest_baseline.value.is_finite()
        or not open_interest.value.is_finite()
        or interest_baseline.value <= 0
        or open_interest.value <= 0
    ):
        values["open_interest_change_15m_pct"] = _unavailable_feature(
            "open_interest_change_15m_pct",
            as_of=inputs.as_of,
            reason="Open-interest values must be finite and positive.",
            status="invalid_data",
            source_as_of=open_interest.event_time if open_interest else None,
            window_start=cutoff,
            sample_count=2,
        )
    else:
        values["open_interest_change_15m_pct"] = _feature(
            "open_interest_change_15m_pct",
            float((open_interest.value / interest_baseline.value - 1) * 100),
            as_of=inputs.as_of,
            source_as_of=open_interest.event_time,
            window_start=interest_baseline.event_time,
            sample_count=2,
        )

    mark_valid = (
        mark is not None
        and mark.event_time <= inputs.as_of
        and inputs.as_of - mark.event_time <= METRIC_FRESHNESS
    )
    index_valid = (
        index is not None
        and index.event_time <= inputs.as_of
        and inputs.as_of - index.event_time <= METRIC_FRESHNESS
    )
    mark_index_synchronized = (
        mark_valid
        and index_valid
        and mark is not None
        and index is not None
        and abs(mark.event_time - index.event_time) <= MAX_METRIC_SKEW
    )
    if not mark_index_synchronized:
        latest_as_of = min(
            (sample.event_time for sample in (mark, index) if sample is not None),
            default=None,
        )
        values["mark_index_basis_pct"] = _unavailable_feature(
            "mark_index_basis_pct",
            as_of=inputs.as_of,
            reason=(
                "Fresh mark and index observations within 1 minute of each other "
                "are required for a synchronized basis."
            ),
            status="missing"
            if mark is None or index is None
            else "stale",
            source_as_of=latest_as_of,
            sample_count=int(mark is not None) + int(index is not None),
        )
    elif (
        mark is None
        or index is None
        or not mark.value.is_finite()
        or not index.value.is_finite()
        or mark.value <= 0
        or index.value <= 0
    ):
        values["mark_index_basis_pct"] = _unavailable_feature(
            "mark_index_basis_pct",
            as_of=inputs.as_of,
            reason="Mark and index prices must be finite and positive.",
            status="invalid_data",
            sample_count=int(mark is not None) + int(index is not None),
        )
    else:
        values["mark_index_basis_pct"] = _feature(
            "mark_index_basis_pct",
            float((mark.value / index.value - 1) * 100),
            as_of=inputs.as_of,
            source_as_of=min(mark.event_time, index.event_time),
            window_start=min(mark.event_time, index.event_time),
            sample_count=2,
        )

    index_baseline = inputs.baseline_metrics.get((index_symbol, "index_price"))
    index_current_valid = (
        index is not None
        and index.event_time <= inputs.as_of
        and inputs.as_of - index.event_time <= METRIC_FRESHNESS
    )
    index_baseline_valid = (
        index_baseline is not None
        and index_baseline.event_time <= cutoff
        and cutoff - index_baseline.event_time <= MAX_EVENT_WINDOW_EDGE_AGE
    )
    if not index_current_valid or not index_baseline_valid:
        values["index_return_15m_pct"] = _unavailable_feature(
            "index_return_15m_pct",
            as_of=inputs.as_of,
            reason="Fresh index prices bracketing the 15-minute window are required.",
            status="insufficient_data",
            source_as_of=index.event_time if index else None,
            window_start=cutoff,
            sample_count=int(index_baseline is not None) + int(index is not None),
        )
    elif (
        index is None
        or index_baseline is None
        or not index.value.is_finite()
        or not index_baseline.value.is_finite()
        or index.value <= 0
        or index_baseline.value <= 0
    ):
        values["index_return_15m_pct"] = _unavailable_feature(
            "index_return_15m_pct",
            as_of=inputs.as_of,
            reason="Index prices must be finite and positive.",
            status="invalid_data",
            source_as_of=index.event_time if index else None,
            window_start=cutoff,
            sample_count=2,
        )
    else:
        values["index_return_15m_pct"] = _feature(
            "index_return_15m_pct",
            float((index.value / index_baseline.value - 1) * 100),
            as_of=inputs.as_of,
            source_as_of=index.event_time,
            window_start=index_baseline.event_time,
            sample_count=2,
        )


def _hourly_return(
    candles: list[FeatureCandle],
    *,
    as_of: datetime,
) -> tuple[float | None, datetime | None, str | None]:
    reference_end = as_of.replace(minute=0, second=0, microsecond=0)
    expected = [reference_end - timedelta(hours=2), reference_end - timedelta(hours=1)]
    bars_by_open = {candle.open_time: candle for candle in candles}
    if not all(open_time in bars_by_open for open_time in expected):
        return None, None, "Two contiguous completed 1h candles are required."
    first_close = bars_by_open[expected[0]].close
    latest_close = bars_by_open[expected[1]].close
    if first_close <= 0:
        return None, reference_end, "Hourly return denominator must be positive."
    return float((latest_close / first_close - 1) * 100), reference_end, None


def _add_market_context_features(
    inputs: FeatureInputs,
    values: dict[str, FeatureValue],
) -> None:
    returns: dict[Asset, float | None] = {}
    source_times: dict[Asset, datetime | None] = {}
    for asset in ASSET_SYMBOLS:
        value, source_time, error = _hourly_return(
            inputs.hourly_candles.get(asset, []),
            as_of=inputs.as_of,
        )
        returns[asset] = value
        source_times[asset] = source_time
        name = f"{asset.lower()}_return_1h_pct"
        if error is not None:
            values[name] = _unavailable_feature(
                name,
                as_of=inputs.as_of,
                reason=error,
                status="insufficient_data",
                source_as_of=source_time,
                window_start=inputs.as_of.replace(minute=0, second=0, microsecond=0)
                - timedelta(hours=2),
                sample_count=len(inputs.hourly_candles.get(asset, [])),
            )
        else:
            values[name] = _feature(
                name,
                value,
                as_of=inputs.as_of,
                source_as_of=source_time,
                window_start=source_time - timedelta(hours=2) if source_time else None,
                sample_count=2,
            )

    if returns["BTC"] is None or returns["ETH"] is None:
        values["btc_minus_eth_return_1h_pct"] = _unavailable_feature(
            "btc_minus_eth_return_1h_pct",
            as_of=inputs.as_of,
            reason="Both BTC and ETH 1h returns are required.",
            status="insufficient_data",
            source_as_of=None,
            sample_count=0,
        )
    else:
        values["btc_minus_eth_return_1h_pct"] = _feature(
            "btc_minus_eth_return_1h_pct",
            returns["BTC"] - returns["ETH"],
            as_of=inputs.as_of,
            source_as_of=min(
                source_times["BTC"] or inputs.as_of,
                source_times["ETH"] or inputs.as_of,
            ),
            window_start=inputs.as_of.replace(minute=0, second=0, microsecond=0)
            - timedelta(hours=2),
            sample_count=4,
        )


def build_market_feature_vector(inputs: FeatureInputs) -> MarketFeatureVector:
    schema_version = inputs.schema_version or FEATURE_SCHEMA_VERSION
    order = FEATURE_SCHEMAS.get(schema_version, FEATURE_ORDER)
    values: dict[str, FeatureValue] = {}
    _add_price_features(inputs, values)
    if schema_version == FEATURE_SCHEMA_VERSION:
        _add_order_flow_features(inputs, values)
        _add_derivative_features(inputs, values)
    _add_market_context_features(inputs, values)
    ordered_features = [values[name] for name in order]
    return MarketFeatureVector(
        schema_version=schema_version,
        asset=inputs.asset,
        symbol=ASSET_SYMBOLS[inputs.asset][0],
        resolution=inputs.resolution,
        as_of=inputs.as_of,
        computed_at=inputs.computed_at,
        feature_order=list(order),
        features=ordered_features,
        interpretation=(
            "Deterministic, ordered research features with per-value provenance. "
            "Unavailable inputs remain null; trade volume is unsigned and the "
            "exchange trade-role field is not interpreted as buy/sell direction."
        ),
    )


async def _load_event_window(
    connection: AsyncConnection,
    *,
    symbol: str,
    event_type: str,
    as_of: datetime,
    include_samples: bool,
) -> EventWindow:
    window_start = as_of - ORDER_FLOW_WINDOW
    criteria = (
        MarketEvent.venue == "delta",
        MarketEvent.symbol == symbol,
        MarketEvent.event_type == event_type,
        MarketEvent.event_time >= window_start,
        MarketEvent.event_time <= as_of,
    )
    summary = await connection.execute(
        select(
            func.count(MarketEvent.id),
            func.min(MarketEvent.event_time),
            func.max(MarketEvent.event_time),
        ).where(*criteria)
    )
    sample_count, first_at, last_at = summary.one()
    first_payload = None
    last_payload = None
    if sample_count:
        first_result = await connection.execute(
            select(MarketEvent.payload)
            .where(*criteria)
            .order_by(MarketEvent.event_time, MarketEvent.id)
            .limit(1)
        )
        last_result = await connection.execute(
            select(MarketEvent.payload)
            .where(*criteria)
            .order_by(MarketEvent.event_time.desc(), MarketEvent.id.desc())
            .limit(1)
        )
        first_payload = first_result.scalar_one()
        last_payload = last_result.scalar_one()
    samples: tuple[tuple[datetime, dict[str, object]], ...] = ()
    if include_samples and sample_count:
        samples_result = await connection.execute(
            select(MarketEvent.event_time, MarketEvent.payload)
            .where(*criteria)
            .order_by(MarketEvent.event_time, MarketEvent.id)
        )
        samples = tuple((row[0], row[1]) for row in samples_result.all())
    return EventWindow(
        sample_count=int(sample_count or 0),
        first_at=first_at,
        last_at=last_at,
        first_payload=first_payload,
        last_payload=last_payload,
        samples=samples,
    )


async def _load_metrics(
    connection: AsyncConnection,
    *,
    as_of: datetime,
) -> tuple[dict[tuple[str, str], MetricSample], dict[tuple[str, str], MetricSample]]:
    symbols = [
        symbol for symbol_pair in ASSET_SYMBOLS.values() for symbol in symbol_pair
    ]
    metrics = ("funding_rate", "open_interest", "mark_price", "index_price")
    latest_result = await connection.execute(
        select(
            MarketMetric.symbol,
            MarketMetric.metric,
            MarketMetric.value,
            MarketMetric.event_time,
        )
        .where(
            MarketMetric.venue == "delta",
            MarketMetric.symbol.in_(symbols),
            MarketMetric.metric.in_(metrics),
            MarketMetric.event_time >= as_of - FUNDING_FRESHNESS,
            MarketMetric.event_time <= as_of,
        )
        .distinct(MarketMetric.symbol, MarketMetric.metric)
        .order_by(
            MarketMetric.symbol,
            MarketMetric.metric,
            MarketMetric.event_time.desc(),
        )
    )
    latest_metrics = {
        (row[0], row[1]): MetricSample(value=row[2], event_time=row[3])
        for row in latest_result.all()
    }

    cutoff = as_of - ORDER_FLOW_WINDOW
    baseline_result = await connection.execute(
        select(
            MarketMetric.symbol,
            MarketMetric.metric,
            MarketMetric.value,
            MarketMetric.event_time,
        )
        .where(
            MarketMetric.venue == "delta",
            MarketMetric.symbol.in_(symbols),
            MarketMetric.metric.in_(("open_interest", "index_price")),
            MarketMetric.event_time >= cutoff - MAX_EVENT_WINDOW_EDGE_AGE,
            MarketMetric.event_time <= cutoff,
        )
        .distinct(MarketMetric.symbol, MarketMetric.metric)
        .order_by(
            MarketMetric.symbol,
            MarketMetric.metric,
            MarketMetric.event_time.desc(),
        )
    )
    baseline_metrics = {
        (row[0], row[1]): MetricSample(value=row[2], event_time=row[3])
        for row in baseline_result.all()
    }
    return latest_metrics, baseline_metrics


@asynccontextmanager
async def _feature_connection(
    database_engine: AsyncEngine,
    connection: AsyncConnection | None,
) -> AsyncGenerator[AsyncConnection, None]:
    if connection is not None:
        yield connection
        return
    async with database_engine.connect() as owned_connection:
        yield owned_connection


async def get_market_feature_vector(
    database_engine: AsyncEngine,
    *,
    asset: Asset,
    resolution: FeatureResolution = "15m",
    as_of: datetime | None = None,
    now: datetime | None = None,
    connection: AsyncConnection | None = None,
    schema_version: str = FEATURE_SCHEMA_VERSION,
) -> MarketFeatureVector:
    computed_at = now or datetime.now(UTC)
    feature_as_of = aligned_as_of(
        resolution=resolution,
        now=computed_at,
        requested=as_of,
    )
    duration = timedelta(seconds=RESOLUTION_SECONDS[resolution])
    symbols = [symbol_pair[0] for symbol_pair in ASSET_SYMBOLS.values()]
    candles_by_asset: dict[Asset, list[FeatureCandle]] = {
        selected_asset: [] for selected_asset in ASSET_SYMBOLS
    }
    hourly_by_asset: dict[Asset, list[FeatureCandle]] = {
        selected_asset: [] for selected_asset in ASSET_SYMBOLS
    }
    symbol_to_asset = {
        symbol_pair[0]: selected_asset
        for selected_asset, symbol_pair in ASSET_SYMBOLS.items()
    }
    async with _feature_connection(database_engine, connection) as connection:
        candles_result = await connection.execute(
            select(
                Candle.symbol,
                Candle.resolution,
                Candle.open_time,
                Candle.open,
                Candle.high,
                Candle.low,
                Candle.close,
                Candle.volume,
            )
            .where(
                Candle.venue == "delta",
                Candle.symbol.in_(symbols),
                Candle.resolution == resolution,
                Candle.open_time >= feature_as_of - duration * MAX_CANDLE_HISTORY,
                Candle.open_time < feature_as_of,
            )
            .order_by(Candle.symbol, Candle.open_time)
        )
        for row in candles_result.all():
            candle = FeatureCandle(
                open_time=row[2],
                open=row[3],
                high=row[4],
                low=row[5],
                close=row[6],
                volume=row[7],
            )
            candles_by_asset[symbol_to_asset[row[0]]].append(candle)
            if resolution == "1h":
                hourly_by_asset[symbol_to_asset[row[0]]].append(candle)

        if resolution != "1h":
            reference_as_of = feature_as_of.replace(
                minute=0, second=0, microsecond=0
            )
            hourly_result = await connection.execute(
                select(
                    Candle.symbol,
                    Candle.open_time,
                    Candle.open,
                    Candle.high,
                    Candle.low,
                    Candle.close,
                    Candle.volume,
                )
                .where(
                    Candle.venue == "delta",
                    Candle.symbol.in_(symbols),
                    Candle.resolution == "1h",
                    Candle.open_time >= reference_as_of - timedelta(hours=3),
                    Candle.open_time < reference_as_of,
                )
                .order_by(Candle.symbol, Candle.open_time)
            )
            for row in hourly_result.all():
                hourly_by_asset[symbol_to_asset[row[0]]].append(
                    FeatureCandle(
                        open_time=row[1],
                        open=row[2],
                        high=row[3],
                        low=row[4],
                        close=row[5],
                        volume=row[6],
                    )
                )

        if schema_version == FEATURE_SCHEMA_VERSION:
            latest_metrics, baseline_metrics = await _load_metrics(
                connection,
                as_of=feature_as_of,
            )
            order_book = await _load_event_window(
                connection,
                symbol=ASSET_SYMBOLS[asset][0],
                event_type="ob_l1",
                as_of=feature_as_of,
                include_samples=False,
            )
            trades = await _load_event_window(
                connection,
                symbol=ASSET_SYMBOLS[asset][0],
                event_type="trades",
                as_of=feature_as_of,
                include_samples=True,
            )
        else:
            latest_metrics = {}
            baseline_metrics = {}
            order_book = EventWindow(sample_count=0, first_at=None, last_at=None)
            trades = EventWindow(sample_count=0, first_at=None, last_at=None)

    return build_market_feature_vector(
        FeatureInputs(
            schema_version=schema_version,
            asset=asset,
            resolution=resolution,
            as_of=feature_as_of,
            computed_at=computed_at.astimezone(UTC),
            candles=candles_by_asset[asset],
            hourly_candles=hourly_by_asset,
            latest_metrics=latest_metrics,
            baseline_metrics=baseline_metrics,
            order_book=order_book,
            trades=trades,
        )
    )
