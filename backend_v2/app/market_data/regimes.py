import math
import statistics
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine

from app.models import Candle, MarketMetric

REGIME_ALGORITHM_VERSION = "ohlcv-regime-v1"
Asset = Literal["BTC", "ETH"]
ASSET_SYMBOLS: dict[Asset, tuple[str, str]] = {
    "BTC": ("BTCUSD", ".DEXBTUSD"),
    "ETH": ("ETHUSD", ".DEETHUSD"),
}
MINIMUM_HOURLY_CANDLES = 50
MAX_CANDLE_AGE = timedelta(hours=2)
TREND_SPREAD_THRESHOLD_PERCENT = 0.15
TREND_CONSISTENCY_THRESHOLD = 0.60
HIGH_VOLATILITY_THRESHOLD_PERCENT = 5.0
HOUR = timedelta(hours=1)

RegimeLabel = Literal[
    "trend_up",
    "trend_down",
    "range",
    "high_volatility",
    "unavailable",
]


@dataclass(frozen=True)
class HourlyCandle:
    open_time: datetime
    close: Decimal
    volume: Decimal


@dataclass(frozen=True)
class DerivativeContext:
    mark_price: float | None = None
    mark_price_as_of: datetime | None = None
    index_price: float | None = None
    index_price_as_of: datetime | None = None
    funding_rate: float | None = None
    funding_rate_as_of: datetime | None = None
    open_interest: float | None = None
    open_interest_as_of: datetime | None = None
    open_interest_change_percent: float | None = None
    open_interest_change_hours: float | None = None


class RegimeFeatures(BaseModel):
    ema_20: float | None = None
    ema_50: float | None = None
    ema_spread_percent: float | None = None
    return_24h_percent: float | None = None
    realized_volatility_24h_percent: float | None = None
    directional_consistency_12h: float | None = None
    volume_ratio_24h_vs_prior_6d: float | None = None
    mark_price: float | None = None
    mark_price_as_of: datetime | None = None
    index_price: float | None = None
    index_price_as_of: datetime | None = None
    mark_index_basis_percent: float | None = None
    funding_rate: float | None = None
    funding_rate_as_of: datetime | None = None
    open_interest: float | None = None
    open_interest_as_of: datetime | None = None
    open_interest_change_percent: float | None = None
    open_interest_change_hours: float | None = None


class MarketRegimeSnapshot(BaseModel):
    asset: Asset
    regime: RegimeLabel
    confidence: float
    algorithm_version: str
    computed_at: datetime
    data_as_of: datetime | None
    features: RegimeFeatures
    evidence: list[str]
    invalidation_conditions: list[str]
    interpretation: str


def _ema(values: list[float], period: int) -> float:
    if len(values) < period:
        raise ValueError(f"At least {period} values are required for EMA")
    current = sum(values[:period]) / period
    alpha = 2 / (period + 1)
    for value in values[period:]:
        current = alpha * value + (1 - alpha) * current
    return current


def _base_features(derivatives: DerivativeContext) -> RegimeFeatures:
    mark_index_basis = None
    if (
        derivatives.mark_price is not None
        and derivatives.index_price is not None
        and derivatives.index_price > 0
    ):
        mark_index_basis = (
            derivatives.mark_price / derivatives.index_price - 1
        ) * 100
    return RegimeFeatures(
        mark_price=derivatives.mark_price,
        mark_price_as_of=derivatives.mark_price_as_of,
        index_price=derivatives.index_price,
        index_price_as_of=derivatives.index_price_as_of,
        mark_index_basis_percent=mark_index_basis,
        funding_rate=derivatives.funding_rate,
        funding_rate_as_of=derivatives.funding_rate_as_of,
        open_interest=derivatives.open_interest,
        open_interest_as_of=derivatives.open_interest_as_of,
        open_interest_change_percent=derivatives.open_interest_change_percent,
        open_interest_change_hours=derivatives.open_interest_change_hours,
    )


def _unavailable(
    asset: Asset,
    computed_at: datetime,
    derivatives: DerivativeContext,
    reason: str,
    data_as_of: datetime | None = None,
) -> MarketRegimeSnapshot:
    return MarketRegimeSnapshot(
        asset=asset,
        regime="unavailable",
        confidence=0.0,
        algorithm_version=REGIME_ALGORITHM_VERSION,
        computed_at=computed_at,
        data_as_of=data_as_of,
        features=_base_features(derivatives),
        evidence=[reason],
        invalidation_conditions=[
            f"Restore at least {MINIMUM_HOURLY_CANDLES} contiguous, completed 1h candles.",
            "Refresh hourly candles so the latest completed candle is no more than 2 hours old.",
        ],
        interpretation=(
            "No regime classification is available; downstream analysis must not "
            "treat this as a directional signal."
        ),
    )


def classify_hourly_regime(
    asset: Literal["BTC", "ETH"],
    candles: list[HourlyCandle],
    *,
    derivatives: DerivativeContext | None = None,
    computed_at: datetime | None = None,
) -> MarketRegimeSnapshot:
    timestamp = computed_at or datetime.now(UTC)
    if timestamp.tzinfo is None:
        raise ValueError("Regime computation time must be timezone-aware")
    timestamp = timestamp.astimezone(UTC)
    derivative_values = derivatives or DerivativeContext()

    completed = sorted(
        (
            candle
            for candle in candles
            if candle.open_time + HOUR <= timestamp
        ),
        key=lambda candle: candle.open_time,
    )
    if len(completed) < MINIMUM_HOURLY_CANDLES:
        return _unavailable(
            asset,
            timestamp,
            derivative_values,
            f"Only {len(completed)} completed hourly candles are available.",
        )

    sample = completed[-168:]
    last_bar = sample[-1]
    data_as_of = last_bar.open_time + HOUR
    if timestamp - data_as_of > MAX_CANDLE_AGE:
        return _unavailable(
            asset,
            timestamp,
            derivative_values,
            "The latest completed hourly candle is stale.",
            data_as_of,
        )

    recent = sample[-MINIMUM_HOURLY_CANDLES:]
    if any(
        later.open_time - earlier.open_time != HOUR
        for earlier, later in zip(recent, recent[1:])
    ):
        return _unavailable(
            asset,
            timestamp,
            derivative_values,
            "A gap exists in the 50-candle regime calculation window.",
            data_as_of,
        )
    if any(candle.close <= 0 or candle.volume < 0 for candle in sample):
        return _unavailable(
            asset,
            timestamp,
            derivative_values,
            "Hourly data contains a nonpositive close or negative volume.",
            data_as_of,
        )

    closes = [float(candle.close) for candle in sample]
    ema_20 = _ema(closes, 20)
    ema_50 = _ema(closes, 50)
    ema_spread = (ema_20 / ema_50 - 1) * 100
    return_24h = (closes[-1] / closes[-25] - 1) * 100
    hourly_log_returns = [
        math.log(closes[index] / closes[index - 1])
        for index in range(len(closes) - 24, len(closes))
    ]
    realized_volatility = statistics.stdev(hourly_log_returns) * math.sqrt(24) * 100

    recent_returns = [
        closes[index] / closes[index - 1] - 1
        for index in range(len(closes) - 12, len(closes))
    ]
    direction = 1 if ema_spread > 0 else -1 if ema_spread < 0 else 0
    consistency = (
        sum(value * direction > 0 for value in recent_returns) / len(recent_returns)
        if direction
        else 0.5
    )
    volume_ratio = None
    if len(sample) == 168 and all(
        later.open_time - earlier.open_time == HOUR
        for earlier, later in zip(sample, sample[1:])
    ):
        recent_volume = sum(float(candle.volume) for candle in sample[-24:]) / 24
        baseline_volume = sum(float(candle.volume) for candle in sample[:-24]) / 144
        if baseline_volume > 0:
            volume_ratio = recent_volume / baseline_volume

    features = _base_features(derivative_values).model_copy(
        update={
            "ema_20": ema_20,
            "ema_50": ema_50,
            "ema_spread_percent": ema_spread,
            "return_24h_percent": return_24h,
            "realized_volatility_24h_percent": realized_volatility,
            "directional_consistency_12h": consistency,
            "volume_ratio_24h_vs_prior_6d": volume_ratio,
        }
    )
    evidence = [
        f"EMA20/EMA50 spread: {ema_spread:.3f}% (trend threshold +/-{TREND_SPREAD_THRESHOLD_PERCENT:.2f}%).",
        f"24h return: {return_24h:.3f}%; annualization-free 24h realized volatility: {realized_volatility:.3f}%.",
        f"12h directional consistency: {consistency:.3f}; completed candles through {data_as_of.isoformat()}.",
    ]
    if volume_ratio is not None:
        evidence.append(
            f"24h average volume is {volume_ratio:.3f}x the preceding 6d hourly average."
        )
    if derivative_values.funding_rate is not None:
        evidence.append(
            f"Latest funding rate: {derivative_values.funding_rate:.8f}."
        )
    if derivative_values.open_interest_change_percent is not None:
        evidence.append(
            "Open interest changed "
            f"{derivative_values.open_interest_change_percent:.3f}% over the observed "
            f"{derivative_values.open_interest_change_hours:.2f}h window."
        )

    # Differentiate strong directional trend momentum from non-directional erratic volatility
    is_trend_up = (
        ema_spread >= TREND_SPREAD_THRESHOLD_PERCENT
        and consistency >= TREND_CONSISTENCY_THRESHOLD
    ) or (
        ema_spread >= 1.0 and return_24h >= 2.5 and consistency >= 0.50
    )

    is_trend_down = (
        ema_spread <= -TREND_SPREAD_THRESHOLD_PERCENT
        and consistency >= TREND_CONSISTENCY_THRESHOLD
    ) or (
        ema_spread <= -1.0 and return_24h <= -2.5 and consistency >= 0.50
    )

    if is_trend_up:
        regime: RegimeLabel = "trend_up"
        confidence = (
            0.5 + 0.5 * min(abs(ema_spread) / 0.5, 1.0)
        ) * consistency
        invalidation = [
            "Reassess if EMA20 crosses below EMA50.",
            f"Reassess if 12h directional consistency falls below {TREND_CONSISTENCY_THRESHOLD:.2f}.",
            f"Reclassify as high volatility if 24h realized volatility reaches {HIGH_VOLATILITY_THRESHOLD_PERCENT:.2f}%.",
        ]
    elif is_trend_down:
        regime: RegimeLabel = "trend_down"
        confidence = (
            0.5 + 0.5 * min(abs(ema_spread) / 0.5, 1.0)
        ) * consistency
        invalidation = [
            "Reassess if EMA20 crosses above EMA50.",
            f"Reassess if 12h directional consistency falls below {TREND_CONSISTENCY_THRESHOLD:.2f}.",
            f"Reclassify as high volatility if 24h realized volatility reaches {HIGH_VOLATILITY_THRESHOLD_PERCENT:.2f}%.",
        ]
    elif realized_volatility >= HIGH_VOLATILITY_THRESHOLD_PERCENT:
        regime = "high_volatility"
        confidence = min(
            1.0,
            0.5
            + (realized_volatility - HIGH_VOLATILITY_THRESHOLD_PERCENT)
            / HIGH_VOLATILITY_THRESHOLD_PERCENT,
        )
        invalidation = [
            f"Reassess when 24h realized volatility falls below {HIGH_VOLATILITY_THRESHOLD_PERCENT:.2f}%.",
            "Invalidate this snapshot if completed hourly candles become stale or discontinuous.",
        ]
    else:
        regime = "range"
        confidence = (
            0.5 * (1 - min(abs(ema_spread) / TREND_SPREAD_THRESHOLD_PERCENT, 1.0))
            + 0.5 * (1 - consistency)
        )
        invalidation = [
            f"Reclassify as trend_up if EMA spread exceeds {TREND_SPREAD_THRESHOLD_PERCENT:.2f}% and consistency reaches {TREND_CONSISTENCY_THRESHOLD:.2f}.",
            f"Reclassify as trend_down if EMA spread falls below -{TREND_SPREAD_THRESHOLD_PERCENT:.2f}% and consistency reaches {TREND_CONSISTENCY_THRESHOLD:.2f}.",
            f"Reclassify as high volatility if 24h realized volatility reaches {HIGH_VOLATILITY_THRESHOLD_PERCENT:.2f}%.",
        ]

    return MarketRegimeSnapshot(
        asset=asset,
        regime=regime,
        confidence=max(0.0, min(confidence, 1.0)),
        algorithm_version=REGIME_ALGORITHM_VERSION,
        computed_at=timestamp,
        data_as_of=data_as_of,
        features=features,
        evidence=evidence,
        invalidation_conditions=invalidation,
        interpretation=(
            "Research-only OHLCV regime baseline. Derivative values are contextual "
            "and do not alter the regime label."
        ),
    )


async def classify_current_regimes(
    database_engine: AsyncEngine,
    *,
    computed_at: datetime | None = None,
) -> dict[str, MarketRegimeSnapshot]:
    timestamp = computed_at or datetime.now(UTC)
    if timestamp.tzinfo is None:
        raise ValueError("Regime computation time must be timezone-aware")
    timestamp = timestamp.astimezone(UTC)
    symbols = [symbol for pair in ASSET_SYMBOLS.values() for symbol in pair]
    candle_symbols = [pair[0] for pair in ASSET_SYMBOLS.values()]
    candle_result_by_asset: dict[Asset, list[HourlyCandle]] = {
        asset: [] for asset in ASSET_SYMBOLS
    }
    metric_samples: dict[Asset, dict[tuple[str, str], list[tuple[datetime, Decimal]]]] = {
        asset: {} for asset in ASSET_SYMBOLS
    }
    symbol_to_asset = {
        symbol: asset
        for asset, symbol_pair in ASSET_SYMBOLS.items()
        for symbol in symbol_pair
    }

    async with database_engine.connect() as connection:
        candle_result = await connection.execute(
            select(Candle.symbol, Candle.open_time, Candle.close, Candle.volume)
            .where(
                Candle.venue == "delta",
                Candle.symbol.in_(candle_symbols),
                Candle.resolution == "1h",
                Candle.open_time >= timestamp - timedelta(days=8),
            )
            .order_by(Candle.symbol, Candle.open_time)
        )
        for row in candle_result.all():
            asset = symbol_to_asset[row[0]]
            candle_result_by_asset[asset].append(
                HourlyCandle(open_time=row[1], close=row[2], volume=row[3])
            )

        metric_result = await connection.execute(
            select(
                MarketMetric.symbol,
                MarketMetric.metric,
                MarketMetric.value,
                MarketMetric.event_time,
            )
            .where(
                MarketMetric.venue == "delta",
                MarketMetric.symbol.in_(symbols),
                MarketMetric.metric.in_(
                    ("mark_price", "index_price", "funding_rate", "open_interest")
                ),
                MarketMetric.event_time >= timestamp - timedelta(hours=24),
                MarketMetric.event_time <= timestamp,
            )
            .order_by(MarketMetric.symbol, MarketMetric.metric, MarketMetric.event_time)
        )
        for row in metric_result.all():
            asset = symbol_to_asset[row[0]]
            metric_samples[asset].setdefault((row[0], row[1]), []).append(
                (row[3], row[2])
            )

    snapshots = {}
    for asset, (perpetual_symbol, index_symbol) in ASSET_SYMBOLS.items():
        samples = metric_samples[asset]

        def latest(metric: str, symbol: str) -> tuple[datetime, Decimal] | None:
            values = samples.get((symbol, metric), [])
            matched = [
                (event_time, value)
                for event_time, value in values
                if event_time is not None
            ]
            return matched[-1] if matched else None

        mark = latest("mark_price", perpetual_symbol)
        index = latest("index_price", index_symbol)
        funding = latest("funding_rate", perpetual_symbol)
        interest_values = samples.get((perpetual_symbol, "open_interest"), [])
        interest = latest("open_interest", perpetual_symbol)
        interest_change = None
        interest_hours = None
        perpetual_interest = [
            (event_time, value)
            for event_time, value in interest_values
            if event_time is not None
        ]
        if len(perpetual_interest) >= 2:
            first_time, first_value = perpetual_interest[0]
            last_time, last_value = perpetual_interest[-1]
            observed_hours = (last_time - first_time).total_seconds() / 3600
            if observed_hours > 0 and first_value > 0:
                interest_change = float((last_value / first_value - 1) * 100)
                interest_hours = observed_hours
        derivatives = DerivativeContext(
            mark_price=float(mark[1]) if mark else None,
            mark_price_as_of=mark[0] if mark else None,
            index_price=float(index[1]) if index else None,
            index_price_as_of=index[0] if index else None,
            funding_rate=float(funding[1]) if funding else None,
            funding_rate_as_of=funding[0] if funding else None,
            open_interest=float(interest[1]) if interest else None,
            open_interest_as_of=interest[0] if interest else None,
            open_interest_change_percent=interest_change,
            open_interest_change_hours=interest_hours,
        )
        snapshots[asset] = classify_hourly_regime(
            asset,
            candle_result_by_asset[asset],
            derivatives=derivatives,
            computed_at=timestamp,
        )
    return snapshots
