import math
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from app.market_data.backfill import RESOLUTION_SECONDS
from app.market_data.features import FeatureResolution
from app.market_data.regimes import Asset

TRIPLE_BARRIER_VERSION = "triple-barrier-v1"

Barrier = Literal["profit", "stop", "vertical", "ambiguous"]
LabelStatus = Literal[
    "labeled",
    "ambiguous",
    "gap",
    "insufficient_data",
    "invalid_data",
]
DirectionLabel = Literal[-1, 0, 1]


class TripleBarrierParameters(BaseModel):
    profit_target_bps: Decimal = Field(gt=0)
    stop_loss_bps: Decimal = Field(gt=0)
    round_trip_cost_bps: Decimal = Field(ge=0)
    max_holding_bars: int = Field(gt=0)

    @model_validator(mode="after")
    def stop_must_exceed_cost(self) -> "TripleBarrierParameters":
        if self.stop_loss_bps <= self.round_trip_cost_bps:
            raise ValueError("stop_loss_bps must exceed round_trip_cost_bps")
        if self.stop_loss_bps - self.round_trip_cost_bps >= Decimal(10_000):
            raise ValueError(
                "The net stop distance must be less than 10,000 basis points"
            )
        return self


@dataclass(frozen=True)
class BarrierCandle:
    open_time: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal


class TripleBarrierLabel(BaseModel):
    algorithm_version: str
    asset: Asset
    resolution: FeatureResolution
    parameters: TripleBarrierParameters
    decision_time: datetime
    entry_time: datetime | None
    label_end_time: datetime | None
    entry_price: float | None
    exit_price: float | None
    holding_bars: int
    gross_return_bps: float | None
    net_return_bps: float | None
    round_trip_cost_bps: float
    label: DirectionLabel | None
    barrier: Barrier | None
    status: LabelStatus
    reason: str | None

    @field_validator(
        "entry_price",
        "exit_price",
        "gross_return_bps",
        "net_return_bps",
        "round_trip_cost_bps",
    )
    @classmethod
    def numeric_values_must_be_finite(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("Label values must be finite")
        return value

    @model_validator(mode="after")
    def status_matches_label(self) -> "TripleBarrierLabel":
        if self.status == "labeled":
            if self.label is None or self.barrier is None:
                raise ValueError("Labeled outcomes require a label and barrier")
            if self.reason is not None:
                raise ValueError("Labeled outcomes must not have a failure reason")
        elif self.label is not None or not self.reason:
            raise ValueError("Unlabeled outcomes require a reason and null label")
        return self


def _invalid_result(
    *,
    asset: Asset,
    resolution: FeatureResolution,
    decision_time: datetime,
    parameters: TripleBarrierParameters,
    status: Literal["ambiguous", "gap", "insufficient_data", "invalid_data"],
    reason: str,
    entry_time: datetime | None = None,
    label_end_time: datetime | None = None,
    entry_price: Decimal | None = None,
    holding_bars: int = 0,
    barrier: Barrier | None = None,
) -> TripleBarrierLabel:
    return TripleBarrierLabel(
        algorithm_version=TRIPLE_BARRIER_VERSION,
        asset=asset,
        resolution=resolution,
        parameters=parameters,
        decision_time=decision_time,
        entry_time=entry_time,
        label_end_time=label_end_time,
        entry_price=float(entry_price) if entry_price is not None else None,
        exit_price=None,
        holding_bars=holding_bars,
        gross_return_bps=None,
        net_return_bps=None,
        round_trip_cost_bps=float(parameters.round_trip_cost_bps),
        label=None,
        barrier=barrier,
        status=status,
        reason=reason,
    )


def _valid_ohlc(candle: BarrierCandle) -> bool:
    prices = (candle.open, candle.high, candle.low, candle.close)
    return (
        all(price.is_finite() and price > 0 for price in prices)
        and candle.high >= max(candle.open, candle.close)
        and candle.low <= min(candle.open, candle.close)
        and candle.high >= candle.low
    )


def _labeled_result(
    *,
    asset: Asset,
    resolution: FeatureResolution,
    decision_time: datetime,
    entry_time: datetime,
    label_end_time: datetime,
    entry_price: Decimal,
    exit_price: Decimal,
    holding_bars: int,
    parameters: TripleBarrierParameters,
    barrier: Literal["profit", "stop", "vertical"],
    label: DirectionLabel,
) -> TripleBarrierLabel:
    gross_return_bps = (exit_price / entry_price - 1) * Decimal(10_000)
    net_return_bps = gross_return_bps - parameters.round_trip_cost_bps
    return TripleBarrierLabel(
        algorithm_version=TRIPLE_BARRIER_VERSION,
        asset=asset,
        resolution=resolution,
        parameters=parameters,
        decision_time=decision_time,
        entry_time=entry_time,
        label_end_time=label_end_time,
        entry_price=float(entry_price),
        exit_price=float(exit_price),
        holding_bars=holding_bars,
        gross_return_bps=float(gross_return_bps),
        net_return_bps=float(net_return_bps),
        round_trip_cost_bps=float(parameters.round_trip_cost_bps),
        label=label,
        barrier=barrier,
        status="labeled",
        reason=None,
    )


def label_triple_barrier(
    candles: list[BarrierCandle],
    *,
    asset: Asset,
    resolution: FeatureResolution,
    decision_time: datetime,
    parameters: TripleBarrierParameters,
) -> TripleBarrierLabel:
    if decision_time.tzinfo is None:
        raise ValueError("decision_time must include a timezone")
    decision_time = decision_time.astimezone(UTC)
    interval = timedelta(seconds=RESOLUTION_SECONDS[resolution])
    if int(decision_time.timestamp()) % int(interval.total_seconds()) != 0:
        raise ValueError("decision_time must align to the resolution's UTC boundary")
    if not candles:
        return _invalid_result(
            asset=asset,
            resolution=resolution,
            decision_time=decision_time,
            parameters=parameters,
            status="insufficient_data",
            reason="No entry candle is available at decision_time.",
        )

    first_time = candles[0].open_time
    if first_time.tzinfo is None:
        return _invalid_result(
            asset=asset,
            resolution=resolution,
            decision_time=decision_time,
            parameters=parameters,
            status="invalid_data",
            reason="Candle timestamps must include a timezone.",
        )
    entry_time = first_time.astimezone(UTC)
    if entry_time != decision_time:
        return _invalid_result(
            asset=asset,
            resolution=resolution,
            decision_time=decision_time,
            parameters=parameters,
            status="gap",
            reason="The first future candle does not open at decision_time.",
            entry_time=entry_time,
        )

    entry_price = candles[0].open
    if not _valid_ohlc(candles[0]):
        return _invalid_result(
            asset=asset,
            resolution=resolution,
            decision_time=decision_time,
            parameters=parameters,
            status="invalid_data",
            reason="Entry candle contains invalid OHLC values.",
            entry_time=entry_time,
        )

    scale = Decimal(10_000)
    upper_price = entry_price * (
        Decimal(1)
        + (
            parameters.profit_target_bps + parameters.round_trip_cost_bps
        )
        / scale
    )
    lower_price = entry_price * (
        Decimal(1)
        + (
            parameters.round_trip_cost_bps - parameters.stop_loss_bps
        )
        / scale
    )
    observed_bars = min(len(candles), parameters.max_holding_bars)
    for index in range(observed_bars):
        candle = candles[index]
        if candle.open_time.tzinfo is None:
            return _invalid_result(
                asset=asset,
                resolution=resolution,
                decision_time=decision_time,
                parameters=parameters,
                status="invalid_data",
                reason="Candle timestamps must include a timezone.",
                entry_time=entry_time,
                label_end_time=(
                    candles[index - 1].open_time.astimezone(UTC) + interval
                    if index > 0
                    else None
                ),
                entry_price=entry_price,
                holding_bars=index,
            )
        expected_open = entry_time + interval * index
        actual_open = candle.open_time.astimezone(UTC)
        previous_close_time = (
            candles[index - 1].open_time.astimezone(UTC) + interval
            if index > 0
            else None
        )
        if actual_open != expected_open:
            is_gap = actual_open > expected_open
            return _invalid_result(
                asset=asset,
                resolution=resolution,
                decision_time=decision_time,
                parameters=parameters,
                status="gap" if is_gap else "invalid_data",
                reason=(
                    "A missing candle interrupts the label horizon."
                    if is_gap
                    else "Future candles are not in chronological order."
                ),
                entry_time=entry_time,
                label_end_time=previous_close_time,
                entry_price=entry_price,
                holding_bars=index,
            )
        if not _valid_ohlc(candle):
            return _invalid_result(
                asset=asset,
                resolution=resolution,
                decision_time=decision_time,
                parameters=parameters,
                status="invalid_data",
                reason="A future candle contains invalid OHLC values.",
                entry_time=entry_time,
                label_end_time=actual_open + interval,
                entry_price=entry_price,
                holding_bars=index + 1,
            )

        if candle.open >= upper_price:
            return _labeled_result(
                asset=asset,
                resolution=resolution,
                decision_time=decision_time,
                entry_time=entry_time,
                label_end_time=actual_open + interval,
                entry_price=entry_price,
                exit_price=candle.open,
                holding_bars=index + 1,
                parameters=parameters,
                barrier="profit",
                label=1,
            )
        if candle.open <= lower_price:
            return _labeled_result(
                asset=asset,
                resolution=resolution,
                decision_time=decision_time,
                entry_time=entry_time,
                label_end_time=actual_open + interval,
                entry_price=entry_price,
                exit_price=candle.open,
                holding_bars=index + 1,
                parameters=parameters,
                barrier="stop",
                label=-1,
            )

        profit_touched = candle.high >= upper_price
        stop_touched = candle.low <= lower_price
        if profit_touched and stop_touched:
            return _invalid_result(
                asset=asset,
                resolution=resolution,
                decision_time=decision_time,
                parameters=parameters,
                status="ambiguous",
                reason=(
                    "Both barriers were touched within one OHLC candle; "
                    "intrabar ordering is unavailable."
                ),
                entry_time=entry_time,
                label_end_time=actual_open + interval,
                entry_price=entry_price,
                holding_bars=index + 1,
                barrier="ambiguous",
            )
        if profit_touched:
            return _labeled_result(
                asset=asset,
                resolution=resolution,
                decision_time=decision_time,
                entry_time=entry_time,
                label_end_time=actual_open + interval,
                entry_price=entry_price,
                exit_price=upper_price,
                holding_bars=index + 1,
                parameters=parameters,
                barrier="profit",
                label=1,
            )
        if stop_touched:
            return _labeled_result(
                asset=asset,
                resolution=resolution,
                decision_time=decision_time,
                entry_time=entry_time,
                label_end_time=actual_open + interval,
                entry_price=entry_price,
                exit_price=lower_price,
                holding_bars=index + 1,
                parameters=parameters,
                barrier="stop",
                label=-1,
            )

    if len(candles) < parameters.max_holding_bars:
        last_candle = candles[observed_bars - 1]
        last_close_time = last_candle.open_time.astimezone(UTC) + interval
        return _invalid_result(
            asset=asset,
            resolution=resolution,
            decision_time=decision_time,
            parameters=parameters,
            status="insufficient_data",
            reason="The maximum holding horizon is not fully observed.",
            entry_time=entry_time,
            label_end_time=last_close_time,
            entry_price=entry_price,
            holding_bars=observed_bars,
        )

    final_candle = candles[parameters.max_holding_bars - 1]
    final_close_time = final_candle.open_time.astimezone(UTC) + interval
    final_close = final_candle.close
    gross_return_bps = (final_close / entry_price - 1) * scale
    net_return_bps = gross_return_bps - parameters.round_trip_cost_bps
    label: DirectionLabel = (
        1 if net_return_bps > 0 else -1 if net_return_bps < 0 else 0
    )
    return _labeled_result(
        asset=asset,
        resolution=resolution,
        decision_time=decision_time,
        entry_time=entry_time,
        label_end_time=final_close_time,
        entry_price=entry_price,
        exit_price=final_close,
        holding_bars=parameters.max_holding_bars,
        parameters=parameters,
        barrier="vertical",
        label=label,
    )
