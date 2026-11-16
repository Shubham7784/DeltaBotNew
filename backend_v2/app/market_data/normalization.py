from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, model_validator

EpochUnit = Literal["seconds", "milliseconds", "microseconds"]


class NormalizedCandle(BaseModel):
    model_config = ConfigDict(frozen=True)

    open_time: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal

    @model_validator(mode="after")
    def validate_ohlc(self) -> "NormalizedCandle":
        if not all(
            value.is_finite()
            for value in (self.open, self.high, self.low, self.close, self.volume)
        ):
            raise ValueError("Candle prices and volume must be finite")
        if min(self.open, self.high, self.low, self.close, self.volume) < 0:
            raise ValueError("Candle prices and volume must be non-negative")
        if self.high < max(self.open, self.close, self.low):
            raise ValueError("Candle high must be >= open, close, and low")
        if self.low > min(self.open, self.close, self.high):
            raise ValueError("Candle low must be <= open, close, and high")
        return self


def epoch_to_datetime(value: int | float | str, unit: EpochUnit) -> datetime:
    try:
        timestamp = Decimal(str(value))
    except InvalidOperation as error:
        raise ValueError(f"Invalid {unit} timestamp: {value!r}") from error

    divisor = {
        "seconds": Decimal(1),
        "milliseconds": Decimal(1_000),
        "microseconds": Decimal(1_000_000),
    }[unit]
    return datetime.fromtimestamp(float(timestamp / divisor), tz=UTC)


def normalize_candle(payload: dict[str, Any]) -> NormalizedCandle:
    return NormalizedCandle(
        open_time=epoch_to_datetime(payload["time"], "seconds"),
        open=Decimal(str(payload["open"])),
        high=Decimal(str(payload["high"])),
        low=Decimal(str(payload["low"])),
        close=Decimal(str(payload["close"])),
        volume=Decimal(str(payload.get("volume", 0))),
    )


def normalize_websocket_candle(payload: dict[str, Any]) -> NormalizedCandle:
    try:
        return NormalizedCandle(
            open_time=epoch_to_datetime(payload["cst"], "microseconds"),
            open=Decimal(str(payload["o"])),
            high=Decimal(str(payload["h"])),
            low=Decimal(str(payload["l"])),
            close=Decimal(str(payload["c"])),
            volume=Decimal(str(payload["v"])),
        )
    except KeyError as error:
        raise ValueError(
            f"WebSocket candle is missing the {error.args[0]!r} field"
        ) from error
    except InvalidOperation as error:
        raise ValueError("WebSocket candle contains an invalid decimal value") from error
