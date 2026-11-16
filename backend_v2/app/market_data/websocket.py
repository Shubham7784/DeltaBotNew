import asyncio
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.dialects.postgresql import insert as pg_insert
from websockets.asyncio.client import connect
from websockets.exceptions import WebSocketException

from app.core.config import Settings, get_settings
from app.database import redis_client
from app.market_data.backfill import build_candle_upsert_statement
from app.models import MarketEvent, MarketMetric
from app.market_data.normalization import epoch_to_datetime, normalize_websocket_candle

logger = logging.getLogger(__name__)
SUPPORTED_RESOLUTIONS = ("5m", "15m", "1h")
DELTA_INDEX_SYMBOLS = {"BTCUSD": ".DEXBTUSD", "ETHUSD": ".DEETHUSD"}


class SequenceGapError(RuntimeError):
    pass


class DeltaSubscriptionError(RuntimeError):
    pass


@dataclass(frozen=True)
class MarketEventData:
    event_type: str
    symbol: str
    event_time: datetime
    payload: dict[str, Any]


@dataclass(frozen=True)
class MarketMetricData:
    symbol: str
    metric: str
    value: Decimal


@dataclass
class OrderBookState:
    symbol: str
    sequence: int | None = None
    asks: dict[Decimal, Decimal] = field(default_factory=dict)
    bids: dict[Decimal, Decimal] = field(default_factory=dict)
    synchronized: bool = False

    def apply(self, message: dict[str, Any]) -> None:
        action = message.get("action")
        sequence = message.get("seq")
        if not isinstance(sequence, int):
            raise ValueError("Order-book message is missing integer sequence")

        if action == "snapshot":
            self.asks = self._levels(message.get("a"))
            self.bids = self._levels(message.get("b"))
            self.sequence = sequence
            self.synchronized = True
            return

        if action != "update":
            raise ValueError(f"Unsupported order-book action: {action!r}")
        if not self.synchronized or self.sequence is None:
            raise SequenceGapError(f"{self.symbol} has no initial order-book snapshot")
        if sequence != self.sequence + 1:
            self.synchronized = False
            raise SequenceGapError(
                f"{self.symbol} sequence gap: expected {self.sequence + 1}, got {sequence}"
            )

        self._apply_levels(self.asks, message.get("a"))
        self._apply_levels(self.bids, message.get("b"))
        self.sequence = sequence

    @staticmethod
    def _levels(raw_levels: Any) -> dict[Decimal, Decimal]:
        levels: dict[Decimal, Decimal] = {}
        if not isinstance(raw_levels, list):
            raise ValueError("Order-book side must be a list")
        for level in raw_levels:
            if not isinstance(level, list) or len(level) != 2:
                raise ValueError("Order-book level must contain price and size")
            price, size = Decimal(str(level[0])), Decimal(str(level[1]))
            if not price.is_finite() or not size.is_finite() or price <= 0 or size < 0:
                raise ValueError("Order-book levels must have positive prices and nonnegative sizes")
            if size > 0:
                levels[price] = size
        return levels

    @classmethod
    def _apply_levels(cls, current: dict[Decimal, Decimal], raw_levels: Any) -> None:
        for price, size in cls._levels(raw_levels).items():
            current[price] = size
        if not isinstance(raw_levels, list):
            raise ValueError("Order-book updates must be a list")
        for level in raw_levels:
            price, size = Decimal(str(level[0])), Decimal(str(level[1]))
            if size == 0:
                current.pop(price, None)

    def snapshot(self) -> dict[str, Any]:
        if not self.synchronized or self.sequence is None:
            raise SequenceGapError(f"{self.symbol} order book is not synchronized")
        return {
            "sequence": self.sequence,
            "asks": [[str(price), str(size)] for price, size in sorted(self.asks.items())],
            "bids": [
                [str(price), str(size)]
                for price, size in sorted(self.bids.items(), reverse=True)
            ],
        }


def normalize_ws_message(payload: dict[str, Any]) -> MarketEventData | None:
    event_type = payload.get("type")
    if not isinstance(event_type, str) or event_type in {
        "subscriptions",
        "unsubscriptions",
        "error",
        "pong",
    }:
        return None

    symbol = payload.get("sy") or payload.get("symbol")
    if not isinstance(symbol, str):
        return None
    timestamp = payload.get("ts")
    if timestamp is None:
        timestamp = payload.get("timestamp")
    if timestamp is None and event_type == "spot_price":
        return MarketEventData(
            event_type=event_type,
            symbol=symbol,
            event_time=datetime.now(UTC),
            payload=payload,
        )
    if not isinstance(timestamp, (int, float, str)):
        raise ValueError(f"WebSocket {event_type} event has no timestamp")
    return MarketEventData(
        event_type=event_type,
        symbol=symbol.removeprefix("MARK:"),
        event_time=epoch_to_datetime(timestamp, "microseconds"),
        payload=payload,
    )


def subscription_message(symbols: list[str]) -> dict[str, Any]:
    if not symbols or any(not symbol for symbol in symbols):
        raise ValueError("At least one non-empty market symbol is required")
    unknown_symbols = set(symbols) - DELTA_INDEX_SYMBOLS.keys()
    if unknown_symbols:
        raise ValueError(
            f"Missing Delta index symbols for: {', '.join(sorted(unknown_symbols))}"
        )
    channels: list[dict[str, Any]] = [
        {"name": name, "symbols": symbols}
        for name in (
            "ticker",
            "trades",
            "ob_l1",
            "ob_updates",
            "mark_price",
            "funding_rate",
        )
    ]
    channels.append(
        {
            "name": "spot_price",
            "symbols": [DELTA_INDEX_SYMBOLS[symbol] for symbol in symbols],
        }
    )
    channels.extend(
        {"name": f"candlestick_{resolution}", "symbols": symbols}
        for resolution in SUPPORTED_RESOLUTIONS
    )
    return {"type": "subscribe", "payload": {"channels": channels}}


def normalize_market_metrics(event: MarketEventData) -> list[MarketMetricData]:
    metrics: list[MarketMetricData] = []
    payload = event.payload

    def add(symbol: str, metric: str, raw_value: Any) -> None:
        try:
            value = Decimal(str(raw_value))
        except (InvalidOperation, ValueError) as exc:
            raise ValueError(f"Invalid {metric} value for {symbol}") from exc
        if not value.is_finite():
            raise ValueError(f"Non-finite {metric} value for {symbol}")
        if metric in {"mark_price", "index_price"} and value <= 0:
            raise ValueError(f"{metric} must be positive for {symbol}")
        if metric == "open_interest" and value < 0:
            raise ValueError(f"open_interest must be nonnegative for {symbol}")
        metrics.append(MarketMetricData(symbol=symbol, metric=metric, value=value))

    if event.event_type == "mark_price":
        add(event.symbol, "mark_price", payload.get("p"))
    elif event.event_type == "funding_rate":
        add(event.symbol, "funding_rate", payload.get("fr"))
    elif event.event_type == "spot_price":
        add(event.symbol, "index_price", payload.get("p"))
    elif event.event_type == "ticker":
        details = payload.get("d")
        if not isinstance(details, list):
            raise ValueError("Delta ticker event is missing its data list")
        for detail in details:
            if not isinstance(detail, dict):
                raise ValueError("Delta ticker data items must be objects")
            symbol = detail.get("s")
            if not isinstance(symbol, str):
                raise ValueError("Delta ticker data item is missing its symbol")
            if "m" in detail:
                add(symbol, "mark_price", detail["m"])
            open_interest = detail.get("oi")
            if (
                isinstance(open_interest, list)
                and open_interest
                and open_interest[0] is not None
            ):
                add(symbol, "open_interest", open_interest[0])

    return metrics


class DeltaWebSocketCollector:
    def __init__(
        self,
        settings: Settings,
        symbols: list[str],
        on_event: Callable[[MarketEventData], Any] | None = None,
    ) -> None:
        self._settings = settings
        self._symbols = symbols
        self._on_event = on_event
        self._stop = asyncio.Event()
        self.orderbooks = {symbol: OrderBookState(symbol) for symbol in symbols}

    def stop(self) -> None:
        self._stop.set()

    async def run_forever(self) -> None:
        delay = 1
        while not self._stop.is_set():
            try:
                await self._run_connection()
            except (
                WebSocketException,
                OSError,
                asyncio.TimeoutError,
                SequenceGapError,
                DeltaSubscriptionError,
                ValueError,
            ):
                if self._stop.is_set():
                    break
                logger.exception("Delta public market-data connection failed")
            if self._stop.is_set():
                break
            await self._wait_before_reconnect(delay)
            delay = min(delay * 2, 30)

    async def _wait_before_reconnect(self, delay: int) -> None:
        try:
            await asyncio.wait_for(self._stop.wait(), timeout=delay)
        except asyncio.TimeoutError:
            pass

    async def _run_connection(self) -> None:
        async with connect(
            self._settings.delta_public_ws_url,
            open_timeout=self._settings.delta_request_timeout_seconds,
            ping_interval=20,
            ping_timeout=20,
            max_queue=1_000,
        ) as websocket:
            await websocket.send(json.dumps(subscription_message(self._symbols)))
            while not self._stop.is_set():
                try:
                    raw_message = await asyncio.wait_for(websocket.recv(), timeout=1)
                except asyncio.TimeoutError:
                    continue
                decoded = json.loads(raw_message)
                if not isinstance(decoded, dict):
                    raise ValueError("Delta WebSocket message must be a JSON object")
                event_type = decoded.get("type")
                if event_type == "error":
                    raise DeltaSubscriptionError(str(decoded))
                if event_type == "subscriptions":
                    channels = decoded.get("channels")
                    rejected = (
                        [
                            channel
                            for channel in channels
                            if isinstance(channel, dict) and channel.get("error")
                        ]
                        if isinstance(channels, list)
                        else []
                    )
                    if rejected:
                        raise DeltaSubscriptionError(
                            f"Delta rejected WebSocket subscriptions: {rejected}"
                        )
                    continue
                symbol = decoded.get("sy") or decoded.get("symbol")
                if event_type == "ob_updates" and isinstance(symbol, str):
                    book = self.orderbooks.get(symbol)
                    if book is not None:
                        book.apply(decoded)
                event = normalize_ws_message(decoded)
                if event is not None and self._on_event is not None:
                    result = self._on_event(event)
                    if asyncio.iscoroutine(result):
                        await result


async def persist_market_event(
    session_factory: async_sessionmaker[AsyncSession],
    event: MarketEventData,
    *,
    venue: str = "delta",
) -> None:
    async with session_factory() as session:
        async with session.begin():
            session.add(
                MarketEvent(
                    venue=venue,
                    event_type=event.event_type,
                    symbol=event.symbol,
                    event_time=event.event_time,
                    payload=event.payload,
                )
            )
            metrics = normalize_market_metrics(event)
            if event.event_type.startswith("candlestick_"):
                resolution = event.event_type.removeprefix("candlestick_")
                if resolution not in SUPPORTED_RESOLUTIONS:
                    raise ValueError(f"Unsupported candle resolution: {resolution}")
                if event.payload.get("res") != resolution:
                    raise ValueError("Candle resolution does not match event type")
                candle = normalize_websocket_candle(event.payload)
                await session.execute(
                    build_candle_upsert_statement(
                        [
                            {
                                "venue": "delta",
                                "symbol": event.symbol,
                                "resolution": resolution,
                                "open_time": candle.open_time,
                                "open": candle.open,
                                "high": candle.high,
                                "low": candle.low,
                                "close": candle.close,
                                "volume": candle.volume,
                                "source": "delta_ws",
                            }
                        ]
                    )
                )
            if metrics:
                statement = pg_insert(MarketMetric).values(
                    [
                        {
                            "venue": venue,
                            "symbol": metric.symbol,
                            "metric": metric.metric,
                            "event_time": event.event_time,
                            "value": metric.value,
                            "source_event_type": event.event_type,
                            "payload": event.payload,
                        }
                        for metric in metrics
                    ]
                )
                statement = statement.on_conflict_do_update(
                    index_elements=["venue", "symbol", "metric", "event_time"],
                    set_={
                        "value": statement.excluded.value,
                        "source_event_type": statement.excluded.source_event_type,
                        "payload": statement.excluded.payload,
                    },
                )
                await session.execute(statement)
    await redis_client.set(
        f"market-data:last-received:{event.symbol}",
        str(datetime.now(UTC).timestamp()),
        ex=get_settings().market_data_stale_after_seconds * 2,
    )
