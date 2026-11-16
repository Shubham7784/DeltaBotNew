from datetime import UTC, datetime
from decimal import Decimal

import pytest
from sqlalchemy.dialects import postgresql

import app.market_data.websocket as websocket
from app.market_data.websocket import (
    DeltaWebSocketCollector,
    MarketEventData,
    OrderBookState,
    SequenceGapError,
    normalize_market_metrics,
    normalize_ws_message,
    persist_market_event,
    subscription_message,
)


def test_orderbook_snapshot_and_incremental_updates() -> None:
    book = OrderBookState("BTCUSD")
    book.apply(
        {
            "action": "snapshot",
            "seq": 10,
            "a": [["101", "2"], ["102", "1"]],
            "b": [["100", "3"]],
        }
    )
    book.apply(
        {
            "action": "update",
            "seq": 11,
            "a": [["101", "0"], ["103", "4"]],
            "b": [["100", "5"]],
        }
    )

    snapshot = book.snapshot()
    assert book.sequence == 11
    assert book.asks == {Decimal("102"): Decimal("1"), Decimal("103"): Decimal("4")}
    assert snapshot["bids"] == [["100", "5"]]


def test_orderbook_sequence_gap_invalidates_book() -> None:
    book = OrderBookState("ETHUSD")
    book.apply({"action": "snapshot", "seq": 4, "a": [], "b": []})
    with pytest.raises(SequenceGapError, match="sequence gap"):
        book.apply({"action": "update", "seq": 6, "a": [], "b": []})
    assert not book.synchronized


def test_orderbook_snapshot_recovers_after_sequence_gap() -> None:
    book = OrderBookState("ETHUSD")
    book.apply({"action": "snapshot", "seq": 4, "a": [], "b": []})
    with pytest.raises(SequenceGapError):
        book.apply({"action": "update", "seq": 6, "a": [], "b": []})

    book.apply({"action": "snapshot", "seq": 20, "a": [["101", "1"]], "b": []})

    assert book.snapshot()["sequence"] == 20
    assert book.synchronized


def test_orderbook_rejects_non_finite_levels() -> None:
    book = OrderBookState("BTCUSD")
    with pytest.raises(ValueError, match="positive prices"):
        book.apply(
            {"action": "snapshot", "seq": 1, "a": [["NaN", "1"]], "b": []}
        )


def test_ws_timestamp_is_normalized_from_microseconds() -> None:
    event = normalize_ws_message(
        {"type": "ticker", "sy": "BTCUSD", "ts": 1_700_000_000_000_000}
    )
    assert event is not None
    assert event.symbol == "BTCUSD"
    assert event.event_time.isoformat() == "2023-11-14T22:13:20+00:00"


def test_spot_price_without_exchange_timestamp_uses_receive_time() -> None:
    before = datetime.now(UTC)
    event = normalize_ws_message(
        {"type": "spot_price", "sy": ".DEXBTUSD", "ts": None, "p": "65000"}
    )
    after = datetime.now(UTC)
    assert event is not None
    assert before <= event.event_time <= after


def test_subscriptions_use_current_public_channels() -> None:
    message = subscription_message(["BTCUSD", "ETHUSD"])
    names = [channel["name"] for channel in message["payload"]["channels"]]
    assert names == [
        "ticker",
        "trades",
        "ob_l1",
        "ob_updates",
        "mark_price",
        "funding_rate",
        "spot_price",
        "candlestick_5m",
        "candlestick_15m",
        "candlestick_1h",
    ]
    assert message["payload"]["channels"][6]["symbols"] == [
        ".DEXBTUSD",
        ".DEETHUSD",
    ]


def test_normalizes_funding_spot_mark_and_ticker_open_interest() -> None:
    timestamp = normalize_ws_message(
        {
            "type": "funding_rate",
            "sy": "BTCUSD",
            "ts": 1_700_000_000_000_000,
            "fr": "-0.0001",
        }
    )
    assert timestamp is not None
    assert normalize_market_metrics(timestamp)[0].value == Decimal("-0.0001")

    for event_type, symbol, field, metric in (
        ("mark_price", "BTCUSD", "p", "mark_price"),
        ("spot_price", ".DEXBTUSD", "p", "index_price"),
    ):
        event = MarketEventData(
            event_type=event_type,
            symbol=symbol,
            event_time=timestamp.event_time,
            payload={"type": event_type, "sy": symbol, field: "65000.25"},
        )
        normalized = normalize_market_metrics(event)
        assert [(item.metric, item.value) for item in normalized] == [
            (metric, Decimal("65000.25"))
        ]

    ticker = MarketEventData(
        event_type="ticker",
        symbol="ETHUSD",
        event_time=timestamp.event_time,
        payload={
            "type": "ticker",
            "d": [{"s": "ETHUSD", "m": "1933.25", "oi": ["1810145", "-12"]}],
        },
    )
    metrics = normalize_market_metrics(ticker)
    assert [(metric.metric, metric.value) for metric in metrics] == [
        ("mark_price", Decimal("1933.25")),
        ("open_interest", Decimal("1810145")),
    ]


def test_subscription_rejects_unknown_index_symbol() -> None:
    with pytest.raises(ValueError, match="Missing Delta index symbols"):
        subscription_message(["SOLUSD"])


class FakePersistenceSession:
    def __init__(self, factory: "FakePersistenceSessionFactory") -> None:
        self.factory = factory

    async def __aenter__(self) -> "FakePersistenceSession":
        return self

    async def __aexit__(self, *args: object) -> None:
        _ = args
        return None

    def begin(self) -> "FakePersistenceSession":
        return self

    def add(self, event: object) -> None:
        self.factory.events.append(event)

    async def execute(self, statement: object) -> None:
        self.factory.statements.append(statement)


class FakePersistenceSessionFactory:
    def __init__(self) -> None:
        self.events: list[object] = []
        self.statements: list[object] = []

    def __call__(self) -> FakePersistenceSession:
        return FakePersistenceSession(self)


class FakeRedis:
    def __init__(self) -> None:
        self.values: dict[str, tuple[str, int]] = {}

    async def set(self, key: str, value: str, *, ex: int) -> None:
        self.values[key] = (value, ex)


@pytest.mark.asyncio
async def test_live_candlestick_event_upserts_normalized_candle(monkeypatch) -> None:
    redis = FakeRedis()
    monkeypatch.setattr(websocket, "redis_client", redis)
    session_factory = FakePersistenceSessionFactory()
    event_time = datetime(2023, 11, 14, 22, 14, tzinfo=UTC)
    event = MarketEventData(
        event_type="candlestick_1h",
        symbol="BTCUSD",
        event_time=event_time,
        payload={
            "type": "candlestick_1h",
            "sy": "BTCUSD",
            "res": "1h",
            "cst": 1_700_000_000_000_000,
            "o": "35000",
            "h": "35100",
            "l": "34900",
            "c": "35050",
            "v": "1.25",
        },
    )

    await persist_market_event(session_factory, event)

    assert len(session_factory.events) == 1
    assert len(session_factory.statements) == 1
    sql = str(
        session_factory.statements[0].compile(dialect=postgresql.dialect())
    )
    assert "ON CONFLICT (venue, symbol, resolution, open_time) DO UPDATE" in sql
    assert "source" in sql
    assert redis.values["market-data:last-received:BTCUSD"][0]


@pytest.mark.asyncio
async def test_collector_reconnects_after_sequence_gap(monkeypatch) -> None:
    from app.core.config import Settings

    collector = DeltaWebSocketCollector(Settings(), ["BTCUSD"])
    attempts = 0
    waits: list[int] = []

    async def run_connection() -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise SequenceGapError("test sequence gap")
        collector.stop()

    async def wait_before_reconnect(delay: int) -> None:
        waits.append(delay)

    monkeypatch.setattr(collector, "_run_connection", run_connection)
    monkeypatch.setattr(collector, "_wait_before_reconnect", wait_before_reconnect)
    await collector.run_forever()

    assert attempts == 2
    assert waits == [1]
