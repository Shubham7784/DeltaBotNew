import pytest
from sqlalchemy.dialects import postgresql

from app.market_data.backfill import upsert_candles


class FakeSession:
    def __init__(self, statements: list[object]) -> None:
        self.statements = statements

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args) -> None:
        return None

    def begin(self):
        return self

    async def execute(self, statement) -> None:
        self.statements.append(statement)


class FakeSessionFactory:
    def __init__(self) -> None:
        self.statements: list[object] = []

    def __call__(self) -> FakeSession:
        return FakeSession(self.statements)


@pytest.mark.asyncio
async def test_repeated_backfill_uses_conflict_update_for_same_candle() -> None:
    session_factory = FakeSessionFactory()
    candle = {
        "time": 1_700_000_000,
        "open": "35000",
        "high": "35002",
        "low": "34999",
        "close": "35001",
        "volume": "0.001",
    }

    for _ in range(2):
        count = await upsert_candles(
            session_factory,
            venue="delta",
            symbol="BTCUSD",
            resolution="1h",
            source="delta_rest",
            candles=[candle],
        )
        assert count == 1

    assert len(session_factory.statements) == 2
    for statement in session_factory.statements:
        sql = str(statement.compile(dialect=postgresql.dialect()))
        assert "ON CONFLICT (venue, symbol, resolution, open_time) DO UPDATE" in sql
        assert "SET open =" in sql
