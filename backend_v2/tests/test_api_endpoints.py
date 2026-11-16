"""Tests for new REST API endpoints (Trading, Supervisor, Analytics)."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
import pytest
from httpx import ASGITransport, AsyncClient

from app.database import engine
from app.main import app


@asynccontextmanager
async def get_test_client() -> AsyncGenerator[AsyncClient, None]:
    await engine.dispose()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac
    await engine.dispose()


@pytest.mark.asyncio
async def test_trading_account_endpoint() -> None:
    async with get_test_client() as client:
        res = await client.get("/api/v1/trading/account")
        assert res.status_code == 200
        data = res.json()
        assert "cash_balance" in data
        assert "equity" in data
        assert "is_halted" in data
        assert data["currency"] == "USD"


@pytest.mark.asyncio
async def test_trading_positions_endpoint() -> None:
    async with get_test_client() as client:
        res = await client.get("/api/v1/trading/positions")
        assert res.status_code == 200
        positions = res.json()
        assert isinstance(positions, list)


@pytest.mark.asyncio
async def test_trading_orders_endpoint() -> None:
    async with get_test_client() as client:
        res = await client.get("/api/v1/trading/orders")
        assert res.status_code == 200
        orders = res.json()
        assert isinstance(orders, list)


@pytest.mark.asyncio
async def test_supervisor_sessions_endpoint() -> None:
    async with get_test_client() as client:
        res = await client.get("/api/v1/supervisor/sessions")
        assert res.status_code == 200
        sessions = res.json()
        assert isinstance(sessions, list)


@pytest.mark.asyncio
async def test_supervisor_traces_endpoint() -> None:
    async with get_test_client() as client:
        res = await client.get("/api/v1/supervisor/traces")
        assert res.status_code == 200
        traces = res.json()
        assert isinstance(traces, list)


@pytest.mark.asyncio
async def test_supervisor_cycle_endpoint() -> None:
    async with get_test_client() as client:
        res = await client.post("/api/v1/supervisor/cycle", json={"asset": "BTC", "stage": "monitor"})
        assert res.status_code == 200
        data = res.json()
        assert data["asset"] == "BTC"
        assert data["stage"] == "monitor"
        assert "result" in data


@pytest.mark.asyncio
async def test_analytics_performance_endpoint() -> None:
    async with get_test_client() as client:
        res = await client.get("/api/v1/analytics/performance")
        assert res.status_code == 200
        data = res.json()
        assert "sharpe_ratio" in data
        assert "max_drawdown" in data
        assert "win_rate" in data


@pytest.mark.asyncio
async def test_analytics_report_endpoint() -> None:
    async with get_test_client() as client:
        res = await client.get("/api/v1/analytics/report")
        assert res.status_code == 200
        data = res.json()
        assert "summary" in data
        assert "by_asset" in data
        assert "by_direction" in data


@pytest.mark.asyncio
async def test_analytics_worker_drift_endpoint() -> None:
    async with get_test_client() as client:
        res = await client.get("/api/v1/analytics/worker-drift")
        assert res.status_code == 200
        data = res.json()
        assert "total_cycles" in data
        assert "approval_rate" in data
        assert "rejection_rate" in data
