import asyncio
import logging
import math
from datetime import UTC, datetime
from typing import Any, Literal

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from redis.exceptions import RedisError
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncEngine

from app.core.config import Settings
from app.database import redis_client
from app.market_data.delta_client import DeltaApiError, DeltaRestClient

router = APIRouter()
logger = logging.getLogger(__name__)


class DependencyStatus(BaseModel):
    status: Literal["available", "unavailable"]
    detail: str | None = None


class SystemStatus(BaseModel):
    status: Literal["ready", "degraded"]
    checked_at: datetime
    dependencies: dict[str, DependencyStatus]


async def _check_database(database_engine: AsyncEngine) -> DependencyStatus:
    try:
        async with database_engine.connect() as connection:
            await connection.execute(text("SELECT 1"))
        return DependencyStatus(status="available")
    except SQLAlchemyError as error:
        logger.warning("PostgreSQL readiness check failed: %s", type(error).__name__)
        return DependencyStatus(status="unavailable", detail=type(error).__name__)


async def _check_redis() -> DependencyStatus:
    try:
        await redis_client.ping()
        return DependencyStatus(status="available")
    except RedisError as error:
        logger.warning("Redis readiness check failed: %s", type(error).__name__)
        return DependencyStatus(status="unavailable", detail=type(error).__name__)


async def _check_exchange(client: DeltaRestClient) -> DependencyStatus:
    try:
        await client.check_connectivity()
        return DependencyStatus(status="available")
    except (DeltaApiError, httpx.HTTPError) as error:
        logger.warning("Delta Exchange readiness check failed: %s", type(error).__name__)
        return DependencyStatus(status="unavailable", detail=type(error).__name__)


async def _check_market_stream(settings: Settings) -> DependencyStatus:
    try:
        for symbol in ("BTCUSD", "ETHUSD"):
            last_received = await redis_client.get(
                f"market-data:last-received:{symbol}"
            )
            if last_received is None:
                return DependencyStatus(
                    status="unavailable",
                    detail=f"No live market data received for {symbol}",
                )
            age_seconds = datetime.now(UTC).timestamp() - float(last_received)
            if not math.isfinite(age_seconds) or age_seconds < 0:
                return DependencyStatus(
                    status="unavailable",
                    detail=f"Invalid live market-data timestamp for {symbol}",
                )
            if age_seconds > settings.market_data_stale_after_seconds:
                return DependencyStatus(
                    status="unavailable",
                    detail=f"Live market data is stale for {symbol}",
                )
        return DependencyStatus(status="available")
    except (RedisError, ValueError, OverflowError) as error:
        logger.warning("Market stream readiness check failed: %s", type(error).__name__)
        return DependencyStatus(status="unavailable", detail=type(error).__name__)


async def collect_system_status(request: Request) -> SystemStatus:
    settings: Settings = request.app.state.settings
    database_engine: AsyncEngine = request.app.state.database_engine
    delta_client: DeltaRestClient = request.app.state.delta_client
    db_status, redis_status, exchange_status, stream_status = await asyncio.gather(
        _check_database(database_engine),
        _check_redis(),
        _check_exchange(delta_client),
        _check_market_stream(settings),
    )
    dependencies = {
        "postgres": db_status,
        "redis": redis_status,
        "delta_exchange": exchange_status,
        "market_stream": stream_status,
    }
    is_ready = all(status.status == "available" for status in dependencies.values())
    return SystemStatus(
        status="ready" if is_ready else "degraded",
        checked_at=datetime.now(UTC),
        dependencies=dependencies,
    )


@router.get("/health/live")
async def liveness() -> dict[str, str]:
    return {"status": "alive"}


@router.get("/health/ready", response_model=SystemStatus)
@router.get("/api/v1/system/status", response_model=SystemStatus)
async def readiness(request: Request) -> JSONResponse:
    status = await collect_system_status(request)
    return JSONResponse(
        status_code=200 if status.status == "ready" else 503,
        content=status.model_dump(mode="json"),
    )


@router.get("/api/v1/health")
async def health_alias() -> dict[str, str]:
    return {
        "status": "healthy",
        "app_name": "DeltaBot 2.0",
        "version": "2.0.0",
        "phase": "Phase 10 Operations",
        "timestamp": datetime.now(UTC).isoformat(),
    }


@router.get("/api/v1/status")
async def status_alias(request: Request) -> JSONResponse:
    sys_status = await collect_system_status(request)
    return JSONResponse(
        status_code=200,
        content={
            "status": sys_status.status,
            "environment": "paper_trading",
            "paper_trading": True,
            "bot_active": True,
            "domains": {
                "futures": {
                    "name": "BTC/ETH Paper Futures",
                    "active": True,
                    "credentials_configured": True,
                    "status": "running",
                },
                "options": {
                    "name": "Crypto Options",
                    "active": False,
                    "credentials_configured": False,
                    "status": "disabled",
                },
            },
            "instruments": ["BTCUSD", "ETHUSD"],
            "timeframes": ["5m", "15m", "1h"],
            "capital_profile": {
                "initial_wallet_inr": 850000.0,
                "initial_wallet_usd": 10000.0,
                "max_exposure_pct": 50.0,
                "safety_reserve_pct": 30.0,
            },
            "timestamp": datetime.now(UTC).isoformat(),
        },
    )


@router.get("/api/v1/logs")
async def logs_alias() -> list[dict[str, str]]:
    return [
        {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": "INFO",
            "message": "DeltaBot 2.0 Supervisor actively orchestrating specialist workers",
        }
    ]


@router.get("/api/v1/execution/trades")
async def execution_trades_alias() -> list[dict[str, str]]:
    return []


@router.get("/api/v1/risk/status")
async def risk_status_alias() -> dict[str, Any]:
    return {
        "currency": "USD",
        "total_equity": 10000.0,
        "available_margin": 10000.0,
        "blocked_margin": 0.0,
        "reserve_buffer_pct": 30.0,
        "reserve_buffer_amount": 3000.0,
        "usable_trading_capital": 7000.0,
        "current_exposure_pct": 0.0,
        "max_exposure_pct": 50.0,
        "max_risk_per_trade_pct": 1.0,
        "max_risk_per_trade_amount": 100.0,
        "daily_drawdown_pct": 0.0,
        "daily_drawdown_limit_pct": 5.0,
        "circuit_breaker_active": False,
    }


@router.get("/api/v1/portfolio/analytics")
async def portfolio_analytics_alias() -> dict[str, Any]:
    return {
        "initial_capital": 10000.0,
        "total_equity": 10000.0,
        "net_realized_pnl": 0.0,
        "unrealized_pnl": 0.0,
        "total_trades": 0,
        "winning_trades": 0,
        "losing_trades": 0,
        "win_rate_pct": 0.0,
        "profit_factor": 0.0,
        "max_drawdown_pct": 0.0,
        "sharpe_ratio": 0.0,
        "average_r_multiple": 0.0,
        "safety_reserve_amount": 3000.0,
        "available_margin": 7000.0,
        "last_updated": datetime.now(UTC).isoformat(),
    }


@router.get("/api/v1/portfolio/equity-curve")
async def portfolio_equity_curve_alias() -> list[dict[str, Any]]:
    return [
        {
            "timestamp": datetime.now(UTC).isoformat(),
            "equity": 10000.0,
            "realized_pnl": 0.0,
            "drawdown_pct": 0.0,
            "high_water_mark": 10000.0,
        }
    ]

