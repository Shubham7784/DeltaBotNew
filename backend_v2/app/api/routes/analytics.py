"""REST API endpoints for performance analytics, metrics, and worker drift."""

from typing import Any

from fastapi import APIRouter, Query
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.analytics.metrics import calculate_worker_drift_metrics
from app.analytics.reporting import generate_context_report
from app.database import engine
from app.models import PaperAccount, PaperPosition, WorkerDecisionTrace

router = APIRouter(prefix="/api/v1/analytics", tags=["analytics"])


class PerformanceSummaryResponse(BaseModel):
    initial_balance: float
    final_equity: float
    net_pnl: float
    total_trades: int
    win_rate: float
    profit_factor: float
    avg_trade_pnl: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_pct: float
    mean_r_multiple: float


class ContextReportResponse(BaseModel):
    summary: PerformanceSummaryResponse
    by_asset: dict[str, Any]
    by_direction: dict[str, Any]
    by_exit_reason: dict[str, int]


class WorkerDriftResponse(BaseModel):
    total_cycles: int
    approval_rate: float
    rejection_rate: float
    waiting_cycles: int
    ml_technical_agreements: int
    mean_ml_latency_ms: float


@router.get("/performance", response_model=PerformanceSummaryResponse)
async def get_performance_summary() -> PerformanceSummaryResponse:
    """Retrieve portfolio performance metrics including Sharpe, Sortino, Drawdown, and Win Rate."""
    async with AsyncSession(engine, expire_on_commit=False) as session:
        # Get account balance
        stmt_acc = select(PaperAccount).where(PaperAccount.name == "default_paper_account")
        res_acc = await session.execute(stmt_acc)
        account = res_acc.scalar_one_or_none()
        initial_balance = float(account.initial_balance) if account else 10000.0

        # Get positions
        stmt_pos = select(PaperPosition).where(PaperPosition.status == "CLOSED")
        res_pos = await session.execute(stmt_pos)
        positions = [
            {
                "asset": p.asset,
                "direction": p.direction,
                "status": p.status,
                "entry_price": float(p.entry_price),
                "stop_loss": float(p.stop_loss),
                "exit_price": float(p.exit_price or 0.0),
                "realized_pnl": float(p.realized_pnl),
                "exit_reason": p.exit_reason or "UNKNOWN",
            }
            for p in res_pos.scalars().all()
        ]

        report = generate_context_report(positions, initial_balance=initial_balance)
        return PerformanceSummaryResponse(**report["summary"])


@router.get("/report", response_model=ContextReportResponse)
async def get_context_report() -> ContextReportResponse:
    """Retrieve full performance metrics broken down by asset, trade direction, and exit reason."""
    async with AsyncSession(engine, expire_on_commit=False) as session:
        stmt_acc = select(PaperAccount).where(PaperAccount.name == "default_paper_account")
        res_acc = await session.execute(stmt_acc)
        account = res_acc.scalar_one_or_none()
        initial_balance = float(account.initial_balance) if account else 10000.0

        stmt_pos = select(PaperPosition).where(PaperPosition.status == "CLOSED")
        res_pos = await session.execute(stmt_pos)
        positions = [
            {
                "asset": p.asset,
                "direction": p.direction,
                "status": p.status,
                "entry_price": float(p.entry_price),
                "stop_loss": float(p.stop_loss),
                "exit_price": float(p.exit_price or 0.0),
                "realized_pnl": float(p.realized_pnl),
                "exit_reason": p.exit_reason or "UNKNOWN",
            }
            for p in res_pos.scalars().all()
        ]

        report = generate_context_report(positions, initial_balance=initial_balance)
        return ContextReportResponse(**report)


@router.get("/worker-drift", response_model=WorkerDriftResponse)
async def get_worker_drift_metrics(limit: int = Query(default=100, ge=1, le=500)) -> WorkerDriftResponse:
    """Retrieve worker disagreement, rejection rates, approval ratios, and ML latency statistics."""
    async with AsyncSession(engine, expire_on_commit=False) as session:
        stmt = (
            select(WorkerDecisionTrace)
            .where(WorkerDecisionTrace.event_type == "monitoring_cycle")
            .order_by(WorkerDecisionTrace.recorded_at.desc())
            .limit(limit)
        )
        res = await session.execute(stmt)
        traces = [
            {
                "event_type": t.event_type,
                "details": dict(t.details),
            }
            for t in res.scalars().all()
        ]

        drift = calculate_worker_drift_metrics(traces)
        return WorkerDriftResponse(**drift)
