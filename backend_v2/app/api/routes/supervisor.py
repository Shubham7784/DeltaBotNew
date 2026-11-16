"""REST API endpoints for Supervisor sessions, decision traces, and on-demand workflow cycles."""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
import uuid

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import engine
from app.market_data.regimes import Asset
from app.ml.inference import MLInferenceService
from app.ml.registry import ModelRegistry
from app.models import WorkerDecisionTrace, WorkerSession
from app.workers.workflow import AssetWorkflowCoordinator

router = APIRouter(prefix="/api/v1/supervisor", tags=["supervisor"])


class WorkerSessionResponse(BaseModel):
    id: uuid.UUID
    asset: str
    status: str
    required_roles: list[str]
    created_at: datetime
    updated_at: datetime
    suspended_at: datetime | None
    suspension_reason: str | None


class WorkerDecisionTraceResponse(BaseModel):
    id: uuid.UUID
    session_id: uuid.UUID
    cycle_id: uuid.UUID | None
    event_type: str
    from_status: str | None
    to_status: str
    recorded_at: datetime
    details: dict[str, Any]


class TriggerCycleRequest(BaseModel):
    asset: Asset
    stage: Literal["init", "monitor"] = "monitor"


class TriggerCycleResponse(BaseModel):
    asset: Asset
    stage: str
    result: dict[str, Any]


@router.get("/sessions", response_model=list[WorkerSessionResponse])
async def get_worker_sessions() -> list[WorkerSessionResponse]:
    """Retrieve active and recent supervisor worker sessions per asset."""
    async with AsyncSession(engine, expire_on_commit=False) as session:
        query = select(WorkerSession).order_by(WorkerSession.updated_at.desc())
        res = await session.execute(query)
        sessions = res.scalars().all()

        return [
            WorkerSessionResponse(
                id=s.id,
                asset=s.asset,
                status=s.status,
                required_roles=list(s.required_roles),
                created_at=s.created_at,
                updated_at=s.updated_at,
                suspended_at=s.suspended_at,
                suspension_reason=s.suspension_reason,
            )
            for s in sessions
        ]


@router.get("/traces", response_model=list[WorkerDecisionTraceResponse])
async def get_decision_traces(
    session_id: uuid.UUID | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
) -> list[WorkerDecisionTraceResponse]:
    """Retrieve immutable audit traces of supervisor and worker state transitions."""
    async with AsyncSession(engine, expire_on_commit=False) as session:
        query = select(WorkerDecisionTrace)
        if session_id is not None:
            query = query.where(WorkerDecisionTrace.session_id == session_id)
        query = query.order_by(WorkerDecisionTrace.recorded_at.desc()).limit(limit)

        res = await session.execute(query)
        traces = res.scalars().all()

        return [
            WorkerDecisionTraceResponse(
                id=t.id,
                session_id=t.session_id,
                cycle_id=t.cycle_id,
                event_type=t.event_type,
                from_status=t.from_status,
                to_status=t.to_status,
                recorded_at=t.recorded_at,
                details=dict(t.details),
            )
            for t in traces
        ]


@router.post("/cycle", response_model=TriggerCycleResponse)
async def trigger_supervisor_cycle(payload: TriggerCycleRequest) -> TriggerCycleResponse:
    """Trigger an on-demand session initialization or monitoring cycle for an asset."""
    registry = ModelRegistry(Path("model-registry"))
    ml_service = MLInferenceService(registry, engine)
    coord = AssetWorkflowCoordinator(engine, ml_service)
    now = datetime.now(UTC)

    async with AsyncSession(engine, expire_on_commit=False) as session:
        if payload.stage == "init":
            result = await coord.run_session_initialization(payload.asset, now, session)
        else:
            result = await coord.run_monitoring_cycle(payload.asset, now, session)

        return TriggerCycleResponse(
            asset=payload.asset,
            stage=payload.stage,
            result=result,
        )
