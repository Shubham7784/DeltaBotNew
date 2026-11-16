"""Asset Workflow Coordinator (Phase 7: p7-workflow).

Coordinates Stage 1 (Session Initialization) and Stage 2 (Continuous Monitoring Loop)
independently per asset (BTC/ETH) using the deterministic Supervisor and specialist workers.
"""

from datetime import UTC, datetime, timedelta
import logging
import uuid
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from app.market_data.regimes import Asset, classify_current_regimes
from app.market_data.session_bias import assess_session_biases, get_current_session_biases
from app.ml.inference import MLInferenceService
from app.models import SessionBias, WorkerDecisionTrace, WorkerSession
from app.workers.ml_agent import MLPredictionAgent
from app.workers.order_flow_agent import OrderFlowAgent
from app.workers.performance_agent import PerformanceAnalystAgent
from app.workers.supervisor import (
    SupervisorSessionStatus,
    WorkerAttemptPolicy,
    WorkerDefinition,
    WorkerExecution,
    WorkerOutput,
    advance_session_status,
    execute_worker,
)
from app.workers.technical_agent import TechnicalAnalysisAgent
from app.workers.tools import WorkerRole
from app.workers.validator_agent import TradeProposal, TradeValidatorAgent

logger = logging.getLogger(__name__)

DEFAULT_WORKER_POLICY = WorkerAttemptPolicy(
    timeout=timedelta(seconds=15),
    max_attempts=2,
    max_output_age=timedelta(minutes=5),
)


class AssetWorkflowCoordinator:
    """Coordinates specialist workers and session state machines per asset."""

    def __init__(
        self,
        engine: AsyncEngine,
        ml_service: MLInferenceService,
    ):
        self.engine = engine
        self.ml_service = ml_service
        self.technical_agent = TechnicalAnalysisAgent(engine)
        self.ml_agent = MLPredictionAgent(ml_service)
        self.order_flow_agent = OrderFlowAgent(engine)
        self.validator_agent = TradeValidatorAgent()
        self.performance_agent = PerformanceAnalystAgent()

    async def get_or_create_session(
        self,
        asset: Asset,
        session: AsyncSession,
        *,
        required_roles: list[str] | None = None,
    ) -> WorkerSession:
        """Fetch active session or initialize a new one."""
        roles = required_roles or [
            WorkerRole.NEWS.value,
            WorkerRole.MACHINE_LEARNING.value,
            WorkerRole.TECHNICAL.value,
            WorkerRole.ORDER_FLOW.value,
            WorkerRole.VALIDATOR.value,
        ]
        stmt = (
            select(WorkerSession)
            .where(
                WorkerSession.asset == asset,
                WorkerSession.status.in_(["initializing", "active", "suspended"]),
            )
            .order_by(WorkerSession.created_at.desc())
            .limit(1)
        )
        res = await session.execute(stmt)
        worker_session = res.scalar_one_or_none()

        if worker_session is None:
            now = datetime.now(UTC)
            worker_session = WorkerSession(
                id=uuid.uuid4(),
                asset=asset,
                status=SupervisorSessionStatus.INITIALIZING.value,
                required_roles=roles,
                created_at=now,
                updated_at=now,
            )
            session.add(worker_session)
            await session.commit()
            await session.refresh(worker_session)

        return worker_session

    async def record_decision_trace(
        self,
        session_id: uuid.UUID,
        cycle_id: uuid.UUID,
        event_type: str,
        from_status: str | None,
        to_status: str,
        details: dict[str, Any],
        session: AsyncSession,
    ) -> WorkerDecisionTrace:
        """Persist an immutable audit trace of supervisor/worker transitions."""
        trace = WorkerDecisionTrace(
            id=uuid.uuid4(),
            session_id=session_id,
            cycle_id=cycle_id,
            event_type=event_type,
            from_status=from_status,
            to_status=to_status,
            recorded_at=datetime.now(UTC),
            details=details,
        )
        session.add(trace)
        await session.commit()
        return trace

    async def run_session_initialization(
        self,
        asset: Asset,
        as_of: datetime,
        session: AsyncSession,
    ) -> dict[str, Any]:
        """Stage 1: Initialize session, establish and lock directional bias."""
        cycle_id = uuid.uuid4()
        worker_session = await self.get_or_create_session(asset, session)
        session_id = worker_session.id
        old_status = str(worker_session.status)

        # 1. Market Regimes
        regime_dict = await classify_current_regimes(
            self.engine,
            computed_at=as_of,
        )
        regime_info = regime_dict.get(asset)
        regime_name = regime_info.regime if regime_info else "TRANSITIONAL"

        # 2. Session Bias (assesses and locks in DB)
        bias_snapshots = await assess_session_biases(
            self.engine,
            assessed_at=as_of,
        )
        asset_bias = next((b for b in bias_snapshots if b.asset == asset), None)
        bias_direction = asset_bias.direction if asset_bias else "NO_TRADE"
        bias_confidence = asset_bias.confidence if asset_bias else 0.0
        evidence = asset_bias.evidence if asset_bias else []

        now = datetime.now(UTC)
        await session.refresh(worker_session)
        worker_session.status = SupervisorSessionStatus.ACTIVE.value
        worker_session.updated_at = now
        await session.commit()

        await self.record_decision_trace(
            session_id=session_id,
            cycle_id=cycle_id,
            event_type="session_init",
            from_status=old_status,
            to_status=SupervisorSessionStatus.ACTIVE.value,
            details={
                "asset": asset,
                "regime": regime_name,
                "bias_direction": bias_direction,
                "bias_confidence": bias_confidence,
                "evidence": evidence,
            },
            session=session,
        )

        return {
            "session_id": str(session_id),
            "asset": asset,
            "status": SupervisorSessionStatus.ACTIVE.value,
            "regime": regime_name,
            "bias_direction": bias_direction,
            "bias_confidence": bias_confidence,
        }

    async def run_monitoring_cycle(
        self,
        asset: Asset,
        as_of: datetime,
        session: AsyncSession,
        *,
        model_id: str | None = None,
        confidence_threshold: float = 0.45,
    ) -> dict[str, Any]:
        """Stage 2: Run continuous multi-agent monitoring cycle and candidate trade validation."""
        cycle_id = uuid.uuid4()
        worker_session = await self.get_or_create_session(asset, session)

        if worker_session.status == SupervisorSessionStatus.SUSPENDED.value:
            return {
                "cycle_id": str(cycle_id),
                "asset": asset,
                "status": "SUSPENDED",
                "proposal": None,
                "reason": f"Session {worker_session.id} is suspended: {worker_session.suspension_reason}",
            }

        # 1. Check existing session bias
        stmt = (
            select(SessionBias)
            .where(
                SessionBias.asset == asset,
                SessionBias.session_date == as_of.date(),
            )
            .order_by(SessionBias.updated_at.desc())
            .limit(1)
        )
        res = await session.execute(stmt)
        bias_row = res.scalar_one_or_none()
        current_bias = bias_row.direction if bias_row else "NO_TRADE"
        bias_confidence = bias_row.confidence if bias_row else 0.0

        # Invalidate/suspend check if bias became suspended
        if bias_row and bias_row.status == "suspended":
            worker_session.status = SupervisorSessionStatus.SUSPENDED.value
            worker_session.suspended_at = datetime.now(UTC)
            worker_session.suspension_reason = bias_row.suspension_reason or "Session bias invalidated"
            await session.commit()
            return {
                "cycle_id": str(cycle_id),
                "asset": asset,
                "status": "SUSPENDED",
                "proposal": None,
                "reason": worker_session.suspension_reason,
            }

        # 2. Worker 2: Current Regime
        regime_dict = await classify_current_regimes(
            self.engine,
            computed_at=as_of,
        )
        regime_info = regime_dict.get(asset)
        regime_name = regime_info.regime if regime_info else "TRANSITIONAL"

        # 3. Worker 4: Technical Analysis
        tech_output = await self.technical_agent.analyze(
            asset,
            as_of,
            connection=await session.connection(),
        )

        # 4. Worker 5: ML Prediction
        ml_output = await self.ml_agent.predict(
            asset,
            as_of,
            model_id=model_id,
            confidence_threshold=confidence_threshold,
            connection=await session.connection(),
        )

        # 5. Worker 6: Order Flow
        of_output = await self.order_flow_agent.analyze(
            asset,
            as_of,
            connection=await session.connection(),
        )

        # 6. Worker 7: Trade Validator
        validator_output = self.validator_agent.validate(
            asset=asset,
            session_bias=current_bias,
            bias_confidence=bias_confidence,
            regime=regime_name,
            technical_output=tech_output.payload,
            ml_output=ml_output.payload,
            order_flow_output=of_output.payload,
            as_of=as_of,
        )

        # Record decision trace
        await self.record_decision_trace(
            session_id=worker_session.id,
            cycle_id=cycle_id,
            event_type="monitoring_cycle",
            from_status=worker_session.status,
            to_status=worker_session.status,
            details={
                "asset": asset,
                "bias": current_bias,
                "regime": regime_name,
                "technical": tech_output.payload,
                "ml": ml_output.payload,
                "order_flow": of_output.payload,
                "validation": validator_output.payload,
            },
            session=session,
        )

        return {
            "cycle_id": str(cycle_id),
            "session_id": str(worker_session.id),
            "asset": asset,
            "session_status": worker_session.status,
            "session_bias": current_bias,
            "regime": regime_name,
            "technical_setup": tech_output.payload.get("setup_type"),
            "ml_decision": ml_output.payload.get("decision"),
            "order_flow_bias": of_output.payload.get("order_flow_bias"),
            "validator_decision": validator_output.payload.get("decision"),
            "rationale": validator_output.payload.get("rationale"),
            "proposal": validator_output.payload.get("proposal"),
        }
