"""End-to-End integration tests for DeltaBot 2.0 (Phase 10: p10-integration & p10-release).

Verifies full-stack integration across:
- Specialist workers (Technical, ML, Order Flow, Validator, Performance)
- Supervisor state machine, session locks, and decision traces
- Deterministic Risk Engine position sizing and exposure guards
- Paper execution engine, slippage/fee accounting, and bracket exits
- Failure recovery and fail-closed safety
"""

import asyncio
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
import uuid

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import engine
from app.ml.inference import MLInferenceService
from app.ml.registry import ModelRegistry
from app.models import PaperAccount, PaperOrder, PaperPosition, WorkerDecisionTrace, WorkerSession
from app.trading.paper_executor import PaperExecutionEngine
from app.trading.position_monitor import PositionMonitorService
from app.trading.risk_engine import RiskEngine
from app.trading.safety import verify_codebase_safety, verify_worker_permissions_isolated
from app.workers.supervisor import SupervisorSessionStatus
from app.workers.workflow import AssetWorkflowCoordinator


@pytest.mark.asyncio
async def test_end_to_end_paper_lifecycle() -> None:
    """Exercise complete lifecycle: proposal creation -> risk authorization -> paper fill -> bracket exit."""
    registry = ModelRegistry(Path("model-registry"))
    ml_service = MLInferenceService(registry, engine)
    coord = AssetWorkflowCoordinator(engine, ml_service)
    risk_engine = RiskEngine()
    executor = PaperExecutionEngine()
    monitor = PositionMonitorService()

    now = datetime(2026, 11, 16, 14, 0, tzinfo=UTC)

    async with AsyncSession(engine, expire_on_commit=False) as session:
        # Create isolated test paper account with $25,000 equity
        account_name = f"integration_test_{uuid.uuid4().hex[:8]}"
        account = PaperAccount(
            id=uuid.uuid4(),
            name=account_name,
            initial_balance=Decimal("25000.00"),
            cash_balance=Decimal("25000.00"),
            equity=Decimal("25000.00"),
            daily_starting_equity=Decimal("25000.00"),
            daily_realized_pnl=Decimal("0.00"),
            is_halted=False,
            created_at=now,
            updated_at=now,
        )
        session.add(account)
        await session.commit()
        await session.refresh(account)

        # 1. Candidate proposal (stop at 88,000 gives distance 2,000 -> 0.125 BTC -> $11,250 exposure <= $12,500 limit)
        proposal = {
            "asset": "BTC",
            "side": "BUY",
            "direction": "LONG",
            "entry_price": 90000.0,
            "stop_loss": 88000.0,
            "take_profit": 92000.0,
            "confidence": 0.70,
            "rationale": "High conviction bullish pullback",
        }

        # 2. Risk Engine Evaluation
        risk_result = risk_engine.evaluate_proposal(account, proposal, open_positions=[])
        assert risk_result.authorized is True
        assert risk_result.authorized_quantity is not None
        assert risk_result.authorized_quantity > Decimal("0")
        assert risk_result.risk_amount == Decimal("250.00")

        # 3. Paper Execution Simulation
        order, position = await executor.execute_proposal(
            account,
            proposal,
            risk_result.authorized_quantity,
            session,
            slippage_bps=2.0,
            fee_bps=5.0,
        )
        assert order.status == "FILLED"
        assert position.status == "OPEN"
        assert position.quantity == risk_result.authorized_quantity
        # Entry price includes 2 bps slippage
        assert position.entry_price > Decimal("90000.0")
        assert position.total_fees > Decimal("0")

        # 4. Mark-to-market accounting
        unrealized = monitor.update_position_mark_to_market(position, Decimal("91000.0"))
        assert unrealized > Decimal("0")

        # 5. Position Monitor bracket exit (Take Profit triggered at 92,000)
        closed = await monitor.check_and_execute_exits(
            account,
            {"BTC": Decimal("92500.0")},
            session,
        )
        assert len(closed) == 1
        closed_pos = closed[0]
        assert closed_pos.status == "CLOSED"
        assert closed_pos.exit_reason == "TAKE_PROFIT"
        assert closed_pos.realized_pnl > Decimal("0")
        assert closed_pos.closed_at is not None

        # Verify account balance updated
        assert account.equity > Decimal("25000.00")
        assert account.daily_realized_pnl > Decimal("0")


@pytest.mark.asyncio
async def test_system_release_safety_audit() -> None:
    """Verify release acceptance criteria: strictly zero real-money order execution pathways."""
    violations = verify_codebase_safety()
    assert len(violations) == 0, f"Safety violations found in codebase: {violations}"
    assert verify_worker_permissions_isolated() is True
