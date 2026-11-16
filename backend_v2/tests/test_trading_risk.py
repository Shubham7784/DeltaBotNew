"""Unit tests for Phase 8: Risk Engine, Paper Execution, Position Monitor, and Safety."""

from datetime import UTC, datetime
from decimal import Decimal
import uuid

import pytest

from app.models import PaperAccount, PaperPosition
from app.trading.paper_executor import PaperExecutionEngine
from app.trading.position_monitor import PositionMonitorService
from app.trading.risk_engine import RiskEngine, RiskParameters
from app.trading.safety import verify_codebase_safety, verify_worker_permissions_isolated


def create_test_account(
    balance: float = 10000.0,
    daily_loss: float = 0.0,
    is_halted: bool = False,
) -> PaperAccount:
    now = datetime(2026, 11, 16, 12, 0, tzinfo=UTC)
    b = Decimal(str(balance))
    return PaperAccount(
        id=uuid.uuid4(),
        name="test_account",
        initial_balance=b,
        cash_balance=b,
        equity=b,
        daily_starting_equity=b,
        daily_realized_pnl=Decimal(str(daily_loss)),
        is_halted=is_halted,
        created_at=now,
        updated_at=now,
    )


def test_risk_position_sizing_calculation() -> None:
    engine = RiskEngine()
    account = create_test_account(10000.0)

    # 1% risk on $10,000 = $100 risk capital
    # Entry: 50,000, Stop: 49,000 -> Distance = 1,000
    # Expected size = 100 / 1000 = 0.1000 BTC
    proposal = {
        "asset": "BTC",
        "entry_price": 50000.0,
        "stop_loss": 49000.0,
        "take_profit": 52000.0,
    }
    result = engine.evaluate_proposal(account, proposal, open_positions=[])

    assert result.authorized is True
    assert result.authorized_quantity == Decimal("0.1000")
    assert result.risk_amount == Decimal("100.00")
    assert result.notional_value == Decimal("5000.00")


def test_risk_rejects_duplicate_asset_position() -> None:
    engine = RiskEngine()
    account = create_test_account(10000.0)

    existing_pos = PaperPosition(
        id=uuid.uuid4(),
        account_id=account.id,
        asset="BTC",
        direction="LONG",
        status="OPEN",
        quantity=Decimal("0.1"),
        entry_price=Decimal("50000.0"),
        stop_loss=Decimal("49000.0"),
        take_profit=Decimal("52000.0"),
        opened_at=datetime.now(UTC),
    )

    proposal = {
        "asset": "BTC",
        "entry_price": 50500.0,
        "stop_loss": 49500.0,
    }
    result = engine.evaluate_proposal(account, proposal, open_positions=[existing_pos])

    assert result.authorized is False
    assert "duplicate entry rejected" in (result.reason or "")


def test_risk_rejects_exceeding_max_exposure() -> None:
    engine = RiskEngine()
    account = create_test_account(10000.0)

    # Sizing: $100 risk capital / $10 distance = 10 units!
    # 10 units * $1000 entry = $10,000 notional (100% exposure > 50% limit)
    proposal = {
        "asset": "ETH",
        "entry_price": 1000.0,
        "stop_loss": 990.0,
    }
    result = engine.evaluate_proposal(account, proposal, open_positions=[])

    assert result.authorized is False
    assert "max allowed 50% limit" in (result.reason or "")


def test_risk_rejects_on_daily_loss_halt() -> None:
    engine = RiskEngine()
    # 3% of $10,000 is -$300. Set daily loss to -$350
    account = create_test_account(balance=10000.0, daily_loss=-350.0)

    proposal = {
        "asset": "BTC",
        "entry_price": 50000.0,
        "stop_loss": 49000.0,
    }
    result = engine.evaluate_proposal(account, proposal, open_positions=[])

    assert result.authorized is False
    assert "Daily loss halt triggered" in (result.reason or "")
    assert account.is_halted is True


def test_position_monitor_mark_to_market() -> None:
    service = PositionMonitorService()
    now = datetime(2026, 11, 16, 12, 0, tzinfo=UTC)

    pos_long = PaperPosition(
        id=uuid.uuid4(),
        account_id=uuid.uuid4(),
        asset="BTC",
        direction="LONG",
        status="OPEN",
        quantity=Decimal("0.5"),
        entry_price=Decimal("90000.0"),
        stop_loss=Decimal("88000.0"),
        take_profit=Decimal("94000.0"),
        opened_at=now,
    )

    # Price moves to 92,000 (+2,000 * 0.5 = +1,000)
    unrealized = service.update_position_mark_to_market(pos_long, Decimal("92000.0"))
    assert unrealized == Decimal("1000.0000")

    pos_short = PaperPosition(
        id=uuid.uuid4(),
        account_id=uuid.uuid4(),
        asset="ETH",
        direction="SHORT",
        status="OPEN",
        quantity=Decimal("2.0"),
        entry_price=Decimal("3000.0"),
        stop_loss=Decimal("3100.0"),
        take_profit=Decimal("2800.0"),
        opened_at=now,
    )

    # Price moves down to 2,900 (+100 * 2.0 = +200)
    unrealized_short = service.update_position_mark_to_market(pos_short, Decimal("2900.0"))
    assert unrealized_short == Decimal("200.0000")


def test_safety_worker_permissions_isolated() -> None:
    """Prove that no worker role has order placement or execution tool capabilities."""
    assert verify_worker_permissions_isolated() is True


def test_safety_codebase_audit() -> None:
    """Prove that no live exchange order APIs or live credentials exist in app/."""
    violations = verify_codebase_safety()
    assert len(violations) == 0, f"Found violations: {violations}"
