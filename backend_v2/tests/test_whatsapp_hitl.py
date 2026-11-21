"""Tests for WhatsApp Web Notification & Human-In-The-Loop (HITL) Alert and Override System."""

from datetime import UTC, datetime
from decimal import Decimal
import uuid
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.notifications.whatsapp_service import WhatsAppBotService
from app.models import PaperAccount, PaperPosition
from app.trading.risk_engine import RiskEngine
from app.trading.paper_executor import PaperExecutionEngine


@pytest.fixture
def mock_account():
    now = datetime.now(UTC)
    b = Decimal("10000.00")
    return PaperAccount(
        id=uuid.uuid4(),
        name="test_account",
        initial_balance=b,
        cash_balance=b,
        equity=b,
        daily_starting_equity=b,
        daily_realized_pnl=Decimal("0"),
        is_halted=False,
        created_at=now,
        updated_at=now,
    )


@pytest.mark.asyncio
async def test_whatsapp_service_configuration():
    service = WhatsAppBotService()
    service.configure(bridge_url="http://whatsapp:3001", target_number="919876543210", enabled=True)
    assert service.bridge_url == "http://whatsapp:3001"
    assert service.target_number == "919876543210"
    assert service.enabled is True


@pytest.mark.asyncio
async def test_whatsapp_send_blocked_trade_alert_and_deduplication():
    service = WhatsAppBotService()
    service.send_message = AsyncMock(return_value=True)

    prop_id = await service.send_blocked_trade_alert(
        asset="BTC",
        direction="SHORT",
        entry_price=57960.0,
        stop_loss=58650.0,
        take_profit=56500.0,
        setup_type="MOMENTUM_CONTINUATION",
        rejection_reasons=["Session bias is NO_TRADE"],
        wait_reasons=["Market regime is high_volatility"],
        worker_summary={
            "regime": "high_volatility",
            "session_bias": "NO_TRADE",
            "technical_summary": "MOMENTUM_CONTINUATION (RSI 56.9)",
            "ml_summary": "SHORT (Conf: 68.4%)",
            "order_flow_summary": "Net selling pressure",
        },
    )

    assert prop_id is not None
    assert prop_id in service.pending_proposals
    proposal = service.pending_proposals[prop_id]
    assert proposal["asset"] == "BTC"
    assert proposal["direction"] == "SHORT"
    assert proposal["status"] == "PENDING"
    assert service.latest_proposal_id == prop_id

    # Deduplication test: sending again immediately should return None
    dup_id = await service.send_blocked_trade_alert(
        asset="BTC",
        direction="SHORT",
        entry_price=57960.0,
        stop_loss=58650.0,
        take_profit=56500.0,
        setup_type="MOMENTUM_CONTINUATION",
        rejection_reasons=["Session bias is NO_TRADE"],
        wait_reasons=[],
        worker_summary={},
    )
    assert dup_id is None


@pytest.mark.asyncio
async def test_whatsapp_execute_override(mock_account):
    service = WhatsAppBotService()
    service.send_message = AsyncMock(return_value=True)

    prop_id = await service.send_blocked_trade_alert(
        asset="BTC",
        direction="SHORT",
        entry_price=58000.0,
        stop_loss=59450.0,
        take_profit=55100.0,
        setup_type="TREND_PULLBACK_BREAKDOWN",
        rejection_reasons=["Session bias NO_TRADE"],
        wait_reasons=[],
        worker_summary={},
    )

    risk_engine = RiskEngine()
    executor = PaperExecutionEngine()

    mock_session = AsyncMock()
    mock_res = MagicMock()
    mock_scalars = MagicMock()
    mock_scalars.all.return_value = []
    mock_res.scalars.return_value = mock_scalars
    mock_session.execute.return_value = mock_res
    mock_session.flush = AsyncMock()

    success, msg, executed_prop = await service.execute_override(
        prop_id,
        mock_account,
        mock_session,
        risk_engine,
        executor,
    )

    assert success is True
    assert "FILLED" in msg
    assert executed_prop is not None
    assert executed_prop["status"] == "EXECUTED"
    assert executed_prop["executed_size"] > 0


@pytest.mark.asyncio
async def test_whatsapp_handle_incoming_messages(mock_account):
    service = WhatsAppBotService()
    service.send_message = AsyncMock(return_value=True)

    prop_id = await service.send_blocked_trade_alert(
        asset="ETH",
        direction="SHORT",
        entry_price=1630.0,
        stop_loss=1670.0,
        take_profit=1550.0,
        setup_type="BEARISH_MOMENTUM",
        rejection_reasons=["Session bias NO_TRADE"],
        wait_reasons=[],
        worker_summary={},
    )

    risk_engine = RiskEngine()
    executor = PaperExecutionEngine()

    mock_session = AsyncMock()
    mock_res = MagicMock()
    mock_scalars = MagicMock()
    mock_scalars.all.return_value = []
    mock_res.scalars.return_value = mock_scalars
    mock_session.execute.return_value = mock_res
    mock_session.flush = AsyncMock()

    # User replies "1" to approve
    reply = await service.handle_incoming_message(
        phone="919876543210",
        text="1",
        session=mock_session,
        risk_engine=risk_engine,
        executor=executor,
        account=mock_account,
    )

    assert "TRADE OVERRIDE EXECUTED" in reply
    assert "ETHUSD" in reply

    # User asks for STATUS
    status_reply = await service.handle_incoming_message(
        phone="919876543210",
        text="STATUS",
        session=mock_session,
        risk_engine=risk_engine,
        executor=executor,
        account=mock_account,
    )
    assert "DeltaBot 2.0 Account Status" in status_reply
    assert "Total Equity:" in status_reply
