"""Tests for Telegram Human-In-The-Loop (HITL) Alert and Override System."""

from datetime import UTC, datetime
from decimal import Decimal
import uuid
import pytest
from unittest.mock import AsyncMock, patch

from app.notifications.telegram_service import TelegramBotService
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
async def test_telegram_service_configuration():
    service = TelegramBotService(bot_token="123:test_token", chat_id="999888")
    assert service.is_configured is True
    assert service.bot_token == "123:test_token"
    assert service.chat_id == "999888"


@pytest.mark.asyncio
async def test_telegram_send_blocked_trade_alert_and_deduplication():
    service = TelegramBotService(bot_token="123:fake", chat_id="999")

    # Mock send_message to prevent actual Telegram network call
    service.send_message = AsyncMock(return_value={"message_id": 1042})

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
    assert proposal["entry_price"] == 57960.0
    assert proposal["status"] == "PENDING"
    assert proposal["telegram_message_id"] == 1042

    # Verify send_message was called with inline keyboard
    service.send_message.assert_called_once()
    call_args = service.send_message.call_args[0]
    call_kwargs = service.send_message.call_args[1]
    msg_text = call_args[0]
    reply_markup = call_kwargs["reply_markup"]

    assert "Trade Opportunity Blocked" in msg_text
    assert "BTCUSD" in msg_text
    assert "SHORT" in msg_text
    assert "Session bias is NO_TRADE" in msg_text
    assert reply_markup["inline_keyboard"][0][0]["callback_data"] == f"override_exec:{prop_id}"
    assert reply_markup["inline_keyboard"][0][1]["callback_data"] == f"override_rej:{prop_id}"

    # Verify deduplication within window returns None on immediate second call
    second_call = await service.send_blocked_trade_alert(
        asset="BTC",
        direction="SHORT",
        entry_price=57950.0,
        stop_loss=58650.0,
        take_profit=56500.0,
        setup_type="MOMENTUM_CONTINUATION",
        rejection_reasons=["Session bias is NO_TRADE"],
        wait_reasons=[],
        worker_summary={},
    )
    assert second_call is None


@pytest.mark.asyncio
async def test_telegram_reject_override():
    service = TelegramBotService(bot_token="123:fake", chat_id="999")
    service.pending_proposals["prop-123"] = {
        "id": "prop-123",
        "asset": "ETH",
        "status": "PENDING",
    }

    success, msg = await service.reject_override("prop-123")
    assert success is True
    assert service.pending_proposals["prop-123"]["status"] == "REJECTED"


@pytest.mark.asyncio
async def test_telegram_execute_override(mock_account):
    service = TelegramBotService(bot_token="123:fake", chat_id="999")
    prop_id = "prop-exec-456"
    service.pending_proposals[prop_id] = {
        "id": prop_id,
        "asset": "BTC",
        "side": "SELL",
        "direction": "SHORT",
        "entry_price": 58000.0,
        "stop_loss": 59450.0,  # 2.5% stop distance ($1450), keeping notional < 50% max exposure
        "take_profit": 56500.0,
        "status": "PENDING",
    }

    risk_engine = RiskEngine()
    executor = PaperExecutionEngine()

    mock_session = AsyncMock()
    mock_session.commit = AsyncMock()

    # Mock execute_proposal
    mock_order = AsyncMock()
    mock_order.id = uuid.uuid4()
    mock_order.executed_price = Decimal("57988.40")

    mock_pos = AsyncMock()
    mock_pos.id = uuid.uuid4()
    mock_pos.direction = "SHORT"
    mock_pos.asset = "BTC"

    executor.execute_proposal = AsyncMock(return_value=(mock_order, mock_pos))

    # Mock query for open positions
    from unittest.mock import MagicMock
    mock_res = MagicMock()
    mock_scalars = MagicMock()
    mock_scalars.all.return_value = []
    mock_res.scalars.return_value = mock_scalars
    mock_session.execute = AsyncMock(return_value=mock_res)

    success, msg, executed_prop = await service.execute_override(
        prop_id,
        mock_account,
        mock_session,
        risk_engine,
        executor,
    )

    assert success is True
    assert "Executed SHORT BTC" in msg
    assert executed_prop["status"] == "EXECUTED"
    assert executed_prop["executed_price"] == 57988.40
    assert executed_prop["order_id"] == str(mock_order.id)
    assert executed_prop["position_id"] == str(mock_pos.id)
