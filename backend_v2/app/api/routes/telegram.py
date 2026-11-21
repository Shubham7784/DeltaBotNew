"""Telegram Notification & Human-In-The-Loop (HITL) Override API Routes."""

from datetime import UTC, datetime
import logging
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import engine, redis_client
import json
from app.notifications.telegram_service import telegram_service
from app.trading.paper_executor import PaperExecutionEngine
from app.trading.risk_engine import RiskEngine
from scripts.daemon_paper_trader import get_or_create_paper_account

router = APIRouter(prefix="/api/v1/telegram", tags=["Telegram & HITL Overrides"])
logger = logging.getLogger(__name__)


class TelegramConfigPayload(BaseModel):
    bot_token: str = Field(..., min_length=1)
    chat_id: str = Field(..., min_length=1)


class OverrideActionPayload(BaseModel):
    proposal_id: str = Field(..., min_length=1)
    action: Literal["approve", "reject"]


@router.get("/status")
async def get_telegram_status() -> dict[str, Any]:
    """Check configuration and operational status of Telegram HITL alerts."""
    return {
        "configured": telegram_service.is_configured,
        "chat_id_configured": bool(telegram_service.chat_id),
        "pending_proposals_count": len(telegram_service.pending_proposals),
        "timestamp": datetime.now(UTC).isoformat(),
    }


@router.post("/config")
async def update_telegram_config(payload: TelegramConfigPayload) -> dict[str, Any]:
    """Dynamically configure Telegram Bot Token and Target Chat ID."""
    telegram_service.bot_token = payload.bot_token.strip()
    telegram_service.chat_id = payload.chat_id.strip()
    logger.info("Updated Telegram bot credentials via API.")
    return {
        "success": True,
        "configured": telegram_service.is_configured,
        "message": "Telegram credentials updated successfully.",
    }


@router.post("/test")
async def send_test_telegram_message() -> dict[str, Any]:
    """Send a test verification message to the configured Telegram chat."""
    if not telegram_service.is_configured:
        raise HTTPException(
            status_code=400,
            detail="Telegram bot_token or chat_id not configured in environment or settings.",
        )

    test_text = (
        "🤖 <b>DeltaBot 2.0 — Telegram Notification Channel Online</b>\n\n"
        "✅ Real-time multi-agent alert & interactive override system is active.\n"
        f"<i>Timestamp: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}</i>"
    )
    res = await telegram_service.send_message(test_text)
    if not res:
        raise HTTPException(
            status_code=502,
            detail="Failed to send message to Telegram API. Check bot token and chat ID.",
        )
    return {"success": True, "result": res}


@router.get("/proposals")
async def list_proposals() -> list[dict[str, Any]]:
    """List all recent candidate trade proposals pending or resolved."""
    proposals = dict(telegram_service.pending_proposals)
    try:
        keys = await redis_client.keys("telegram:proposals:*")
        for key in keys:
            raw = await redis_client.get(key)
            if raw:
                prop = json.loads(raw)
                pid = prop.get("proposal_id")
                if pid and pid not in proposals:
                    proposals[pid] = prop
    except Exception as e:
        logger.warning("Failed scanning Redis for telegram proposals: %s", e)
    sorted_props = sorted(
        proposals.values(),
        key=lambda x: str(x.get("created_at", "")),
        reverse=True,
    )
    return sorted_props


@router.post("/override")
async def execute_manual_override(payload: OverrideActionPayload) -> dict[str, Any]:
    """Execute or reject a candidate trade proposal directly via API or UI."""
    proposal = await telegram_service.get_proposal(payload.proposal_id)
    if not proposal:
        raise HTTPException(status_code=404, detail="Proposal not found or expired.")

    if payload.action == "reject":
        success, msg = await telegram_service.reject_override(payload.proposal_id)
        return {"success": success, "action": "reject", "message": msg}

    # Action is approve
    risk_engine = RiskEngine()
    executor = PaperExecutionEngine()

    async with AsyncSession(engine, expire_on_commit=False) as session:
        account = await get_or_create_paper_account(session)
        success, msg, executed_prop = await telegram_service.execute_override(
            payload.proposal_id,
            account,
            session,
            risk_engine,
            executor,
        )

    if not success:
        raise HTTPException(status_code=400, detail=msg)

    return {
        "success": True,
        "action": "approve",
        "message": msg,
        "proposal": executed_prop,
    }
