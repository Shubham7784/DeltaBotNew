"""WhatsApp Web Notification & Human-In-The-Loop (HITL) API Routes."""

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import engine
from app.notifications.whatsapp_service import whatsapp_service
from app.trading.paper_executor import PaperExecutionEngine
from app.trading.risk_engine import RiskEngine
from scripts.daemon_paper_trader import get_or_create_paper_account

router = APIRouter(prefix="/api/v1/whatsapp", tags=["WhatsApp & HITL Overrides"])
logger = logging.getLogger(__name__)


class WhatsAppConfigPayload(BaseModel):
    bridge_url: str | None = None
    target_number: str | None = None
    enabled: bool | None = None


class WhatsAppIncomingPayload(BaseModel):
    from_jid: str | None = None
    phone: str = Field(..., min_length=1)
    text: str = Field(..., min_length=1)
    timestamp: int | None = None


class WhatsAppStatusReportPayload(BaseModel):
    connected: bool
    phone: str | None = None
    jid: str | None = None


@router.get("/status")
async def get_whatsapp_status() -> dict[str, Any]:
    """Check WhatsApp bridge connection status and linked phone number."""
    bridge_status = await whatsapp_service.get_bridge_status()
    return {
        "enabled": whatsapp_service.enabled,
        "bridge_url": whatsapp_service.bridge_url,
        "target_number": whatsapp_service.target_number,
        "linked_phone": whatsapp_service.linked_phone,
        "bridge": bridge_status,
        "pending_proposals_count": len(whatsapp_service.pending_proposals),
    }


@router.post("/status")
async def report_whatsapp_status(payload: WhatsAppStatusReportPayload) -> dict[str, Any]:
    """Webhook called by whatsapp-bridge when connection state updates."""
    if payload.phone:
        whatsapp_service.linked_phone = payload.phone
    logger.info("WhatsApp bridge state updated: connected=%s, phone=%s", payload.connected, payload.phone)
    return {"status": "ok"}


@router.post("/config")
async def update_whatsapp_config(payload: WhatsAppConfigPayload) -> dict[str, Any]:
    """Configure WhatsApp bridge URL and target phone number."""
    whatsapp_service.configure(
        bridge_url=payload.bridge_url,
        target_number=payload.target_number,
        enabled=payload.enabled,
    )
    return {
        "success": True,
        "bridge_url": whatsapp_service.bridge_url,
        "target_number": whatsapp_service.target_number,
        "enabled": whatsapp_service.enabled,
    }


@router.post("/incoming")
async def handle_whatsapp_incoming(payload: WhatsAppIncomingPayload) -> dict[str, Any]:
    """Process incoming WhatsApp message from user (e.g. '1' to override, '2' to reject)."""
    risk_engine = RiskEngine()
    executor = PaperExecutionEngine()

    async with AsyncSession(engine, expire_on_commit=False) as session:
        account = await get_or_create_paper_account(session)
        reply = await whatsapp_service.handle_incoming_message(
            phone=payload.phone,
            text=payload.text,
            session=session,
            risk_engine=risk_engine,
            executor=executor,
            account=account,
        )

    return {"reply": reply}


@router.post("/test")
async def send_whatsapp_test_message(phone: str | None = None) -> dict[str, Any]:
    """Send a test notification to verify the WhatsApp bridge."""
    target = phone or whatsapp_service.target_number or whatsapp_service.linked_phone
    test_text = (
        "🤖 *DeltaBot 2.0 WhatsApp Alert Test*\n\n"
        "Your WhatsApp is successfully linked to DeltaBot 2.0!\n\n"
        "When specialist workers detect trade opportunities blocked by safety gates, "
        "you will receive interactive alerts here.\n\n"
        "Reply *STATUS* anytime to see live account balance and open positions."
    )
    sent = await whatsapp_service.send_message(test_text, recipient=target)
    if not sent:
        raise HTTPException(
            status_code=502,
            detail="Failed to send message via WhatsApp bridge. Ensure WhatsApp is linked via QR code at http://localhost:3001/qr.",
        )
    return {"success": True, "recipient": target, "message": "Test WhatsApp alert delivered"}
