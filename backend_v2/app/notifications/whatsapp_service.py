"""WhatsApp Web Notification & Human-In-The-Loop (HITL) Alert Service.

Connects to the local WhatsApp Web bridge (Baileys) to send blocked trade candidate alerts
and handle operator overrides when users reply '1' (approve) or '2' (reject).
"""

from datetime import UTC, datetime, timedelta
from decimal import Decimal
import json
import logging
from typing import Any
import uuid

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.database import redis_client
from app.models import PaperAccount, PaperOrder, PaperPosition
from app.trading.paper_executor import PaperExecutionEngine
from app.trading.risk_engine import RiskEngine

logger = logging.getLogger(__name__)

DEDUPLICATION_WINDOW_MINUTES = 15
PROPOSAL_EXPIRY_MINUTES = 30


class WhatsAppBotService:
    """Manages WhatsApp notifications and interactive Human-In-The-Loop trade overrides."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.bridge_url = str(self.settings.whatsapp_bridge_url).rstrip("/")
        self.target_number = self.settings.whatsapp_target_number or ""
        self.enabled = bool(self.settings.whatsapp_enabled)

        # In-memory tracking
        self.pending_proposals: dict[str, dict[str, Any]] = {}
        self.last_alert_time: dict[str, datetime] = {}
        # Stores the latest pending proposal ID (or per sender phone) so users can simply reply "1" or "2"
        self.latest_proposal_id: str | None = None
        self.latest_proposal_per_phone: dict[str, str] = {}
        self.linked_phone: str | None = None

    def configure(self, *, bridge_url: str | None = None, target_number: str | None = None, enabled: bool | None = None) -> None:
        """Update runtime configuration."""
        if bridge_url is not None:
            self.bridge_url = bridge_url.rstrip("/")
        if target_number is not None:
            self.target_number = target_number.strip().replace("+", "").replace(" ", "").replace("-", "")
        if enabled is not None:
            self.enabled = enabled
        logger.info(
            "WhatsApp service reconfigured: bridge_url=%s, target_number=%s, enabled=%s",
            self.bridge_url,
            self.target_number or "Auto-Linked",
            self.enabled,
        )

    async def get_bridge_status(self) -> dict[str, Any]:
        """Fetch live status from the WhatsApp Web bridge."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self.bridge_url}/status")
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("phone"):
                        self.linked_phone = data["phone"]
                    return data
        except Exception as e:
            logger.debug("WhatsApp bridge status check failed: %s", e)
        return {
            "connected": False,
            "phone": self.linked_phone,
            "has_qr": False,
            "error": "WhatsApp bridge unreachable",
        }

    async def send_message(self, text: str, recipient: str | None = None) -> bool:
        """Send a WhatsApp text message via the local bridge."""
        if not self.enabled:
            return False

        target = recipient or self.target_number or self.linked_phone
        payload = {"text": text}
        if target:
            payload["to"] = target

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(f"{self.bridge_url}/send", json=payload)
                if resp.status_code == 200:
                    return True
                logger.warning("WhatsApp bridge returned %d: %s", resp.status_code, resp.text)
        except Exception as e:
            logger.error("Failed sending message via WhatsApp bridge: %s", e)
        return False

    async def send_blocked_trade_alert(
        self,
        *,
        asset: str,
        direction: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        setup_type: str,
        rejection_reasons: list[str],
        wait_reasons: list[str],
        worker_summary: dict[str, Any],
    ) -> str | None:
        """Create and send an interactive WhatsApp alert for a blocked trade candidate."""
        now = datetime.now(UTC)
        dedup_key = f"{asset}:{direction}"

        # Deduplication check: prevent spamming every cycle for the same ongoing setup
        last_sent = self.last_alert_time.get(dedup_key)
        if last_sent and (now - last_sent) < timedelta(minutes=DEDUPLICATION_WINDOW_MINUTES):
            logger.info(
                "Skipping duplicate WhatsApp alert for %s %s (sent %.1f min ago)",
                asset,
                direction,
                (now - last_sent).total_seconds() / 60.0,
            )
            return None

        proposal_id = str(uuid.uuid4())
        side = "BUY" if direction.upper() == "LONG" else "SELL"

        proposal_data = {
            "proposal_id": proposal_id,
            "asset": asset,
            "direction": direction,
            "side": side,
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "setup_type": setup_type,
            "rejection_reasons": rejection_reasons,
            "wait_reasons": wait_reasons,
            "worker_summary": worker_summary,
            "status": "PENDING",
            "created_at": now.isoformat(),
        }

        # Store in-memory and Redis
        self.pending_proposals[proposal_id] = proposal_data
        self.latest_proposal_id = proposal_id
        try:
            await redis_client.set(
                f"telegram:proposals:{proposal_id}",
                json.dumps(proposal_data),
                ex=PROPOSAL_EXPIRY_MINUTES * 60,
            )
            await redis_client.set("whatsapp:latest_proposal", proposal_id, ex=PROPOSAL_EXPIRY_MINUTES * 60)
        except Exception:
            pass

        self.last_alert_time[dedup_key] = now

        # Format worker consensus highlights
        tech_str = worker_summary.get("technical_summary", setup_type)
        ml_str = worker_summary.get("ml_summary", "Directional signal active")
        of_str = worker_summary.get("order_flow_summary", "Order book active")
        regime = worker_summary.get("regime", "N/A")
        session_bias = worker_summary.get("session_bias", "NO_TRADE")

        all_reasons = rejection_reasons + wait_reasons
        reasons_formatted = "\n".join(f"• _{r}_" for r in all_reasons) if all_reasons else "• Safety gate constraint active"

        dir_emoji = "📈 LONG" if direction.upper() == "LONG" else "📉 SHORT"

        message_text = (
            f"🚨 *DeltaBot 2.0 — Trade Blocked by Safety Gate* 🚨\n\n"
            f"*Asset:* {asset}USD\n"
            f"*Direction:* {dir_emoji}\n"
            f"*Entry Ref:* ${entry_price:,.2f}\n"
            f"*Stop Loss:* ${stop_loss:,.2f}\n"
            f"*Take Profit:* ${take_profit:,.2f}\n\n"
            f"💡 *Worker Consensus (Why Workers Want to Trade):*\n"
            f"• *Technical Worker:* {tech_str}\n"
            f"• *ML Prediction:* {ml_str}\n"
            f"• *Order Flow:* {of_str}\n\n"
            f"🛡️ *Safety Gate Interventions (Why Blocked):*\n"
            f"• Current Regime: `{regime}`\n"
            f"• Session Bias: `{session_bias}`\n"
            f"{reasons_formatted}\n\n"
            f"⚡ *OPERATOR DECISION:*\n"
            f"👉 Reply *1* (or *YES*) to *Place / Override Trade*\n"
            f"👉 Reply *2* (or *NO*) to *Reject Proposal*"
        )

        sent = await self.send_message(message_text)
        if sent:
            logger.info("Sent WhatsApp override alert for proposal %s (%s %s)", proposal_id, asset, direction)

        return proposal_id

    async def get_proposal(self, proposal_id: str) -> dict[str, Any] | None:
        """Fetch proposal from memory or Redis."""
        prop = self.pending_proposals.get(proposal_id)
        if prop is not None:
            return prop

        try:
            raw = await redis_client.get(f"telegram:proposals:{proposal_id}")
            if raw:
                prop = json.loads(raw)
                self.pending_proposals[proposal_id] = prop
                return prop
        except Exception:
            pass
        return None

    async def execute_override(
        self,
        proposal_id: str,
        account: PaperAccount,
        session: AsyncSession,
        risk_engine: RiskEngine,
        executor: PaperExecutionEngine,
    ) -> tuple[bool, str, dict[str, Any] | None]:
        """Execute approved proposal with deterministic risk-managed position sizing."""
        proposal = await self.get_proposal(proposal_id)
        if not proposal:
            return False, f"Proposal {proposal_id} not found or expired.", None

        if proposal.get("status") != "PENDING":
            return False, f"Proposal {proposal_id} is already {proposal.get('status')}.", None

        # Query open positions for risk sizing
        stmt = select(PaperPosition).where(
            PaperPosition.account_id == account.id,
            PaperPosition.status == "OPEN",
        )
        res = await session.execute(stmt)
        open_positions = list(res.scalars().all())

        # Construct trade proposal object for Risk Engine
        cand_proposal = {
            "asset": proposal["asset"],
            "side": proposal["side"],
            "direction": proposal["direction"],
            "entry_price": proposal["entry_price"],
            "stop_loss": proposal["stop_loss"],
            "take_profit": proposal["take_profit"],
            "rationale": f"Manual WhatsApp Operator Override (HITL Proposal #{proposal_id[:8]})",
        }

        risk_decision = risk_engine.evaluate_proposal(account, cand_proposal, open_positions)
        if not risk_decision.authorized or not risk_decision.authorized_quantity:
            proposal["status"] = "RISK_REJECTED"
            return False, f"Risk Engine rejected override: {risk_decision.reason}", None

        # Execute paper order and position
        order, pos = await executor.execute_proposal(
            account,
            cand_proposal,
            risk_decision.authorized_quantity,
            session,
        )
        await session.commit()

        proposal["status"] = "EXECUTED"
        proposal["executed_at"] = datetime.now(UTC).isoformat()
        proposal["order_id"] = str(order.id)
        proposal["executed_size"] = float(risk_decision.authorized_quantity)
        proposal["executed_price"] = float(order.executed_price)

        # Update in-memory & Redis
        self.pending_proposals[proposal_id] = proposal
        try:
            await redis_client.set(
                f"telegram:proposals:{proposal_id}",
                json.dumps(proposal),
                ex=PROPOSAL_EXPIRY_MINUTES * 60,
            )
        except Exception:
            pass

        logger.info(
            "WhatsApp operator override executed: Proposal %s -> Order %s (%s %s %s @ %s)",
            proposal_id,
            order.id,
            proposal["side"],
            risk_decision.authorized_quantity,
            f"{proposal['asset']}USD",
            order.executed_price,
        )
        return True, f"Paper trade executed successfully. Order #{order.id} FILLED.", proposal

    async def reject_override(self, proposal_id: str) -> tuple[bool, str]:
        """Mark proposal as rejected by operator."""
        proposal = await self.get_proposal(proposal_id)
        if not proposal:
            return False, f"Proposal {proposal_id} not found or expired."

        if proposal.get("status") != "PENDING":
            return False, f"Proposal {proposal_id} is already {proposal.get('status')}."

        proposal["status"] = "REJECTED"
        proposal["rejected_at"] = datetime.now(UTC).isoformat()

        self.pending_proposals[proposal_id] = proposal
        try:
            await redis_client.set(
                f"telegram:proposals:{proposal_id}",
                json.dumps(proposal),
                ex=PROPOSAL_EXPIRY_MINUTES * 60,
            )
        except Exception:
            pass

        logger.info("WhatsApp operator rejected proposal %s", proposal_id)
        return True, f"Proposal {proposal_id} rejected by operator."

    async def handle_incoming_message(
        self,
        phone: str,
        text: str,
        session: AsyncSession,
        risk_engine: RiskEngine,
        executor: PaperExecutionEngine,
        account: PaperAccount,
    ) -> str:
        """Process inbound text commands from the authorized WhatsApp user."""
        cmd = text.strip().upper()
        self.latest_proposal_per_phone[phone] = self.latest_proposal_id or ""

        # Option 1: Approve / Override Trade
        if cmd in {"1", "YES", "APPROVE", "BUY", "SELL", "EXECUTE", "OVERRIDE", "PLACE"}:
            proposal_id = self.latest_proposal_per_phone.get(phone) or self.latest_proposal_id
            if not proposal_id:
                # Check Redis
                try:
                    raw_id = await redis_client.get("whatsapp:latest_proposal")
                    if raw_id:
                        proposal_id = raw_id.decode() if isinstance(raw_id, bytes) else str(raw_id)
                except Exception:
                    pass

            if not proposal_id:
                return "⚠️ No pending trade proposals to approve. When specialist workers detect an opportunity, you'll receive an alert here."

            success, msg, executed = await self.execute_override(
                proposal_id,
                account,
                session,
                risk_engine,
                executor,
            )

            if success and executed:
                asset = executed.get("asset", "")
                direction = executed.get("direction", "")
                size = executed.get("executed_size", 0.0)
                px = executed.get("executed_price", 0.0)
                order_id = executed.get("order_id", "")
                return (
                    f"🚀 *TRADE OVERRIDE EXECUTED!* 🚀\n\n"
                    f"• *Asset:* {asset}USD\n"
                    f"• *Direction:* {direction}\n"
                    f"• *Order:* #{order_id} (FILLED)\n"
                    f"• *Size:* {size} (${size * px:,.2f})\n"
                    f"• *Execution Price:* ${px:,.2f}\n"
                    f"• *Risk Control:* 1.0% portfolio equity risk strictly maintained.\n"
                    f"• *Status:* Simulated Paper Position OPEN"
                )
            return f"❌ Trade override failed: {msg}"

        # Option 2: Reject Proposal
        if cmd in {"2", "NO", "REJECT", "CANCEL", "DISMISS"}:
            proposal_id = self.latest_proposal_per_phone.get(phone) or self.latest_proposal_id
            if not proposal_id:
                try:
                    raw_id = await redis_client.get("whatsapp:latest_proposal")
                    if raw_id:
                        proposal_id = raw_id.decode() if isinstance(raw_id, bytes) else str(raw_id)
                except Exception:
                    pass

            if not proposal_id:
                return "⚠️ No pending trade proposals to reject."

            success, msg = await self.reject_override(proposal_id)
            if success:
                return "❌ *Proposal Dismissed.* Trade proposal has been rejected. Safety gate remains closed."
            return f"⚠️ {msg}"

        # Status / Account inquiry
        if cmd in {"STATUS", "ACCOUNT", "EQUITY", "BALANCE"}:
            # Query positions
            stmt = select(PaperPosition).where(
                PaperPosition.account_id == account.id,
                PaperPosition.status == "OPEN",
            )
            res = await session.execute(stmt)
            open_pos = list(res.scalars().all())

            pos_desc = f"{len(open_pos)} open position(s)"
            if open_pos:
                pos_lines = [f"• {p.symbol} {p.side} {p.quantity} @ ${p.entry_price:,.2f}" for p in open_pos]
                pos_desc += "\n" + "\n".join(pos_lines)

            return (
                f"📊 *DeltaBot 2.0 Account Status*\n\n"
                f"• *Total Equity:* ${float(account.equity):,.2f}\n"
                f"• *Cash Balance:* ${float(account.cash_balance):,.2f}\n"
                f"• *Positions:* {pos_desc}\n"
                f"• *Mode:* Simulated Paper-Trading\n"
                f"• *Safety Gate:* ACTIVE"
            )

        # Positions inquiry
        if cmd in {"POSITIONS", "POS"}:
            stmt = select(PaperPosition).where(
                PaperPosition.account_id == account.id,
                PaperPosition.status == "OPEN",
            )
            res = await session.execute(stmt)
            open_pos = list(res.scalars().all())

            if not open_pos:
                return "ℹ️ *Open Positions:* 0 active positions. Portfolio is 100% in reserve buffer."

            lines = ["📋 *Active Paper Positions:*"]
            for p in open_pos:
                lines.append(f"• *{p.symbol}* ({p.side}): {p.quantity} @ ${p.entry_price:,.2f}")
            return "\n".join(lines)

        # Help
        return (
            f"🤖 *DeltaBot 2.0 WhatsApp Bot Guide*\n\n"
            f"When a trade candidate is blocked by safety gates, you'll receive an alert here.\n\n"
            f"*Available Commands:*\n"
            f"• *1* or *YES* — Place / Override blocked trade\n"
            f"• *2* or *NO* — Reject & dismiss proposal\n"
            f"• *STATUS* — Check account equity & margin\n"
            f"• *POSITIONS* — View open paper positions\n"
            f"• *HELP* — Show this menu"
        )


whatsapp_service = WhatsAppBotService()
