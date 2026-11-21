"""Telegram Human-In-The-Loop (HITL) Notification and Trade Override Service.

Delivers interactive real-time Telegram alerts when specialist workers detect a candidate trade
opportunity that is blocked by safety gates (regime, session bias, news risk, or validator constraints).
Allows operator to review the exact multi-agent rationale and override or reject the trade via inline buttons.
"""

import asyncio
from datetime import UTC, datetime, timedelta
from decimal import Decimal
import json
import logging
import os
from typing import Any, Callable
import uuid

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.database import redis_client
from app.models import PaperAccount, PaperPosition
from app.trading.paper_executor import PaperExecutionEngine
from app.trading.risk_engine import RiskEngine

logger = logging.getLogger(__name__)

PROPOSAL_EXPIRY_MINUTES = 30
DEDUPLICATION_WINDOW_MINUTES = 15


class TelegramBotService:
    """Asynchronous Telegram client for interactive operator alerts and trade overrides."""

    def __init__(
        self,
        bot_token: str | None = None,
        chat_id: str | None = None,
    ):
        settings = get_settings()
        self.bot_token = (
            bot_token
            or os.environ.get("TELEGRAM_BOT_TOKEN")
            or settings.telegram_bot_token
        ).strip()
        self.chat_id = (
            chat_id
            or os.environ.get("TELEGRAM_CHAT_ID")
            or settings.telegram_chat_id
        ).strip()
        self.pending_proposals: dict[str, dict[str, Any]] = {}
        self.last_alert_time: dict[str, datetime] = {}
        self._last_update_id = 0
        self._is_polling = False

    @property
    def is_configured(self) -> bool:
        """Check if Telegram credentials have been configured."""
        return bool(self.bot_token and self.chat_id)

    def _api_url(self, method: str) -> str:
        return f"https://api.telegram.org/bot{self.bot_token}/{method}"

    async def send_message(
        self,
        text: str,
        reply_markup: dict[str, Any] | None = None,
        parse_mode: str = "HTML",
    ) -> dict[str, Any] | None:
        """Send a formatted message to the configured Telegram chat."""
        if not self.is_configured:
            logger.debug("Telegram not configured; skipping send_message.")
            return None

        payload: dict[str, Any] = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": parse_mode,
        }
        if reply_markup is not None:
            payload["reply_markup"] = reply_markup

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(self._api_url("sendMessage"), json=payload)
                if resp.status_code == 200:
                    return resp.json().get("result")
                logger.error(
                    "Telegram sendMessage failed [%d]: %s",
                    resp.status_code,
                    resp.text,
                )
        except Exception as e:
            logger.error("Telegram sendMessage exception: %s", e)
        return None

    async def edit_message_text(
        self,
        chat_id: str | int,
        message_id: int,
        text: str,
        reply_markup: dict[str, Any] | None = None,
        parse_mode: str = "HTML",
    ) -> bool:
        """Edit an existing Telegram message in place (e.g. after button click)."""
        if not self.is_configured:
            return False

        payload: dict[str, Any] = {
            "chat_id": chat_id,
            "message_id": message_id,
            "text": text,
            "parse_mode": parse_mode,
        }
        if reply_markup is not None:
            payload["reply_markup"] = reply_markup

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(self._api_url("editMessageText"), json=payload)
                return resp.status_code == 200
        except Exception as e:
            logger.error("Telegram editMessageText exception: %s", e)
        return False

    async def answer_callback_query(
        self,
        callback_query_id: str,
        text: str = "",
        show_alert: bool = False,
    ) -> bool:
        """Acknowledge button tap to stop the loading spinner on user client."""
        if not self.is_configured:
            return False

        payload = {
            "callback_query_id": callback_query_id,
            "text": text,
            "show_alert": show_alert,
        }
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(self._api_url("answerCallbackQuery"), json=payload)
                return resp.status_code == 200
        except Exception as e:
            logger.error("Telegram answerCallbackQuery exception: %s", e)
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
        """Create and send an interactive Telegram alert for a blocked trade candidate."""
        now = datetime.now(UTC)
        dedup_key = f"{asset}:{direction}"

        # Deduplication check: prevent spamming every 60s for the same ongoing setup
        last_sent = self.last_alert_time.get(dedup_key)
        if last_sent and (now - last_sent) < timedelta(minutes=DEDUPLICATION_WINDOW_MINUTES):
            logger.info(
                "Skipping duplicate Telegram alert for %s %s (sent %.1f min ago)",
                asset,
                direction,
                (now - last_sent).total_seconds() / 60.0,
            )
            return None

        proposal_id = str(uuid.uuid4())
        side = "BUY" if direction.upper() == "LONG" else "SELL"

        proposal_data = {
            "id": proposal_id,
            "asset": asset,
            "side": side,
            "direction": direction.upper(),
            "entry_price": float(entry_price),
            "stop_loss": float(stop_loss),
            "take_profit": float(take_profit),
            "setup_type": setup_type,
            "rejection_reasons": rejection_reasons,
            "wait_reasons": wait_reasons,
            "worker_summary": worker_summary,
            "status": "PENDING",
            "created_at": now.isoformat(),
        }

        # Store in-memory and Redis
        self.pending_proposals[proposal_id] = proposal_data
        try:
            await redis_client.set(
                f"telegram:proposals:{proposal_id}",
                json.dumps(proposal_data),
                ex=PROPOSAL_EXPIRY_MINUTES * 60,
            )
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
        reasons_formatted = "\n".join(f"• <i>{r}</i>" for r in all_reasons) if all_reasons else "• Safety gate constraint active"

        dir_emoji = "📈 LONG" if direction.upper() == "LONG" else "📉 SHORT"

        message_text = (
            f"🚨 <b>DeltaBot 2.0 — Trade Opportunity Blocked by Safety Gate</b> 🚨\n\n"
            f"<b>Asset:</b> {asset}USD\n"
            f"<b>Direction:</b> {dir_emoji}\n"
            f"<b>Entry Ref:</b> ${entry_price:,.2f}\n"
            f"<b>Stop Loss:</b> ${stop_loss:,.2f}\n"
            f"<b>Take Profit:</b> ${take_profit:,.2f}\n\n"
            f"💡 <b>Worker Consensus (Why Workers Want to Trade):</b>\n"
            f"• <b>Technical Worker:</b> {tech_str}\n"
            f"• <b>ML Prediction:</b> {ml_str}\n"
            f"• <b>Order Flow:</b> {of_str}\n\n"
            f"🛡️ <b>Safety Gate Interventions (Why Blocked):</b>\n"
            f"• Current Regime: <code>{regime}</code>\n"
            f"• Session Bias: <code>{session_bias}</code>\n"
            f"{reasons_formatted}\n\n"
            f"⚠️ <b>Operator Decision Required:</b>\n"
            f"Would you like to override the safety gate and execute this simulated paper trade?"
        )

        reply_markup = {
            "inline_keyboard": [
                [
                    {
                        "text": "🚀 Place / Override Trade",
                        "callback_data": f"override_exec:{proposal_id}",
                    },
                    {
                        "text": "❌ Reject Trade",
                        "callback_data": f"override_rej:{proposal_id}",
                    },
                ]
            ]
        }

        sent = await self.send_message(message_text, reply_markup=reply_markup)
        if sent:
            proposal_data["telegram_message_id"] = sent.get("message_id")
            logger.info("Sent interactive Telegram override alert for proposal %s (%s %s)", proposal_id, asset, direction)

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
            "rationale": f"Manual Telegram Operator Override (HITL Proposal #{proposal_id[:8]})",
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
        proposal["executed_price"] = float(order.executed_price)
        proposal["order_id"] = str(order.id)
        proposal["position_id"] = str(pos.id)
        proposal["quantity"] = float(risk_decision.authorized_quantity)

        # Update Redis
        try:
            await redis_client.set(
                f"telegram:proposals:{proposal_id}",
                json.dumps(proposal),
                ex=86400,
            )
        except Exception:
            pass

        return True, (
            f"Executed {pos.direction} {pos.asset} at ${order.executed_price:,.2f} "
            f"(Qty: {risk_decision.authorized_quantity}, Pos ID: #{str(pos.id)[:8]})"
        ), proposal

    async def reject_override(self, proposal_id: str) -> tuple[bool, str]:
        """Mark proposal as rejected by operator."""
        proposal = await self.get_proposal(proposal_id)
        if not proposal:
            return False, f"Proposal {proposal_id} not found."

        proposal["status"] = "REJECTED"
        proposal["rejected_at"] = datetime.now(UTC).isoformat()
        try:
            await redis_client.set(
                f"telegram:proposals:{proposal_id}",
                json.dumps(proposal),
                ex=86400,
            )
        except Exception:
            pass
        return True, "Proposal dismissed by operator."

    async def poll_updates_once(
        self,
        get_account_and_session: Callable[[], Any],
        risk_engine: RiskEngine,
        executor: PaperExecutionEngine,
    ) -> None:
        """Poll Telegram getUpdates and dispatch button callback queries."""
        if not self.is_configured:
            return

        params = {"timeout": 15, "offset": self._last_update_id + 1}
        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                resp = await client.get(self._api_url("getUpdates"), params=params)
                if resp.status_code != 200:
                    return
                updates = resp.json().get("result", [])
                for u in updates:
                    self._last_update_id = max(self._last_update_id, u.get("update_id", 0))
                    cb = u.get("callback_query")
                    if not cb:
                        continue

                    cb_id = cb.get("id")
                    data = str(cb.get("data", ""))
                    msg = cb.get("message", {})
                    msg_id = msg.get("message_id")
                    chat = msg.get("chat", {})
                    chat_id = chat.get("id")

                    if data.startswith("override_exec:"):
                        prop_id = data.split("override_exec:")[1]
                        logger.info("Operator triggered Telegram OVERRIDE EXECUTE for proposal %s", prop_id)

                        session, account = await get_account_and_session()
                        async with session:
                            success, message, executed_prop = await self.execute_override(
                                prop_id, account, session, risk_engine, executor
                            )

                        if success and executed_prop:
                            await self.answer_callback_query(cb_id, text="🚀 Trade executed in paper account!", show_alert=True)
                            updated_text = (
                                f"✅ <b>TRADE OVERRIDDEN & EXECUTED BY OPERATOR</b>\n\n"
                                f"<b>Asset:</b> {executed_prop['asset']}USD\n"
                                f"<b>Side:</b> {executed_prop['side']} ({executed_prop['direction']})\n"
                                f"<b>Filled Price:</b> ${executed_prop['executed_price']:,.2f}\n"
                                f"<b>Quantity:</b> {executed_prop['quantity']}\n"
                                f"<b>Stop Loss:</b> ${executed_prop['stop_loss']:,.2f}\n"
                                f"<b>Take Profit:</b> ${executed_prop['take_profit']:,.2f}\n"
                                f"<b>Order ID:</b> <code>{executed_prop['order_id']}</code>\n"
                                f"<b>Position ID:</b> <code>{executed_prop['position_id']}</code>\n"
                                f"<b>Executed At:</b> {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}\n\n"
                                f"<i>Position is now actively managed by Bracket Monitor.</i>"
                            )
                            if chat_id and msg_id:
                                await self.edit_message_text(chat_id, msg_id, updated_text)
                        else:
                            await self.answer_callback_query(cb_id, text=f"Execution failed: {message}", show_alert=True)

                    elif data.startswith("override_rej:"):
                        prop_id = data.split("override_rej:")[1]
                        logger.info("Operator triggered Telegram OVERRIDE REJECT for proposal %s", prop_id)
                        await self.reject_override(prop_id)
                        await self.answer_callback_query(cb_id, text="❌ Proposal rejected.")
                        rejected_text = (
                            f"❌ <b>TRADE PROPOSAL REJECTED BY OPERATOR</b>\n\n"
                            f"Proposal ID: <code>{prop_id}</code>\n"
                            f"Action: Dismissed\n"
                            f"Time: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}\n\n"
                            f"<i>Capital preserved. Gating rules maintained.</i>"
                        )
                        if chat_id and msg_id:
                            await self.edit_message_text(chat_id, msg_id, rejected_text)

        except Exception as e:
            logger.debug("Telegram poll_updates_once error: %s", e)


# Global singleton instance
telegram_service = TelegramBotService()
