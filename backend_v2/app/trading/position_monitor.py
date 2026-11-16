"""Paper Position Monitor and Accounting Engine (Phase 8: p8-monitor).

Performs mark-to-market accounting on open paper positions, detects stop-loss and take-profit
bracket triggers, executes simulated exit orders, and updates portfolio equity.
"""

from datetime import UTC, datetime
from decimal import Decimal
import logging
import uuid
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import PaperAccount, PaperOrder, PaperPosition

logger = logging.getLogger(__name__)


class PositionMonitorService:
    """Monitors paper positions, evaluates exit conditions, and performs portfolio accounting."""

    def __init__(self, *, default_taker_fee_bps: float = 5.0):
        self.default_taker_fee_bps = default_taker_fee_bps

    def update_position_mark_to_market(
        self,
        position: PaperPosition,
        current_price: Decimal,
    ) -> Decimal:
        """Calculate and update unrealized P&L for an open position."""
        if position.status != "OPEN":
            return Decimal("0")

        entry = Decimal(str(position.entry_price))
        qty = Decimal(str(position.quantity))

        if position.direction == "LONG":
            unrealized = (current_price - entry) * qty
        else:
            unrealized = (entry - current_price) * qty

        position.unrealized_pnl = unrealized.quantize(Decimal("0.0001"))
        return position.unrealized_pnl

    async def check_and_execute_exits(
        self,
        account: PaperAccount,
        current_prices: dict[str, Decimal],
        session: AsyncSession,
        *,
        as_of: datetime | None = None,
    ) -> list[PaperPosition]:
        """Check all open positions against stop-loss and take-profit thresholds."""
        now = as_of or datetime.now(UTC)

        stmt = select(PaperPosition).where(
            PaperPosition.account_id == account.id,
            PaperPosition.status == "OPEN",
        )
        res = await session.execute(stmt)
        open_positions = list(res.scalars().all())

        closed_positions: list[PaperPosition] = []

        for pos in open_positions:
            price = current_prices.get(pos.asset)
            if price is None:
                continue

            # Update mark-to-market
            self.update_position_mark_to_market(pos, price)

            entry = Decimal(str(pos.entry_price))
            qty = Decimal(str(pos.quantity))
            stop = Decimal(str(pos.stop_loss))
            tp = Decimal(str(pos.take_profit))

            exit_triggered = False
            exit_reason: str | None = None
            exit_price = price

            if pos.direction == "LONG":
                if price <= stop:
                    exit_triggered = True
                    exit_reason = "STOP_LOSS"
                    exit_price = stop
                elif price >= tp:
                    exit_triggered = True
                    exit_reason = "TAKE_PROFIT"
                    exit_price = tp
            else:  # SHORT
                if price >= stop:
                    exit_triggered = True
                    exit_reason = "STOP_LOSS"
                    exit_price = stop
                elif price <= tp:
                    exit_triggered = True
                    exit_reason = "TAKE_PROFIT"
                    exit_price = tp

            if exit_triggered:
                # Calculate exit fee
                fee_mult = Decimal(str(self.default_taker_fee_bps)) / Decimal("10000")
                exit_notional = exit_price * qty
                exit_fee = (exit_notional * fee_mult).quantize(Decimal("0.0001"))

                # Gross P&L
                if pos.direction == "LONG":
                    gross_pnl = (exit_price - entry) * qty
                else:
                    gross_pnl = (entry - exit_price) * qty

                total_fees = Decimal(str(pos.total_fees)) + exit_fee
                net_pnl = gross_pnl - exit_fee

                # Update position
                pos.status = "CLOSED"
                pos.exit_price = exit_price
                pos.exit_reason = exit_reason
                pos.realized_pnl = net_pnl.quantize(Decimal("0.0001"))
                pos.unrealized_pnl = Decimal("0")
                pos.total_fees = total_fees.quantize(Decimal("0.0001"))
                pos.closed_at = now

                # Update account accounting
                account.cash_balance += (entry * qty + net_pnl)
                account.equity += net_pnl
                account.daily_realized_pnl += net_pnl
                account.updated_at = now

                # Create closing order
                closing_side = "SELL" if pos.direction == "LONG" else "BUY"
                closing_order = PaperOrder(
                    id=uuid.uuid4(),
                    account_id=account.id,
                    position_id=pos.id,
                    asset=pos.asset,
                    side=closing_side,
                    order_type="MARKET",
                    status="FILLED",
                    requested_price=exit_price,
                    executed_price=exit_price,
                    quantity=qty,
                    fee_paid=exit_fee,
                    slippage_bps=0.0,
                    rejection_reason=None,
                    created_at=now,
                    executed_at=now,
                )
                session.add(closing_order)
                closed_positions.append(pos)

        if closed_positions:
            await session.commit()

        return closed_positions
