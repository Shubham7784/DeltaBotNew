"""Deterministic Paper Execution Engine for DeltaBot 2.0.

Simulates market and limit order execution against live/historical market data,
applying explicit slippage and fee models, and managing paper position lifecycles.
Strict invariant: No exchange execution credentials or live trading endpoints exist.
"""

from datetime import UTC, datetime
from decimal import Decimal
import logging
import uuid
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.models import PaperAccount, PaperOrder, PaperPosition

logger = logging.getLogger(__name__)


class PaperExecutionEngine:
    """Executes authorized trade proposals exclusively in simulated paper environment."""

    def __init__(
        self,
        *,
        default_slippage_bps: float = 2.0,
        default_taker_fee_bps: float = 5.0,
    ):
        self.default_slippage_bps = default_slippage_bps
        self.default_taker_fee_bps = default_taker_fee_bps

    async def execute_proposal(
        self,
        account: PaperAccount,
        proposal: dict[str, Any],
        authorized_quantity: Decimal,
        session: AsyncSession,
        *,
        slippage_bps: float | None = None,
        fee_bps: float | None = None,
    ) -> tuple[PaperOrder, PaperPosition]:
        """Execute risk-authorized proposal into a simulated order and position."""
        now = datetime.now(UTC)
        used_slippage = slippage_bps if slippage_bps is not None else self.default_slippage_bps
        used_fee = fee_bps if fee_bps is not None else self.default_taker_fee_bps

        asset = str(proposal["asset"])
        side = str(proposal["side"]).upper()
        direction = str(proposal["direction"]).upper()
        requested_price = Decimal(str(proposal["entry_price"]))
        stop_loss = Decimal(str(proposal["stop_loss"]))
        take_profit = Decimal(str(proposal["take_profit"]))

        # Apply simulated slippage: BUY gets filled higher, SELL gets filled lower
        slip_mult = Decimal(str(used_slippage)) / Decimal("10000")
        if side == "BUY":
            executed_price = requested_price * (Decimal("1") + slip_mult)
        else:
            executed_price = requested_price * (Decimal("1") - slip_mult)

        executed_price = executed_price.quantize(Decimal("0.01"))

        # Calculate transaction fee
        fee_mult = Decimal(str(used_fee)) / Decimal("10000")
        notional = executed_price * authorized_quantity
        fee_paid = (notional * fee_mult).quantize(Decimal("0.0001"))

        # Deduct transaction fee from cash balance
        account.cash_balance -= fee_paid
        account.equity -= fee_paid
        account.updated_at = now

        # Create PaperPosition record
        position_id = uuid.uuid4()
        position = PaperPosition(
            id=position_id,
            account_id=account.id,
            asset=asset,
            direction=direction,
            status="OPEN",
            quantity=authorized_quantity,
            entry_price=executed_price,
            exit_price=None,
            stop_loss=stop_loss,
            take_profit=take_profit,
            unrealized_pnl=Decimal("0"),
            realized_pnl=Decimal("0"),
            total_fees=fee_paid,
            exit_reason=None,
            opened_at=now,
            closed_at=None,
            metadata_payload={
                "requested_price": float(requested_price),
                "slippage_bps": used_slippage,
                "fee_bps": used_fee,
                "rationale": proposal.get("rationale", ""),
            },
        )
        session.add(position)
        await session.flush()

        # Create PaperOrder record (FILLED)
        order = PaperOrder(
            id=uuid.uuid4(),
            account_id=account.id,
            position_id=position_id,
            asset=asset,
            side=side,
            order_type="MARKET",
            status="FILLED",
            requested_price=requested_price,
            executed_price=executed_price,
            quantity=authorized_quantity,
            fee_paid=fee_paid,
            slippage_bps=used_slippage,
            rejection_reason=None,
            created_at=now,
            executed_at=now,
        )
        session.add(order)

        await session.commit()
        await session.refresh(position)
        await session.refresh(order)

        return order, position
