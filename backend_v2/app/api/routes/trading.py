"""REST API endpoints for Paper Trading account, positions, and orders."""

from datetime import datetime
from decimal import Decimal
from typing import Any, Literal
import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from app.database import engine
from app.market_data.regimes import Asset
from app.models import PaperAccount, PaperOrder, PaperPosition

router = APIRouter(prefix="/api/v1/trading", tags=["trading"])


class PaperAccountResponse(BaseModel):
    id: uuid.UUID
    name: str
    initial_balance: float
    cash_balance: float
    equity: float
    daily_starting_equity: float
    daily_realized_pnl: float
    is_halted: bool
    halt_reason: str | None
    currency: str
    updated_at: datetime


class PaperPositionResponse(BaseModel):
    id: uuid.UUID
    account_id: uuid.UUID
    asset: str
    direction: str
    status: str
    quantity: float
    entry_price: float
    exit_price: float | None
    stop_loss: float
    take_profit: float
    unrealized_pnl: float
    realized_pnl: float
    total_fees: float
    exit_reason: str | None
    opened_at: datetime
    closed_at: datetime | None
    metadata_payload: dict[str, Any]


class PaperOrderResponse(BaseModel):
    id: uuid.UUID
    account_id: uuid.UUID
    position_id: uuid.UUID | None
    asset: str
    side: str
    order_type: str
    status: str
    requested_price: float
    executed_price: float | None
    quantity: float
    fee_paid: float
    slippage_bps: float
    rejection_reason: str | None
    created_at: datetime
    executed_at: datetime | None


async def _get_or_create_account(session: AsyncSession) -> PaperAccount:
    stmt = select(PaperAccount).where(PaperAccount.name == "default_paper_account")
    res = await session.execute(stmt)
    account = res.scalar_one_or_none()
    if account is None:
        from datetime import UTC
        now = datetime.now(UTC)
        b = Decimal("10000.00")
        account = PaperAccount(
            id=uuid.uuid4(),
            name="default_paper_account",
            initial_balance=b,
            cash_balance=b,
            equity=b,
            daily_starting_equity=b,
            daily_realized_pnl=Decimal("0.00"),
            is_halted=False,
            created_at=now,
            updated_at=now,
        )
        session.add(account)
        await session.commit()
        await session.refresh(account)
    return account


@router.get("/account", response_model=PaperAccountResponse)
async def get_paper_account() -> PaperAccountResponse:
    """Retrieve simulated paper trading account balance, equity, and risk halt status."""
    async with AsyncSession(engine, expire_on_commit=False) as session:
        account = await _get_or_create_account(session)
        return PaperAccountResponse(
            id=account.id,
            name=account.name,
            initial_balance=float(account.initial_balance),
            cash_balance=float(account.cash_balance),
            equity=float(account.equity),
            daily_starting_equity=float(account.daily_starting_equity),
            daily_realized_pnl=float(account.daily_realized_pnl),
            is_halted=account.is_halted,
            halt_reason=account.halt_reason,
            currency=account.currency,
            updated_at=account.updated_at,
        )


@router.get("/positions", response_model=list[PaperPositionResponse])
async def get_paper_positions(
    status: Literal["OPEN", "CLOSED", "ALL"] = Query(default="ALL"),
    asset: Asset | None = Query(default=None),
) -> list[PaperPositionResponse]:
    """List open and closed paper trading positions with mark-to-market P&L."""
    async with AsyncSession(engine, expire_on_commit=False) as session:
        query = select(PaperPosition)
        if status != "ALL":
            query = query.where(PaperPosition.status == status)
        if asset is not None:
            query = query.where(PaperPosition.asset == asset)
        query = query.order_by(PaperPosition.opened_at.desc())

        res = await session.execute(query)
        positions = res.scalars().all()

        return [
            PaperPositionResponse(
                id=p.id,
                account_id=p.account_id,
                asset=p.asset,
                direction=p.direction,
                status=p.status,
                quantity=float(p.quantity),
                entry_price=float(p.entry_price),
                exit_price=float(p.exit_price) if p.exit_price is not None else None,
                stop_loss=float(p.stop_loss),
                take_profit=float(p.take_profit),
                unrealized_pnl=float(p.unrealized_pnl),
                realized_pnl=float(p.realized_pnl),
                total_fees=float(p.total_fees),
                exit_reason=p.exit_reason,
                opened_at=p.opened_at,
                closed_at=p.closed_at,
                metadata_payload=dict(p.metadata_payload or {}),
            )
            for p in positions
        ]


@router.get("/orders", response_model=list[PaperOrderResponse])
async def get_paper_orders(
    asset: Asset | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
) -> list[PaperOrderResponse]:
    """List paper trading simulated orders, execution prices, slippage, and fees."""
    async with AsyncSession(engine, expire_on_commit=False) as session:
        query = select(PaperOrder)
        if asset is not None:
            query = query.where(PaperOrder.asset == asset)
        query = query.order_by(PaperOrder.created_at.desc()).limit(limit)

        res = await session.execute(query)
        orders = res.scalars().all()

        return [
            PaperOrderResponse(
                id=o.id,
                account_id=o.account_id,
                position_id=o.position_id,
                asset=o.asset,
                side=o.side,
                order_type=o.order_type,
                status=o.status,
                requested_price=float(o.requested_price),
                executed_price=float(o.executed_price) if o.executed_price is not None else None,
                quantity=float(o.quantity),
                fee_paid=float(o.fee_paid),
                slippage_bps=o.slippage_bps,
                rejection_reason=o.rejection_reason,
                created_at=o.created_at,
                executed_at=o.executed_at,
            )
            for o in orders
        ]
