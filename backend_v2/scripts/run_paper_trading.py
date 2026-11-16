"""CLI Runner for DeltaBot 2.0 End-to-End Paper Trading (Phase 10).

Executes deterministic Supervisor workflows, specialist workers, risk validation,
and paper-only order simulation for BTC and ETH without placing any real-money orders.
"""

import argparse
import asyncio
from datetime import UTC, datetime, timedelta
from decimal import Decimal
import logging
from pathlib import Path
import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import engine
from app.market_data.regimes import Asset
from app.ml.inference import MLInferenceService
from app.ml.registry import ModelRegistry
from app.models import PaperAccount, PaperPosition
from app.trading.paper_executor import PaperExecutionEngine
from app.trading.position_monitor import PositionMonitorService
from app.trading.risk_engine import RiskEngine
from app.workers.workflow import AssetWorkflowCoordinator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("deltabot.paper_runner")


async def get_or_create_paper_account(session: AsyncSession, initial_balance: float = 10000.0) -> PaperAccount:
    stmt = select(PaperAccount).where(PaperAccount.name == "default_paper_account")
    res = await session.execute(stmt)
    account = res.scalar_one_or_none()

    if account is None:
        now = datetime.now(UTC)
        b = Decimal(str(initial_balance))
        account = PaperAccount(
            id=uuid.uuid4(),
            name="default_paper_account",
            initial_balance=b,
            cash_balance=b,
            equity=b,
            daily_starting_equity=b,
            daily_realized_pnl=Decimal("0"),
            is_halted=False,
            created_at=now,
            updated_at=now,
        )
        session.add(account)
        await session.commit()
        await session.refresh(account)
    return account


async def run_paper_trading_session(
    assets: list[Asset] = ["BTC", "ETH"],
    as_of: datetime | None = None,
    registry_path: Path = Path("model-registry"),
) -> None:
    now = as_of or datetime.now(UTC)
    logger.info("=" * 70)
    logger.info("STARTING DELTABOT 2.0 PAPER TRADING SESSION")
    logger.info("Timestamp: %s | Venue: Delta Exchange (Simulated) | Assets: %s", now, assets)
    logger.info("=" * 70)

    registry = ModelRegistry(registry_path)
    ml_service = MLInferenceService(registry, engine)
    workflow = AssetWorkflowCoordinator(engine, ml_service)
    risk_engine = RiskEngine()
    executor = PaperExecutionEngine()
    monitor = PositionMonitorService()

    async with AsyncSession(engine, expire_on_commit=False) as session:
        account = await get_or_create_paper_account(session)
        logger.info(
            "Paper Account Balance: Cash=$%.2f | Equity=$%.2f | Halted=%s",
            float(account.cash_balance),
            float(account.equity),
            account.is_halted,
        )

        for asset in assets:
            logger.info("\n--- Processing %s Session Initialization ---", asset)
            init_res = await workflow.run_session_initialization(asset, now, session)
            logger.info(
                "%s Init Status: %s | Regime: %s | Session Bias: %s (Confidence: %.2f)",
                asset,
                init_res["status"],
                init_res["regime"],
                init_res["bias_direction"],
                init_res["bias_confidence"],
            )

            logger.info("--- Executing %s Continuous Monitoring Cycle ---", asset)
            mon_res = await workflow.run_monitoring_cycle(asset, now, session)
            logger.info(
                "%s Validator Decision: %s | Rationale: %s",
                asset,
                mon_res["validator_decision"],
                mon_res["rationale"],
            )

            proposal = mon_res.get("proposal")
            if proposal is not None and mon_res["validator_decision"] == "APPROVE":
                logger.info("Evaluating candidate trade proposal through Risk Engine...")
                stmt = select(PaperPosition).where(
                    PaperPosition.account_id == account.id,
                    PaperPosition.status == "OPEN",
                )
                res = await session.execute(stmt)
                open_pos = list(res.scalars().all())

                risk_decision = risk_engine.evaluate_proposal(account, proposal, open_pos)
                if risk_decision.authorized and risk_decision.authorized_quantity is not None:
                    logger.info(
                        "Risk Engine AUTHORIZED trade: Qty=%s %s | Notional=$%.2f | Risk Capital=$%.2f",
                        risk_decision.authorized_quantity,
                        asset,
                        float(risk_decision.notional_value or 0),
                        float(risk_decision.risk_amount or 0),
                    )
                    order, position = await executor.execute_proposal(
                        account,
                        proposal,
                        risk_decision.authorized_quantity,
                        session,
                    )
                    logger.info(
                        "Simulated Order FILLED: Order ID=%s | Position ID=%s | Fill Price=$%.2f | Fee Paid=$%.4f",
                        order.id,
                        position.id,
                        float(order.executed_price or 0),
                        float(order.fee_paid),
                    )
                else:
                    logger.warning("Risk Engine REJECTED trade proposal: %s", risk_decision.reason)

        # Check bracket exits and mark-to-market
        stmt = select(PaperPosition).where(
            PaperPosition.account_id == account.id,
            PaperPosition.status == "OPEN",
        )
        res = await session.execute(stmt)
        open_pos = list(res.scalars().all())
        logger.info("\nTotal Open Paper Positions: %d", len(open_pos))
        for p in open_pos:
            logger.info(
                "Open Position: %s %s | Qty=%s | Entry=$%.2f | SL=$%.2f | TP=$%.2f | Unrealized P&L=$%.2f",
                p.asset,
                p.direction,
                p.quantity,
                float(p.entry_price),
                float(p.stop_loss),
                float(p.take_profit),
                float(p.unrealized_pnl),
            )

    logger.info("\n" + "=" * 70)
    logger.info("DELTABOT 2.0 PAPER TRADING SESSION CYCLE COMPLETED")
    logger.info("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(description="DeltaBot 2.0 Paper Trading Runner")
    parser.add_argument("--assets", nargs="+", default=["BTC", "ETH"], choices=["BTC", "ETH"])
    args = parser.parse_args()
    asyncio.run(run_paper_trading_session(assets=args.assets))


if __name__ == "__main__":
    main()
