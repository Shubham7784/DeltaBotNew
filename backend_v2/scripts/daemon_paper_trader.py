"""Autonomous Continuous Paper Trading Daemon (Phase 10: Daemon Service).

Runs a resilient background event loop orchestrating:
1. Session Initialization (Stage 1) at session boundaries (daily 00:00 UTC).
2. Continuous Multi-Agent Monitoring Loop (Stage 2) every N seconds.
3. Mark-to-market position monitoring and stop-loss / take-profit bracket exits.
4. Deterministic Risk Engine evaluation and simulated paper-order execution.
5. Graceful shutdown on SIGTERM / SIGINT and fail-closed safety.
"""

import argparse
import asyncio
from datetime import UTC, date, datetime
from decimal import Decimal
import logging
import os
from pathlib import Path
import signal
import sys
from typing import Any
import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import close_database_resources, engine
from app.market_data.regimes import Asset
from app.ml.inference import MLInferenceService
from app.ml.registry import ModelRegistry
from app.models import Candle, PaperAccount, PaperPosition
from app.trading.paper_executor import PaperExecutionEngine
from app.trading.position_monitor import PositionMonitorService
from app.trading.risk_engine import RiskEngine
from app.workers.workflow import AssetWorkflowCoordinator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("deltabot.daemon_paper_trader")


async def get_or_create_paper_account(
    session: AsyncSession,
    account_name: str = "default_paper_account",
    initial_balance: float = 10000.0,
) -> PaperAccount:
    """Retrieve existing paper account or initialize a default paper portfolio."""
    stmt = select(PaperAccount).where(PaperAccount.name == account_name)
    res = await session.execute(stmt)
    account = res.scalar_one_or_none()

    if account is None:
        now = datetime.now(UTC)
        b = Decimal(str(initial_balance))
        account = PaperAccount(
            id=uuid.uuid4(),
            name=account_name,
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
        logger.info("Initialized new paper account '%s' with equity $%s", account_name, b)
    return account


async def get_latest_prices(
    assets: list[Asset],
    session: AsyncSession,
    as_of: datetime,
) -> dict[str, Decimal]:
    """Retrieve latest known candle close prices for specified assets for mark-to-market."""
    prices: dict[str, Decimal] = {}
    for asset in assets:
        symbol = f"{asset}USD"
        stmt = (
            select(Candle.close)
            .where(Candle.symbol == symbol, Candle.open_time <= as_of)
            .order_by(Candle.open_time.desc())
            .limit(1)
        )
        res = await session.execute(stmt)
        val = res.scalar_one_or_none()
        if val is not None:
            prices[asset] = Decimal(str(val))
            prices[symbol] = Decimal(str(val))
    return prices


class ContinuousPaperTrader:
    """Daemon runner maintaining the continuous paper-trading lifecycle."""

    def __init__(
        self,
        assets: list[Asset] | None = None,
        interval_seconds: int = 60,
        registry_path: Path = Path("model-registry"),
        account_name: str = "default_paper_account",
        initial_balance: float = 10000.0,
        max_cycles: int | None = None,
    ):
        self.assets: list[Asset] = assets or ["BTC", "ETH"]
        self.interval_seconds = interval_seconds
        self.registry_path = registry_path
        self.account_name = account_name
        self.initial_balance = initial_balance
        self.max_cycles = max_cycles
        self.stop_event = asyncio.Event()

    def stop(self) -> None:
        """Signal the daemon to stop gracefully."""
        self.stop_event.set()

    async def run(self) -> None:
        """Execute the continuous paper trading loop."""
        logger.info("=" * 70)
        logger.info("DELTABOT 2.0 CONTINUOUS PAPER TRADER DAEMON STARTING")
        logger.info("Assets: %s | Loop Interval: %ds | Registry: %s", self.assets, self.interval_seconds, self.registry_path)
        logger.info("Execution Mode: STRICT SIMULATED PAPER TRADING (No Live Capital)")
        logger.info("=" * 70)

        registry = ModelRegistry(self.registry_path)
        ml_service = MLInferenceService(registry, engine)
        workflow = AssetWorkflowCoordinator(engine, ml_service)
        risk_engine = RiskEngine()
        executor = PaperExecutionEngine()
        monitor = PositionMonitorService()

        active_session_date: date | None = None
        cycles_completed = 0

        try:
            while not self.stop_event.is_set():
                now = datetime.now(UTC)
                current_date = now.date()

                async with AsyncSession(engine, expire_on_commit=False) as session:
                    account = await get_or_create_paper_account(session, self.account_name, self.initial_balance)

                    # Stage 1: Daily Session Initialization
                    if active_session_date != current_date:
                        logger.info("\n>>> Daily Session Rollover: Initializing Stage 1 for %s (UTC Date: %s) <<<", self.assets, current_date)
                        account.daily_starting_equity = account.equity
                        account.daily_realized_pnl = Decimal("0")
                        await session.commit()

                        for asset in self.assets:
                            try:
                                init_res = await workflow.run_session_initialization(asset, now, session)
                                logger.info(
                                    "Session Init %s: Regime=%s | Bias=%s (Conf: %.2f)",
                                    asset,
                                    init_res["regime"],
                                    init_res["bias_direction"],
                                    init_res["bias_confidence"],
                                )
                            except Exception as e:
                                logger.error("Failed to initialize session for %s: %s", asset, e, exc_info=True)
                        active_session_date = current_date

                    # Mark-to-market and bracket exits
                    try:
                        current_prices = await get_latest_prices(self.assets, session, now)
                        closed_positions = await monitor.check_and_execute_exits(account, current_prices, session, as_of=now)
                        if closed_positions:
                            logger.info(
                                "Bracket Monitor: Closed %d position(s) on SL/TP triggers.",
                                len(closed_positions),
                            )
                            for cp in closed_positions:
                                logger.info(
                                    "Closed %s %s: Exit Price=$%s | Net P&L=$%s | Reason=%s",
                                    cp.asset,
                                    cp.direction,
                                    cp.exit_price,
                                    cp.realized_pnl,
                                    cp.exit_reason,
                                )
                    except Exception as e:
                        logger.error("Bracket monitor evaluation error: %s", e, exc_info=True)

                    # Stage 2: Continuous Multi-Agent Monitoring Loop
                    for asset in self.assets:
                        try:
                            mon_res = await workflow.run_monitoring_cycle(asset, now, session)
                            decision = mon_res.get("validator_decision", "NO_TRADE")
                            proposal = mon_res.get("proposal")

                            if decision == "APPROVE" and proposal is not None:
                                logger.info(
                                    "Workflow validator APPROVED candidate proposal for %s: Setup=%s | Direction=%s",
                                    asset,
                                    proposal.setup_type,
                                    proposal.direction,
                                )
                                stmt = select(PaperPosition).where(
                                    PaperPosition.account_id == account.id,
                                    PaperPosition.status == "OPEN",
                                )
                                res = await session.execute(stmt)
                                open_positions = list(res.scalars().all())

                                risk_decision = risk_engine.evaluate_proposal(account, proposal, open_positions)
                                if risk_decision.authorized and risk_decision.authorized_quantity:
                                    logger.info(
                                        "Risk Engine AUTHORIZED: Qty=%s %s | Stop=%s | TP=%s",
                                        risk_decision.authorized_quantity,
                                        asset,
                                        proposal.stop_loss,
                                        proposal.take_profit,
                                    )
                                    order, pos = await executor.execute_proposal(
                                        account,
                                        proposal,
                                        risk_decision.authorized_quantity,
                                        session,
                                    )
                                    logger.info(
                                        "Simulated Paper Order FILLED: Order ID=%s | Position ID=%s | Fill Price=$%s",
                                        order.id,
                                        pos.id,
                                        order.executed_price,
                                    )
                                else:
                                    logger.warning("Risk Engine REJECTED trade proposal: %s", risk_decision.reason)
                        except Exception as e:
                            logger.error("Monitoring cycle failed for %s: %s", asset, e, exc_info=True)

                    # Check open positions summary
                    stmt = select(PaperPosition).where(
                        PaperPosition.account_id == account.id,
                        PaperPosition.status == "OPEN",
                    )
                    res = await session.execute(stmt)
                    open_positions = list(res.scalars().all())
                    logger.info(
                        "Status Check: Cash=$%.2f | Equity=$%.2f | Open Positions: %d | Halted: %s",
                        float(account.cash_balance),
                        float(account.equity),
                        len(open_positions),
                        account.is_halted,
                    )

                cycles_completed += 1
                if self.max_cycles is not None and cycles_completed >= self.max_cycles:
                    logger.info("Reached maximum requested cycles (%d). Stopping daemon.", self.max_cycles)
                    break

                # Sleep until next cycle or stop signal
                try:
                    await asyncio.wait_for(self.stop_event.wait(), timeout=self.interval_seconds)
                except TimeoutError:
                    pass

        finally:
            logger.info("Shutting down Continuous Paper Trader Daemon. Cleaning up database connections...")
            await close_database_resources()
            logger.info("Continuous Paper Trader Daemon stopped cleanly.")


def main() -> None:
    parser = argparse.ArgumentParser(description="DeltaBot 2.0 Continuous Paper Trader Daemon")
    parser.add_argument(
        "--assets",
        nargs="+",
        default=os.getenv("TRADING_ASSETS", "BTC,ETH").split(","),
        help="Cryptocurrency assets to trade (default: BTC ETH)",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=int(os.getenv("MONITOR_INTERVAL_SECONDS", "60")),
        help="Monitoring loop interval in seconds (default: 60)",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path(os.getenv("MODEL_REGISTRY_PATH", "model-registry")),
        help="Path to trained model registry directory",
    )
    parser.add_argument(
        "--account-name",
        type=str,
        default="default_paper_account",
        help="Paper trading account identifier",
    )
    parser.add_argument(
        "--initial-balance",
        type=float,
        default=float(os.getenv("INITIAL_BALANCE", "10000.0")),
        help="Initial simulated cash balance",
    )
    parser.add_argument(
        "--max-cycles",
        type=int,
        default=None,
        help="Optional maximum number of monitoring cycles to run before exiting",
    )
    args = parser.parse_args()

    assets = [a.strip().upper() for a in args.assets if a.strip()]
    daemon = ContinuousPaperTrader(
        assets=assets,  # type: ignore[arg-type]
        interval_seconds=args.interval,
        registry_path=args.registry,
        account_name=args.account_name,
        initial_balance=args.initial_balance,
        max_cycles=args.max_cycles,
    )

    def _handle_signal(sig: int, frame: Any) -> None:
        logger.info("Signal %s received. Stopping daemon loop...", sig)
        daemon.stop()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    asyncio.run(daemon.run())


if __name__ == "__main__":
    main()
