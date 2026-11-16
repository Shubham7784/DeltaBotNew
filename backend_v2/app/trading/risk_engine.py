"""Deterministic Risk Management Engine for DeltaBot 2.0.

Enforces strict mathematical invariants:
1. Max 1% portfolio risk per trade.
2. Max 50% gross portfolio exposure across all open positions.
3. Minimum 30% unencumbered safety reserve buffer (cash).
4. Max 3% daily cumulative loss halt (closes entries for the remainder of UTC day).
5. Max open positions limit and duplicate signal rejection per asset.
6. Absolute separation from agents: Workers and Supervisor cannot bypass this engine.
"""

from dataclasses import dataclass
from decimal import Decimal, ROUND_DOWN
import logging
from typing import Any

from app.models import PaperAccount, PaperPosition

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RiskParameters:
    max_risk_pct: Decimal = Decimal("0.01")  # 1% equity at risk per trade
    max_exposure_pct: Decimal = Decimal("0.50")  # 50% max gross exposure
    safety_reserve_pct: Decimal = Decimal("0.30")  # 30% cash buffer
    max_daily_loss_pct: Decimal = Decimal("0.03")  # 3% daily loss halt
    max_open_positions_total: int = 2
    max_open_positions_per_asset: int = 1
    quantity_precision: int = 4


@dataclass(frozen=True)
class RiskAuthorizationResult:
    authorized: bool
    reason: str | None
    authorized_quantity: Decimal | None = None
    notional_value: Decimal | None = None
    risk_amount: Decimal | None = None


class RiskEngine:
    """Independent deterministic risk authorization engine."""

    def __init__(self, params: RiskParameters | None = None):
        self.params = params or RiskParameters()

    def evaluate_proposal(
        self,
        account: PaperAccount,
        proposal: dict[str, Any],
        open_positions: list[PaperPosition],
    ) -> RiskAuthorizationResult:
        """Evaluate a candidate trade proposal against all risk limits."""
        # 1. Emergency Halt Check
        if account.is_halted:
            return RiskAuthorizationResult(
                authorized=False,
                reason=f"Account emergency halt is active: {account.halt_reason}",
            )

        # 2. Daily Loss Limit Check
        daily_loss_threshold = account.daily_starting_equity * self.params.max_daily_loss_pct
        if account.daily_realized_pnl <= -daily_loss_threshold:
            account.is_halted = True
            account.halt_reason = (
                f"Daily realized loss of {account.daily_realized_pnl} reached max threshold of -{daily_loss_threshold}"
            )
            return RiskAuthorizationResult(
                authorized=False,
                reason="Daily loss halt triggered (exceeded 3% starting equity)",
            )

        # 3. Open Positions Limit Check
        if len(open_positions) >= self.params.max_open_positions_total:
            return RiskAuthorizationResult(
                authorized=False,
                reason=f"Maximum open positions limit ({self.params.max_open_positions_total}) reached",
            )

        # 4. Duplicate Asset Position Check
        asset = str(proposal.get("asset", ""))
        for pos in open_positions:
            if pos.asset == asset:
                return RiskAuthorizationResult(
                    authorized=False,
                    reason=f"Position already open for asset {asset} (duplicate entry rejected)",
                )

        # 5. Stop Loss Distance Validation
        try:
            entry_price = Decimal(str(proposal["entry_price"]))
            stop_loss = Decimal(str(proposal["stop_loss"]))
        except (KeyError, ValueError) as exc:
            return RiskAuthorizationResult(
                authorized=False,
                reason=f"Malformed proposal price values: {exc}",
            )

        stop_distance = abs(entry_price - stop_loss)
        if stop_distance <= Decimal("0"):
            return RiskAuthorizationResult(
                authorized=False,
                reason="Stop loss distance must be greater than zero",
            )

        # 6. Position Sizing Formula: Quantity = (Equity * RiskFraction) / StopDistance
        risk_capital = account.equity * self.params.max_risk_pct
        raw_quantity = risk_capital / stop_distance

        # Quantize to configured precision
        step = Decimal("10") ** -self.params.quantity_precision
        quantity = raw_quantity.quantize(step, rounding=ROUND_DOWN)

        if quantity <= Decimal("0"):
            return RiskAuthorizationResult(
                authorized=False,
                reason=f"Calculated size {raw_quantity} below minimum contract precision {step}",
            )

        notional_value = quantity * entry_price

        # 7. Gross Exposure Check (Sum of all open notional values + new position)
        current_exposure = sum(
            Decimal(str(pos.quantity)) * Decimal(str(pos.entry_price))
            for pos in open_positions
        )
        total_projected_exposure = current_exposure + notional_value
        max_allowed_exposure = account.equity * self.params.max_exposure_pct

        if total_projected_exposure > max_allowed_exposure:
            return RiskAuthorizationResult(
                authorized=False,
                reason=(
                    f"Projected exposure {total_projected_exposure:.2f} exceeds "
                    f"max allowed 50% limit ({max_allowed_exposure:.2f})"
                ),
            )

        # 8. Unencumbered Safety Reserve Buffer Check (30% Cash Buffer)
        available_cash = account.cash_balance - notional_value
        required_safety_reserve = account.equity * self.params.safety_reserve_pct

        if available_cash < required_safety_reserve:
            return RiskAuthorizationResult(
                authorized=False,
                reason=(
                    f"Available cash after trade ({available_cash:.2f}) dips below "
                    f"mandatory 30% safety reserve buffer ({required_safety_reserve:.2f})"
                ),
            )

        return RiskAuthorizationResult(
            authorized=True,
            reason=None,
            authorized_quantity=quantity,
            notional_value=notional_value,
            risk_amount=risk_capital,
        )
