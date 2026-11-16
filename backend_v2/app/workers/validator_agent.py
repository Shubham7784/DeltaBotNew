"""Worker 7: Trade Validator Agent for DeltaBot 2.0.

Synthesizes specialist worker intelligence (Session Bias, ML Prediction, Technical Setup,
Market Regime, Order Flow, and News Risk) into a candidate trade proposal decision:
APPROVE, REJECT, or WAIT.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

from app.market_data.regimes import Asset
from app.workers.supervisor import WorkerOutput

ValidationDecision = Literal["APPROVE", "REJECT", "WAIT"]
TradeSide = Literal["BUY", "SELL"]


@dataclass(frozen=True)
class TradeProposal:
    asset: Asset
    side: TradeSide
    direction: Literal["LONG", "SHORT"]
    entry_price: float
    stop_loss: float
    take_profit: float
    confidence: float
    rationale: str
    proposed_at: datetime
    metadata: dict[str, Any]


class TradeValidatorAgent:
    """Specialist worker evaluating synthesized cross-agent signals."""

    def validate(
        self,
        *,
        asset: Asset,
        session_bias: str,  # LONG_ONLY, SHORT_ONLY, NO_TRADE
        bias_confidence: float,
        regime: str,  # STRONG_BULLISH, STRONG_BEARISH, RANGE_BOUND, HIGH_VOLATILITY, TRANSITIONAL
        technical_output: dict[str, Any],
        ml_output: dict[str, Any],
        order_flow_output: dict[str, Any],
        news_output: dict[str, Any] | None = None,
        as_of: datetime,
    ) -> WorkerOutput:
        rejection_reasons: list[str] = []
        wait_reasons: list[str] = []

        # 1. Session Bias Check
        if session_bias == "NO_TRADE":
            rejection_reasons.append("Session bias is NO_TRADE")
        elif session_bias not in ("LONG_ONLY", "SHORT_ONLY"):
            rejection_reasons.append(f"Invalid session bias: {session_bias}")

        # 2. News Risk Check
        if news_output is not None:
            event_risk = str(news_output.get("event_risk", "low")).lower()
            if event_risk in ("high", "extreme"):
                rejection_reasons.append(f"Major market risk event detected: {event_risk}")

        # 3. Regime Compatibility Check
        if regime == "HIGH_VOLATILITY":
            wait_reasons.append("Market regime is HIGH_VOLATILITY; waiting for stabilization")
        elif session_bias == "LONG_ONLY" and regime == "STRONG_BEARISH":
            rejection_reasons.append("Regime STRONG_BEARISH contradicts LONG_ONLY bias")
        elif session_bias == "SHORT_ONLY" and regime == "STRONG_BULLISH":
            rejection_reasons.append("Regime STRONG_BULLISH contradicts SHORT_ONLY bias")

        # 4. Technical Setup Check
        tech_setup = technical_output.get("setup_detected", False)
        tech_dir = technical_output.get("direction", "NEUTRAL")
        if not tech_setup:
            wait_reasons.append("No active technical setup detected")
        elif session_bias == "LONG_ONLY" and tech_dir != "LONG":
            rejection_reasons.append(f"Technical direction {tech_dir} does not match LONG_ONLY bias")
        elif session_bias == "SHORT_ONLY" and tech_dir != "SHORT":
            rejection_reasons.append(f"Technical direction {tech_dir} does not match SHORT_ONLY bias")

        # 5. ML Prediction Check
        ml_decision = ml_output.get("decision", "NO_TRADE")
        ml_confidence = float(ml_output.get("confidence") or 0.0)
        if ml_decision == "NO_TRADE":
            wait_reasons.append(f"ML model output is NO_TRADE (confidence: {ml_confidence:.2f})")
        elif session_bias == "LONG_ONLY" and ml_decision != "LONG":
            rejection_reasons.append("ML model predicts contrary to LONG_ONLY session bias")
        elif session_bias == "SHORT_ONLY" and ml_decision != "SHORT":
            rejection_reasons.append("ML model predicts contrary to SHORT_ONLY session bias")

        # 6. Order Flow Check
        of_status = order_flow_output.get("status", "UNAVAILABLE")
        of_bias = order_flow_output.get("order_flow_bias", "NEUTRAL")
        if of_status == "AVAILABLE":
            if session_bias == "LONG_ONLY" and of_bias == "BEARISH":
                rejection_reasons.append("Order flow book imbalance contradicts LONG direction")
            elif session_bias == "SHORT_ONLY" and of_bias == "BULLISH":
                rejection_reasons.append("Order flow book imbalance contradicts SHORT direction")

        # Determine Final Decision
        if rejection_reasons:
            decision: ValidationDecision = "REJECT"
            proposal = None
            rationale = "Rejected: " + "; ".join(rejection_reasons)
        elif wait_reasons:
            decision = "WAIT"
            proposal = None
            rationale = "Waiting: " + "; ".join(wait_reasons)
        else:
            decision = "APPROVE"
            side: TradeSide = "BUY" if session_bias == "LONG_ONLY" else "SELL"
            direction = "LONG" if session_bias == "LONG_ONLY" else "SHORT"
            entry_price = float(technical_output.get("entry_reference_price") or 0.0)
            stop_loss = float(technical_output.get("stop_reference_price") or 0.0)
            take_profit = float(technical_output.get("target_reference_price") or 0.0)
            combined_confidence = round((bias_confidence * 0.3 + ml_confidence * 0.4 + 0.3 * technical_output.get("confidence", 0.5)), 4)

            rationale = (
                f"Approved {direction} trade proposal. Aligned with {session_bias} bias, "
                f"ML prediction ({ml_decision} @ {ml_confidence:.2f}), "
                f"and {technical_output.get('setup_type')} setup."
            )

            proposal = {
                "asset": asset,
                "side": side,
                "direction": direction,
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "confidence": combined_confidence,
                "rationale": rationale,
                "proposed_at": as_of.isoformat(),
            }

        return WorkerOutput(
            output_as_of=as_of,
            payload={
                "asset": asset,
                "decision": decision,
                "rationale": rationale,
                "rejection_reasons": rejection_reasons,
                "wait_reasons": wait_reasons,
                "proposal": proposal,
            },
        )
