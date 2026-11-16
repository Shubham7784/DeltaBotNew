"""Worker 8: Performance Analyst Agent for DeltaBot 2.0.

Analyzes completed paper trades, ML prediction accuracy, regime-specific win rates,
worker agreement, and generates descriptive improvement recommendations.
Strict invariant: Never modifies active models or risk parameters autonomously.
"""

from datetime import datetime
from typing import Any

from app.workers.supervisor import WorkerOutput


class PerformanceAnalystAgent:
    """Specialist worker computing system metrics and research improvement recommendations."""

    def analyze_performance(
        self,
        *,
        trades: list[dict[str, Any]],
        predictions: list[dict[str, Any]],
        as_of: datetime,
    ) -> WorkerOutput:
        total_trades = len(trades)
        wins = sum(1 for t in trades if float(t.get("realized_pnl", 0.0)) > 0)
        losses = sum(1 for t in trades if float(t.get("realized_pnl", 0.0)) < 0)
        win_rate = (wins / total_trades) if total_trades > 0 else 0.0

        pnl_values = [float(t.get("realized_pnl", 0.0)) for t in trades]
        total_pnl = sum(pnl_values)
        gross_profit = sum(p for p in pnl_values if p > 0)
        gross_loss = abs(sum(p for p in pnl_values if p < 0))
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (1.0 if gross_profit == 0 else float("inf"))

        # ML Accuracy analysis
        evaluated_predictions = 0
        correct_predictions = 0
        for p in predictions:
            if p.get("decision") in ("LONG", "SHORT") and "actual_outcome" in p:
                evaluated_predictions += 1
                if p["decision"] == p["actual_outcome"]:
                    correct_predictions += 1
        ml_accuracy = (correct_predictions / evaluated_predictions) if evaluated_predictions > 0 else 0.0

        # Formulate non-autonomous recommendations
        recommendations: list[str] = []
        if total_trades >= 10 and win_rate < 0.45:
            recommendations.append("Win rate below 45%; consider increasing confidence threshold in Trade Validator")
        if total_trades >= 10 and profit_factor < 1.0:
            recommendations.append("Profit factor below 1.0; review risk/reward ratio on technical setups")
        if evaluated_predictions >= 20 and ml_accuracy < 0.52:
            recommendations.append("ML out-of-sample accuracy degraded below 52%; suggest retraining candidate models with recent data")
        if not recommendations:
            recommendations.append("System performance metrics within normal parameters; maintain current configuration")

        return WorkerOutput(
            output_as_of=as_of,
            payload={
                "total_trades": total_trades,
                "win_rate": round(win_rate, 4),
                "profit_factor": round(profit_factor, 2) if profit_factor != float("inf") else 999.0,
                "total_pnl": round(total_pnl, 2),
                "evaluated_predictions": evaluated_predictions,
                "ml_accuracy": round(ml_accuracy, 4),
                "recommendations": recommendations,
                "status": "COMPLETED",
            },
        )
