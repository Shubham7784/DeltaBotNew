"""Quantitative metrics calculation for DeltaBot 2.0 (Phase 9: p9-metrics & p9-agents).

Calculates Sharpe ratio, Sortino ratio, max drawdown, expectancy, R-multiples,
worker disagreement rates, rejection ratios, and model drift.
"""

import math
from typing import Any

import numpy as np


def calculate_sharpe_ratio(
    returns: list[float],
    annualization_factor: float = math.sqrt(365 * 24),
    risk_free_rate: float = 0.0,
) -> float:
    """Calculate annualized Sharpe ratio from trade or bar returns."""
    if len(returns) < 2:
        return 0.0
    arr = np.array(returns, dtype=np.float64)
    std = float(np.std(arr, ddof=1))
    if std <= 1e-8:
        return 0.0
    mean_excess = float(np.mean(arr)) - risk_free_rate
    return float((mean_excess / std) * annualization_factor)


def calculate_sortino_ratio(
    returns: list[float],
    annualization_factor: float = math.sqrt(365 * 24),
    risk_free_rate: float = 0.0,
) -> float:
    """Calculate annualized Sortino ratio using downside standard deviation."""
    if len(returns) < 2:
        return 0.0
    arr = np.array(returns, dtype=np.float64)
    downside = arr[arr < risk_free_rate]
    if len(downside) < 1:
        return float("inf") if np.mean(arr) > 0 else 0.0
    downside_std = float(np.std(downside, ddof=1)) if len(downside) > 1 else float(abs(downside[0]))
    if downside_std <= 1e-8:
        return 0.0
    mean_excess = float(np.mean(arr)) - risk_free_rate
    return float((mean_excess / downside_std) * annualization_factor)


def calculate_drawdown(equity_curve: list[float]) -> dict[str, float]:
    """Calculate maximum drawdown and maximum drawdown percentage."""
    if not equity_curve:
        return {"max_drawdown": 0.0, "max_drawdown_pct": 0.0}
    arr = np.array(equity_curve, dtype=np.float64)
    running_max = np.maximum.accumulate(arr)
    drawdowns = running_max - arr
    max_dd = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0

    dd_pcts = np.zeros_like(arr)
    mask = running_max > 0
    dd_pcts[mask] = (running_max[mask] - arr[mask]) / running_max[mask]
    max_dd_pct = float(np.max(dd_pcts)) if len(dd_pcts) > 0 else 0.0

    return {
        "max_drawdown": max_dd,
        "max_drawdown_pct": max_dd_pct * 100.0,
    }


def calculate_r_multiples(trades: list[dict[str, Any]]) -> list[float]:
    """Calculate R-multiple for each trade: (exit - entry) / (stop distance)."""
    r_multiples: list[float] = []
    for t in trades:
        entry = float(t.get("entry_price") or 0.0)
        exit_p = float(t.get("exit_price") or 0.0)
        stop = float(t.get("stop_loss") or 0.0)
        direction = str(t.get("direction", "LONG")).upper()

        risk_dist = abs(entry - stop)
        if risk_dist <= 0.0:
            continue

        if direction == "LONG":
            r = (exit_p - entry) / risk_dist
        else:
            r = (entry - exit_p) / risk_dist

        r_multiples.append(round(r, 2))
    return r_multiples


def calculate_worker_drift_metrics(traces: list[dict[str, Any]]) -> dict[str, Any]:
    """Measure worker disagreement, rejection rates, and latency stats."""
    total_monitoring_cycles = 0
    approved_proposals = 0
    rejected_proposals = 0
    waiting_cycles = 0
    ml_tech_agreements = 0
    latencies: list[float] = []

    for trace in traces:
        if trace.get("event_type") != "monitoring_cycle":
            continue
        total_monitoring_cycles += 1
        details = trace.get("details", {})
        val = details.get("validation", {})
        dec = val.get("decision", "WAIT")
        if dec == "APPROVE":
            approved_proposals += 1
        elif dec == "REJECT":
            rejected_proposals += 1
        else:
            waiting_cycles += 1

        ml_info = details.get("ml", {})
        tech_info = details.get("technical", {})
        ml_dec = ml_info.get("decision")
        tech_dir = tech_info.get("direction")
        if ml_dec in ("LONG", "SHORT") and tech_dir in ("LONG", "SHORT"):
            if ml_dec == tech_dir:
                ml_tech_agreements += 1

        if "latency_ms" in ml_info:
            latencies.append(float(ml_info["latency_ms"]))

    rejection_rate = (rejected_proposals / total_monitoring_cycles) if total_monitoring_cycles > 0 else 0.0
    approval_rate = (approved_proposals / total_monitoring_cycles) if total_monitoring_cycles > 0 else 0.0
    mean_latency = float(np.mean(latencies)) if latencies else 0.0

    return {
        "total_cycles": total_monitoring_cycles,
        "approval_rate": round(approval_rate, 4),
        "rejection_rate": round(rejection_rate, 4),
        "waiting_cycles": waiting_cycles,
        "ml_technical_agreements": ml_tech_agreements,
        "mean_ml_latency_ms": round(mean_latency, 2),
    }
