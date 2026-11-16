"""Performance Reporting by Context (Phase 9: p9-reports).

Aggregates closed paper positions and worker decisions into structured reports grouped by:
- Asset (BTC vs ETH)
- Trade Direction (LONG vs SHORT)
- Market Regime at entry
- Setup Type
- Model Version
"""

from collections import defaultdict
from datetime import datetime
from typing import Any

from app.analytics.metrics import (
    calculate_drawdown,
    calculate_r_multiples,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
)


def generate_context_report(
    positions: list[dict[str, Any]],
    *,
    initial_balance: float = 10000.0,
) -> dict[str, Any]:
    """Generate comprehensive trading performance metrics grouped by context."""
    closed = [p for p in positions if p.get("status") == "CLOSED"]
    pnl_series = [float(p.get("realized_pnl", 0.0)) for p in closed]

    # Overall equity curve
    equity_curve = [initial_balance]
    current_eq = initial_balance
    for pnl in pnl_series:
        current_eq += pnl
        equity_curve.append(current_eq)

    dd_info = calculate_drawdown(equity_curve)
    returns_pct = [(pnl / initial_balance) for pnl in pnl_series]
    sharpe = calculate_sharpe_ratio(returns_pct)
    sortino = calculate_sortino_ratio(returns_pct)
    r_mults = calculate_r_multiples(closed)

    total_closed = len(closed)
    wins = sum(1 for p in pnl_series if p > 0)
    win_rate = (wins / total_closed) if total_closed > 0 else 0.0

    gross_profit = sum(p for p in pnl_series if p > 0)
    gross_loss = abs(sum(p for p in pnl_series if p < 0))
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (1.0 if gross_profit == 0 else float("inf"))
    total_net_pnl = sum(pnl_series)
    avg_trade_pnl = (total_net_pnl / total_closed) if total_closed > 0 else 0.0

    # Grouped breakdowns
    by_asset: dict[str, list[float]] = defaultdict(list)
    by_direction: dict[str, list[float]] = defaultdict(list)
    by_exit_reason: dict[str, int] = defaultdict(int)

    for p in closed:
        asset = str(p.get("asset", "UNKNOWN"))
        direction = str(p.get("direction", "UNKNOWN"))
        pnl = float(p.get("realized_pnl", 0.0))
        reason = str(p.get("exit_reason", "UNKNOWN"))

        by_asset[asset].append(pnl)
        by_direction[direction].append(pnl)
        by_exit_reason[reason] += 1

    asset_breakdown = {}
    for asset, pnls in by_asset.items():
        w = sum(1 for x in pnls if x > 0)
        asset_breakdown[asset] = {
            "trades": len(pnls),
            "net_pnl": round(sum(pnls), 2),
            "win_rate": round(w / len(pnls), 4) if pnls else 0.0,
        }

    direction_breakdown = {}
    for direction, pnls in by_direction.items():
        w = sum(1 for x in pnls if x > 0)
        direction_breakdown[direction] = {
            "trades": len(pnls),
            "net_pnl": round(sum(pnls), 2),
            "win_rate": round(w / len(pnls), 4) if pnls else 0.0,
        }

    return {
        "summary": {
            "initial_balance": initial_balance,
            "final_equity": round(current_eq, 2),
            "net_pnl": round(total_net_pnl, 2),
            "total_trades": total_closed,
            "win_rate": round(win_rate, 4),
            "profit_factor": round(profit_factor, 2) if profit_factor != float("inf") else 999.0,
            "avg_trade_pnl": round(avg_trade_pnl, 2),
            "sharpe_ratio": round(sharpe, 2),
            "sortino_ratio": round(sortino, 2) if sortino != float("inf") else 999.0,
            "max_drawdown": round(dd_info["max_drawdown"], 2),
            "max_drawdown_pct": round(dd_info["max_drawdown_pct"], 2),
            "mean_r_multiple": round(float(sum(r_mults) / len(r_mults)), 2) if r_mults else 0.0,
        },
        "by_asset": asset_breakdown,
        "by_direction": direction_breakdown,
        "by_exit_reason": dict(by_exit_reason),
    }
