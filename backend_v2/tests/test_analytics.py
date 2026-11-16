"""Unit tests for Phase 9: Analytics, performance metrics, and reporting."""

import pytest

from app.analytics.metrics import (
    calculate_drawdown,
    calculate_r_multiples,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_worker_drift_metrics,
)
from app.analytics.reporting import generate_context_report


def test_sharpe_and_sortino_ratio() -> None:
    # Steady positive returns
    returns = [0.01, 0.02, 0.015, 0.01, 0.025, 0.012]
    sharpe = calculate_sharpe_ratio(returns)
    assert sharpe > 1.0

    # Sortino should be high when no downside
    sortino = calculate_sortino_ratio(returns)
    assert sortino > sharpe


def test_drawdown_calculation() -> None:
    # Curve starts at 1000, drops to 800 (200 dd), peaks at 1200, drops to 900 (300 dd)
    equity = [1000.0, 900.0, 800.0, 1100.0, 1200.0, 900.0, 1050.0]
    dd = calculate_drawdown(equity)

    assert dd["max_drawdown"] == 300.0
    # 300 / 1200 = 25%
    assert dd["max_drawdown_pct"] == 25.0


def test_r_multiples() -> None:
    trades = [
        # Long: entry 100, stop 90 (risk 10), exit 120 -> +2.0R
        {"direction": "LONG", "entry_price": 100.0, "stop_loss": 90.0, "exit_price": 120.0},
        # Long: entry 100, stop 90 (risk 10), exit 85 -> -1.5R
        {"direction": "LONG", "entry_price": 100.0, "stop_loss": 90.0, "exit_price": 85.0},
        # Short: entry 100, stop 110 (risk 10), exit 80 -> +2.0R
        {"direction": "SHORT", "entry_price": 100.0, "stop_loss": 110.0, "exit_price": 80.0},
    ]
    r_vals = calculate_r_multiples(trades)
    assert r_vals == [2.0, -1.5, 2.0]


def test_generate_context_report() -> None:
    positions = [
        {
            "asset": "BTC",
            "direction": "LONG",
            "status": "CLOSED",
            "entry_price": 90000.0,
            "stop_loss": 89000.0,
            "exit_price": 92000.0,
            "realized_pnl": 200.0,
            "exit_reason": "TAKE_PROFIT",
        },
        {
            "asset": "ETH",
            "direction": "SHORT",
            "status": "CLOSED",
            "entry_price": 3100.0,
            "stop_loss": 3150.0,
            "exit_price": 3160.0,
            "realized_pnl": -60.0,
            "exit_reason": "STOP_LOSS",
        },
    ]

    report = generate_context_report(positions, initial_balance=10000.0)
    summary = report["summary"]

    assert summary["total_trades"] == 2
    assert summary["win_rate"] == 0.50
    assert summary["net_pnl"] == 140.0
    assert summary["final_equity"] == 10140.0
    assert summary["profit_factor"] > 1.0

    # Breakdowns
    assert "BTC" in report["by_asset"]
    assert "ETH" in report["by_asset"]
    assert report["by_asset"]["BTC"]["net_pnl"] == 200.0
    assert report["by_asset"]["ETH"]["net_pnl"] == -60.0

    assert "LONG" in report["by_direction"]
    assert "SHORT" in report["by_direction"]
    assert report["by_exit_reason"]["TAKE_PROFIT"] == 1
    assert report["by_exit_reason"]["STOP_LOSS"] == 1


def test_worker_drift_metrics() -> None:
    traces = [
        {
            "event_type": "monitoring_cycle",
            "details": {
                "validation": {"decision": "APPROVE"},
                "ml": {"decision": "LONG", "latency_ms": 15.2},
                "technical": {"direction": "LONG"},
            },
        },
        {
            "event_type": "monitoring_cycle",
            "details": {
                "validation": {"decision": "REJECT"},
                "ml": {"decision": "SHORT", "latency_ms": 18.0},
                "technical": {"direction": "LONG"},
            },
        },
    ]
    drift = calculate_worker_drift_metrics(traces)
    assert drift["total_cycles"] == 2
    assert drift["approval_rate"] == 0.50
    assert drift["rejection_rate"] == 0.50
    assert drift["ml_technical_agreements"] == 1
    assert drift["mean_ml_latency_ms"] == 16.6
