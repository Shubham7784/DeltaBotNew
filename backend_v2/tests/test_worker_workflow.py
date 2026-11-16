"""Tests for Phase 7: Technical Agent, Order Flow Agent, Trade Validator, and Workflow Coordinator."""

from datetime import UTC, datetime
from typing import Any
import uuid

import pytest

from app.workers.order_flow_agent import OrderFlowAgent
from app.workers.performance_agent import PerformanceAnalystAgent
from app.workers.technical_agent import (
    CandleBar,
    calculate_ema,
    calculate_rsi,
    calculate_vwap,
)
from app.workers.validator_agent import TradeValidatorAgent


def test_technical_indicator_calculations() -> None:
    prices = [100.0 + i for i in range(30)]
    ema_10 = calculate_ema(prices, 10)
    assert len(ema_10) == 30
    assert not any(p < 100.0 for p in ema_10[10:])

    rsi_14 = calculate_rsi(prices, 14)
    assert len(rsi_14) == 30
    # Consistently increasing prices should give high RSI (> 70)
    assert rsi_14[-1] > 70.0

    candles = [
        CandleBar(
            open_time=datetime(2026, 1, 1, i, tzinfo=UTC),
            open=100.0 + i,
            high=102.0 + i,
            low=99.0 + i,
            close=101.0 + i,
            volume=10.0,
        )
        for i in range(5)
    ]
    vwap = calculate_vwap(candles)
    assert vwap is not None
    assert 100.0 <= vwap <= 106.0


def test_trade_validator_rejection_on_conflicting_bias() -> None:
    validator = TradeValidatorAgent()
    now = datetime(2026, 11, 16, 12, 0, tzinfo=UTC)

    # Bias is SHORT_ONLY, but technical and ML are LONG
    result = validator.validate(
        asset="BTC",
        session_bias="SHORT_ONLY",
        bias_confidence=0.75,
        regime="RANGE_BOUND",
        technical_output={"setup_detected": True, "direction": "LONG", "setup_type": "EMA_PULLBACK"},
        ml_output={"decision": "LONG", "confidence": 0.65},
        order_flow_output={"status": "AVAILABLE", "order_flow_bias": "NEUTRAL"},
        as_of=now,
    )

    assert result.payload["decision"] == "REJECT"
    assert result.payload["proposal"] is None
    assert any("SHORT_ONLY" in r for r in result.payload["rejection_reasons"])


def test_trade_validator_wait_on_no_technical_setup() -> None:
    validator = TradeValidatorAgent()
    now = datetime(2026, 11, 16, 12, 0, tzinfo=UTC)

    # Bias is LONG_ONLY, ML is LONG, but no technical setup
    result = validator.validate(
        asset="BTC",
        session_bias="LONG_ONLY",
        bias_confidence=0.80,
        regime="STRONG_BULLISH",
        technical_output={"setup_detected": False, "direction": "NEUTRAL"},
        ml_output={"decision": "LONG", "confidence": 0.68},
        order_flow_output={"status": "AVAILABLE", "order_flow_bias": "BULLISH"},
        as_of=now,
    )

    assert result.payload["decision"] == "WAIT"
    assert result.payload["proposal"] is None
    assert any("technical setup" in r for r in result.payload["wait_reasons"])


def test_trade_validator_approves_aligned_setup() -> None:
    validator = TradeValidatorAgent()
    now = datetime(2026, 11, 16, 12, 0, tzinfo=UTC)

    # Everything is aligned LONG
    result = validator.validate(
        asset="BTC",
        session_bias="LONG_ONLY",
        bias_confidence=0.80,
        regime="STRONG_BULLISH",
        technical_output={
            "setup_detected": True,
            "setup_type": "EMA_PULLBACK",
            "direction": "LONG",
            "confidence": 0.65,
            "entry_reference_price": 91000.0,
            "stop_reference_price": 90200.0,
            "target_reference_price": 92500.0,
        },
        ml_output={"decision": "LONG", "confidence": 0.65},
        order_flow_output={"status": "AVAILABLE", "order_flow_bias": "BULLISH"},
        news_output={"event_risk": "low"},
        as_of=now,
    )

    assert result.payload["decision"] == "APPROVE"
    proposal = result.payload["proposal"]
    assert proposal is not None
    assert proposal["asset"] == "BTC"
    assert proposal["direction"] == "LONG"
    assert proposal["side"] == "BUY"
    assert proposal["entry_price"] == 91000.0
    assert proposal["stop_loss"] == 90200.0
    assert proposal["take_profit"] == 92500.0


def test_performance_analyst_agent() -> None:
    agent = PerformanceAnalystAgent()
    now = datetime(2026, 11, 16, 12, 0, tzinfo=UTC)

    mock_trades = [
        {"realized_pnl": 120.0},
        {"realized_pnl": -50.0},
        {"realized_pnl": 80.0},
        {"realized_pnl": -40.0},
        {"realized_pnl": 90.0},
    ]
    mock_predictions = [
        {"decision": "LONG", "actual_outcome": "LONG"},
        {"decision": "SHORT", "actual_outcome": "SHORT"},
        {"decision": "LONG", "actual_outcome": "SHORT"},
    ]

    out = agent.analyze_performance(trades=mock_trades, predictions=mock_predictions, as_of=now)
    assert out.payload["total_trades"] == 5
    assert out.payload["win_rate"] == 0.60
    assert out.payload["profit_factor"] > 1.0
    assert abs(out.payload["ml_accuracy"] - 2 / 3) < 1e-3
    assert len(out.payload["recommendations"]) > 0
