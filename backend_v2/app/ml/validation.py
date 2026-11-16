"""Model validation, probability calibration evaluation, and after-cost trading metrics."""

import math
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    matthews_corrcoef,
)

from app.ml.models import (
    CLASS_TO_LABEL,
    LABEL_TO_CLASS,
    TrainedModelArtifact,
    map_labels_to_classes,
)


def compute_classification_metrics(
    y_true: np.ndarray,
    probs: np.ndarray,
    *,
    label_to_class: dict[int, int] | None = None,
) -> dict[str, float]:
    """Compute standard classification metrics."""
    if label_to_class is not None:
        y_classes = np.array([label_to_class[val] for val in y_true], dtype=np.int64)
    else:
        y_classes = map_labels_to_classes(y_true)
    y_pred = np.argmax(probs, axis=1)

    acc = float(accuracy_score(y_classes, y_pred))
    f1 = float(f1_score(y_classes, y_pred, average="macro", zero_division=0))
    mcc = float(matthews_corrcoef(y_classes, y_pred))

    try:
        loss = float(log_loss(y_classes, probs))
    except Exception:
        loss = 0.0

    # Brier score: mean squared difference from one-hot target
    one_hot = np.zeros_like(probs)
    for i, c in enumerate(y_classes):
        one_hot[i, c] = 1.0
    brier = float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))

    return {
        "accuracy": acc,
        "f1_macro": f1,
        "mcc": mcc,
        "log_loss": loss,
        "brier_score": brier,
    }


def compute_trading_metrics(
    decisions: list[dict[str, Any]],
    net_returns_bps: np.ndarray,
    sample_metadata: list[dict[str, Any]],
) -> dict[str, Any]:
    """Simulate trade returns on predicted signals after cost and compute expectancy and drawdown."""
    trade_returns: list[float] = []
    trade_count = 0
    wins = 0

    for i, dec in enumerate(decisions):
        direction = dec["direction"]
        if direction == 0:
            continue  # NO_TRADE

        trade_count += 1
        true_label = sample_metadata[i]["label"]
        raw_net_return = float(net_returns_bps[i])

        # If model goes LONG (+1), return is raw_net_return
        # If model goes SHORT (-1), return is opposite of gross return minus round_trip_cost
        if direction == 1:
            trade_ret = raw_net_return
        else:
            gross = sample_metadata[i].get("gross_return_bps") or 0.0
            trade_ret = float(-gross - 10.0)  # 10 bps round trip cost

        trade_returns.append(trade_ret)
        if trade_ret > 0:
            wins += 1

    if trade_count == 0:
        return {
            "total_trades": 0,
            "win_rate": 0.0,
            "net_expectancy_bps": 0.0,
            "profit_factor": 0.0,
            "max_drawdown_bps": 0.0,
            "net_pnl_bps": 0.0,
        }

    returns_arr = np.array(trade_returns, dtype=np.float64)
    win_rate = float(wins / trade_count)
    expectancy = float(np.mean(returns_arr))
    net_pnl = float(np.sum(returns_arr))

    gross_profit = float(np.sum(returns_arr[returns_arr > 0]))
    gross_loss = float(abs(np.sum(returns_arr[returns_arr < 0])))

    if gross_loss > 0:
        profit_factor = float(gross_profit / gross_loss)
    else:
        profit_factor = float(gross_profit) if gross_profit > 0 else 1.0

    # Max Drawdown calculation
    cumulative = np.cumsum(returns_arr)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = running_max - cumulative
    max_dd = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0

    return {
        "total_trades": trade_count,
        "win_rate": win_rate,
        "net_expectancy_bps": expectancy,
        "profit_factor": profit_factor,
        "max_drawdown_bps": max_dd,
        "net_pnl_bps": net_pnl,
    }


def evaluate_model(
    model: TrainedModelArtifact,
    X: np.ndarray,
    y: np.ndarray,
    net_returns_bps: np.ndarray,
    sample_metadata: list[dict[str, Any]],
    *,
    confidence_threshold: float = 0.40,
) -> dict[str, Any]:
    """Run full evaluation suite returning classification, trade expectancy, and acceptance gates."""
    probs = model.predict_proba(X)
    decisions = model.predict_direction(X, confidence_threshold=confidence_threshold)

    clf_metrics = compute_classification_metrics(y, probs, label_to_class=model.label_to_class)
    trade_metrics = compute_trading_metrics(decisions, net_returns_bps, sample_metadata)

    # Acceptance criteria from Section 10:
    # 1. Positive out-of-sample expectancy after costs
    # 2. Profit factor above 1.0
    # 3. Executed at least 5 trades
    passes_acceptance = (
        trade_metrics["total_trades"] >= 5
        and trade_metrics["net_expectancy_bps"] > 0
        and trade_metrics["profit_factor"] > 1.0
    )

    return {
        "model_type": model.model_type,
        "calibration_method": model.calibration_method,
        "confidence_threshold": confidence_threshold,
        "classification": clf_metrics,
        "trading": trade_metrics,
        "passes_acceptance": passes_acceptance,
    }
