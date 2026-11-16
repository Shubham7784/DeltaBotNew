"""Tests for ML inference service, training-inference parity, and fail-closed safety."""

from datetime import UTC, datetime
from pathlib import Path
import uuid

import numpy as np
import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.ml.inference import MLInferenceService, MLPredictionResult
from app.ml.loader import load_split
from app.ml.models import train_model
from app.ml.registry import ModelRegistry
from app.models import MLPrediction


@pytest.fixture
def mock_registry(tmp_path: Path) -> ModelRegistry:
    reg_dir = tmp_path / "test_registry"
    registry = ModelRegistry(reg_dir)
    np.random.seed(42)
    X = np.random.randn(50, 4)
    y = np.random.choice([-1, 1], size=50)
    feat_names = ["f1", "f2", "f3", "f4"]

    artifact = train_model(
        model_type="logistic_regression",
        X_train=X,
        y_train=y,
        calibration="none",
        feature_names=feat_names,
    )
    m_id = registry.register_model(
        artifact,
        dataset_version="ds-test",
        feature_schema_version="schema-test",
        evaluation_metrics={"accuracy": 0.8},
        description="test model",
    )
    registry.approve_model(m_id, approved_by="test_reviewer")
    return registry


def test_training_inference_parity() -> None:
    """Verify that an individual feature row produces identical predictions whether batch or single."""
    np.random.seed(42)
    X = np.random.randn(20, 5)
    y = np.random.choice([-1, 1], size=20)
    features = ["c1", "c2", "c3", "c4", "c5"]

    artifact = train_model(
        model_type="lightgbm",
        X_train=X,
        y_train=y,
        calibration="sigmoid",
        feature_names=features,
    )

    batch_decisions = artifact.predict_direction(X, confidence_threshold=0.45)

    for i in range(len(X)):
        single_row = X[i : i + 1]
        single_decision = artifact.predict_direction(single_row, confidence_threshold=0.45)[0]

        assert single_decision["decision"] == batch_decisions[i]["decision"]
        assert single_decision["direction"] == batch_decisions[i]["direction"]
        assert np.isclose(single_decision["confidence"], batch_decisions[i]["confidence"], atol=1e-6)
        for cls_name in ["SHORT", "NEUTRAL", "LONG"]:
            assert np.isclose(
                single_decision["probabilities"][cls_name],
                batch_decisions[i]["probabilities"][cls_name],
                atol=1e-6,
            )


def test_unapproved_model_raises_permission_error(mock_registry: ModelRegistry) -> None:
    # Register an unapproved model
    np.random.seed(42)
    artifact = train_model(
        model_type="random_forest",
        X_train=np.random.randn(30, 4),
        y_train=np.random.choice([-1, 1], size=30),
        calibration="none",
        feature_names=["f1", "f2", "f3", "f4"],
    )
    unapproved_id = mock_registry.register_model(
        artifact,
        dataset_version="ds-test",
        feature_schema_version="schema-test",
        evaluation_metrics={},
    )

    service = MLInferenceService(mock_registry, engine=None)  # type: ignore
    with pytest.raises(PermissionError, match="not been approved"):
        service.load_model(unapproved_id)


def test_prediction_result_schema_validation() -> None:
    res = MLPredictionResult(
        id=uuid.uuid4(),
        model_id="model-test-1",
        asset="BTC",
        as_of=datetime(2026, 11, 16, 12, 0, tzinfo=UTC),
        predicted_at=datetime(2026, 11, 16, 12, 0, 5, tzinfo=UTC),
        decision="LONG",
        trade_direction=1,
        confidence=0.68,
        probabilities={"SHORT": 0.32, "NEUTRAL": 0.0, "LONG": 0.68},
        feature_schema_version="candle-market-features-v1",
        features_used={"close_price": 90500.0},
        latency_ms=12.4,
        status="success",
        error_reason=None,
    )
    assert res.decision == "LONG"
    assert res.trade_direction == 1
    assert res.status == "success"
