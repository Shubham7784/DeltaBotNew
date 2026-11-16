"""Unit tests for ML loader, models, validation, and registry."""

import numpy as np
import pytest

from app.ml.models import (
    CLASS_NAMES,
    TrainedModelArtifact,
    create_base_estimator,
    map_classes_to_labels,
    map_labels_to_classes,
    train_model,
)
from app.ml.registry import ModelRegistry
from app.ml.validation import (
    compute_classification_metrics,
    compute_trading_metrics,
    evaluate_model,
)


def test_label_mapping_bijection() -> None:
    y_raw = np.array([-1, 0, 1, 1, -1, 0, 1])
    classes = map_labels_to_classes(y_raw)
    assert list(classes) == [0, 1, 2, 2, 0, 1, 2]
    reconstructed = map_classes_to_labels(classes)
    assert np.array_equal(y_raw, reconstructed)


def test_model_training_and_calibration(tmp_path: pytest.TempPathFactory) -> None:
    np.random.seed(42)
    X_train = np.random.randn(100, 18)
    y_train = np.random.choice([-1, 0, 1], size=100)
    X_val = np.random.randn(40, 18)
    y_val = np.random.choice([-1, 0, 1], size=40)
    feature_names = [f"f_{i}" for i in range(18)]

    for model_type in ["logistic_regression", "random_forest", "lightgbm", "xgboost"]:
        artifact = train_model(
            model_type=model_type,  # type: ignore
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            calibration="sigmoid",
            feature_names=feature_names,
        )

        probs = artifact.predict_proba(X_val)
        assert probs.shape == (40, 3)
        assert np.allclose(np.sum(probs, axis=1), 1.0)

        decisions = artifact.predict_direction(X_val, confidence_threshold=0.35)
        assert len(decisions) == 40
        for dec in decisions:
            assert dec["decision"] in ["LONG", "SHORT", "NO_TRADE"]
            assert dec["direction"] in [-1, 0, 1]
            assert 0.0 <= dec["confidence"] <= 1.0


def test_binary_label_training_and_evaluation() -> None:
    """Ensure training and evaluating with binary labels (e.g. [-1, 1]) works without shape or index errors."""
    np.random.seed(42)
    X_train = np.random.randn(80, 18)
    y_train = np.random.choice([-1, 1], size=80)
    X_val = np.random.randn(30, 18)
    y_val = np.random.choice([-1, 1], size=30)
    feature_names = [f"f_{i}" for i in range(18)]

    for model_type in ["logistic_regression", "random_forest", "lightgbm", "xgboost"]:
        artifact = train_model(
            model_type=model_type,  # type: ignore
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            calibration="sigmoid",
            feature_names=feature_names,
        )

        decisions = artifact.predict_direction(X_val, confidence_threshold=0.40)
        assert len(decisions) == 30
        for dec in decisions:
            assert dec["decision"] in ["LONG", "SHORT", "NO_TRADE"]
            assert dec["direction"] in [-1, 0, 1]
            assert "SHORT" in dec["probabilities"]
            assert "NEUTRAL" in dec["probabilities"]
            assert "LONG" in dec["probabilities"]

        sample_meta = [{"label": y, "gross_return_bps": 50.0} for y in y_val]
        eval_result = evaluate_model(
            artifact,
            X_val,
            y_val,
            np.full(30, 40.0),
            sample_meta,
            confidence_threshold=0.40,
        )
        assert "classification" in eval_result
        assert "trading" in eval_result


def test_evaluation_metrics() -> None:
    y_true = np.array([-1, 1, 0, 1, -1])
    probs = np.array(
        [
            [0.7, 0.2, 0.1],  # true -1 (pred SHORT)
            [0.1, 0.2, 0.7],  # true 1 (pred LONG)
            [0.2, 0.6, 0.2],  # true 0 (pred NEUTRAL)
            [0.1, 0.1, 0.8],  # true 1 (pred LONG)
            [0.6, 0.3, 0.1],  # true -1 (pred SHORT)
        ]
    )
    clf = compute_classification_metrics(y_true, probs)
    assert clf["accuracy"] == 1.0
    assert clf["f1_macro"] == 1.0
    assert clf["brier_score"] < 0.2


def test_model_registry_lifecycle(tmp_path: pytest.TempPathFactory) -> None:
    registry_dir = tmp_path / "registry"
    registry = ModelRegistry(registry_dir)

    np.random.seed(42)
    X_train = np.random.randn(60, 18)
    y_train = np.random.choice([-1, 0, 1], size=60)
    feature_names = [f"feat_{i}" for i in range(18)]

    artifact = train_model(
        model_type="logistic_regression",
        X_train=X_train,
        y_train=y_train,
        calibration="none",
        feature_names=feature_names,
    )

    dummy_eval = {
        "classification": {"accuracy": 0.85},
        "trading": {"net_expectancy_bps": 12.5, "profit_factor": 1.4},
        "passes_acceptance": True,
    }

    model_id = registry.register_model(
        artifact,
        dataset_version="dataset-test-123",
        feature_schema_version="candle-market-features-v1",
        evaluation_metrics=dummy_eval,
        description="Unit test model",
    )

    models = registry.list_models()
    assert len(models) == 1
    assert models[0]["model_id"] == model_id
    assert models[0]["approved"] is False  # unapproved by default

    # Load and verify
    loaded = registry.load_model(model_id)
    assert loaded.model_type == "logistic_regression"
    assert loaded.feature_names == feature_names

    # Approve model
    approved_meta = registry.approve_model(model_id, approved_by="senior_quant")
    assert approved_meta["approved"] is True
    assert approved_meta["approved_by"] == "senior_quant"

    active_id, active_model = registry.get_latest_approved_model()  # type: ignore
    assert active_id == model_id
    assert active_model.model_type == "logistic_regression"
