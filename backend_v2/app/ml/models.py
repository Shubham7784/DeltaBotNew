"""Model architectures, training wrappers, and calibration."""

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

ModelType = Literal["logistic_regression", "random_forest", "lightgbm", "xgboost"]
CalibrationMethod = Literal["isotonic", "sigmoid", "none"]

LABEL_TO_CLASS = {-1: 0, 0: 1, 1: 2}
CLASS_TO_LABEL = {0: -1, 1: 0, 2: 1}
CLASS_NAMES = ["SHORT", "NEUTRAL", "LONG"]


def map_labels_to_classes(y: np.ndarray) -> np.ndarray:
    """Map directional labels {-1, 0, 1} to class indices {0, 1, 2}."""
    return np.array([LABEL_TO_CLASS[val] for val in y], dtype=np.int64)


def map_classes_to_labels(classes: np.ndarray) -> np.ndarray:
    """Map class indices {0, 1, 2} back to directional labels {-1, 0, 1}."""
    return np.array([CLASS_TO_LABEL[c] for c in classes], dtype=np.int64)


def create_base_estimator(model_type: ModelType, random_state: int = 42) -> Any:
    """Instantiate the base estimator with reproducible hyperparameters."""
    if model_type == "logistic_regression":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=1000,
                        C=0.1,
                        random_state=random_state,
                        class_weight="balanced",
                    ),
                ),
            ]
        )
    elif model_type == "random_forest":
        return RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )
    elif model_type == "lightgbm":
        return LGBMClassifier(
            n_estimators=100,
            max_depth=4,
            num_leaves=15,
            learning_rate=0.03,
            min_child_samples=20,
            class_weight="balanced",
            random_state=random_state,
            verbose=-1,
            n_jobs=-1,
        )
    elif model_type == "xgboost":
        return XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            eval_metric="mlogloss",
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


@dataclass
class TrainedModelArtifact:
    model_type: ModelType
    feature_names: list[str]
    estimator: Any
    calibration_method: CalibrationMethod
    hyperparameters: dict[str, Any]
    label_to_class: dict[int, int]
    class_to_label: dict[int, int]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return calibrated class probabilities."""
        return self.estimator.predict_proba(X)

    def predict_direction(
        self,
        X: np.ndarray,
        *,
        confidence_threshold: float = 0.40,
    ) -> list[dict[str, Any]]:
        """Generate prediction decisions with explicit NO_TRADE fail-closed logic."""
        probs = self.predict_proba(X)
        results = []
        for prob_vec in probs:
            best_idx = int(np.argmax(prob_vec))
            confidence = float(prob_vec[best_idx])
            best_label = self.class_to_label[best_idx]

            # Enforce NO_TRADE if best label is NEUTRAL or confidence is below threshold
            if best_label == 0 or confidence < confidence_threshold:
                final_decision = "NO_TRADE"
                trade_direction = 0
            elif best_label == 1:
                final_decision = "LONG"
                trade_direction = 1
            else:
                final_decision = "SHORT"
                trade_direction = -1

            short_idx = self.label_to_class.get(-1)
            neutral_idx = self.label_to_class.get(0)
            long_idx = self.label_to_class.get(1)

            results.append(
                {
                    "decision": final_decision,
                    "direction": trade_direction,
                    "confidence": confidence,
                    "probabilities": {
                        "SHORT": float(prob_vec[short_idx]) if short_idx is not None else 0.0,
                        "NEUTRAL": float(prob_vec[neutral_idx]) if neutral_idx is not None else 0.0,
                        "LONG": float(prob_vec[long_idx]) if long_idx is not None else 0.0,
                    },
                }
            )
        return results


def train_model(
    model_type: ModelType,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    *,
    calibration: CalibrationMethod = "sigmoid",
    feature_names: list[str],
    random_state: int = 42,
) -> TrainedModelArtifact:
    """Train base model and optionally calibrate probabilities on validation data."""
    unique_labels = sorted(list(set(y_train)))
    label_to_class = {label: i for i, label in enumerate(unique_labels)}
    class_to_label = {i: label for i, label in enumerate(unique_labels)}

    y_train_classes = np.array([label_to_class[val] for val in y_train], dtype=np.int64)
    estimator = create_base_estimator(model_type, random_state=random_state)
    estimator.fit(X_train, y_train_classes)

    if calibration != "none" and X_val is not None and y_val is not None and len(y_val) >= 30:
        y_val_classes = np.array([label_to_class[val] for val in y_val], dtype=np.int64)
        try:
            from sklearn.frozen import FrozenEstimator

            calibrated = CalibratedClassifierCV(
                estimator=FrozenEstimator(estimator),
                method=calibration,
            )
            calibrated.fit(X_val, y_val_classes)
            final_estimator = calibrated
        except ImportError:
            calibrated = CalibratedClassifierCV(
                estimator=estimator,
                method=calibration,
                cv="prefit",
            )
            calibrated.fit(X_val, y_val_classes)
            final_estimator = calibrated
    else:
        final_estimator = estimator
        calibration = "none"

    return TrainedModelArtifact(
        model_type=model_type,
        feature_names=feature_names,
        estimator=final_estimator,
        calibration_method=calibration,
        hyperparameters={"model_type": model_type, "random_state": random_state},
        label_to_class=label_to_class,
        class_to_label=class_to_label,
    )
