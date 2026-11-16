"""Model Registry: versioned storage, auditability, checksums, and explicit approval."""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib

from app.ml.models import TrainedModelArtifact


class ModelRegistry:
    def __init__(self, registry_dir: Path | str = "model-registry") -> None:
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)

    def register_model(
        self,
        model: TrainedModelArtifact,
        *,
        dataset_version: str,
        feature_schema_version: str,
        evaluation_metrics: dict[str, Any],
        description: str = "",
    ) -> str:
        """Save model artifact, metadata, and checksums. Defaults to approved=False."""
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        raw_hash = hashlib.sha256(
            f"{model.model_type}:{dataset_version}:{timestamp}".encode("utf-8")
        ).hexdigest()[:12]
        model_id = f"model-{model.model_type}-{timestamp}-{raw_hash}"

        model_dir = self.registry_dir / model_id
        model_dir.mkdir(parents=True, exist_ok=True)

        artifact_path = model_dir / "model.joblib"
        joblib.dump(model, artifact_path)

        artifact_bytes = artifact_path.read_bytes()
        artifact_hash = hashlib.sha256(artifact_bytes).hexdigest()

        metadata: dict[str, Any] = {
            "model_id": model_id,
            "model_type": model.model_type,
            "description": description,
            "dataset_version": dataset_version,
            "feature_schema_version": feature_schema_version,
            "feature_names": model.feature_names,
            "calibration_method": model.calibration_method,
            "hyperparameters": model.hyperparameters,
            "evaluation_metrics": evaluation_metrics,
            "registered_at": datetime.now(UTC).isoformat(),
            "approved": False,
            "approved_at": None,
            "approved_by": None,
            "artifact_file": "model.joblib",
            "artifact_sha256": artifact_hash,
        }

        meta_path = model_dir / "metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, sort_keys=True)

        return model_id

    def list_models(self) -> list[dict[str, Any]]:
        """List metadata for all registered models sorted by registered_at desc."""
        models: list[dict[str, Any]] = []
        for meta_file in self.registry_dir.glob("*/metadata.json"):
            try:
                with open(meta_file, "r", encoding="utf-8") as f:
                    models.append(json.load(f))
            except Exception:
                continue
        models.sort(key=lambda m: m.get("registered_at", ""), reverse=True)
        return models

    def get_metadata(self, model_id: str) -> dict[str, Any]:
        """Get model metadata by model ID."""
        meta_path = self.registry_dir / model_id / "metadata.json"
        if not meta_path.is_file():
            raise FileNotFoundError(f"Model metadata not found for: {model_id}")
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def load_model(self, model_id: str) -> TrainedModelArtifact:
        """Load model artifact after verifying its SHA-256 checksum."""
        meta = self.get_metadata(model_id)
        artifact_path = self.registry_dir / model_id / meta["artifact_file"]
        if not artifact_path.is_file():
            raise FileNotFoundError(f"Model artifact file not found for: {model_id}")

        current_hash = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
        if current_hash != meta["artifact_sha256"]:
            raise ValueError(
                f"Model artifact checksum mismatch for {model_id}! "
                f"Expected {meta['artifact_sha256']}, got {current_hash}"
            )

        loaded = joblib.load(artifact_path)
        if not isinstance(loaded, TrainedModelArtifact):
            raise TypeError(f"Loaded object is not a TrainedModelArtifact: {type(loaded)}")
        return loaded

    def approve_model(self, model_id: str, *, approved_by: str = "operator") -> dict[str, Any]:
        """Explicit human approval gate for promoting a model to approved status."""
        meta_path = self.registry_dir / model_id / "metadata.json"
        meta = self.get_metadata(model_id)
        meta["approved"] = True
        meta["approved_at"] = datetime.now(UTC).isoformat()
        meta["approved_by"] = approved_by

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, sort_keys=True)
        return meta

    def get_latest_approved_model(
        self, model_type: str | None = None
    ) -> tuple[str, TrainedModelArtifact] | None:
        """Find the latest explicitly approved model."""
        for meta in self.list_models():
            if not meta.get("approved"):
                continue
            if model_type is not None and meta.get("model_type") != model_type:
                continue
            model_id = meta["model_id"]
            model = self.load_model(model_id)
            return model_id, model
        return None
