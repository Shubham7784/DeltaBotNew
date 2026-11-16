"""Live ML inference service and persistence for DeltaBot 2.0.

Loads approved models from the registry, validates feature vector completeness and order,
enforces fail-closed NO_TRADE behaviour, and persists versioned predictions to PostgreSQL.
"""

from datetime import UTC, datetime
import logging
import time
import uuid
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine, AsyncSession

from app.market_data.features import (
    CANDLE_FEATURE_SCHEMA_VERSION,
    FeatureResolution,
    get_market_feature_vector,
)
from app.market_data.regimes import Asset
from app.ml.models import TrainedModelArtifact
from app.ml.registry import ModelRegistry
from app.models import MLPrediction

logger = logging.getLogger(__name__)

PredictionDecision = Literal["LONG", "SHORT", "NO_TRADE"]
PredictionStatus = Literal["success", "no_trade", "error"]


class MLPredictionResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: uuid.UUID
    model_id: str
    asset: Asset
    as_of: datetime
    predicted_at: datetime
    decision: PredictionDecision
    trade_direction: int
    confidence: float
    probabilities: dict[str, float]
    feature_schema_version: str
    features_used: dict[str, float]
    latency_ms: float
    status: PredictionStatus
    error_reason: str | None = None


class MLInferenceService:
    """Service to execute calibrated ML inference on fresh market feature vectors."""

    def __init__(self, registry: ModelRegistry, engine: AsyncEngine):
        self.registry = registry
        self.engine = engine
        self._cached_model_id: str | None = None
        self._cached_artifact: TrainedModelArtifact | None = None

    def load_model(self, model_id: str | None = None) -> tuple[str, TrainedModelArtifact]:
        """Load approved model from registry, using cache if model_id is unchanged."""
        if model_id is not None:
            if self._cached_model_id == model_id and self._cached_artifact is not None:
                return self._cached_model_id, self._cached_artifact
            artifact = self.registry.load_model(model_id)
            meta = self.registry.get_metadata(model_id)
            if not meta.get("approved", False):
                raise PermissionError(f"Model {model_id} has not been approved for inference")
            self._cached_model_id = model_id
            self._cached_artifact = artifact
            return model_id, artifact

        # Load latest approved model
        result = self.registry.get_latest_approved_model()
        if result is None:
            raise FileNotFoundError("No approved model found in registry")
        m_id, artifact = result
        self._cached_model_id = m_id
        self._cached_artifact = artifact
        return m_id, artifact

    async def predict(
        self,
        asset: Asset,
        as_of: datetime | None = None,
        *,
        model_id: str | None = None,
        resolution: FeatureResolution = "1h",
        confidence_threshold: float = 0.40,
        connection: AsyncConnection | None = None,
    ) -> MLPredictionResult:
        """Run ML inference fail-closed for asset at as_of time."""
        start_time = time.perf_counter()
        prediction_id = uuid.uuid4()
        now = datetime.now(UTC)
        effective_as_of = as_of or now

        # 1. Load approved model
        try:
            active_model_id, artifact = self.load_model(model_id)
            meta = self.registry.get_metadata(active_model_id)
            feature_schema = meta.get("feature_schema_version", CANDLE_FEATURE_SCHEMA_VERSION)
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            logger.warning("ML inference model load failure: %s", exc)
            return MLPredictionResult(
                id=prediction_id,
                model_id=model_id or "none",
                asset=asset,
                as_of=effective_as_of,
                predicted_at=now,
                decision="NO_TRADE",
                trade_direction=0,
                confidence=0.0,
                probabilities={"SHORT": 0.0, "NEUTRAL": 1.0, "LONG": 0.0},
                feature_schema_version="none",
                features_used={},
                latency_ms=elapsed_ms,
                status="error",
                error_reason=f"Model load error: {exc}",
            )

        # Align as_of to resolution boundary
        duration = 3600 if resolution == "1h" else (900 if resolution == "15m" else 300)
        epoch = int(effective_as_of.timestamp())
        boundary_as_of = datetime.fromtimestamp(epoch - epoch % duration, UTC)

        # 2. Extract feature vector
        try:
            vector = await get_market_feature_vector(
                self.engine,
                asset=asset,
                resolution=resolution,
                as_of=boundary_as_of,
                now=effective_as_of,
                connection=connection,
                schema_version=feature_schema,
            )
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            logger.warning("Feature vector extraction error for %s: %s", asset, exc)
            return MLPredictionResult(
                id=prediction_id,
                model_id=active_model_id,
                asset=asset,
                as_of=effective_as_of,
                predicted_at=now,
                decision="NO_TRADE",
                trade_direction=0,
                confidence=0.0,
                probabilities={"SHORT": 0.0, "NEUTRAL": 1.0, "LONG": 0.0},
                feature_schema_version=feature_schema,
                features_used={},
                latency_ms=elapsed_ms,
                status="error",
                error_reason=f"Feature extraction failed: {exc}",
            )

        # 3. Validate feature availability (Fail closed)
        unavailable = [f for f in vector.features if f.status != "available"]
        if unavailable:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            reason = f"Missing features: {', '.join(f.name for f in unavailable[:3])}"
            return MLPredictionResult(
                id=prediction_id,
                model_id=active_model_id,
                asset=asset,
                as_of=vector.as_of,
                predicted_at=now,
                decision="NO_TRADE",
                trade_direction=0,
                confidence=0.0,
                probabilities={"SHORT": 0.0, "NEUTRAL": 1.0, "LONG": 0.0},
                feature_schema_version=feature_schema,
                features_used={},
                latency_ms=elapsed_ms,
                status="no_trade",
                error_reason=reason,
            )

        # 4. Check feature name and order compatibility with trained artifact
        feature_dict = {f.name: float(f.value) for f in vector.features if f.value is not None}
        missing_keys = [k for k in artifact.feature_names if k not in feature_dict]
        if missing_keys:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return MLPredictionResult(
                id=prediction_id,
                model_id=active_model_id,
                asset=asset,
                as_of=vector.as_of,
                predicted_at=now,
                decision="NO_TRADE",
                trade_direction=0,
                confidence=0.0,
                probabilities={"SHORT": 0.0, "NEUTRAL": 1.0, "LONG": 0.0},
                feature_schema_version=feature_schema,
                features_used={},
                latency_ms=elapsed_ms,
                status="error",
                error_reason=f"Model requires missing features: {missing_keys}",
            )

        # Build ordered input row
        X_row = np.array([[feature_dict[name] for name in artifact.feature_names]], dtype=np.float64)

        # 5. Execute inference
        decisions = artifact.predict_direction(X_row, confidence_threshold=confidence_threshold)
        decision_info = decisions[0]
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return MLPredictionResult(
            id=prediction_id,
            model_id=active_model_id,
            asset=asset,
            as_of=vector.as_of,
            predicted_at=now,
            decision=decision_info["decision"],
            trade_direction=decision_info["direction"],
            confidence=decision_info["confidence"],
            probabilities=decision_info["probabilities"],
            feature_schema_version=feature_schema,
            features_used=feature_dict,
            latency_ms=elapsed_ms,
            status="success" if decision_info["decision"] != "NO_TRADE" else "no_trade",
            error_reason=None if decision_info["decision"] != "NO_TRADE" else "Confidence below threshold or neutral",
        )

    async def persist_prediction(
        self,
        result: MLPredictionResult,
        session: AsyncSession,
    ) -> MLPrediction:
        """Persist prediction outcome to ml_predictions table."""
        record = MLPrediction(
            id=result.id,
            model_id=result.model_id,
            asset=result.asset,
            as_of=result.as_of,
            predicted_at=result.predicted_at,
            decision=result.decision,
            trade_direction=result.trade_direction,
            confidence=result.confidence,
            probabilities=result.probabilities,
            feature_schema_version=result.feature_schema_version,
            features_used=result.features_used,
            latency_ms=result.latency_ms,
            status=result.status,
            error_reason=result.error_reason,
        )
        session.add(record)
        await session.commit()
        return record

    async def predict_and_persist(
        self,
        asset: Asset,
        as_of: datetime | None = None,
        *,
        model_id: str | None = None,
        resolution: FeatureResolution = "1h",
        confidence_threshold: float = 0.40,
    ) -> MLPredictionResult:
        """Convenience function to predict and persist in one database session."""
        async with AsyncSession(self.engine) as session:
            result = await self.predict(
                asset,
                as_of=as_of,
                model_id=model_id,
                resolution=resolution,
                confidence_threshold=confidence_threshold,
                connection=await session.connection(),
            )
            await self.persist_prediction(result, session)
            return result
