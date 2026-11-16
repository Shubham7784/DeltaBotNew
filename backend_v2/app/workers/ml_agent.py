"""Worker 5: ML Prediction Agent for DeltaBot 2.0.

Queries the registered, approved machine learning model on the latest market features,
returns calibrated directional probability and confidence, and enforces fail-closed NO_TRADE.
"""

from datetime import datetime
from typing import Literal

from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine

from app.market_data.regimes import Asset
from app.ml.inference import MLInferenceService
from app.ml.registry import ModelRegistry
from app.workers.supervisor import WorkerOutput


class MLPredictionAgent:
    """Specialist worker executing calibrated ML model inference."""

    def __init__(self, service: MLInferenceService):
        self.service = service

    async def predict(
        self,
        asset: Asset,
        as_of: datetime,
        *,
        model_id: str | None = None,
        confidence_threshold: float = 0.45,
        connection: AsyncConnection | None = None,
    ) -> WorkerOutput:
        result = await self.service.predict(
            asset=asset,
            as_of=as_of,
            model_id=model_id,
            confidence_threshold=confidence_threshold,
            connection=connection,
        )

        return WorkerOutput(
            output_as_of=result.as_of,
            payload={
                "asset": asset,
                "model_id": result.model_id,
                "decision": result.decision,
                "trade_direction": result.trade_direction,
                "confidence": result.confidence,
                "probabilities": result.probabilities,
                "feature_schema_version": result.feature_schema_version,
                "latency_ms": result.latency_ms,
                "status": result.status,
                "error_reason": result.error_reason,
            },
        )
