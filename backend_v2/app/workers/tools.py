from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from types import MappingProxyType
from typing import Mapping

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.market_data.features import FeatureResolution
from app.market_data.regimes import Asset


class WorkerRole(StrEnum):
    NEWS = "news"
    MACHINE_LEARNING = "machine_learning"
    TECHNICAL = "technical"
    ORDER_FLOW = "order_flow"
    VALIDATOR = "validator"
    PERFORMANCE = "performance"


class WorkerTool(StrEnum):
    READ_MARKET_FEATURES = "read_market_features"
    READ_MARKET_REGIMES = "read_market_regimes"
    READ_SESSION_BIAS = "read_session_bias"
    READ_NEWS_QUALITY = "read_news_quality"
    READ_BIAS_EVALUATION = "read_bias_evaluation"


class _ToolArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ReadMarketFeaturesArguments(_ToolArguments):
    asset: Asset
    resolution: FeatureResolution = "15m"
    as_of: datetime | None = None

    @field_validator("as_of")
    @classmethod
    def as_of_must_be_timezone_aware(
        cls, value: datetime | None
    ) -> datetime | None:
        if value is not None and value.tzinfo is None:
            raise ValueError("as_of must include a timezone")
        return value


class ReadMarketRegimesArguments(_ToolArguments):
    pass


class ReadSessionBiasArguments(_ToolArguments):
    pass


class ReadNewsQualityArguments(_ToolArguments):
    window_hours: int = Field(default=24, ge=1, le=720, strict=True)


class ReadBiasEvaluationArguments(_ToolArguments):
    days: int = Field(default=30, ge=7, le=90, strict=True)


ToolArguments = (
    ReadMarketFeaturesArguments
    | ReadMarketRegimesArguments
    | ReadSessionBiasArguments
    | ReadNewsQualityArguments
    | ReadBiasEvaluationArguments
)

_ARGUMENT_MODELS: Mapping[WorkerTool, type[_ToolArguments]] = MappingProxyType(
    {
        WorkerTool.READ_MARKET_FEATURES: ReadMarketFeaturesArguments,
        WorkerTool.READ_MARKET_REGIMES: ReadMarketRegimesArguments,
        WorkerTool.READ_SESSION_BIAS: ReadSessionBiasArguments,
        WorkerTool.READ_NEWS_QUALITY: ReadNewsQualityArguments,
        WorkerTool.READ_BIAS_EVALUATION: ReadBiasEvaluationArguments,
    }
)

WORKER_TOOL_ALLOWLIST: Mapping[WorkerRole, frozenset[WorkerTool]] = (
    MappingProxyType(
        {
            WorkerRole.NEWS: frozenset(
                {
                    WorkerTool.READ_NEWS_QUALITY,
                    WorkerTool.READ_MARKET_REGIMES,
                }
            ),
            WorkerRole.MACHINE_LEARNING: frozenset(
                {
                    WorkerTool.READ_MARKET_FEATURES,
                    WorkerTool.READ_MARKET_REGIMES,
                    WorkerTool.READ_SESSION_BIAS,
                    WorkerTool.READ_NEWS_QUALITY,
                }
            ),
            WorkerRole.TECHNICAL: frozenset(
                {
                    WorkerTool.READ_MARKET_FEATURES,
                    WorkerTool.READ_MARKET_REGIMES,
                    WorkerTool.READ_SESSION_BIAS,
                }
            ),
            WorkerRole.ORDER_FLOW: frozenset(
                {WorkerTool.READ_MARKET_FEATURES}
            ),
            WorkerRole.VALIDATOR: frozenset(WorkerTool),
            WorkerRole.PERFORMANCE: frozenset(
                {
                    WorkerTool.READ_NEWS_QUALITY,
                    WorkerTool.READ_BIAS_EVALUATION,
                }
            ),
        }
    )
)


@dataclass(frozen=True)
class AuthorizedWorkerToolCall:
    role: WorkerRole
    tool: WorkerTool
    arguments: ToolArguments


def authorize_worker_tool(
    role: WorkerRole | str,
    tool: WorkerTool | str,
    arguments: object,
) -> AuthorizedWorkerToolCall:
    try:
        worker_role = WorkerRole(role)
    except ValueError as error:
        raise PermissionError(f"Unknown worker role: {role}") from error
    try:
        worker_tool = WorkerTool(tool)
    except ValueError as error:
        raise PermissionError(f"Tool is not available: {tool}") from error

    if worker_tool not in WORKER_TOOL_ALLOWLIST[worker_role]:
        raise PermissionError(
            f"Worker role {worker_role.value} cannot use {worker_tool.value}"
        )

    validated_arguments = _ARGUMENT_MODELS[worker_tool].model_validate(arguments)
    return AuthorizedWorkerToolCall(
        role=worker_role,
        tool=worker_tool,
        arguments=validated_arguments,
    )
