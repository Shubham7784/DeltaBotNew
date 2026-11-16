from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from app.workers.tools import (
    ReadMarketFeaturesArguments,
    ReadNewsQualityArguments,
    WorkerRole,
    WorkerTool,
    authorize_worker_tool,
)


def test_authorizes_and_validates_tool_arguments() -> None:
    call = authorize_worker_tool(
        WorkerRole.MACHINE_LEARNING,
        WorkerTool.READ_MARKET_FEATURES,
        {
            "asset": "BTC",
            "resolution": "1h",
            "as_of": "2026-01-01T00:00:00Z",
        },
    )

    assert call.role is WorkerRole.MACHINE_LEARNING
    assert call.tool is WorkerTool.READ_MARKET_FEATURES
    assert isinstance(call.arguments, ReadMarketFeaturesArguments)
    assert call.arguments.as_of == datetime(2026, 1, 1, tzinfo=UTC)


def test_denies_ungranted_unknown_and_mutating_tools() -> None:
    with pytest.raises(PermissionError, match="cannot use"):
        authorize_worker_tool(
            WorkerRole.NEWS,
            WorkerTool.READ_MARKET_FEATURES,
            {"asset": "BTC"},
        )

    for unsupported_tool in (
        "place_live_order",
        "modify_risk_limits",
        "promote_model",
    ):
        with pytest.raises(PermissionError, match="not available"):
            authorize_worker_tool(
                WorkerRole.VALIDATOR,
                unsupported_tool,
                {},
            )

    with pytest.raises(PermissionError, match="Unknown worker role"):
        authorize_worker_tool("administrator", WorkerTool.READ_NEWS_QUALITY, {})


def test_rejects_invalid_and_extra_arguments() -> None:
    with pytest.raises(ValidationError, match="timezone"):
        authorize_worker_tool(
            WorkerRole.TECHNICAL,
            WorkerTool.READ_MARKET_FEATURES,
            {
                "asset": "BTC",
                "as_of": "2026-01-01T00:00:00",
            },
        )
    with pytest.raises(ValidationError, match="window_hours"):
        authorize_worker_tool(
            WorkerRole.NEWS,
            WorkerTool.READ_NEWS_QUALITY,
            {"window_hours": 721},
        )
    with pytest.raises(ValidationError, match="Extra inputs"):
        authorize_worker_tool(
            WorkerRole.PERFORMANCE,
            WorkerTool.READ_NEWS_QUALITY,
            {"window_hours": 24, "place_order": True},
        )
    with pytest.raises(ValidationError):
        ReadNewsQualityArguments(window_hours=True)
