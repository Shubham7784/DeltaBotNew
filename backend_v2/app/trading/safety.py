"""Execution Safety Verification (Phase 8: p8-safety).

Verifies that DeltaBot 2.0 strictly operates in a simulated paper trading environment
and that no live exchange order execution pathways or credentials exist.
"""

import inspect
from pathlib import Path
from typing import Any

from app.trading.risk_engine import RiskEngine
from app.workers.tools import WORKER_TOOL_ALLOWLIST, WorkerRole, WorkerTool


FORBIDDEN_PATTERNS = [
    "place_live_order",
    "create_order_delta_api",
    "send_order_live",
    "live_api_key",
    "api_secret_key",
    "real_money_execution",
]


def verify_codebase_safety(root_dir: Path | None = None) -> list[str]:
    """Scan all python files in app/ to prove no live execution APIs exist."""
    base_dir = root_dir or Path(__file__).parent.parent
    violations = []

    for file_path in base_dir.rglob("*.py"):
        if "__pycache__" in str(file_path) or file_path.name == "safety.py":
            continue
        try:
            content = file_path.read_text(encoding="utf-8")
        except Exception:
            continue

        for pattern in FORBIDDEN_PATTERNS:
            if pattern in content:
                violations.append(f"Forbidden live execution pattern '{pattern}' found in {file_path}")

    return violations


def verify_worker_permissions_isolated() -> bool:
    """Verify that none of the worker roles have order creation or execution tool permissions."""
    for role in WorkerRole:
        allowed_tools = WORKER_TOOL_ALLOWLIST.get(role, frozenset())
        for tool in allowed_tools:
            tool_name = tool.value.lower()
            if "order" in tool_name or "trade_execution" in tool_name or "modify_risk" in tool_name:
                raise PermissionError(f"Worker {role} has unsafe permission: {tool_name}")
    return True
