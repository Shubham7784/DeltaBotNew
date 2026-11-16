import asyncio
import logging
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.workers.tools import WorkerRole

logger = logging.getLogger(__name__)


class WorkerAttemptAction(StrEnum):
    ACCEPT = "accept"
    RETRY = "retry"
    SUSPEND = "suspend"


class SupervisorSessionStatus(StrEnum):
    INITIALIZING = "initializing"
    ACTIVE = "active"
    SUSPENDED = "suspended"
    STOPPED = "stopped"


class WorkerAttemptPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    timeout: timedelta
    max_attempts: int = Field(ge=1, strict=True)
    max_output_age: timedelta

    @model_validator(mode="after")
    def durations_must_be_positive(self) -> "WorkerAttemptPolicy":
        if self.timeout <= timedelta(0):
            raise ValueError("timeout must be positive")
        if self.max_output_age <= timedelta(0):
            raise ValueError("max_output_age must be positive")
        return self


@dataclass(frozen=True)
class WorkerAttempt:
    role: WorkerRole
    attempt_number: int
    started_at: datetime
    completed_at: datetime
    output_as_of: datetime | None
    error: str | None = None


@dataclass(frozen=True)
class WorkerAttemptDecision:
    role: WorkerRole
    action: WorkerAttemptAction
    reason: str | None
    attempt_number: int
    started_at: datetime | None = None
    completed_at: datetime | None = None
    output_as_of: datetime | None = None


class WorkerOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    output_as_of: datetime | None
    payload: dict[str, object]

    @field_validator("output_as_of")
    @classmethod
    def output_timestamp_must_be_aware(
        cls, value: datetime | None
    ) -> datetime | None:
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError("output_as_of must include a timezone")
        return value


@dataclass(frozen=True)
class WorkerDefinition:
    role: WorkerRole
    policy: WorkerAttemptPolicy
    run: Callable[[], Awaitable[WorkerOutput]]


@dataclass(frozen=True)
class WorkerExecution:
    role: WorkerRole
    output: WorkerOutput | None
    attempts: tuple[WorkerAttemptDecision, ...]

    def __post_init__(self) -> None:
        if not self.attempts:
            raise ValueError("Worker execution must contain at least one attempt")
        if any(attempt.role is not self.role for attempt in self.attempts):
            raise ValueError("Worker attempt role must match its execution")
        accepted = self.attempts[-1].action is WorkerAttemptAction.ACCEPT
        if accepted != (self.output is not None):
            raise ValueError("Only accepted worker executions may contain output")

    @property
    def decision(self) -> WorkerAttemptDecision:
        return self.attempts[-1]


def _require_aware(timestamp: datetime, name: str) -> None:
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError(f"{name} must include a timezone")


def assess_worker_attempt(
    attempt: WorkerAttempt,
    *,
    policy: WorkerAttemptPolicy,
) -> WorkerAttemptDecision:
    _require_aware(attempt.started_at, "started_at")
    _require_aware(attempt.completed_at, "completed_at")
    if attempt.output_as_of is not None:
        _require_aware(attempt.output_as_of, "output_as_of")
    if attempt.completed_at < attempt.started_at:
        raise ValueError("completed_at cannot precede started_at")
    if attempt.attempt_number < 1:
        raise ValueError("attempt_number must be positive")
    if attempt.attempt_number > policy.max_attempts:
        raise ValueError("attempt_number exceeds the configured maximum")

    reason: str | None = None
    deadline = attempt.started_at + policy.timeout
    if attempt.completed_at > deadline:
        reason = "deadline_exceeded"
    elif attempt.error is not None:
        reason = "worker_error"
    elif attempt.output_as_of is None:
        reason = "missing_output_timestamp"
    elif attempt.output_as_of > attempt.completed_at:
        reason = "output_timestamp_in_future"
    elif attempt.completed_at - attempt.output_as_of > policy.max_output_age:
        reason = "stale_output"

    if reason is None:
        action = WorkerAttemptAction.ACCEPT
    elif attempt.attempt_number < policy.max_attempts:
        action = WorkerAttemptAction.RETRY
    else:
        action = WorkerAttemptAction.SUSPEND

    return WorkerAttemptDecision(
        role=attempt.role,
        action=action,
        reason=reason,
        attempt_number=attempt.attempt_number,
        started_at=attempt.started_at,
        completed_at=attempt.completed_at,
        output_as_of=attempt.output_as_of,
    )


async def execute_worker(
    worker: WorkerDefinition,
    *,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> WorkerExecution:
    attempts: list[WorkerAttemptDecision] = []
    accepted_output: WorkerOutput | None = None
    for attempt_number in range(1, worker.policy.max_attempts + 1):
        started_at = now()
        error: str | None = None
        output: WorkerOutput | None = None
        try:
            raw_output = await asyncio.wait_for(
                worker.run(), timeout=worker.policy.timeout.total_seconds()
            )
            output = WorkerOutput.model_validate(raw_output)
        except TimeoutError:
            error = "worker_timeout"
        except Exception as exception:
            error = f"worker_error:{type(exception).__name__}"
            logger.error(
                "Worker %s failed on attempt %s: %s",
                worker.role.value,
                attempt_number,
                type(exception).__name__,
            )
        completed_at = now()
        decision = assess_worker_attempt(
            WorkerAttempt(
                role=worker.role,
                attempt_number=attempt_number,
                started_at=started_at,
                completed_at=completed_at,
                output_as_of=output.output_as_of if output is not None else None,
                error=error,
            ),
            policy=worker.policy,
        )
        attempts.append(decision)
        if decision.action is WorkerAttemptAction.ACCEPT:
            accepted_output = output
            break
        if decision.action is WorkerAttemptAction.SUSPEND:
            break
    return WorkerExecution(
        role=worker.role,
        output=accepted_output,
        attempts=tuple(attempts),
    )


async def execute_worker_cycle(
    workers: Mapping[WorkerRole, WorkerDefinition],
) -> dict[WorkerRole, WorkerExecution]:
    if not workers:
        raise ValueError("At least one worker is required")
    if any(role is not worker.role for role, worker in workers.items()):
        raise ValueError("Worker role must match its mapping key")
    results = await asyncio.gather(
        *(execute_worker(worker) for worker in workers.values())
    )
    return {result.role: result for result in results}


def advance_session_status(
    current: SupervisorSessionStatus | str,
    *,
    required_roles: frozenset[WorkerRole],
    decisions: Mapping[WorkerRole, WorkerAttemptDecision],
    resume_requested: bool = False,
) -> SupervisorSessionStatus:
    try:
        current_status = SupervisorSessionStatus(current)
    except ValueError as error:
        raise ValueError(f"Unknown session status: {current}") from error
    if current_status is SupervisorSessionStatus.STOPPED:
        return current_status
    if not required_roles:
        raise ValueError("At least one worker role is required")
    if any(role is not decision.role for role, decision in decisions.items()):
        raise ValueError("Worker decision role must match its mapping key")
    if any(
        decision.action is WorkerAttemptAction.SUSPEND
        for role, decision in decisions.items()
        if role in required_roles
    ):
        return SupervisorSessionStatus.SUSPENDED

    cycle_is_fresh = all(
        role in decisions
        and decisions[role].action is WorkerAttemptAction.ACCEPT
        for role in required_roles
    )
    if not cycle_is_fresh:
        if current_status is SupervisorSessionStatus.INITIALIZING:
            return current_status
        return SupervisorSessionStatus.SUSPENDED
    if (
        current_status is SupervisorSessionStatus.SUSPENDED
        and not resume_requested
    ):
        return current_status
    return SupervisorSessionStatus.ACTIVE
