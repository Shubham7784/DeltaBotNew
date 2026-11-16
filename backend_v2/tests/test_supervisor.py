import asyncio
from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from app.workers.supervisor import (
    SupervisorSessionStatus,
    WorkerAttempt,
    WorkerAttemptAction,
    WorkerAttemptDecision,
    WorkerAttemptPolicy,
    WorkerDefinition,
    WorkerOutput,
    advance_session_status,
    assess_worker_attempt,
    execute_worker,
)
from app.workers.tools import WorkerRole

STARTED_AT = datetime(2026, 11, 15, 6, 0, tzinfo=UTC)
POLICY = WorkerAttemptPolicy(
    timeout=timedelta(seconds=30),
    max_attempts=2,
    max_output_age=timedelta(minutes=5),
)


def attempt(
    *,
    role: WorkerRole = WorkerRole.TECHNICAL,
    attempt_number: int = 1,
    elapsed: timedelta = timedelta(seconds=1),
    output_age: timedelta = timedelta(seconds=1),
    error: str | None = None,
) -> WorkerAttempt:
    completed_at = STARTED_AT + elapsed
    return WorkerAttempt(
        role=role,
        attempt_number=attempt_number,
        started_at=STARTED_AT,
        completed_at=completed_at,
        output_as_of=completed_at - output_age,
        error=error,
    )


def test_accepts_fresh_output_and_retries_deadline_or_worker_errors() -> None:
    accepted = assess_worker_attempt(attempt(), policy=POLICY)
    deadline_retry = assess_worker_attempt(
        attempt(elapsed=timedelta(seconds=31)),
        policy=POLICY,
    )
    error_suspend = assess_worker_attempt(
        attempt(
            attempt_number=2,
            error="worker failed",
        ),
        policy=POLICY,
    )

    assert accepted.action is WorkerAttemptAction.ACCEPT
    assert deadline_retry.action is WorkerAttemptAction.RETRY
    assert deadline_retry.reason == "deadline_exceeded"
    assert error_suspend.action is WorkerAttemptAction.SUSPEND
    assert error_suspend.reason == "worker_error"


def test_stale_and_future_outputs_are_rejected() -> None:
    stale = assess_worker_attempt(
        attempt(output_age=timedelta(minutes=6)),
        policy=POLICY,
    )
    future = assess_worker_attempt(
        attempt(output_age=-timedelta(seconds=1)),
        policy=POLICY,
    )

    assert stale.action is WorkerAttemptAction.RETRY
    assert stale.reason == "stale_output"
    assert future.action is WorkerAttemptAction.RETRY
    assert future.reason == "output_timestamp_in_future"


def test_rejects_invalid_timestamps_attempts_and_policies() -> None:
    with pytest.raises(ValueError, match="timezone"):
        assess_worker_attempt(
            WorkerAttempt(
                role=WorkerRole.TECHNICAL,
                attempt_number=1,
                started_at=STARTED_AT.replace(tzinfo=None),
                completed_at=STARTED_AT,
                output_as_of=STARTED_AT,
            ),
            policy=POLICY,
        )
    with pytest.raises(ValueError, match="exceeds"):
        assess_worker_attempt(
            attempt(attempt_number=3),
            policy=POLICY,
        )
    with pytest.raises(ValidationError, match="max_output_age"):
        WorkerAttemptPolicy(
            timeout=timedelta(seconds=30),
            max_attempts=1,
            max_output_age=timedelta(0),
        )


def test_suspends_on_incomplete_active_cycle_and_requires_explicit_recovery() -> None:
    required = frozenset({WorkerRole.TECHNICAL, WorkerRole.NEWS})
    fresh_technical = WorkerAttemptDecision(
        role=WorkerRole.TECHNICAL,
        action=WorkerAttemptAction.ACCEPT,
        reason=None,
        attempt_number=1,
    )
    fresh_news = WorkerAttemptDecision(
        role=WorkerRole.NEWS,
        action=WorkerAttemptAction.ACCEPT,
        reason=None,
        attempt_number=1,
    )
    failed_news = WorkerAttemptDecision(
        role=WorkerRole.NEWS,
        action=WorkerAttemptAction.SUSPEND,
        reason="deadline_exceeded",
        attempt_number=2,
    )
    incomplete = {WorkerRole.TECHNICAL: fresh_technical}
    complete = {
        WorkerRole.TECHNICAL: fresh_technical,
        WorkerRole.NEWS: fresh_news,
    }

    assert advance_session_status(
        SupervisorSessionStatus.INITIALIZING,
        required_roles=required,
        decisions=incomplete,
    ) is SupervisorSessionStatus.INITIALIZING
    assert advance_session_status(
        SupervisorSessionStatus.ACTIVE,
        required_roles=required,
        decisions=incomplete,
    ) is SupervisorSessionStatus.SUSPENDED
    assert advance_session_status(
        SupervisorSessionStatus.ACTIVE,
        required_roles=required,
        decisions={**complete, WorkerRole.NEWS: failed_news},
    ) is SupervisorSessionStatus.SUSPENDED
    assert advance_session_status(
        SupervisorSessionStatus.SUSPENDED,
        required_roles=required,
        decisions=complete,
    ) is SupervisorSessionStatus.SUSPENDED
    assert advance_session_status(
        SupervisorSessionStatus.SUSPENDED,
        required_roles=required,
        decisions=complete,
        resume_requested=True,
    ) is SupervisorSessionStatus.ACTIVE


def test_stopped_session_is_terminal() -> None:
    assert advance_session_status(
        "stopped",
        required_roles=frozenset({WorkerRole.NEWS}),
        decisions={},
    ) is SupervisorSessionStatus.STOPPED
    with pytest.raises(ValueError, match="Unknown session status"):
        advance_session_status(
            "unknown",
            required_roles=frozenset({WorkerRole.NEWS}),
            decisions={},
        )


def test_rejects_mismatched_worker_decision_keys() -> None:
    decision = WorkerAttemptDecision(
        role=WorkerRole.TECHNICAL,
        action=WorkerAttemptAction.ACCEPT,
        reason=None,
        attempt_number=1,
    )
    with pytest.raises(ValueError, match="mapping key"):
        advance_session_status(
            SupervisorSessionStatus.ACTIVE,
            required_roles=frozenset({WorkerRole.NEWS}),
            decisions={WorkerRole.NEWS: decision},
        )


@pytest.mark.asyncio
async def test_worker_executor_retries_errors_then_accepts_fresh_output() -> None:
    call_count = 0

    async def run_worker() -> WorkerOutput:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("temporary failure")
        return WorkerOutput(
            output_as_of=STARTED_AT + timedelta(seconds=3),
            payload={"result": "ok"},
        )

    clock_values = iter(
        STARTED_AT + timedelta(seconds=offset)
        for offset in range(4)
    )
    execution = await execute_worker(
        WorkerDefinition(
            role=WorkerRole.TECHNICAL,
            policy=POLICY,
            run=run_worker,
        ),
        now=lambda: next(clock_values),
    )

    assert call_count == 2
    assert [attempt.action for attempt in execution.attempts] == [
        WorkerAttemptAction.RETRY,
        WorkerAttemptAction.ACCEPT,
    ]
    assert execution.output is not None
    assert execution.output.payload == {"result": "ok"}


@pytest.mark.asyncio
async def test_worker_executor_times_out_and_suspends_at_attempt_limit() -> None:
    async def slow_worker() -> WorkerOutput:
        await asyncio.sleep(0.05)
        return WorkerOutput(output_as_of=STARTED_AT, payload={})

    execution = await execute_worker(
        WorkerDefinition(
            role=WorkerRole.NEWS,
            policy=WorkerAttemptPolicy(
                timeout=timedelta(milliseconds=1),
                max_attempts=1,
                max_output_age=timedelta(minutes=1),
            ),
            run=slow_worker,
        )
    )

    assert execution.decision.action is WorkerAttemptAction.SUSPEND
    assert execution.decision.reason == "deadline_exceeded"
    assert execution.output is None

