from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import UUID, uuid4

from sqlalchemy import func, insert, select, update
from sqlalchemy.ext.asyncio import AsyncEngine

from app.models import WorkerDecisionTrace, WorkerSession
from app.workers.supervisor import (
    SupervisorSessionStatus,
    WorkerAttemptAction,
    WorkerExecution,
    WorkerDefinition,
    execute_worker_cycle,
    advance_session_status,
)
from app.workers.tools import WorkerRole


@dataclass(frozen=True)
class WorkerSessionSnapshot:
    id: UUID
    asset: str
    status: SupervisorSessionStatus
    required_roles: frozenset[WorkerRole]
    created_at: datetime
    updated_at: datetime
    suspended_at: datetime | None
    suspension_reason: str | None
    stopped_at: datetime | None


@dataclass(frozen=True)
class WorkerDecisionTraceSnapshot:
    id: UUID
    cycle_id: UUID | None
    event_type: str
    from_status: SupervisorSessionStatus | None
    to_status: SupervisorSessionStatus
    recorded_at: datetime
    details: dict[str, object]


def _require_aware(timestamp: datetime, name: str) -> None:
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError(f"{name} must include a timezone")


@asynccontextmanager
async def _session_advisory_lock(
    database_engine: AsyncEngine,
    session_id: UUID,
) -> AsyncGenerator[None, None]:
    advisory_key = session_id.int & ((1 << 63) - 1)
    async with database_engine.connect() as lock_connection:
        acquired = await lock_connection.scalar(
            select(func.pg_try_advisory_lock(advisory_key))
        )
        await lock_connection.commit()
        if not acquired:
            raise RuntimeError("A worker operation is already running for this session")
        try:
            yield
        finally:
            unlocked = await lock_connection.scalar(
                select(func.pg_advisory_unlock(advisory_key))
            )
            await lock_connection.commit()
            if not unlocked:
                raise RuntimeError("Worker session advisory lock was not released")


def _snapshot(row: object) -> WorkerSessionSnapshot:
    return WorkerSessionSnapshot(
        id=row[0],
        asset=row[1],
        status=SupervisorSessionStatus(row[2]),
        required_roles=frozenset(WorkerRole(role) for role in row[3]),
        created_at=row[4],
        updated_at=row[5],
        suspended_at=row[6],
        suspension_reason=row[7],
        stopped_at=row[8],
    )


async def run_and_record_worker_cycle(
    database_engine: AsyncEngine,
    *,
    session_id: UUID,
    workers: dict[WorkerRole, WorkerDefinition],
    resume_requested: bool = False,
) -> WorkerSessionSnapshot:
    if not workers:
        raise ValueError("At least one worker is required")
    session = await get_worker_session(database_engine, session_id=session_id)
    if frozenset(workers) != session.required_roles:
        raise ValueError("Workers must exactly match the session's required roles")
    if session.status is SupervisorSessionStatus.STOPPED:
        raise ValueError("Cannot run workers for a stopped session")

    async with _session_advisory_lock(database_engine, session_id):
        session = await get_worker_session(
            database_engine,
            session_id=session_id,
        )
        if session.status is SupervisorSessionStatus.STOPPED:
            raise ValueError("Cannot run workers for a stopped session")
        executions = await execute_worker_cycle(workers)
        return await _record_worker_cycle(
            database_engine,
            session_id=session_id,
            cycle_id=uuid4(),
            executions=executions,
            now=datetime.now(UTC),
            resume_requested=resume_requested,
        )


async def create_worker_session(
    database_engine: AsyncEngine,
    *,
    asset: str,
    required_roles: frozenset[WorkerRole],
    now: datetime,
) -> WorkerSessionSnapshot:
    _require_aware(now, "now")
    if asset not in {"BTC", "ETH"}:
        raise ValueError("asset must be BTC or ETH")
    if not required_roles:
        raise ValueError("At least one worker role is required")
    if any(not isinstance(role, WorkerRole) for role in required_roles):
        raise ValueError("required_roles must contain valid worker roles")

    session_id = uuid4()
    roles = sorted(role.value for role in required_roles)
    async with database_engine.begin() as connection:
        await connection.execute(
            insert(WorkerSession).values(
                id=session_id,
                asset=asset,
                status=SupervisorSessionStatus.INITIALIZING.value,
                required_roles=roles,
                created_at=now,
                updated_at=now,
            )
        )
        await connection.execute(
            insert(WorkerDecisionTrace).values(
                id=uuid4(),
                session_id=session_id,
                cycle_id=None,
                event_type="session_initialized",
                from_status=None,
                to_status=SupervisorSessionStatus.INITIALIZING.value,
                recorded_at=now,
                details={"asset": asset, "required_roles": roles},
            )
        )
        result = await connection.execute(
            select(
                WorkerSession.id,
                WorkerSession.asset,
                WorkerSession.status,
                WorkerSession.required_roles,
                WorkerSession.created_at,
                WorkerSession.updated_at,
                WorkerSession.suspended_at,
                WorkerSession.suspension_reason,
                WorkerSession.stopped_at,
            ).where(WorkerSession.id == session_id)
        )
        return _snapshot(result.one())


async def get_worker_session(
    database_engine: AsyncEngine,
    *,
    session_id: UUID,
) -> WorkerSessionSnapshot:
    async with database_engine.connect() as connection:
        result = await connection.execute(
            select(
                WorkerSession.id,
                WorkerSession.asset,
                WorkerSession.status,
                WorkerSession.required_roles,
                WorkerSession.created_at,
                WorkerSession.updated_at,
                WorkerSession.suspended_at,
                WorkerSession.suspension_reason,
                WorkerSession.stopped_at,
            ).where(WorkerSession.id == session_id)
        )
        row = result.one_or_none()
    if row is None:
        raise LookupError(f"Worker session not found: {session_id}")
    return _snapshot(row)


async def get_worker_decision_traces(
    database_engine: AsyncEngine,
    *,
    session_id: UUID,
    limit: int = 100,
) -> list[WorkerDecisionTraceSnapshot]:
    if not 1 <= limit <= 1_000:
        raise ValueError("limit must be between 1 and 1000")
    async with database_engine.connect() as connection:
        session_result = await connection.execute(
            select(WorkerSession.id).where(WorkerSession.id == session_id)
        )
        if session_result.scalar_one_or_none() is None:
            raise LookupError(f"Worker session not found: {session_id}")
        result = await connection.execute(
            select(
                WorkerDecisionTrace.id,
                WorkerDecisionTrace.cycle_id,
                WorkerDecisionTrace.event_type,
                WorkerDecisionTrace.from_status,
                WorkerDecisionTrace.to_status,
                WorkerDecisionTrace.recorded_at,
                WorkerDecisionTrace.details,
            )
            .where(WorkerDecisionTrace.session_id == session_id)
            .order_by(
                WorkerDecisionTrace.recorded_at.desc(),
                WorkerDecisionTrace.id.desc(),
            )
            .limit(limit)
        )
        rows = result.all()
    return [
        WorkerDecisionTraceSnapshot(
            id=row[0],
            cycle_id=row[1],
            event_type=row[2],
            from_status=(
                SupervisorSessionStatus(row[3]) if row[3] is not None else None
            ),
            to_status=SupervisorSessionStatus(row[4]),
            recorded_at=row[5],
            details=row[6],
        )
        for row in rows
    ]


async def _record_worker_cycle(
    database_engine: AsyncEngine,
    *,
    session_id: UUID,
    cycle_id: UUID,
    executions: dict[WorkerRole, WorkerExecution],
    now: datetime,
    resume_requested: bool = False,
) -> WorkerSessionSnapshot:
    _require_aware(now, "now")
    if not executions:
        raise ValueError("At least one worker execution is required")
    if any(role is not execution.role for role, execution in executions.items()):
        raise ValueError("Worker execution role must match its mapping key")
    decisions = {
        role: execution.decision for role, execution in executions.items()
    }
    async with database_engine.begin() as connection:
        result = await connection.execute(
            select(
                WorkerSession.id,
                WorkerSession.asset,
                WorkerSession.status,
                WorkerSession.required_roles,
                WorkerSession.created_at,
                WorkerSession.updated_at,
                WorkerSession.suspended_at,
                WorkerSession.suspension_reason,
                WorkerSession.stopped_at,
            )
            .where(WorkerSession.id == session_id)
            .with_for_update()
        )
        row = result.one_or_none()
        if row is None:
            raise LookupError(f"Worker session not found: {session_id}")

        snapshot = _snapshot(row)
        if snapshot.status is SupervisorSessionStatus.STOPPED:
            raise ValueError("Cannot evaluate a cycle for a stopped session")
        if now < snapshot.updated_at:
            raise ValueError("Cycle timestamp cannot precede the session update")
        existing_cycle = await connection.execute(
            select(WorkerDecisionTrace.id).where(
                WorkerDecisionTrace.session_id == session_id,
                WorkerDecisionTrace.cycle_id == cycle_id,
            )
        )
        if existing_cycle.scalar_one_or_none() is not None:
            raise ValueError(f"Cycle has already been recorded: {cycle_id}")
        next_status = advance_session_status(
            snapshot.status,
            required_roles=snapshot.required_roles,
            decisions=decisions,
            resume_requested=resume_requested,
        )
        reason_parts = [
            f"{role.value}:{decisions[role].reason or decisions[role].action.value}"
            for role in sorted(snapshot.required_roles, key=lambda value: value.value)
            if role in decisions
            and decisions[role].action is not WorkerAttemptAction.ACCEPT
        ]
        missing_roles = snapshot.required_roles - decisions.keys()
        reason_parts.extend(
            f"{role.value}:missing_result"
            for role in sorted(missing_roles, key=lambda value: value.value)
        )
        suspension_reason = ";".join(reason_parts) or None
        entering_suspension = (
            next_status is SupervisorSessionStatus.SUSPENDED
            and snapshot.status is not SupervisorSessionStatus.SUSPENDED
        )
        values: dict[str, object] = {
            "status": next_status.value,
            "updated_at": now,
            "suspension_reason": (
                suspension_reason
                if next_status is SupervisorSessionStatus.SUSPENDED
                else None
            ),
        }
        if entering_suspension:
            values["suspended_at"] = now
        elif next_status is SupervisorSessionStatus.ACTIVE:
            values["suspended_at"] = None
        await connection.execute(
            update(WorkerSession)
            .where(WorkerSession.id == session_id)
            .values(**values)
        )
        trace_decisions = {
            role.value: {
                "action": decision.action.value,
                "reason": decision.reason,
                "attempt_number": decision.attempt_number,
                "attempts": [
                    {
                        "action": attempt.action.value,
                        "reason": attempt.reason,
                        "attempt_number": attempt.attempt_number,
                        "started_at": (
                            attempt.started_at.isoformat()
                            if attempt.started_at
                            else None
                        ),
                        "completed_at": (
                            attempt.completed_at.isoformat()
                            if attempt.completed_at
                            else None
                        ),
                        "output_as_of": (
                            attempt.output_as_of.isoformat()
                            if attempt.output_as_of
                            else None
                        ),
                    }
                    for attempt in executions[role].attempts
                ],
            }
            for role, decision in sorted(
                decisions.items(),
                key=lambda item: item[0].value,
            )
        }
        await connection.execute(
            insert(WorkerDecisionTrace).values(
                id=uuid4(),
                session_id=session_id,
                cycle_id=cycle_id,
                event_type="cycle_evaluated",
                from_status=snapshot.status.value,
                to_status=next_status.value,
                recorded_at=now,
                details={
                    "required_roles": sorted(
                        role.value for role in snapshot.required_roles
                    ),
                    "decisions": trace_decisions,
                    "resume_requested": resume_requested,
                    "suspension_reason": suspension_reason,
                },
            )
        )
        updated = await connection.execute(
            select(
                WorkerSession.id,
                WorkerSession.asset,
                WorkerSession.status,
                WorkerSession.required_roles,
                WorkerSession.created_at,
                WorkerSession.updated_at,
                WorkerSession.suspended_at,
                WorkerSession.suspension_reason,
                WorkerSession.stopped_at,
            ).where(WorkerSession.id == session_id)
        )
        return _snapshot(updated.one())


async def stop_worker_session(
    database_engine: AsyncEngine,
    *,
    session_id: UUID,
    now: datetime,
    reason: str,
) -> WorkerSessionSnapshot:
    _require_aware(now, "now")
    if not reason.strip():
        raise ValueError("A stop reason is required")
    async with _session_advisory_lock(database_engine, session_id):
        async with database_engine.begin() as connection:
            result = await connection.execute(
                select(
                    WorkerSession.id,
                    WorkerSession.asset,
                    WorkerSession.status,
                    WorkerSession.required_roles,
                    WorkerSession.created_at,
                    WorkerSession.updated_at,
                    WorkerSession.suspended_at,
                    WorkerSession.suspension_reason,
                    WorkerSession.stopped_at,
                )
                .where(WorkerSession.id == session_id)
                .with_for_update()
            )
            row = result.one_or_none()
            if row is None:
                raise LookupError(f"Worker session not found: {session_id}")
            snapshot = _snapshot(row)
            if snapshot.status is SupervisorSessionStatus.STOPPED:
                return snapshot
            if now < snapshot.updated_at:
                raise ValueError("Stop timestamp cannot precede the session update")

            now_status = SupervisorSessionStatus.STOPPED
            await connection.execute(
                update(WorkerSession)
                .where(WorkerSession.id == session_id)
                .values(
                    status=now_status.value,
                    updated_at=now,
                    stopped_at=now,
                )
            )
            await connection.execute(
                insert(WorkerDecisionTrace).values(
                    id=uuid4(),
                    session_id=session_id,
                    cycle_id=None,
                    event_type="session_stopped",
                    from_status=snapshot.status.value,
                    to_status=now_status.value,
                    recorded_at=now,
                    details={"reason": reason},
                )
            )
            updated = await connection.execute(
                select(
                    WorkerSession.id,
                    WorkerSession.asset,
                    WorkerSession.status,
                    WorkerSession.required_roles,
                    WorkerSession.created_at,
                    WorkerSession.updated_at,
                    WorkerSession.suspended_at,
                    WorkerSession.suspension_reason,
                    WorkerSession.stopped_at,
                ).where(WorkerSession.id == session_id)
            )
            return _snapshot(updated.one())
