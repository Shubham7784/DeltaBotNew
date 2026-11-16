"""Persist worker sessions and immutable decision traces."""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0007_worker_supervisor"
down_revision = "0006_session_bias"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "worker_sessions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("asset", sa.String(length=8), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("required_roles", postgresql.JSONB(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("suspended_at", sa.DateTime(timezone=True)),
        sa.Column("suspension_reason", sa.Text()),
        sa.Column("stopped_at", sa.DateTime(timezone=True)),
        sa.CheckConstraint(
            "asset IN ('BTC', 'ETH')",
            name="ck_worker_sessions_asset",
        ),
        sa.CheckConstraint(
            "status IN ('initializing', 'active', 'suspended', 'stopped')",
            name="ck_worker_sessions_status",
        ),
    )
    op.create_index(
        "ix_worker_sessions_status_updated_at",
        "worker_sessions",
        ["status", "updated_at"],
    )
    op.create_table(
        "worker_decision_traces",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "session_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("worker_sessions.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("cycle_id", postgresql.UUID(as_uuid=True)),
        sa.Column("event_type", sa.String(length=32), nullable=False),
        sa.Column("from_status", sa.String(length=16)),
        sa.Column("to_status", sa.String(length=16), nullable=False),
        sa.Column("recorded_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("details", postgresql.JSONB(), nullable=False),
        sa.UniqueConstraint(
            "session_id",
            "cycle_id",
            name="uq_worker_decision_traces_session_cycle",
        ),
    )
    op.create_index(
        "ix_worker_decision_traces_session_recorded_at",
        "worker_decision_traces",
        ["session_id", "recorded_at"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_worker_decision_traces_session_recorded_at",
        table_name="worker_decision_traces",
    )
    op.drop_table("worker_decision_traces")
    op.drop_index(
        "ix_worker_sessions_status_updated_at",
        table_name="worker_sessions",
    )
    op.drop_table("worker_sessions")
