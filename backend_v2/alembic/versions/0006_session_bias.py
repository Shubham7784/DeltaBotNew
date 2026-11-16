"""Persist locked daily session-bias decisions."""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0006_session_bias"
down_revision = "0005_news_quality"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "session_biases",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("asset", sa.String(length=8), nullable=False),
        sa.Column("session_date", sa.Date(), nullable=False),
        sa.Column("direction", sa.String(length=16), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("algorithm_version", sa.String(length=64), nullable=False),
        sa.Column("decided_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_checked_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("suspended_at", sa.DateTime(timezone=True)),
        sa.Column("suspension_reason", sa.Text()),
        sa.Column("evidence", postgresql.JSONB(), nullable=False),
        sa.Column("invalidation_conditions", postgresql.JSONB(), nullable=False),
        sa.Column("decision_context", postgresql.JSONB(), nullable=False),
        sa.Column("latest_context", postgresql.JSONB(), nullable=False),
        sa.UniqueConstraint(
            "asset",
            "session_date",
            name="uq_session_biases_asset_session_date",
        ),
        sa.CheckConstraint(
            "asset IN ('BTC', 'ETH')",
            name="ck_session_biases_asset",
        ),
        sa.CheckConstraint(
            "direction IN ('LONG_ONLY', 'SHORT_ONLY', 'NO_TRADE')",
            name="ck_session_biases_direction",
        ),
        sa.CheckConstraint(
            "status IN ('active', 'suspended', 'no_trade')",
            name="ck_session_biases_status",
        ),
        sa.CheckConstraint(
            "(direction = 'NO_TRADE' AND status = 'no_trade') OR "
            "(direction IN ('LONG_ONLY', 'SHORT_ONLY') AND "
            "status IN ('active', 'suspended'))",
            name="ck_session_biases_direction_status",
        ),
        sa.CheckConstraint("version > 0", name="ck_session_biases_version"),
        sa.CheckConstraint(
            "confidence >= 0 AND confidence <= 1",
            name="ck_session_biases_confidence",
        ),
    )
    op.create_index(
        "ix_session_biases_session_date",
        "session_biases",
        ["session_date"],
    )


def downgrade() -> None:
    op.drop_index("ix_session_biases_session_date", table_name="session_biases")
    op.drop_table("session_biases")
