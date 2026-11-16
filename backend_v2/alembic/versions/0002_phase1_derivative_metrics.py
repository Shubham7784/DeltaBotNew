"""Persist normalized derivative market metrics."""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0002_phase1_derivative_metrics"
down_revision = "0001_phase1_market_data"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "market_metrics",
        sa.Column("venue", sa.String(length=32), primary_key=True),
        sa.Column("symbol", sa.String(length=64), primary_key=True),
        sa.Column("metric", sa.String(length=32), primary_key=True),
        sa.Column("event_time", sa.DateTime(timezone=True), primary_key=True),
        sa.Column("value", sa.Numeric(38, 18), nullable=False),
        sa.Column("source_event_type", sa.String(length=32), nullable=False),
        sa.Column("payload", postgresql.JSONB(), nullable=False),
    )
    op.create_index(
        "ix_market_metrics_symbol_metric_time",
        "market_metrics",
        ["symbol", "metric", "event_time"],
    )


def downgrade() -> None:
    op.drop_index("ix_market_metrics_symbol_metric_time", table_name="market_metrics")
    op.drop_table("market_metrics")
