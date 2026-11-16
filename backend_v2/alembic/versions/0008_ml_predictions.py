"""Persist ML model prediction outputs and inference metadata.

Revision ID: 0008_ml_predictions
Revises: 0007_worker_supervisor
Create Date: 2026-11-16
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0008_ml_predictions"
down_revision = "0007_worker_supervisor"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "ml_predictions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("model_id", sa.String(length=100), nullable=False),
        sa.Column("asset", sa.String(length=8), nullable=False),
        sa.Column("as_of", sa.DateTime(timezone=True), nullable=False),
        sa.Column("predicted_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("decision", sa.String(length=16), nullable=False),
        sa.Column("trade_direction", sa.Integer(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("probabilities", postgresql.JSONB(), nullable=False),
        sa.Column("feature_schema_version", sa.String(length=64), nullable=False),
        sa.Column("features_used", postgresql.JSONB(), nullable=False),
        sa.Column("latency_ms", sa.Float(), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("error_reason", sa.Text()),
        sa.CheckConstraint(
            "asset IN ('BTC', 'ETH')",
            name="ck_ml_predictions_asset",
        ),
        sa.CheckConstraint(
            "decision IN ('LONG', 'SHORT', 'NO_TRADE')",
            name="ck_ml_predictions_decision",
        ),
    )
    op.create_index(
        "ix_ml_predictions_asset_as_of",
        "ml_predictions",
        ["asset", "as_of"],
    )
    op.create_index(
        "ix_ml_predictions_model_as_of",
        "ml_predictions",
        ["model_id", "as_of"],
    )


def downgrade() -> None:
    op.drop_index("ix_ml_predictions_model_as_of", table_name="ml_predictions")
    op.drop_index("ix_ml_predictions_asset_as_of", table_name="ml_predictions")
    op.drop_table("ml_predictions")
