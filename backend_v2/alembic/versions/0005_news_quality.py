"""Record RSS feed polling outcomes for news quality reporting."""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0005_news_quality"
down_revision = "0004_news_sentiment"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "news_feed_runs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("feed_url", sa.Text(), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("items_processed", sa.Integer()),
        sa.Column("error_type", sa.String(length=100)),
    )
    op.create_index(
        "ix_news_feed_runs_feed_finished_at",
        "news_feed_runs",
        ["feed_url", "finished_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_news_feed_runs_feed_finished_at", table_name="news_feed_runs")
    op.drop_table("news_feed_runs")
