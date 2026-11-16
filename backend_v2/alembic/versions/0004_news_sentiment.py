"""Persist deterministic news sentiment analyses."""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0004_news_sentiment"
down_revision = "0003_news_collection"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "news_sentiment",
        sa.Column(
            "article_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("news_articles.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("asset", sa.String(length=8), primary_key=True),
        sa.Column("model_revision", sa.String(length=64), primary_key=True),
        sa.Column("model_id", sa.String(length=100), nullable=False),
        sa.Column("label", sa.String(length=16), nullable=False),
        sa.Column("negative_probability", sa.Numeric(8, 7), nullable=False),
        sa.Column("neutral_probability", sa.Numeric(8, 7), nullable=False),
        sa.Column("positive_probability", sa.Numeric(8, 7), nullable=False),
        sa.Column("confidence", sa.Numeric(8, 7), nullable=False),
        sa.Column("relevance_weight", sa.Numeric(8, 7), nullable=False),
        sa.Column("weighted_sentiment", sa.Numeric(9, 7), nullable=False),
        sa.Column("event_category", sa.String(length=32), nullable=False),
        sa.Column("risk_severity", sa.String(length=16), nullable=False),
        sa.Column("scored_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "ix_news_sentiment_asset_scored_at",
        "news_sentiment",
        ["asset", "scored_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_news_sentiment_asset_scored_at", table_name="news_sentiment")
    op.drop_table("news_sentiment")
