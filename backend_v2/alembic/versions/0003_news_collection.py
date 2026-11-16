"""Add deduplicated news articles and source observations."""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0003_news_collection"
down_revision = "0002_phase1_derivative_metrics"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "news_articles",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("dedup_key", sa.String(length=64), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("article_url", sa.Text()),
        sa.Column("published_at", sa.DateTime(timezone=True)),
        sa.Column(
            "first_seen_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.UniqueConstraint("dedup_key", name="uq_news_articles_dedup_key"),
    )
    op.create_index(
        "ix_news_articles_published_at", "news_articles", ["published_at"]
    )
    op.create_table(
        "news_article_sources",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "article_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("news_articles.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("feed_url", sa.Text(), nullable=False),
        sa.Column("external_id", sa.Text(), nullable=False),
        sa.Column(
            "observed_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    op.create_index(
        "ix_news_article_sources_article_id",
        "news_article_sources",
        ["article_id"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_news_article_sources_article_id", table_name="news_article_sources"
    )
    op.drop_table("news_article_sources")
    op.drop_index("ix_news_articles_published_at", table_name="news_articles")
    op.drop_table("news_articles")
