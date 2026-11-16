"""Create Phase 1 market-data schema."""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0001_phase1_market_data"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "assets",
        sa.Column("symbol", sa.String(length=12), primary_key=True),
        sa.Column("name", sa.String(length=80), nullable=False),
    )
    op.create_table(
        "market_products",
        sa.Column("venue", sa.String(length=32), primary_key=True),
        sa.Column("product_id", sa.Integer(), primary_key=True),
        sa.Column("symbol", sa.String(length=64), nullable=False),
        sa.Column(
            "underlying_asset",
            sa.String(length=12),
            sa.ForeignKey("assets.symbol"),
            nullable=False,
        ),
        sa.Column("quoting_asset", sa.String(length=12), nullable=False),
        sa.Column("settling_asset", sa.String(length=12), nullable=False),
        sa.Column("contract_type", sa.String(length=40), nullable=False),
        sa.Column("state", sa.String(length=24), nullable=False),
        sa.Column("tick_size", sa.Numeric(30, 12), nullable=False),
        sa.Column("contract_value", sa.Numeric(30, 12)),
        sa.Column("launch_time", sa.DateTime(timezone=True)),
        sa.Column(
            "observed_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("raw_payload", postgresql.JSONB(), nullable=False),
        sa.UniqueConstraint("venue", "symbol", name="uq_market_products_venue_symbol"),
    )
    op.create_table(
        "candles",
        sa.Column("venue", sa.String(length=32), primary_key=True),
        sa.Column("symbol", sa.String(length=64), primary_key=True),
        sa.Column("resolution", sa.String(length=8), primary_key=True),
        sa.Column("open_time", sa.DateTime(timezone=True), primary_key=True),
        sa.Column("open", sa.Numeric(30, 12), nullable=False),
        sa.Column("high", sa.Numeric(30, 12), nullable=False),
        sa.Column("low", sa.Numeric(30, 12), nullable=False),
        sa.Column("close", sa.Numeric(30, 12), nullable=False),
        sa.Column("volume", sa.Numeric(38, 12), nullable=False),
        sa.Column("source", sa.String(length=32), nullable=False),
        sa.Column(
            "ingested_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    op.create_index(
        "ix_candles_symbol_resolution_time",
        "candles",
        ["symbol", "resolution", "open_time"],
    )
    op.create_table(
        "market_events",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("venue", sa.String(length=32), nullable=False),
        sa.Column("event_type", sa.String(length=32), nullable=False),
        sa.Column("symbol", sa.String(length=64), nullable=False),
        sa.Column("event_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "received_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("payload", postgresql.JSONB(), nullable=False),
    )
    op.create_index(
        "ix_market_events_symbol_event_time",
        "market_events",
        ["symbol", "event_time"],
    )
    op.create_index(
        "ix_market_events_event_type_event_time",
        "market_events",
        ["event_type", "event_time"],
    )
    op.create_table(
        "ingest_runs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("venue", sa.String(length=32), nullable=False),
        sa.Column("symbol", sa.String(length=64)),
        sa.Column("resolution", sa.String(length=8)),
        sa.Column("start_time", sa.DateTime(timezone=True)),
        sa.Column("end_time", sa.DateTime(timezone=True)),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("record_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("error", sa.Text()),
        sa.Column(
            "started_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("finished_at", sa.DateTime(timezone=True)),
    )
    op.create_table(
        "data_quality_events",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("venue", sa.String(length=32), nullable=False),
        sa.Column("symbol", sa.String(length=64)),
        sa.Column("event_type", sa.String(length=40), nullable=False),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("details", postgresql.JSONB(), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("data_quality_events")
    op.drop_table("ingest_runs")
    op.drop_index("ix_market_events_event_type_event_time", table_name="market_events")
    op.drop_index("ix_market_events_symbol_event_time", table_name="market_events")
    op.drop_table("market_events")
    op.drop_index("ix_candles_symbol_resolution_time", table_name="candles")
    op.drop_table("candles")
    op.drop_table("market_products")
    op.drop_table("assets")
