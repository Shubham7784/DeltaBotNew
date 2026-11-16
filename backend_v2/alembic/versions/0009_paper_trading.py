"""Create paper trading tables: accounts, orders, positions, and trades.

Revision ID: 0009_paper_trading
Revises: 0008_ml_predictions
Create Date: 2026-11-16
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0009_paper_trading"
down_revision = "0008_ml_predictions"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 1. paper_accounts
    op.create_table(
        "paper_accounts",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("name", sa.String(length=64), nullable=False, unique=True),
        sa.Column("initial_balance", sa.Numeric(30, 8), nullable=False),
        sa.Column("cash_balance", sa.Numeric(30, 8), nullable=False),
        sa.Column("equity", sa.Numeric(30, 8), nullable=False),
        sa.Column("daily_starting_equity", sa.Numeric(30, 8), nullable=False),
        sa.Column("daily_realized_pnl", sa.Numeric(30, 8), nullable=False, server_default="0"),
        sa.Column("is_halted", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("halt_reason", sa.Text()),
        sa.Column("currency", sa.String(length=12), nullable=False, server_default="USD"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )

    # 2. paper_positions
    op.create_table(
        "paper_positions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("account_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("paper_accounts.id"), nullable=False),
        sa.Column("asset", sa.String(length=8), nullable=False),
        sa.Column("direction", sa.String(length=8), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),  # OPEN, CLOSED
        sa.Column("quantity", sa.Numeric(30, 8), nullable=False),
        sa.Column("entry_price", sa.Numeric(30, 8), nullable=False),
        sa.Column("exit_price", sa.Numeric(30, 8)),
        sa.Column("stop_loss", sa.Numeric(30, 8), nullable=False),
        sa.Column("take_profit", sa.Numeric(30, 8), nullable=False),
        sa.Column("unrealized_pnl", sa.Numeric(30, 8), nullable=False, server_default="0"),
        sa.Column("realized_pnl", sa.Numeric(30, 8), nullable=False, server_default="0"),
        sa.Column("total_fees", sa.Numeric(30, 8), nullable=False, server_default="0"),
        sa.Column("exit_reason", sa.String(length=32)),
        sa.Column("opened_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("closed_at", sa.DateTime(timezone=True)),
        sa.Column("metadata", postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.CheckConstraint("asset IN ('BTC', 'ETH')", name="ck_paper_positions_asset"),
        sa.CheckConstraint("direction IN ('LONG', 'SHORT')", name="ck_paper_positions_direction"),
        sa.CheckConstraint("status IN ('OPEN', 'CLOSED')", name="ck_paper_positions_status"),
    )
    op.create_index("ix_paper_positions_account_status", "paper_positions", ["account_id", "status"])
    op.create_index("ix_paper_positions_asset_status", "paper_positions", ["asset", "status"])

    # 3. paper_orders
    op.create_table(
        "paper_orders",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("account_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("paper_accounts.id"), nullable=False),
        sa.Column("position_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("paper_positions.id")),
        sa.Column("asset", sa.String(length=8), nullable=False),
        sa.Column("side", sa.String(length=8), nullable=False),  # BUY, SELL
        sa.Column("order_type", sa.String(length=16), nullable=False),  # MARKET, LIMIT
        sa.Column("status", sa.String(length=16), nullable=False),  # FILLED, REJECTED, CANCELLED
        sa.Column("requested_price", sa.Numeric(30, 8), nullable=False),
        sa.Column("executed_price", sa.Numeric(30, 8)),
        sa.Column("quantity", sa.Numeric(30, 8), nullable=False),
        sa.Column("fee_paid", sa.Numeric(30, 8), nullable=False, server_default="0"),
        sa.Column("slippage_bps", sa.Float(), nullable=False, server_default="0"),
        sa.Column("rejection_reason", sa.Text()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("executed_at", sa.DateTime(timezone=True)),
        sa.CheckConstraint("side IN ('BUY', 'SELL')", name="ck_paper_orders_side"),
    )
    op.create_index("ix_paper_orders_account_created", "paper_orders", ["account_id", "created_at"])


def downgrade() -> None:
    op.drop_index("ix_paper_orders_account_created", table_name="paper_orders")
    op.drop_table("paper_orders")
    op.drop_index("ix_paper_positions_asset_status", table_name="paper_positions")
    op.drop_index("ix_paper_positions_account_status", table_name="paper_positions")
    op.drop_table("paper_positions")
    op.drop_table("paper_accounts")
