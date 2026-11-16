import uuid
from datetime import date, datetime
from decimal import Decimal

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Asset(Base):
    __tablename__ = "assets"

    symbol: Mapped[str] = mapped_column(String(12), primary_key=True)
    name: Mapped[str] = mapped_column(String(80), nullable=False)


class MarketProduct(Base):
    __tablename__ = "market_products"
    __table_args__ = (
        UniqueConstraint("venue", "symbol", name="uq_market_products_venue_symbol"),
    )

    venue: Mapped[str] = mapped_column(String(32), primary_key=True)
    product_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    symbol: Mapped[str] = mapped_column(String(64), nullable=False)
    underlying_asset: Mapped[str] = mapped_column(
        ForeignKey("assets.symbol"), nullable=False
    )
    quoting_asset: Mapped[str] = mapped_column(String(12), nullable=False)
    settling_asset: Mapped[str] = mapped_column(String(12), nullable=False)
    contract_type: Mapped[str] = mapped_column(String(40), nullable=False)
    state: Mapped[str] = mapped_column(String(24), nullable=False)
    tick_size: Mapped[Decimal] = mapped_column(Numeric(30, 12), nullable=False)
    contract_value: Mapped[Decimal | None] = mapped_column(Numeric(30, 12))
    launch_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    observed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    raw_payload: Mapped[dict[str, object]] = mapped_column(JSONB, nullable=False)


class Candle(Base):
    __tablename__ = "candles"
    __table_args__ = (
        Index("ix_candles_symbol_resolution_time", "symbol", "resolution", "open_time"),
    )

    venue: Mapped[str] = mapped_column(String(32), primary_key=True)
    symbol: Mapped[str] = mapped_column(String(64), primary_key=True)
    resolution: Mapped[str] = mapped_column(String(8), primary_key=True)
    open_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), primary_key=True)
    open: Mapped[Decimal] = mapped_column(Numeric(30, 12), nullable=False)
    high: Mapped[Decimal] = mapped_column(Numeric(30, 12), nullable=False)
    low: Mapped[Decimal] = mapped_column(Numeric(30, 12), nullable=False)
    close: Mapped[Decimal] = mapped_column(Numeric(30, 12), nullable=False)
    volume: Mapped[Decimal] = mapped_column(Numeric(38, 12), nullable=False)
    source: Mapped[str] = mapped_column(String(32), nullable=False)
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class MarketEvent(Base):
    __tablename__ = "market_events"
    __table_args__ = (
        Index("ix_market_events_symbol_event_time", "symbol", "event_time"),
        Index("ix_market_events_event_type_event_time", "event_type", "event_time"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    venue: Mapped[str] = mapped_column(String(32), nullable=False)
    event_type: Mapped[str] = mapped_column(String(32), nullable=False)
    symbol: Mapped[str] = mapped_column(String(64), nullable=False)
    event_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    received_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    payload: Mapped[dict[str, object]] = mapped_column(JSONB, nullable=False)


class MarketMetric(Base):
    __tablename__ = "market_metrics"
    __table_args__ = (
        Index("ix_market_metrics_symbol_metric_time", "symbol", "metric", "event_time"),
    )

    venue: Mapped[str] = mapped_column(String(32), primary_key=True)
    symbol: Mapped[str] = mapped_column(String(64), primary_key=True)
    metric: Mapped[str] = mapped_column(String(32), primary_key=True)
    event_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), primary_key=True)
    value: Mapped[Decimal] = mapped_column(Numeric(38, 18), nullable=False)
    source_event_type: Mapped[str] = mapped_column(String(32), nullable=False)
    payload: Mapped[dict[str, object]] = mapped_column(JSONB, nullable=False)


class IngestRun(Base):
    __tablename__ = "ingest_runs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    venue: Mapped[str] = mapped_column(String(32), nullable=False)
    symbol: Mapped[str | None] = mapped_column(String(64))
    resolution: Mapped[str | None] = mapped_column(String(8))
    start_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    end_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    record_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    error: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class DataQualityEvent(Base):
    __tablename__ = "data_quality_events"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    venue: Mapped[str] = mapped_column(String(32), nullable=False)
    symbol: Mapped[str | None] = mapped_column(String(64))
    event_type: Mapped[str] = mapped_column(String(40), nullable=False)
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    details: Mapped[dict[str, object]] = mapped_column(JSONB, nullable=False)


class NewsArticle(Base):
    __tablename__ = "news_articles"
    __table_args__ = (
        UniqueConstraint("dedup_key", name="uq_news_articles_dedup_key"),
        Index("ix_news_articles_published_at", "published_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    dedup_key: Mapped[str] = mapped_column(String(64), nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    article_url: Mapped[str | None] = mapped_column(Text)
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    first_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class NewsArticleSource(Base):
    __tablename__ = "news_article_sources"
    __table_args__ = (Index("ix_news_article_sources_article_id", "article_id"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    article_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("news_articles.id", ondelete="CASCADE"), nullable=False
    )
    feed_url: Mapped[str] = mapped_column(Text, nullable=False)
    external_id: Mapped[str] = mapped_column(Text, nullable=False)
    observed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class NewsFeedRun(Base):
    __tablename__ = "news_feed_runs"
    __table_args__ = (
        Index("ix_news_feed_runs_feed_finished_at", "feed_url", "finished_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    feed_url: Mapped[str] = mapped_column(Text, nullable=False)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    finished_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    items_processed: Mapped[int | None] = mapped_column(Integer)
    error_type: Mapped[str | None] = mapped_column(String(100))


class NewsSentiment(Base):
    __tablename__ = "news_sentiment"
    __table_args__ = (
        Index("ix_news_sentiment_asset_scored_at", "asset", "scored_at"),
    )

    article_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("news_articles.id", ondelete="CASCADE"), primary_key=True
    )
    asset: Mapped[str] = mapped_column(String(8), primary_key=True)
    model_revision: Mapped[str] = mapped_column(String(64), primary_key=True)
    model_id: Mapped[str] = mapped_column(String(100), nullable=False)
    label: Mapped[str] = mapped_column(String(16), nullable=False)
    negative_probability: Mapped[Decimal] = mapped_column(Numeric(8, 7), nullable=False)
    neutral_probability: Mapped[Decimal] = mapped_column(Numeric(8, 7), nullable=False)
    positive_probability: Mapped[Decimal] = mapped_column(Numeric(8, 7), nullable=False)
    confidence: Mapped[Decimal] = mapped_column(Numeric(8, 7), nullable=False)
    relevance_weight: Mapped[Decimal] = mapped_column(Numeric(8, 7), nullable=False)
    weighted_sentiment: Mapped[Decimal] = mapped_column(Numeric(9, 7), nullable=False)
    event_category: Mapped[str] = mapped_column(String(32), nullable=False)
    risk_severity: Mapped[str] = mapped_column(String(16), nullable=False)
    scored_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class SessionBias(Base):
    __tablename__ = "session_biases"
    __table_args__ = (
        UniqueConstraint(
            "asset",
            "session_date",
            name="uq_session_biases_asset_session_date",
        ),
        CheckConstraint(
            "direction IN ('LONG_ONLY', 'SHORT_ONLY', 'NO_TRADE')",
            name="ck_session_biases_direction",
        ),
        CheckConstraint(
            "asset IN ('BTC', 'ETH')",
            name="ck_session_biases_asset",
        ),
        CheckConstraint(
            "status IN ('active', 'suspended', 'no_trade')",
            name="ck_session_biases_status",
        ),
        CheckConstraint(
            "(direction = 'NO_TRADE' AND status = 'no_trade') OR "
            "(direction IN ('LONG_ONLY', 'SHORT_ONLY') AND "
            "status IN ('active', 'suspended'))",
            name="ck_session_biases_direction_status",
        ),
        CheckConstraint("version > 0", name="ck_session_biases_version"),
        CheckConstraint(
            "confidence >= 0 AND confidence <= 1",
            name="ck_session_biases_confidence",
        ),
        Index("ix_session_biases_session_date", "session_date"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    asset: Mapped[str] = mapped_column(String(8), nullable=False)
    session_date: Mapped[date] = mapped_column(Date, nullable=False)
    direction: Mapped[str] = mapped_column(String(16), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    algorithm_version: Mapped[str] = mapped_column(String(64), nullable=False)
    decided_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    last_checked_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    suspended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    suspension_reason: Mapped[str | None] = mapped_column(Text)
    evidence: Mapped[list[str]] = mapped_column(JSONB, nullable=False)
    invalidation_conditions: Mapped[list[str]] = mapped_column(JSONB, nullable=False)
    decision_context: Mapped[dict[str, object]] = mapped_column(JSONB, nullable=False)
    latest_context: Mapped[dict[str, object]] = mapped_column(JSONB, nullable=False)


class WorkerSession(Base):
    __tablename__ = "worker_sessions"
    __table_args__ = (
        CheckConstraint(
            "asset IN ('BTC', 'ETH')",
            name="ck_worker_sessions_asset",
        ),
        CheckConstraint(
            "status IN ('initializing', 'active', 'suspended', 'stopped')",
            name="ck_worker_sessions_status",
        ),
        Index("ix_worker_sessions_status_updated_at", "status", "updated_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    asset: Mapped[str] = mapped_column(String(8), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    required_roles: Mapped[list[str]] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    suspended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    suspension_reason: Mapped[str | None] = mapped_column(Text)
    stopped_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class WorkerDecisionTrace(Base):
    __tablename__ = "worker_decision_traces"
    __table_args__ = (
        UniqueConstraint(
            "session_id",
            "cycle_id",
            name="uq_worker_decision_traces_session_cycle",
        ),
        Index(
            "ix_worker_decision_traces_session_recorded_at",
            "session_id",
            "recorded_at",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    session_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("worker_sessions.id", ondelete="RESTRICT"),
        nullable=False,
    )
    cycle_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    event_type: Mapped[str] = mapped_column(String(32), nullable=False)
    from_status: Mapped[str | None] = mapped_column(String(16))
    to_status: Mapped[str] = mapped_column(String(16), nullable=False)
    recorded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    details: Mapped[dict[str, object]] = mapped_column(JSONB, nullable=False)


class MLPrediction(Base):
    __tablename__ = "ml_predictions"
    __table_args__ = (
        CheckConstraint(
            "asset IN ('BTC', 'ETH')",
            name="ck_ml_predictions_asset",
        ),
        CheckConstraint(
            "decision IN ('LONG', 'SHORT', 'NO_TRADE')",
            name="ck_ml_predictions_decision",
        ),
        Index("ix_ml_predictions_asset_as_of", "asset", "as_of"),
        Index("ix_ml_predictions_model_as_of", "model_id", "as_of"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    model_id: Mapped[str] = mapped_column(String(100), nullable=False)
    asset: Mapped[str] = mapped_column(String(8), nullable=False)
    as_of: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    predicted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    decision: Mapped[str] = mapped_column(String(16), nullable=False)
    trade_direction: Mapped[int] = mapped_column(Integer, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    probabilities: Mapped[dict[str, object]] = mapped_column(JSONB, nullable=False)
    feature_schema_version: Mapped[str] = mapped_column(String(64), nullable=False)
    features_used: Mapped[dict[str, object]] = mapped_column(JSONB, nullable=False)
    latency_ms: Mapped[float] = mapped_column(Float, nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    error_reason: Mapped[str | None] = mapped_column(Text)


class PaperAccount(Base):
    __tablename__ = "paper_accounts"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    name: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    initial_balance: Mapped[Decimal] = mapped_column(Numeric(30, 8), nullable=False)
    cash_balance: Mapped[Decimal] = mapped_column(Numeric(30, 8), nullable=False)
    equity: Mapped[Decimal] = mapped_column(Numeric(30, 8), nullable=False)
    daily_starting_equity: Mapped[Decimal] = mapped_column(Numeric(30, 8), nullable=False)
    daily_realized_pnl: Mapped[Decimal] = mapped_column(Numeric(30, 8), nullable=False, default=Decimal(0))
    is_halted: Mapped[bool] = mapped_column(default=False, nullable=False)
    halt_reason: Mapped[str | None] = mapped_column(Text)
    currency: Mapped[str] = mapped_column(String(12), nullable=False, default="USD")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class PaperPosition(Base):
    __tablename__ = "paper_positions"
    __table_args__ = (
        CheckConstraint("asset IN ('BTC', 'ETH')", name="ck_paper_positions_asset"),
        CheckConstraint("direction IN ('LONG', 'SHORT')", name="ck_paper_positions_direction"),
        CheckConstraint("status IN ('OPEN', 'CLOSED')", name="ck_paper_positions_status"),
        Index("ix_paper_positions_account_status", "account_id", "status"),
        Index("ix_paper_positions_asset_status", "asset", "status"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    account_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("paper_accounts.id"), nullable=False)
    asset: Mapped[str] = mapped_column(String(8), nullable=False)
    direction: Mapped[str] = mapped_column(String(8), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    quantity: Mapped[Decimal] = mapped_column(Numeric(30, 8), nullable=False)
    entry_price: Mapped[Decimal] = mapped_column(Numeric(30, 8), nullable=False)
    exit_price: Mapped[Decimal | None] = mapped_column(Numeric(30, 8))
    stop_loss: Mapped[Decimal] = mapped_column(Numeric(30, 8), nullable=False)
    take_profit: Mapped[Decimal] = mapped_column(Numeric(30, 8), nullable=False)
    unrealized_pnl: Mapped[Decimal] = mapped_column(Numeric(30, 8), nullable=False, default=Decimal(0))
    realized_pnl: Mapped[Decimal] = mapped_column(Numeric(30, 8), nullable=False, default=Decimal(0))
    total_fees: Mapped[Decimal] = mapped_column(Numeric(30, 8), nullable=False, default=Decimal(0))
    exit_reason: Mapped[str | None] = mapped_column(String(32))
    opened_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    closed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    metadata_payload: Mapped[dict[str, object]] = mapped_column("metadata", JSONB, nullable=False, default=dict)


class PaperOrder(Base):
    __tablename__ = "paper_orders"
    __table_args__ = (
        CheckConstraint("side IN ('BUY', 'SELL')", name="ck_paper_orders_side"),
        Index("ix_paper_orders_account_created", "account_id", "created_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    account_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("paper_accounts.id"), nullable=False)
    position_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("paper_positions.id"))
    asset: Mapped[str] = mapped_column(String(8), nullable=False)
    side: Mapped[str] = mapped_column(String(8), nullable=False)
    order_type: Mapped[str] = mapped_column(String(16), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    requested_price: Mapped[Decimal] = mapped_column(Numeric(30, 8), nullable=False)
    executed_price: Mapped[Decimal | None] = mapped_column(Numeric(30, 8))
    quantity: Mapped[Decimal] = mapped_column(Numeric(30, 8), nullable=False)
    fee_paid: Mapped[Decimal] = mapped_column(Numeric(30, 8), nullable=False, default=Decimal(0))
    slippage_bps: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    rejection_reason: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    executed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


