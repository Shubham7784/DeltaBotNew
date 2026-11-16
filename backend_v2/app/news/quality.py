from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

from pydantic import BaseModel
from sqlalchemy import case, func, select
from sqlalchemy.ext.asyncio import AsyncEngine

from app.models import NewsArticle, NewsFeedRun, NewsSentiment
from app.news.collector import feed_identifier
from app.news.sentiment import MODEL_REVISION

LOW_CONFIDENCE_THRESHOLD = Decimal("0.55")


class NewsFeedQuality(BaseModel):
    source: str
    attempts: int
    successes: int
    failures: int
    failure_rate_percent: float
    last_status: str | None
    last_finished_at: datetime | None
    last_success_items: int | None
    last_error_type: str | None


class AssetNewsQuality(BaseModel):
    sentiment_articles: int
    coverage_percent: float
    average_publish_to_ingest_seconds: float | None
    max_publish_to_ingest_seconds: float | None
    average_scoring_latency_seconds: float | None
    max_scoring_latency_seconds: float | None
    mean_confidence: float | None
    low_confidence_articles: int
    event_categories: dict[str, int]
    risk_severities: dict[str, int]


class NewsQualityReport(BaseModel):
    checked_at: datetime
    window_start: datetime
    window_hours: int
    model_revision: str
    total_articles: int
    scored_articles: int
    pending_articles: int
    score_coverage_percent: float
    sources: list[NewsFeedQuality]
    assets: dict[str, AssetNewsQuality]
    evaluation_note: str


def _as_float(value: Decimal | float | int | None) -> float | None:
    return None if value is None else float(value)


async def get_news_quality_report(
    database_engine: AsyncEngine,
    *,
    window_hours: int = 24,
) -> NewsQualityReport:
    if window_hours < 1 or window_hours > 720:
        raise ValueError("window_hours must be between 1 and 720")
    checked_at = datetime.now(UTC)
    window_start = checked_at - timedelta(hours=window_hours)

    ingestion_lag = case(
        (
            NewsArticle.published_at.is_not(None)
            & (NewsArticle.first_seen_at >= NewsArticle.published_at),
            func.extract(
                "epoch",
                NewsArticle.first_seen_at - NewsArticle.published_at,
            ),
        ),
        else_=None,
    )
    scoring_lag = case(
        (
            NewsSentiment.scored_at >= NewsArticle.first_seen_at,
            func.extract(
                "epoch",
                NewsSentiment.scored_at - NewsArticle.first_seen_at,
            ),
        ),
        else_=None,
    )

    async with database_engine.connect() as connection:
        totals_result = await connection.execute(
            select(
                func.count(func.distinct(NewsArticle.id)),
                func.count(func.distinct(NewsSentiment.article_id)),
            )
            .select_from(NewsArticle)
            .outerjoin(
                NewsSentiment,
                (NewsSentiment.article_id == NewsArticle.id)
                & (NewsSentiment.model_revision == MODEL_REVISION),
            )
            .where(NewsArticle.first_seen_at >= window_start)
        )
        total_articles, scored_articles = totals_result.one()

        feed_result = await connection.execute(
            select(
                NewsFeedRun.feed_url,
                NewsFeedRun.status,
                NewsFeedRun.items_processed,
                NewsFeedRun.error_type,
                NewsFeedRun.started_at,
                NewsFeedRun.finished_at,
            )
            .where(NewsFeedRun.finished_at >= window_start)
            .order_by(NewsFeedRun.feed_url, NewsFeedRun.finished_at)
        )
        feed_rows = feed_result.all()

        asset_result = await connection.execute(
            select(
                NewsSentiment.asset,
                func.count(func.distinct(NewsSentiment.article_id)),
                func.avg(ingestion_lag),
                func.max(ingestion_lag),
                func.avg(scoring_lag),
                func.max(scoring_lag),
                func.avg(NewsSentiment.confidence),
                func.sum(
                    case(
                        (NewsSentiment.confidence < LOW_CONFIDENCE_THRESHOLD, 1),
                        else_=0,
                    )
                ),
            )
            .select_from(NewsArticle)
            .join(NewsSentiment, NewsSentiment.article_id == NewsArticle.id)
            .where(
                NewsArticle.first_seen_at >= window_start,
                NewsSentiment.model_revision == MODEL_REVISION,
                NewsSentiment.asset.in_(("BTC", "ETH")),
            )
            .group_by(NewsSentiment.asset)
        )
        asset_rows = asset_result.all()

        distribution_result = await connection.execute(
            select(
                NewsSentiment.asset,
                NewsSentiment.event_category,
                NewsSentiment.risk_severity,
                func.count(),
            )
            .select_from(NewsArticle)
            .join(NewsSentiment, NewsSentiment.article_id == NewsArticle.id)
            .where(
                NewsArticle.first_seen_at >= window_start,
                NewsSentiment.model_revision == MODEL_REVISION,
                NewsSentiment.asset.in_(("BTC", "ETH")),
            )
            .group_by(
                NewsSentiment.asset,
                NewsSentiment.event_category,
                NewsSentiment.risk_severity,
            )
        )
        distribution_rows = distribution_result.all()

    total_articles = int(total_articles or 0)
    scored_articles = int(scored_articles or 0)
    source_rows: dict[str, list[Any]] = {}
    for row in feed_rows:
        source_rows.setdefault(row[0], []).append(row)
    sources = []
    for source_url, attempts in source_rows.items():
        failures = sum(row[1] == "failed" for row in attempts)
        latest = attempts[-1]
        latest_success = next(
            (row for row in reversed(attempts) if row[1] == "success"),
            None,
        )
        sources.append(
            NewsFeedQuality(
                source=feed_identifier(source_url),
                attempts=len(attempts),
                successes=len(attempts) - failures,
                failures=failures,
                failure_rate_percent=failures / len(attempts) * 100,
                last_status=latest[1],
                last_finished_at=latest[5],
                last_success_items=latest_success[2] if latest_success else None,
                last_error_type=latest[3] if latest[1] == "failed" else None,
            )
        )

    assets: dict[str, AssetNewsQuality] = {}
    for row in asset_rows:
        asset = row[0]
        articles_count = int(row[1])
        assets[asset] = AssetNewsQuality(
            sentiment_articles=articles_count,
            coverage_percent=(
                articles_count / total_articles * 100 if total_articles else 0.0
            ),
            average_publish_to_ingest_seconds=_as_float(row[2]),
            max_publish_to_ingest_seconds=_as_float(row[3]),
            average_scoring_latency_seconds=_as_float(row[4]),
            max_scoring_latency_seconds=_as_float(row[5]),
            mean_confidence=_as_float(row[6]),
            low_confidence_articles=int(row[7] or 0),
            event_categories={},
            risk_severities={},
        )

    for asset in ("BTC", "ETH"):
        if asset not in assets:
            assets[asset] = AssetNewsQuality(
                sentiment_articles=0,
                coverage_percent=0.0,
                average_publish_to_ingest_seconds=None,
                max_publish_to_ingest_seconds=None,
                average_scoring_latency_seconds=None,
                max_scoring_latency_seconds=None,
                mean_confidence=None,
                low_confidence_articles=0,
                event_categories={},
                risk_severities={},
            )

    for row in distribution_rows:
        asset = assets[row[0]]
        asset.event_categories[row[1]] = int(
            asset.event_categories.get(row[1], 0) + row[3]
        )
        asset.risk_severities[row[2]] = int(
            asset.risk_severities.get(row[2], 0) + row[3]
        )

    return NewsQualityReport(
        checked_at=checked_at,
        window_start=window_start,
        window_hours=window_hours,
        model_revision=MODEL_REVISION,
        total_articles=total_articles,
        scored_articles=scored_articles,
        pending_articles=max(total_articles - scored_articles, 0),
        score_coverage_percent=(
            scored_articles / total_articles * 100 if total_articles else 0.0
        ),
        sources=sources,
        assets=assets,
        evaluation_note=(
            "Per-asset coverage uses all collected articles in the window; low "
            "confidence means below 0.55. These operational metrics do not measure "
            "sentiment accuracy, which requires an independently labeled set."
        ),
    )
