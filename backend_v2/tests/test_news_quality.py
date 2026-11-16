from datetime import UTC, datetime, timedelta

import pytest

from app.news.quality import get_news_quality_report


class FakeResult:
    def __init__(self, rows) -> None:
        self.rows = rows

    def one(self):
        return self.rows

    def all(self):
        return self.rows


class FakeConnection:
    def __init__(self, results) -> None:
        self.results = list(results)

    async def __aenter__(self) -> "FakeConnection":
        return self

    async def __aexit__(self, *_args) -> None:
        return None

    async def execute(self, _statement) -> FakeResult:
        return FakeResult(self.results.pop(0))


class FakeEngine:
    def __init__(self, results) -> None:
        self.connection = FakeConnection(results)

    def connect(self) -> FakeConnection:
        return self.connection


@pytest.mark.asyncio
async def test_news_quality_report_aggregates_coverage_latency_and_source_failures() -> None:
    now = datetime.now(UTC)
    source_url = "https://feeds.example/rss?token=redacted"
    engine = FakeEngine(
        [
            (10, 8),
            [
                (
                    source_url,
                    "failed",
                    None,
                    "TimeoutException",
                    now - timedelta(minutes=10),
                    now - timedelta(minutes=9),
                ),
                (
                    source_url,
                    "success",
                    5,
                    None,
                    now - timedelta(minutes=5),
                    now - timedelta(minutes=4),
                ),
            ],
            [
                ("BTC", 4, 10, 20, 1.5, 4, 0.75, 1),
                ("ETH", 2, None, None, 2, 3, 0.60, 0),
            ],
            [
                ("BTC", "security", "high", 2),
                ("BTC", "adoption", "low", 1),
                ("ETH", "macro", "medium", 1),
            ],
        ]
    )

    report = await get_news_quality_report(engine, window_hours=24)

    assert report.total_articles == 10
    assert report.scored_articles == 8
    assert report.pending_articles == 2
    assert report.score_coverage_percent == 80
    assert len(report.sources) == 1
    source = report.sources[0]
    assert source.source == "https://feeds.example/rss"
    assert source.attempts == 2
    assert source.failures == 1
    assert source.failure_rate_percent == 50
    assert source.last_status == "success"
    assert source.last_success_items == 5

    btc = report.assets["BTC"]
    assert btc.sentiment_articles == 4
    assert btc.coverage_percent == 40
    assert btc.average_publish_to_ingest_seconds == 10
    assert btc.max_scoring_latency_seconds == 4
    assert btc.low_confidence_articles == 1
    assert btc.event_categories == {"security": 2, "adoption": 1}
    assert btc.risk_severities == {"high": 2, "low": 1}
    assert report.assets["ETH"].average_publish_to_ingest_seconds is None
    assert "independently labeled" in report.evaluation_note


@pytest.mark.asyncio
async def test_news_quality_report_handles_empty_window_and_rejects_bad_range() -> None:
    report = await get_news_quality_report(
        FakeEngine([(0, 0), [], [], []]),
        window_hours=1,
    )

    assert report.score_coverage_percent == 0
    assert report.assets["BTC"].coverage_percent == 0
    assert report.assets["ETH"].sentiment_articles == 0

    with pytest.raises(ValueError, match="between 1 and 720"):
        await get_news_quality_report(FakeEngine([]), window_hours=0)
