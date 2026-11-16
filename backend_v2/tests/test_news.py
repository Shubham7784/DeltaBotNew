from datetime import UTC, datetime

import httpx
import pytest
from sqlalchemy.dialects import postgresql

import app.news.collector as collector
from app.core.config import Settings
from app.news.collector import NewsFeedError, parse_news_feed, store_news_articles


RSS_FEED = b"""\
<rss version="2.0">
  <channel>
    <item>
      <title>  Bitcoin   market update </title>
      <guid>story-1</guid>
      <link>https://news.example/story?utm_source=rss&amp;id=1</link>
      <pubDate>Fri, 13 Nov 2026 16:00:00 +0000</pubDate>
      <description>This publisher excerpt is intentionally not persisted.</description>
    </item>
    <item>
      <title>Bitcoin market update</title>
      <guid>story-2</guid>
      <link>https://news.example/story?id=1</link>
      <pubDate>Fri, 13 Nov 2026 16:00:00 +0000</pubDate>
    </item>
  </channel>
</rss>
"""

ATOM_FEED = b"""\
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>https://news.example/atom-story</id>
    <title>Ethereum update</title>
    <link rel="alternate" href="https://news.example/atom-story" />
    <published>2026-11-13T16:00:00Z</published>
  </entry>
</feed>
"""


def test_rss_parser_normalizes_tracking_urls_and_deduplicates_identity() -> None:
    articles = parse_news_feed(RSS_FEED, "https://feeds.example/rss")

    assert len(articles) == 2
    assert articles[0].dedup_key == articles[1].dedup_key
    assert articles[0].id == articles[1].id
    assert articles[0].article_url == "https://news.example/story?id=1"
    assert articles[0].published_at == datetime(2026, 11, 13, 16, tzinfo=UTC)
    assert "description" not in articles[0].__dict__


def test_atom_parser_reads_namespace_and_published_timestamp() -> None:
    articles = parse_news_feed(ATOM_FEED, "https://feeds.example/atom")

    assert len(articles) == 1
    assert articles[0].title == "Ethereum update"
    assert articles[0].article_url == "https://news.example/atom-story"
    assert articles[0].published_at == datetime(2026, 11, 13, 16, tzinfo=UTC)


def test_news_feed_allowlist_is_configurable_from_environment(monkeypatch) -> None:
    monkeypatch.setenv(
        "NEWS_FEED_URLS",
        '["https://news.example/custom.rss"]',
    )

    settings = Settings(_env_file=None)

    assert [str(url) for url in settings.news_feed_urls] == [
        "https://news.example/custom.rss"
    ]


def test_news_feed_allowlist_rejects_embedded_api_credentials(monkeypatch) -> None:
    monkeypatch.setenv(
        "NEWS_FEED_URLS",
        '["https://news.example/feed.rss?api_key=secret"]',
    )

    with pytest.raises(ValueError, match="API credentials"):
        Settings(_env_file=None)


@pytest.mark.parametrize(
    "content, message",
    [
        (b"<rss><channel>", "well-formed XML"),
        (b"<!DOCTYPE rss><rss/>", "must not contain"),
        (b"<html/>", "not an RSS"),
    ],
)
def test_parser_rejects_malformed_or_unsupported_documents(
    content: bytes,
    message: str,
) -> None:
    with pytest.raises(NewsFeedError, match=message):
        parse_news_feed(content, "https://feeds.example/rss")


class FakeSession:
    def __init__(self, statements: list[object]) -> None:
        self.statements = statements

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args) -> None:
        return None

    def begin(self):
        return self

    async def execute(self, statement) -> None:
        self.statements.append(statement)


class FakeSessionFactory:
    def __init__(self) -> None:
        self.statements: list[object] = []

    def __call__(self) -> FakeSession:
        return FakeSession(self.statements)


@pytest.mark.asyncio
async def test_news_storage_uses_deduplication_and_source_upserts() -> None:
    articles = parse_news_feed(RSS_FEED, "https://feeds.example/rss")
    session_factory = FakeSessionFactory()

    count = await store_news_articles(session_factory, articles)

    assert count == 1
    assert len(session_factory.statements) == 2
    article_sql = str(
        session_factory.statements[0].compile(dialect=postgresql.dialect())
    )
    source_sql = str(
        session_factory.statements[1].compile(dialect=postgresql.dialect())
    )
    assert "ON CONFLICT (dedup_key) DO UPDATE" in article_sql
    assert "ON CONFLICT (id) DO UPDATE" in source_sql


@pytest.mark.asyncio
async def test_poll_continues_after_source_failure_and_records_both_outcomes(
    monkeypatch,
) -> None:
    feeds = [
        "https://good.example/rss?tracking=private",
        "https://broken.example/rss",
    ]
    settings = Settings(_env_file=None, news_feed_urls=feeds)
    session_factory = FakeSessionFactory()
    processed_feeds = []

    async def collect(_client, _factory, feed_url: str) -> int:
        processed_feeds.append(feed_url)
        if "broken" in feed_url:
            raise httpx.TimeoutException("feed timed out")
        return 3

    monkeypatch.setattr(collector, "collect_news_feed", collect)

    with pytest.raises(NewsFeedError, match="1 configured news feed"):
        await collector.poll_news_feeds(
            settings,
            session_factory,
            once=True,
        )

    assert processed_feeds == feeds
    assert len(session_factory.statements) == 2
    success_params = session_factory.statements[0].compile(
        dialect=postgresql.dialect()
    ).params
    failure_params = session_factory.statements[1].compile(
        dialect=postgresql.dialect()
    ).params
    assert success_params["status"] == "success"
    assert success_params["items_processed"] == 3
    assert success_params["feed_url"] == "https://good.example/rss"
    assert failure_params["status"] == "failed"
    assert failure_params["items_processed"] is None
    assert failure_params["error_type"] == "TimeoutException"
