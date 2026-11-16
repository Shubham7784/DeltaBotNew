import asyncio
import hashlib
import logging
import re
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Literal
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from xml.etree import ElementTree

import httpx
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import Settings
from app.models import NewsArticle, NewsArticleSource, NewsFeedRun

logger = logging.getLogger(__name__)
MAX_FEED_BYTES = 5 * 1024 * 1024
MAX_FEED_ITEMS = 1_000
TRACKING_QUERY_KEYS = {"fbclid", "gclid", "mc_cid", "mc_eid"}


class NewsFeedError(RuntimeError):
    pass


@dataclass(frozen=True)
class NewsArticleData:
    id: uuid.UUID
    dedup_key: str
    title: str
    article_url: str | None
    published_at: datetime | None
    feed_url: str
    external_id: str


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].lower()


def _child_text(element: ElementTree.Element, names: set[str]) -> str | None:
    for child in element:
        if _local_name(child.tag) in names:
            value = " ".join(part.strip() for part in child.itertext() if part.strip())
            if value:
                return value
    return None


def _article_link(element: ElementTree.Element) -> str | None:
    for child in element:
        if _local_name(child.tag) != "link":
            continue
        href = child.attrib.get("href")
        if href:
            if child.attrib.get("rel", "alternate").lower() == "alternate":
                return href.strip()
            continue
        value = "".join(child.itertext()).strip()
        if value:
            return value
    return None


def _canonical_url(raw_url: str | None) -> str | None:
    if not raw_url:
        return None
    try:
        parsed = urlsplit(raw_url.strip())
    except ValueError:
        return None
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
        return None
    if parsed.username is not None or parsed.password is not None:
        return None

    query = [
        (key, value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if not key.lower().startswith("utm_") and key.lower() not in TRACKING_QUERY_KEYS
    ]
    return urlunsplit(
        (
            parsed.scheme.lower(),
            parsed.netloc.lower(),
            parsed.path or "/",
            urlencode(sorted(query)),
            "",
        )
    )


def _feed_label(feed_url: str) -> str:
    return urlsplit(feed_url).hostname or "configured feed"


def feed_identifier(feed_url: str) -> str:
    parsed = urlsplit(feed_url)
    return urlunsplit(
        (parsed.scheme.lower(), parsed.netloc.lower(), parsed.path or "/", "", "")
    )


def _published_at(element: ElementTree.Element) -> datetime | None:
    value = _child_text(
        element,
        {"pubdate", "published", "updated", "date", "modified"},
    )
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        try:
            parsed = parsedate_to_datetime(value)
        except (TypeError, ValueError, OverflowError):
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def parse_news_feed(content: bytes, feed_url: str) -> list[NewsArticleData]:
    if len(content) > MAX_FEED_BYTES:
        raise NewsFeedError(f"Feed exceeds the {MAX_FEED_BYTES}-byte limit")
    if re.search(br"<!DOCTYPE|<!ENTITY", content, re.IGNORECASE):
        raise NewsFeedError("Feed must not contain a document type or entity declaration")
    try:
        root = ElementTree.fromstring(content)
    except ElementTree.ParseError as exc:
        raise NewsFeedError("Feed is not well-formed XML") from exc

    entries = [
        element
        for element in root.iter()
        if _local_name(element.tag) in {"item", "entry"}
    ]
    if not entries and _local_name(root.tag) not in {"rss", "feed", "rdf"}:
        raise NewsFeedError("Document is not an RSS, Atom, or RDF feed")
    if len(entries) > MAX_FEED_ITEMS:
        raise NewsFeedError(f"Feed exceeds the {MAX_FEED_ITEMS}-item limit")

    articles: list[NewsArticleData] = []
    for entry in entries:
        title = _child_text(entry, {"title"})
        if not title:
            continue
        title = re.sub(r"\s+", " ", title).strip()
        article_url = _canonical_url(_article_link(entry))
        published_at = _published_at(entry)
        raw_id = _child_text(entry, {"guid", "id"})
        normalized_title = title.casefold()
        external_id = (
            raw_id
            or article_url
            or hashlib.sha256(
                f"{feed_url}\n{title}\n{published_at or ''}".encode("utf-8")
            ).hexdigest()
        )
        identity = article_url or (
            f"{normalized_title}\n"
            f"{published_at.isoformat() if published_at else external_id}"
        )
        dedup_key = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        articles.append(
            NewsArticleData(
                id=uuid.uuid5(uuid.NAMESPACE_URL, f"news:{dedup_key}"),
                dedup_key=dedup_key,
                title=title,
                article_url=article_url,
                published_at=published_at,
                feed_url=feed_url,
                external_id=external_id,
            )
        )
    return articles


async def store_news_articles(
    session_factory: async_sessionmaker[AsyncSession],
    articles: list[NewsArticleData],
) -> int:
    if not articles:
        return 0

    unique_articles = {article.dedup_key: article for article in articles}
    unique_sources = {
        (article.feed_url, article.external_id): article for article in articles
    }
    article_rows = [
        {
            "id": article.id,
            "dedup_key": article.dedup_key,
            "title": article.title,
            "article_url": article.article_url,
            "published_at": article.published_at,
        }
        for article in unique_articles.values()
    ]
    source_rows = [
        {
            "id": uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"news-source:{article.feed_url}:{article.external_id}",
            ),
            "article_id": article.id,
            "feed_url": article.feed_url,
            "external_id": article.external_id,
        }
        for article in unique_sources.values()
    ]

    async with session_factory() as session:
        async with session.begin():
            article_statement = insert(NewsArticle).values(article_rows)
            article_statement = article_statement.on_conflict_do_update(
                index_elements=["dedup_key"],
                set_={
                    "title": article_statement.excluded.title,
                    "article_url": article_statement.excluded.article_url,
                    "published_at": article_statement.excluded.published_at,
                },
            )
            source_statement = insert(NewsArticleSource).values(source_rows)
            source_statement = source_statement.on_conflict_do_update(
                index_elements=["id"],
                set_={"article_id": source_statement.excluded.article_id},
            )
            await session.execute(article_statement)
            await session.execute(source_statement)
    return len(unique_articles)


async def collect_news_feed(
    client: httpx.AsyncClient,
    session_factory: async_sessionmaker[AsyncSession],
    feed_url: str,
) -> int:
    chunks = []
    byte_count = 0
    async with client.stream("GET", feed_url) as response:
        response.raise_for_status()
        async for chunk in response.aiter_bytes():
            byte_count += len(chunk)
            if byte_count > MAX_FEED_BYTES:
                raise NewsFeedError(f"Feed exceeds the {MAX_FEED_BYTES}-byte limit")
            chunks.append(chunk)
    articles = parse_news_feed(b"".join(chunks), feed_url)
    return await store_news_articles(session_factory, articles)


async def store_news_feed_run(
    session_factory: async_sessionmaker[AsyncSession],
    *,
    feed_url: str,
    started_at: datetime,
    finished_at: datetime,
    status: Literal["success", "failed"],
    items_processed: int | None,
    error_type: str | None,
) -> None:
    statement = insert(NewsFeedRun).values(
        id=uuid.uuid4(),
        feed_url=feed_identifier(feed_url),
        started_at=started_at,
        finished_at=finished_at,
        status=status,
        items_processed=items_processed,
        error_type=error_type,
    )
    async with session_factory() as session:
        async with session.begin():
            await session.execute(statement)


async def poll_news_feeds(
    settings: Settings,
    session_factory: async_sessionmaker[AsyncSession],
    *,
    once: bool = False,
) -> None:
    feed_urls = [str(url) for url in settings.news_feed_urls]
    if not feed_urls:
        raise NewsFeedError("Configure at least one NEWS_FEED_URLS entry")

    async with httpx.AsyncClient(
        timeout=settings.news_request_timeout_seconds,
        headers={
            "Accept": (
                "application/rss+xml, application/atom+xml, application/xml, text/xml"
            )
        },
    ) as client:
        while True:
            failures = []
            for feed_url in feed_urls:
                started_at = datetime.now(UTC)
                items_processed = None
                error_type = None
                try:
                    items_processed = await collect_news_feed(
                        client,
                        session_factory,
                        feed_url,
                    )
                    logger.info(
                        "Processed %s news items from %s",
                        items_processed,
                        _feed_label(feed_url),
                    )
                    status: Literal["success", "failed"] = "success"
                except (httpx.HTTPError, NewsFeedError) as error:
                    logger.exception(
                        "Failed to collect news feed from %s", _feed_label(feed_url)
                    )
                    failures.append(feed_url)
                    error_type = type(error).__name__
                    status = "failed"
                await store_news_feed_run(
                    session_factory,
                    feed_url=feed_url,
                    started_at=started_at,
                    finished_at=datetime.now(UTC),
                    status=status,
                    items_processed=items_processed,
                    error_type=error_type,
                )
            if once:
                if failures:
                    raise NewsFeedError(
                        f"Failed to collect {len(failures)} configured news feed(s)"
                    )
                return
            await asyncio.sleep(settings.news_poll_interval_seconds)
