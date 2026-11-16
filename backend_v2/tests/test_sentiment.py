from datetime import UTC, datetime
from decimal import Decimal
from types import SimpleNamespace
import uuid

import pytest
from sqlalchemy.dialects import postgresql

from app.news.sentiment import (
    SentimentPrediction,
    _prediction,
    annotate_article,
    classify_pending_articles,
)


def test_model_logits_become_three_class_probabilities() -> None:
    prediction = _prediction([0.0, 0.0, 2.0])

    assert prediction.label == "positive"
    assert prediction.confidence == prediction.positive
    assert sum(
        (prediction.negative, prediction.neutral, prediction.positive)
    ) == pytest.approx(1.0)


def test_asset_mentions_receive_confident_weighted_sentiment() -> None:
    scored_at = datetime(2026, 11, 13, tzinfo=UTC)
    prediction = SentimentPrediction(
        negative=0.1,
        neutral=0.1,
        positive=0.8,
        label="positive",
        confidence=0.8,
    )

    annotations = annotate_article(
        uuid.uuid4(),
        "Bitcoin and Ethereum adoption rises after a new partnership",
        prediction,
        scored_at=scored_at,
    )

    assert [item.asset for item in annotations] == ["BTC", "ETH"]
    assert all(item.relevance_weight == Decimal("1.0") for item in annotations)
    assert all(item.weighted_sentiment == Decimal("0.56") for item in annotations)
    assert all(item.event_category == "adoption" for item in annotations)
    assert all(item.risk_severity == "low" for item in annotations)
    assert all(item.scored_at == scored_at for item in annotations)


def test_generic_crypto_news_is_weighted_for_both_assets() -> None:
    prediction = SentimentPrediction(
        negative=0.2,
        neutral=0.6,
        positive=0.2,
        label="neutral",
        confidence=0.6,
    )

    annotations = annotate_article(
        uuid.uuid4(),
        "Crypto exchange suffers an exploit; blockchain withdrawals paused",
        prediction,
    )

    assert [(item.asset, item.relevance_weight) for item in annotations] == [
        ("BTC", Decimal("0.5")),
        ("ETH", Decimal("0.5")),
    ]
    assert all(item.event_category == "security" for item in annotations)
    assert all(item.risk_severity == "high" for item in annotations)


def test_unrelated_headline_is_recorded_as_global_not_asset_relevant() -> None:
    prediction = SentimentPrediction(
        negative=0.2,
        neutral=0.6,
        positive=0.2,
        label="neutral",
        confidence=0.6,
    )

    annotation = annotate_article(
        uuid.uuid4(),
        "Regional banking report highlights stable quarterly earnings",
        prediction,
    )[0]

    assert annotation.asset == "GLOBAL"
    assert annotation.relevance_weight == Decimal("0.0")
    assert annotation.weighted_sentiment == Decimal("0.0")


def test_prediction_rejects_inconsistent_probabilities() -> None:
    with pytest.raises(ValueError, match="inconsistent"):
        SentimentPrediction(
            negative=0.5,
            neutral=0.5,
            positive=0.5,
            label="positive",
            confidence=0.5,
        )


class FakeResult:
    def __init__(self, articles: list[object]) -> None:
        self.articles = articles

    def scalars(self) -> list[object]:
        return self.articles


class FakeSession:
    def __init__(self, factory: "FakeSessionFactory") -> None:
        self.factory = factory

    async def __aenter__(self) -> "FakeSession":
        return self

    async def __aexit__(self, *_args: object) -> None:
        return None

    def begin(self) -> "FakeSession":
        return self

    async def execute(self, statement: object) -> FakeResult | None:
        self.factory.statements.append(statement)
        if getattr(statement, "is_select", False):
            return FakeResult(self.factory.articles)
        return None


class FakeSessionFactory:
    def __init__(self, articles: list[object]) -> None:
        self.articles = articles
        self.statements: list[object] = []

    def __call__(self) -> FakeSession:
        return FakeSession(self)


class FakeModel:
    def __init__(self, predictions: list[SentimentPrediction]) -> None:
        self.predictions = predictions
        self.inputs: list[str] = []

    def predict(self, texts: list[str]) -> list[SentimentPrediction]:
        self.inputs = texts
        return self.predictions


@pytest.mark.asyncio
async def test_pending_news_batch_is_saved_idempotently_by_article_asset_revision() -> None:
    article = SimpleNamespace(
        id=uuid.uuid4(),
        title="Bitcoin and Ethereum adoption increases",
        first_seen_at=datetime(2026, 11, 13, tzinfo=UTC),
    )
    prediction = SentimentPrediction(
        negative=0.1,
        neutral=0.1,
        positive=0.8,
        label="positive",
        confidence=0.8,
    )
    session_factory = FakeSessionFactory([article])
    model = FakeModel([prediction])

    count = await classify_pending_articles(session_factory, model)

    assert count == 1
    assert model.inputs == [article.title]
    assert len(session_factory.statements) == 2
    select_sql = str(
        session_factory.statements[0].compile(dialect=postgresql.dialect())
    )
    insert_sql = str(
        session_factory.statements[1].compile(dialect=postgresql.dialect())
    )
    assert "NOT (EXISTS" in select_sql
    assert "model_revision" in select_sql
    assert "ON CONFLICT (article_id, asset, model_revision) DO NOTHING" in insert_sql
