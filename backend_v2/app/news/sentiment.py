import asyncio
import logging
import math
import re
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal

import httpx
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import Settings
from app.models import NewsArticle, NewsSentiment

MODEL_ID = "9mark9/finbert-minilm-sentiment"
MODEL_REVISION = "f1f32e4b7c3aa451bcc3ab1d36e4b7df8680abe2"
MODEL_LICENSE = "MIT"
MAX_ONNX_BYTES = 150 * 1024 * 1024
MAX_TOKENIZER_BYTES = 10 * 1024 * 1024
MAX_BATCH_SIZE = 64
MAX_TOKEN_LENGTH = 128
logger = logging.getLogger(__name__)

SentimentLabel = Literal["negative", "neutral", "positive"]

ASSET_PATTERNS = {
    "BTC": re.compile(r"\b(?:bitcoin|btc)\b", re.IGNORECASE),
    "ETH": re.compile(r"\b(?:ethereum|ether|eth)\b", re.IGNORECASE),
}
GENERIC_CRYPTO_PATTERN = re.compile(
    r"\b(?:crypto(?:currency)?|digital assets?|blockchain)\b",
    re.IGNORECASE,
)
CATEGORY_PATTERNS = (
    (
        "security",
        re.compile(
            r"\b(?:hack(?:ed)?|exploit(?:ed)?|breach|stolen|theft|scam|fraud)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "regulation",
        re.compile(
            r"\b(?:regulat\w*|sec|cftc|lawsuit|legislation|court|ban|investigation)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "market_structure",
        re.compile(
            r"\b(?:etf|exchange|listing|delisting|futures|options|liquidation)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "protocol",
        re.compile(
            r"\b(?:upgrade|fork|mainnet|testnet|validator|network|outage)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "adoption",
        re.compile(
            r"\b(?:adoption|payment|integration|partnership|institution|reserve)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "macro",
        re.compile(
            r"\b(?:inflation|interest rates?|federal reserve|fomc|cpi|economy)\b",
            re.IGNORECASE,
        ),
    ),
)
HIGH_RISK_PATTERN = re.compile(
    r"\b(?:hack(?:ed)?|exploit(?:ed)?|breach|stolen|bankrupt\w*|insolven\w*|depeg|collapse)\b",
    re.IGNORECASE,
)
MEDIUM_RISK_PATTERN = re.compile(
    r"\b(?:lawsuit|investigation|suspend\w*|outage|delist\w*|liquidat\w*)\b",
    re.IGNORECASE,
)


class SentimentModelError(RuntimeError):
    pass


@dataclass(frozen=True)
class SentimentPrediction:
    negative: float
    neutral: float
    positive: float
    label: SentimentLabel
    confidence: float

    def __post_init__(self) -> None:
        probabilities = (self.negative, self.neutral, self.positive)
        if (
            not all(math.isfinite(value) and 0 <= value <= 1 for value in probabilities)
            or abs(sum(probabilities) - 1) > 1e-6
            or not math.isclose(self.confidence, max(probabilities), abs_tol=1e-6)
        ):
            raise ValueError("Sentiment probabilities and confidence are inconsistent")


@dataclass(frozen=True)
class SentimentAnnotation:
    article_id: uuid.UUID
    asset: str
    model_revision: str
    model_id: str
    label: SentimentLabel
    negative_probability: Decimal
    neutral_probability: Decimal
    positive_probability: Decimal
    confidence: Decimal
    relevance_weight: Decimal
    weighted_sentiment: Decimal
    event_category: str
    risk_severity: str
    scored_at: datetime


def _softmax(logits: list[float]) -> tuple[float, float, float]:
    if len(logits) != 3 or not all(math.isfinite(value) for value in logits):
        raise SentimentModelError("Model must return three finite sentiment logits")
    maximum = max(logits)
    exponentials = [math.exp(value - maximum) for value in logits]
    denominator = sum(exponentials)
    return (
        exponentials[0] / denominator,
        exponentials[1] / denominator,
        exponentials[2] / denominator,
    )


def _prediction(logits: list[float]) -> SentimentPrediction:
    negative, neutral, positive = _softmax(logits)
    probabilities = {
        "negative": negative,
        "neutral": neutral,
        "positive": positive,
    }
    if negative >= neutral and negative >= positive:
        label: SentimentLabel = "negative"
    elif neutral >= positive:
        label = "neutral"
    else:
        label = "positive"
    return SentimentPrediction(
        negative=negative,
        neutral=neutral,
        positive=positive,
        label=label,
        confidence=probabilities[label],
    )


def _targets(title: str) -> list[tuple[str, float]]:
    targets = [
        (asset, 1.0)
        for asset, pattern in ASSET_PATTERNS.items()
        if pattern.search(title)
    ]
    if targets:
        return targets
    if GENERIC_CRYPTO_PATTERN.search(title):
        return [("BTC", 0.5), ("ETH", 0.5)]
    return [("GLOBAL", 0.0)]


def _event_category(title: str) -> str:
    for category, pattern in CATEGORY_PATTERNS:
        if pattern.search(title):
            return category
    return "other"


def _risk_severity(title: str) -> str:
    if HIGH_RISK_PATTERN.search(title):
        return "high"
    if MEDIUM_RISK_PATTERN.search(title):
        return "medium"
    return "low"


def annotate_article(
    article_id: uuid.UUID,
    title: str,
    prediction: SentimentPrediction,
    *,
    scored_at: datetime | None = None,
) -> list[SentimentAnnotation]:
    timestamp = scored_at or datetime.now(UTC)
    if timestamp.tzinfo is None:
        raise ValueError("Sentiment scoring time must be timezone-aware")
    category = _event_category(title)
    severity = _risk_severity(title)
    sentiment_delta = prediction.positive - prediction.negative
    annotations = []
    for asset, relevance_weight in _targets(title):
        weighted_sentiment = sentiment_delta * prediction.confidence * relevance_weight
        annotations.append(
            SentimentAnnotation(
                article_id=article_id,
                asset=asset,
                model_revision=MODEL_REVISION,
                model_id=MODEL_ID,
                label=prediction.label,
                negative_probability=Decimal(str(prediction.negative)),
                neutral_probability=Decimal(str(prediction.neutral)),
                positive_probability=Decimal(str(prediction.positive)),
                confidence=Decimal(str(prediction.confidence)),
                relevance_weight=Decimal(str(relevance_weight)),
                weighted_sentiment=Decimal(str(weighted_sentiment)),
                event_category=category,
                risk_severity=severity,
                scored_at=timestamp.astimezone(UTC),
            )
        )
    return annotations


class OnnxSentimentModel:
    def __init__(self, tokenizer: Any, session: Any) -> None:
        self._tokenizer = tokenizer
        self._session = session

    @classmethod
    async def load(cls, cache_dir: Path) -> "OnnxSentimentModel":
        revision_dir = cache_dir / MODEL_REVISION
        revision_dir.mkdir(parents=True, exist_ok=True)
        model_path = revision_dir / "model_quantized.onnx"
        tokenizer_path = revision_dir / "tokenizer.json"
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(120.0),
            follow_redirects=True,
        ) as client:
            await _download_artifact(
                client,
                model_path,
                "onnx/model_quantized.onnx",
                MAX_ONNX_BYTES,
            )
            await _download_artifact(
                client,
                tokenizer_path,
                "tokenizer.json",
                MAX_TOKENIZER_BYTES,
            )
        return await asyncio.to_thread(cls._load_local, tokenizer_path, model_path)

    @classmethod
    def _load_local(cls, tokenizer_path: Path, model_path: Path) -> "OnnxSentimentModel":
        import onnxruntime
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        pad_id = tokenizer.token_to_id("[PAD]")
        if pad_id is None:
            raise SentimentModelError("Model tokenizer has no [PAD] token")
        tokenizer.enable_truncation(max_length=MAX_TOKEN_LENGTH)
        tokenizer.enable_padding(pad_id=pad_id, pad_token="[PAD]")
        session = onnxruntime.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
        )
        return cls(tokenizer, session)

    def predict(self, texts: list[str]) -> list[SentimentPrediction]:
        import numpy

        if not texts:
            return []
        if any(not text.strip() for text in texts):
            raise ValueError("Sentiment input text must not be empty")

        encodings = self._tokenizer.encode_batch(texts)
        arrays = {
            "input_ids": numpy.asarray(
                [encoding.ids for encoding in encodings],
                dtype=numpy.int64,
            ),
            "attention_mask": numpy.asarray(
                [encoding.attention_mask for encoding in encodings],
                dtype=numpy.int64,
            ),
            "token_type_ids": numpy.asarray(
                [encoding.type_ids for encoding in encodings],
                dtype=numpy.int64,
            ),
        }
        inputs = self._session.get_inputs()
        input_names = {item.name for item in inputs}
        unknown_inputs = input_names - arrays.keys()
        if unknown_inputs:
            raise SentimentModelError(
                f"Unsupported model input tensors: {sorted(unknown_inputs)}"
            )
        logits_batch = self._session.run(
            None,
            {name: arrays[name] for name in input_names},
        )[0]
        if len(logits_batch) != len(texts):
            raise SentimentModelError("Model returned the wrong number of predictions")
        return [
            _prediction([float(value) for value in logits])
            for logits in logits_batch
        ]


async def _download_artifact(
    client: httpx.AsyncClient,
    destination: Path,
    relative_path: str,
    max_bytes: int,
) -> None:
    if destination.exists() and 0 < destination.stat().st_size <= max_bytes:
        return

    temporary_path = destination.with_suffix(destination.suffix + ".part")
    url = (
        f"https://huggingface.co/{MODEL_ID}/resolve/{MODEL_REVISION}/{relative_path}"
    )
    byte_count = 0
    try:
        async with client.stream("GET", url) as response:
            response.raise_for_status()
            with temporary_path.open("wb") as file:
                async for chunk in response.aiter_bytes():
                    byte_count += len(chunk)
                    if byte_count > max_bytes:
                        raise SentimentModelError(
                            f"Model artifact exceeds the {max_bytes}-byte limit"
                        )
                    file.write(chunk)
        if byte_count == 0:
            raise SentimentModelError(f"Model artifact is empty: {relative_path}")
        temporary_path.replace(destination)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


async def classify_pending_articles(
    session_factory: async_sessionmaker[AsyncSession],
    model: OnnxSentimentModel,
    *,
    batch_size: int = MAX_BATCH_SIZE,
) -> int:
    if batch_size <= 0 or batch_size > MAX_BATCH_SIZE:
        raise ValueError(f"batch_size must be between 1 and {MAX_BATCH_SIZE}")
    already_scored = (
        select(NewsSentiment.article_id)
        .where(
            NewsSentiment.article_id == NewsArticle.id,
            NewsSentiment.model_revision == MODEL_REVISION,
        )
        .exists()
    )
    async with session_factory() as session:
        result = await session.execute(
            select(NewsArticle)
            .where(~already_scored)
            .order_by(NewsArticle.first_seen_at, NewsArticle.id)
            .limit(batch_size)
        )
        articles = list(result.scalars())
    if not articles:
        return 0

    predictions = await asyncio.to_thread(
        model.predict,
        [article.title for article in articles],
    )
    if len(predictions) != len(articles):
        raise SentimentModelError("Sentiment model returned an incomplete batch")

    scored_at = datetime.now(UTC)
    annotations = [
        annotation
        for article, prediction in zip(articles, predictions, strict=True)
        for annotation in annotate_article(
            article.id,
            article.title,
            prediction,
            scored_at=scored_at,
        )
    ]
    rows = [
        {
            "article_id": annotation.article_id,
            "asset": annotation.asset,
            "model_revision": annotation.model_revision,
            "model_id": annotation.model_id,
            "label": annotation.label,
            "negative_probability": annotation.negative_probability,
            "neutral_probability": annotation.neutral_probability,
            "positive_probability": annotation.positive_probability,
            "confidence": annotation.confidence,
            "relevance_weight": annotation.relevance_weight,
            "weighted_sentiment": annotation.weighted_sentiment,
            "event_category": annotation.event_category,
            "risk_severity": annotation.risk_severity,
            "scored_at": annotation.scored_at,
        }
        for annotation in annotations
    ]
    if rows:
        statement = insert(NewsSentiment).values(rows)
        statement = statement.on_conflict_do_nothing(
            index_elements=["article_id", "asset", "model_revision"]
        )
        async with session_factory() as session:
            async with session.begin():
                await session.execute(statement)
    return len(articles)


async def classify_news_forever(
    settings: Settings,
    session_factory: async_sessionmaker[AsyncSession],
    *,
    once: bool = False,
) -> None:
    model = await OnnxSentimentModel.load(settings.news_sentiment_model_cache)
    while True:
        count = await classify_pending_articles(session_factory, model)
        if count:
            logger.info("Classified %s news articles", count)
        if once:
            return
        await asyncio.sleep(30)
