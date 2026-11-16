from functools import lru_cache
from pathlib import Path
from urllib.parse import parse_qsl, urlsplit

from pydantic import Field, HttpUrl, RedisDsn, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_env: str = "development"
    log_level: str = "INFO"
    database_url: SecretStr = SecretStr(
        "postgresql+asyncpg://deltabot:deltabot@localhost:5433/deltabot"
    )
    redis_url: RedisDsn = RedisDsn("redis://localhost:6380/0")
    delta_rest_base_url: HttpUrl = HttpUrl("https://api.india.delta.exchange")
    delta_public_ws_url: str = "wss://public-socket.india.delta.exchange"
    delta_request_timeout_seconds: float = Field(default=15, gt=0, le=120)
    delta_max_retries: int = Field(default=3, ge=0, le=10)
    market_data_stale_after_seconds: int = Field(default=30, gt=0)
    news_feed_urls: list[HttpUrl] = Field(
        default_factory=lambda: [
            HttpUrl("https://www.coindesk.com/arc/outboundfeeds/rss"),
            HttpUrl("https://cointelegraph.com/rss"),
        ]
    )
    news_poll_interval_seconds: int = Field(default=300, ge=30, le=3_600)
    news_request_timeout_seconds: float = Field(default=15, gt=0, le=120)
    news_sentiment_model_cache: Path = Path(".model-cache")

    @field_validator("delta_public_ws_url")
    @classmethod
    def validate_websocket_url(cls, value: str) -> str:
        if not value.startswith(("ws://", "wss://")):
            raise ValueError("DELTA_PUBLIC_WS_URL must use ws:// or wss://")
        return value.rstrip("/")

    @field_validator("news_feed_urls")
    @classmethod
    def reject_news_feed_credentials(cls, feeds: list[HttpUrl]) -> list[HttpUrl]:
        credential_names = {
            "api_key",
            "apikey",
            "auth",
            "authorization",
            "access_token",
            "secret",
            "token",
        }
        for feed in feeds:
            parts = urlsplit(str(feed))
            if parts.username is not None or parts.password is not None:
                raise ValueError("NEWS_FEED_URLS must not contain URL credentials")
            if any(
                key.casefold() in credential_names
                for key, _value in parse_qsl(parts.query, keep_blank_values=True)
            ):
                raise ValueError("NEWS_FEED_URLS must not contain API credentials")
        return feeds


@lru_cache
def get_settings() -> Settings:
    return Settings()
