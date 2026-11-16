from sqlalchemy.engine import make_url

from app.core.config import Settings


def test_default_database_and_redis_urls_use_published_host_ports(monkeypatch) -> None:
    monkeypatch.delenv("REDIS_URL", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)

    settings = Settings(_env_file=None)

    assert make_url(settings.database_url.get_secret_value()).port == 5433
    assert str(settings.redis_url) == "redis://localhost:6380/0"
