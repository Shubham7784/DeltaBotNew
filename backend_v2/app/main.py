from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.analytics import router as analytics_router
from app.api.routes.market import router as market_router
from app.api.routes.news import router as news_router
from app.api.routes.supervisor import router as supervisor_router
from app.api.routes.system import router as system_router
from app.api.routes.trading import router as trading_router
from app.core.config import get_settings
from app.database import close_database_resources, engine
from app.market_data.delta_client import DeltaRestClient


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    application.state.settings = settings
    application.state.database_engine = engine
    application.state.delta_client = DeltaRestClient(settings)
    try:
        yield
    finally:
        await application.state.delta_client.close()
        await close_database_resources()


app = FastAPI(
    title="DeltaBot 2.0 Market Analysis & Paper-Trading Service",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost",
        "http://127.0.0.1",
    ],
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:[0-9]+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(system_router)
app.include_router(market_router)
app.include_router(news_router)
app.include_router(trading_router)
app.include_router(supervisor_router)
app.include_router(analytics_router)
