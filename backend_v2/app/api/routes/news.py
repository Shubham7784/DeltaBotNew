from fastapi import APIRouter, Query, Request

from app.news.quality import NewsQualityReport, get_news_quality_report

router = APIRouter(prefix="/api/v1/news", tags=["news"])


@router.get("/quality", response_model=NewsQualityReport)
async def news_quality(
    request: Request,
    window_hours: int = Query(default=24, ge=1, le=720),
) -> NewsQualityReport:
    return await get_news_quality_report(
        request.app.state.database_engine,
        window_hours=window_hours,
    )
