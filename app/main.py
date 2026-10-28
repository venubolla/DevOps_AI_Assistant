from fastapi import FastAPI

from app.api.routes import router
from app.config import settings

app = FastAPI(
    title=settings.app_name,
    description="AI-powered DevOps Operations Assistant",
    version=settings.app_version,
)


@app.get("/")
async def root():
    return {
        "application": settings.app_name,
        "status": "running",
        "version": settings.app_version,
        "environment": settings.environment,
    }


app.include_router(router)
