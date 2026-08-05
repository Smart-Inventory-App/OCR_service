"""Probes de salud. Públicos: no requieren token."""

from fastapi import APIRouter, HTTPException, status

from app.core.config import settings
from app.services import ocr_service, queue

router = APIRouter(prefix="/ocr", tags=["health"])


@router.get("/livez")
async def livez():
    """Liveness: el proceso responde. Sin dependencias externas."""
    return {"status": "ok"}


@router.get("/readyz")
async def readyz():
    """Readiness: Redis accesible y motor OCR ejecutable."""
    checks = {
        "redis": await queue.ping(),
        "engine": ocr_service.engine_available(),
    }

    if not all(checks.values()):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "status": "error",
                "checks": checks,
                "engine_name": settings.OCR_ENGINE,
            },
        )

    return {"status": "ok", "checks": checks, "engine_name": settings.OCR_ENGINE}
