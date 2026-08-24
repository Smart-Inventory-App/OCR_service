"""Probes de salud y estado de los motores. Públicos: no requieren token."""

from fastapi import APIRouter, HTTPException, status

from app.core.config import settings
from app.engines.factory import CONCRETE_ENGINES, build_engine
from app.engines.remote import RemoteEngine
from app.schemas.ocr import EngineStatus
from app.services import ocr_service, queue

router = APIRouter(prefix="/ocr", tags=["health"])


@router.get("/livez")
async def livez():
    """Liveness: el proceso responde. Sin dependencias externas."""
    return {"status": "ok"}


@router.get("/readyz")
async def readyz():
    """Readiness: Redis accesible y el motor configurado puede reconocer.

    Se comprueba el motor **configurado**, no todos: que EasyOCR esté apagado no
    debería marcar el servicio como caído si el que se usa es Tesseract.
    """
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


@router.get("/engines", response_model=list[EngineStatus])
async def engines():
    """Qué motores hay disponibles y qué versión corre cada uno.

    Lo usa el banco de pruebas para no ofrecer un motor que está apagado, y sirve
    de resumen de la comparativa que pide la propuesta.
    """
    return [_status(name) for name in CONCRETE_ENGINES]


def _status(name: str) -> EngineStatus:
    engine = build_engine(name)
    url = engine.base_url if isinstance(engine, RemoteEngine) else ""

    if not engine.is_available():
        return EngineStatus(name=name, url=url, available=False)

    info = engine.info() if isinstance(engine, RemoteEngine) else {}

    return EngineStatus(
        name=name,
        url=url,
        available=True,
        version=info.get("version", "desconocida"),
        languages=info.get("languages", []),
        supports_preprocessing=bool(info.get("supports_preprocessing", False)),
    )
