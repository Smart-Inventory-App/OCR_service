import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.engines import factory
from app.routes.health import router as health_router
from app.routes.ocr import router as ocr_router
from app.routes.playground import router as playground_router
from app.services import queue

logging.basicConfig(
    level=logging.DEBUG if settings.DEBUG else logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    await queue.close()
    # Cierra las conexiones abiertas hacia los servicios de motor.
    factory.reset_cache()

app = FastAPI(
    title="OCR Service — Smart Inventory",
    version="1.0.0",
    debug=settings.DEBUG,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(ocr_router)
# Banco de pruebas manual; la propia ruta devuelve 404 si DEBUG está apagado.
app.include_router(playground_router)
