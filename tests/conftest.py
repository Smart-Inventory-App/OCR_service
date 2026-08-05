"""Fixtures compartidas.

Los tests no dependen del binario de Tesseract ni de un Redis real: el motor se
sustituye por un doble que devuelve palabras con confianza controlada, y Redis
por `fakeredis`. El pipeline OpenCV, los parsers y el scoring se ejercitan de
verdad.
"""

import io
import os

import numpy as np
import pytest
from PIL import Image, ImageDraw

os.environ.setdefault("JWT_SECRET", "test-secret")
os.environ.setdefault("OCR_ENGINE", "tesseract")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")

from app.core.security import get_current_user  # noqa: E402
from app.engines.base import OCRWord, RawOCROutput  # noqa: E402
from app.main import app  # noqa: E402


# ---------------------------------------------------------------------------
# Doble del motor OCR
# ---------------------------------------------------------------------------

class FakeEngine:
    """Motor que devuelve un guion fijo de líneas, con confianza configurable."""

    name = "fake"

    def __init__(self, lines: list[str], confidence: float = 0.95, heights: list[int] | None = None):
        self.lines = lines
        self.confidence = confidence
        self.heights = heights or [30] * len(lines)
        self.calls: list[dict] = []

    def is_available(self) -> bool:
        return True

    def run(self, image, lang: str = "spa", layout: str = "block") -> RawOCROutput:
        self.calls.append({"lang": lang, "layout": layout, "shape": getattr(image, "shape", None)})

        words: list[OCRWord] = []
        for line_num, (line, height) in enumerate(zip(self.lines, self.heights)):
            left = 10
            for token in line.split():
                words.append(
                    OCRWord(
                        text=token,
                        confidence=self.confidence,
                        left=left,
                        top=10 + line_num * (height + 10),
                        width=len(token) * height // 2,
                        height=height,
                        line_num=line_num,
                    )
                )
                left += len(token) * height // 2 + height // 3

        return RawOCROutput.from_words(words, self.name)


def make_output(
    lines: list[str],
    confidence: float = 0.95,
    heights: list[int] | None = None,
) -> RawOCROutput:
    """Salida OCR sintética, sin pasar por el pipeline de imágenes."""
    engine = FakeEngine(lines, confidence, heights)
    return engine.run(np.zeros((10, 10), dtype=np.uint8))


@pytest.fixture
def fake_engine(monkeypatch):
    """Reemplaza el motor real por un `FakeEngine` configurable desde el test."""
    holder: dict[str, FakeEngine] = {}

    def install(lines: list[str], confidence: float = 0.95, heights: list[int] | None = None):
        engine = FakeEngine(lines, confidence, heights)
        holder["engine"] = engine
        monkeypatch.setattr("app.services.ocr_service.build_engine", lambda *_, **__: engine)
        return engine

    install.holder = holder  # type: ignore[attr-defined]
    return install


# ---------------------------------------------------------------------------
# Imágenes de prueba
# ---------------------------------------------------------------------------

def render_image(lines: list[tuple[str, int]], size: tuple[int, int] = (900, 500)) -> bytes:
    """Renderiza líneas de texto (con su tamaño) como JPEG.

    Sirve para ejercitar el pipeline OpenCV completo con una imagen real; el
    contenido reconocido lo determina el motor falso, no esta imagen.
    """
    image = Image.new("RGB", size, color=(255, 255, 255))
    draw = ImageDraw.Draw(image)

    y = 30
    for text, font_size in lines:
        draw.text((30, y), text, fill=(0, 0, 0))
        y += font_size + 20

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=95)
    return buffer.getvalue()


@pytest.fixture
def label_image() -> bytes:
    return render_image([("LECHE ENTERA 1L", 40), ("VTO 15/08/2027", 20)])


@pytest.fixture
def gray_image() -> np.ndarray:
    """Imagen en escala de grises con texto oscuro sobre fondo claro."""
    image = np.full((200, 600), 240, dtype=np.uint8)
    image[80:120, 50:400] = 30
    return image


# ---------------------------------------------------------------------------
# App / Redis
# ---------------------------------------------------------------------------

@pytest.fixture
async def client():
    """Cliente HTTP abierto, con el usuario ya autenticado."""
    from httpx import ASGITransport, AsyncClient

    app.dependency_overrides[get_current_user] = lambda: {"sub": "test-user"}

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
        headers={"Authorization": "Bearer test-token"},
    ) as instance:
        yield instance

    app.dependency_overrides.clear()


@pytest.fixture
async def fake_redis(monkeypatch):
    """Sustituye el cliente Redis por `fakeredis` durante el test."""
    import fakeredis.aioredis

    from app.services import queue

    client = fakeredis.aioredis.FakeRedis(decode_responses=True)
    monkeypatch.setattr(queue, "get_client", lambda: client)

    yield client

    await client.flushall()
    await client.aclose()


@pytest.fixture
async def no_redis(monkeypatch):
    """Simula Redis caído: toda operación levanta `RedisError`."""
    from redis.exceptions import ConnectionError as RedisConnectionError

    from app.services import queue

    class DeadClient:
        def __getattr__(self, _name):
            def fail(*_args, **_kwargs):
                raise RedisConnectionError("connection refused")

            return fail

    monkeypatch.setattr(queue, "get_client", lambda: DeadClient())
    yield
