"""Fixtures compartidas.

Los tests no necesitan levantar ningún servicio de motor ni un Redis real: el
motor se sustituye por un doble que devuelve palabras con confianza controlada y
Redis por `fakeredis`. Lo que sí se ejercita de verdad es todo lo que este
servicio aporta — la estrategia de intentos, los parsers y el scoring —, que es
exactamente el código que quedó acá después de mover los motores a sus propios
repositorios.
"""

import io
import os

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
    """Motor que devuelve un guion fijo de líneas, con confianza configurable.

    Registra cada llamada con la transformación pedida: es lo que permite
    verificar que el coordinador manda el giro, el layout y los filtros que
    corresponden a cada tipo de documento.
    """

    def __init__(
        self,
        lines: list[str],
        confidence: float = 0.95,
        heights: list[int] | None = None,
        name: str = "fake",
        supports_preprocessing: bool = True,
    ):
        self.lines = lines
        self.confidence = confidence
        self.heights = heights or [30] * len(lines)
        self.name = name
        self.supports_preprocessing = supports_preprocessing
        self.calls: list[dict] = []

    def is_available(self) -> bool:
        return True

    def run(self, content, lang: str = "spa", layout: str = "block", transform=None) -> RawOCROutput:
        self.calls.append({
            "lang": lang,
            "layout": layout,
            "transform": transform,
            "rotate": getattr(transform, "rotate", 0),
            "bytes": len(content) if content else 0,
        })

        return build_output(self.lines, self.confidence, self.heights, self.name)


def build_output(
    lines: list[str],
    confidence: float = 0.95,
    heights: list[int] | None = None,
    engine: str = "fake",
) -> RawOCROutput:
    """Arma una salida OCR sintética con cajas coherentes.

    Las cajas importan: el parser de etiquetas agrupa renglones por su altura y
    su posición, así que una salida con geometría inventada no ejercitaría esa
    lógica.
    """
    heights = heights or [30] * len(lines)
    words: list[OCRWord] = []
    top = 10

    for line_num, (line, height) in enumerate(zip(lines, heights)):
        left = 10
        for token in line.split():
            width = max(1, len(token) * height // 2)
            words.append(
                OCRWord(
                    text=token,
                    confidence=confidence,
                    left=left,
                    top=top,
                    width=width,
                    height=height,
                    line_num=line_num,
                )
            )
            left += width + height // 3
        top += height + 10

    return RawOCROutput.from_words(words, engine)


def make_output(
    lines: list[str],
    confidence: float = 0.95,
    heights: list[int] | None = None,
) -> RawOCROutput:
    """Salida OCR sintética, sin pasar por el coordinador."""
    return build_output(lines, confidence, heights)


@pytest.fixture
def fake_engine(monkeypatch):
    """Reemplaza el motor real por un `FakeEngine` configurable desde el test."""
    holder: dict[str, FakeEngine] = {}

    def install(
        lines: list[str],
        confidence: float = 0.95,
        heights: list[int] | None = None,
        supports_preprocessing: bool = True,
    ):
        engine = FakeEngine(
            lines, confidence, heights, supports_preprocessing=supports_preprocessing
        )
        holder["engine"] = engine
        monkeypatch.setattr("app.services.ocr_service.build_engine", lambda *_, **__: engine)
        return engine

    install.holder = holder  # type: ignore[attr-defined]
    return install


# ---------------------------------------------------------------------------
# Imágenes de prueba
# ---------------------------------------------------------------------------

def render_image(lines: list[tuple[str, int]], size: tuple[int, int] = (900, 500)) -> bytes:
    """Renderiza líneas de texto como JPEG.

    Sirve para que las rutas reciban un binario que **es** una imagen válida (se
    valida con Pillow antes de encolar); el texto reconocido lo determina el
    motor falso, no esta imagen.
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


def render_barcode(code: str, kind: str = "ean13") -> Image.Image:
    """Genera un código de barras **real**, con su codificación de barras.

    No es un dibujo aproximado: lo produce `python-barcode` con la misma
    codificación que usa una impresora de etiquetas, así que lo que se ejercita
    en los tests es la decodificación de verdad y no un doble del decodificador.
    """
    import barcode as barcode_lib
    from barcode.writer import ImageWriter

    buffer = io.BytesIO()
    barcode_lib.get_barcode_class(kind)(code, writer=ImageWriter()).write(buffer)

    return Image.open(io.BytesIO(buffer.getvalue())).convert("RGB")


def barcode_photo(
    codes: str | list[str],
    kind: str = "ean13",
    width: int = 420,
    canvas: tuple[int, int] = (1200, 1600),
    blur: float = 0.0,
) -> bytes:
    """Compone códigos sobre un fondo tipo foto y devuelve el PNG.

    `width` es el ancho **del código** en píxeles dentro del cuadro, que es la
    variable de la que depende que se pueda decodificar: es lo que permite
    reproducir en un test los umbrales que se midieron.

    Se guarda en PNG y no en JPEG a propósito: el ruido de compresión movería el
    umbral de un entorno a otro y volvería frágiles justamente los tests que
    fijan dónde está ese umbral.
    """
    from PIL import ImageFilter

    if isinstance(codes, str):
        codes = [codes]

    photo = Image.new("RGB", canvas, (203, 200, 196))
    top = canvas[1] // 6

    for code in codes:
        rendered = render_barcode(code, kind)
        height = int(rendered.height * width / rendered.width)
        rendered = rendered.resize((width, height), Image.LANCZOS)

        if blur:
            rendered = rendered.filter(ImageFilter.GaussianBlur(blur))

        photo.paste(rendered, ((canvas[0] - width) // 2, top))
        top += height + canvas[1] // 8

    buffer = io.BytesIO()
    photo.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def barcode_image() -> bytes:
    """Foto de un EAN-13 argentino, legible en el primer intento."""
    return barcode_photo("779123456789")


@pytest.fixture
def label_image() -> bytes:
    return render_image([("LA SERENISIMA", 40), ("LECHE ENTERA", 30), ("1 litro", 20)])


@pytest.fixture
def stamp_image() -> bytes:
    return render_image([("VTO 15/08/2027", 30)], size=(600, 200))


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
def no_retry(monkeypatch):
    """Deja un único intento por documento.

    Lo usa cualquier test que cuente invocaciones al motor: con la estrategia
    completa, una sola petición puede llamarlo hasta ocho veces.
    """
    from app.core.config import settings

    monkeypatch.setattr(settings, "RETRY_ENABLED", False)
    monkeypatch.setattr(settings, "ROTATION_SEARCH_ENABLED", False)


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
