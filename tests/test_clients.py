"""Clientes HTTP (storage, descarga de imágenes) y loop del worker.

El transporte de `httpx` se mockea: no se hacen llamadas de red reales.
"""

from unittest.mock import AsyncMock, patch

import httpx
import pytest

from app.core.config import settings
from app.schemas.ocr import DocType
from app.services import image_loader, storage_client
from tests.conftest import render_image

IMAGE_URL = "https://res.cloudinary.com/demo/image/upload/bazaar/ocr-temp/scan.jpg"


def _transport(handler) -> httpx.MockTransport:
    return httpx.MockTransport(handler)


def _patched_client(handler):
    """Hace que `httpx.AsyncClient(...)` use un transporte mockeado."""
    original = httpx.AsyncClient

    def factory(*args, **kwargs):
        kwargs["transport"] = _transport(handler)
        return original(*args, **kwargs)

    return patch("httpx.AsyncClient", factory)


# ---------------------------------------------------------------------------
# image_loader
# ---------------------------------------------------------------------------

async def test_from_url_returns_content():
    content = render_image([("LECHE", 40)])

    with _patched_client(lambda _r: httpx.Response(200, content=content)):
        assert await image_loader.from_url(IMAGE_URL) == content


async def test_from_url_raises_on_http_error():
    with _patched_client(lambda _r: httpx.Response(404, text="not found")):
        with pytest.raises(image_loader.ImageLoadError):
            await image_loader.from_url(IMAGE_URL)


async def test_from_url_raises_on_connection_error():
    def boom(_request):
        raise httpx.ConnectError("connection refused")

    with _patched_client(boom):
        with pytest.raises(image_loader.ImageLoadError, match="No se pudo descargar"):
            await image_loader.from_url(IMAGE_URL)


async def test_from_url_rejects_empty_body():
    with _patched_client(lambda _r: httpx.Response(200, content=b"")):
        with pytest.raises(image_loader.ImageLoadError, match="vacía"):
            await image_loader.from_url(IMAGE_URL)


async def test_from_url_rejects_oversized_image(monkeypatch):
    monkeypatch.setattr(image_loader, "MAX_DOWNLOAD_BYTES", 10)

    with _patched_client(lambda _r: httpx.Response(200, content=b"x" * 50)):
        with pytest.raises(image_loader.ImageLoadError, match="límite"):
            await image_loader.from_url(IMAGE_URL)


def test_validate_returns_the_image_size():
    """Se valida sin decodificar la imagen entera: alcanza con la cabecera."""
    width, height = image_loader.validate(render_image([("LECHE", 40)], size=(640, 480)))

    assert (width, height) == (640, 480)


def test_validate_rejects_non_image_bytes():
    with pytest.raises(image_loader.ImageLoadError, match="no es una imagen"):
        image_loader.validate(b"esto no es una imagen")


def test_fingerprint_is_stable_and_content_dependent():
    a = render_image([("LECHE", 40)])
    b = render_image([("PAN", 40)])

    assert image_loader.fingerprint(a) == image_loader.fingerprint(a)
    assert image_loader.fingerprint(a) != image_loader.fingerprint(b)


# ---------------------------------------------------------------------------
# storage_client
# ---------------------------------------------------------------------------

async def test_upload_temp_returns_url_and_sends_token():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(
            200, json={"url": IMAGE_URL, "expires_at": "2026-07-29T21:00:00", "ttl_seconds": 900}
        )

    with _patched_client(handler):
        url = await storage_client.upload_temp(b"bytes", "scan.jpg", "image/jpeg", "tok123")

    assert url == IMAGE_URL
    assert seen["url"] == f"{settings.STORAGE_SERVICE_URL}/storage/temp"
    assert seen["auth"] == "Bearer tok123"


async def test_upload_temp_raises_on_storage_error():
    with _patched_client(lambda _r: httpx.Response(400, text="formato no permitido")):
        with pytest.raises(storage_client.StorageError, match="400"):
            await storage_client.upload_temp(b"bytes", "scan.jpg", "image/jpeg", "tok")


async def test_upload_temp_raises_when_storage_is_unreachable():
    def boom(_request):
        raise httpx.ConnectError("connection refused")

    with _patched_client(boom):
        with pytest.raises(storage_client.StorageError, match="No se pudo contactar"):
            await storage_client.upload_temp(b"bytes", "scan.jpg", "image/jpeg", "tok")


async def test_discard_temp_sends_delete_with_url():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["body"] = request.content.decode()
        return httpx.Response(200, json={"ok": True})

    with _patched_client(handler):
        assert await storage_client.discard_temp(IMAGE_URL, "tok") is True

    assert seen["method"] == "DELETE"
    assert IMAGE_URL in seen["body"]


async def test_discard_temp_is_best_effort_on_failure():
    """Un fallo al descartar no interrumpe el OCR: el storage purga por TTL."""
    with _patched_client(lambda _r: httpx.Response(500, text="boom")):
        assert await storage_client.discard_temp(IMAGE_URL, "tok") is False


# ---------------------------------------------------------------------------
# Loop del worker
# ---------------------------------------------------------------------------

async def test_worker_loop_processes_queued_jobs_then_stops(fake_redis):
    from app import worker
    from app.services import queue

    await queue.enqueue(IMAGE_URL, DocType.PRODUCT_LABEL, "spa", "tok", False)

    processed: list[dict] = []

    async def fake_process(task):
        processed.append(task)
        worker._shutdown.set()  # una sola vuelta

    worker._shutdown.clear()
    with patch.object(worker, "process_job", fake_process), \
         patch.object(worker.queue, "close", AsyncMock()):
        await worker.run()

    assert len(processed) == 1
    assert processed[0]["image_url"] == IMAGE_URL


async def test_worker_loop_retries_when_redis_is_down(no_redis, monkeypatch):
    from app import worker

    sleeps: list[float] = []

    async def fake_sleep(seconds):
        sleeps.append(seconds)
        worker._shutdown.set()  # cortar tras el primer reintento

    worker._shutdown.clear()
    monkeypatch.setattr(worker.asyncio, "sleep", fake_sleep)

    with patch.object(worker.queue, "close", AsyncMock()):
        await worker.run()

    assert sleeps == [worker.RETRY_DELAY_SECONDS]


async def test_worker_loop_ignores_empty_dequeue(fake_redis):
    """Un timeout de BLPOP sin trabajo no debe llamar a process_job."""
    from app import worker

    calls: list = []

    async def dequeue_once(timeout: int):
        worker._shutdown.set()
        return None

    worker._shutdown.clear()
    with patch.object(worker.queue, "dequeue", dequeue_once), \
         patch.object(worker, "process_job", lambda task: calls.append(task)), \
         patch.object(worker.queue, "close", AsyncMock()):
        await worker.run()

    assert calls == []
