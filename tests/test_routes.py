"""Endpoints HTTP: autenticación, flujo asincrónico y vía sincrónica."""

from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.schemas.ocr import DocType, JobStatus
from tests.conftest import render_image

# Una etiqueta leída bien: marca en grande, nombre debajo, contenido chico.
LABEL_LINES = ["La Serenisima", "LECHE ENTERA", "1 litro"]
LABEL_HEIGHTS = [60, 30, 16]
STAMP_LINES = ["VTO: 15/08/2027"]
IMAGE_URL = "https://res.cloudinary.com/demo/image/upload/bazaar/ocr-temp/scan.jpg"


def _image_bytes() -> bytes:
    return render_image([("LA SERENISIMA", 40), ("LECHE ENTERA", 30), ("1 litro", 20)])


def _files(content_type: str = "image/jpeg"):
    return {"file": ("scan.jpg", _image_bytes(), content_type)}


def _anon() -> AsyncClient:
    """Cliente sin credenciales (el fixture `client` ya viene autenticado)."""
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

async def test_livez_needs_no_auth():
    async with _anon() as c:
        r = await c.get("/ocr/livez")

    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


async def test_readyz_ok_when_dependencies_are_up(fake_redis):
    with patch("app.services.ocr_service.engine_available", return_value=True):
        async with _anon() as c:
            r = await c.get("/ocr/readyz")

    assert r.status_code == 200
    assert r.json()["checks"] == {"redis": True, "engine": True}


async def test_readyz_fails_without_redis(no_redis):
    with patch("app.services.ocr_service.engine_available", return_value=True):
        async with _anon() as c:
            r = await c.get("/ocr/readyz")

    assert r.status_code == 503
    assert r.json()["detail"]["checks"]["redis"] is False


async def test_readyz_fails_without_engine(fake_redis):
    with patch("app.services.ocr_service.engine_available", return_value=False):
        async with _anon() as c:
            r = await c.get("/ocr/readyz")

    assert r.status_code == 503
    assert r.json()["detail"]["checks"]["engine"] is False


# ---------------------------------------------------------------------------
# Autenticación — todos los /ocr/* excepto health
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "method, path",
    [
        ("post", "/ocr/jobs"),
        ("post", "/ocr/scan"),
        ("get", "/ocr/jobs/abc123"),
        ("post", "/ocr/sync"),
        ("post", "/ocr/sync/upload"),
    ],
)
async def test_endpoints_require_a_token(method, path):
    async with _anon() as c:
        r = await getattr(c, method)(path)

    assert r.status_code == 401


async def test_invalid_token_is_rejected():
    async with _anon() as c:
        r = await c.post(
            "/ocr/jobs",
            headers={"Authorization": "Bearer no-es-un-jwt"},
            json={"image_url": IMAGE_URL, "doc_type": "product_label"},
        )

    assert r.status_code == 401
    assert r.json()["detail"] == "auth.invalid_token"


async def test_valid_token_is_accepted(fake_redis):
    """Un JWT firmado con el secret compartido pasa la validación real."""
    from jose import jwt

    from app.core.config import settings

    token = jwt.encode({"sub": "user-1"}, settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM)

    async with _anon() as c:
        r = await c.post(
            "/ocr/jobs",
            headers={"Authorization": f"Bearer {token}"},
            json={"image_url": IMAGE_URL, "doc_type": "product_label"},
        )

    assert r.status_code == 202


# ---------------------------------------------------------------------------
# Flujo asincrónico: POST /ocr/jobs → worker → GET /ocr/jobs/{id}
# ---------------------------------------------------------------------------

async def test_create_job_returns_queued(client, fake_redis):
    r = await client.post("/ocr/jobs", json={"image_url": IMAGE_URL, "doc_type": "product_label"})

    assert r.status_code == 202
    body = r.json()
    assert body["status"] == "queued"
    assert body["job_id"]


async def test_create_job_rejects_invalid_url(client, fake_redis):
    r = await client.post("/ocr/jobs", json={"image_url": "no-es-una-url"})

    assert r.status_code == 422


async def test_create_job_without_redis_returns_503(client, no_redis):
    r = await client.post("/ocr/jobs", json={"image_url": IMAGE_URL})

    assert r.status_code == 503


async def test_get_unknown_job_returns_404(client, fake_redis):
    r = await client.get("/ocr/jobs/no-existe")

    assert r.status_code == 404


async def test_full_async_flow(client, fake_redis, fake_engine):
    """Encolar → procesar con el worker → consultar el resultado."""
    from app.services import queue
    from app.worker import process_job

    fake_engine(LABEL_LINES, confidence=0.99, heights=LABEL_HEIGHTS)

    created = await client.post(
        "/ocr/jobs", json={"image_url": IMAGE_URL, "doc_type": "product_label"}
    )
    job_id = created.json()["job_id"]

    task = await queue.dequeue(timeout=1)
    assert task["job_id"] == job_id

    with patch("app.services.image_loader.from_url", AsyncMock(return_value=_image_bytes())):
        await process_job(task)

    body = (await client.get(f"/ocr/jobs/{job_id}")).json()

    assert body["status"] == JobStatus.DONE.value
    assert body["result"]["fields"]["nombre_producto"]["value"] == "LECHE ENTERA"
    assert body["result"]["fields"]["marca"]["value"] == "La Serenisima"
    assert body["result"]["needs_review"] is False


async def test_job_state_never_exposes_the_caller_token(client, fake_redis):
    created = await client.post("/ocr/jobs", json={"image_url": IMAGE_URL})
    job_id = created.json()["job_id"]

    r = await client.get(f"/ocr/jobs/{job_id}")

    assert "token" not in r.text


async def test_worker_marks_job_failed_when_download_fails(client, fake_redis):
    from app.services import image_loader, queue
    from app.worker import process_job

    created = await client.post("/ocr/jobs", json={"image_url": IMAGE_URL})
    job_id = created.json()["job_id"]

    task = await queue.dequeue(timeout=1)

    with patch(
        "app.services.image_loader.from_url",
        AsyncMock(side_effect=image_loader.ImageLoadError("404 not found")),
    ):
        await process_job(task)

    body = (await client.get(f"/ocr/jobs/{job_id}")).json()

    assert body["status"] == JobStatus.FAILED.value
    assert "404" in body["error"]


async def test_worker_marks_job_failed_on_unexpected_error(client, fake_redis, fake_engine):
    from app.services import queue
    from app.worker import process_job

    created = await client.post("/ocr/jobs", json={"image_url": IMAGE_URL})
    job_id = created.json()["job_id"]

    task = await queue.dequeue(timeout=1)

    with patch("app.services.image_loader.from_url", AsyncMock(return_value=_image_bytes())), \
         patch("app.services.ocr_service.process", AsyncMock(side_effect=RuntimeError("boom"))):
        await process_job(task)

    body = (await client.get(f"/ocr/jobs/{job_id}")).json()

    assert body["status"] == JobStatus.FAILED.value


# ---------------------------------------------------------------------------
# POST /ocr/scan — sube la imagen como temporal y la descarta al terminar
# ---------------------------------------------------------------------------

async def test_scan_uploads_temp_image_and_enqueues(client, fake_redis):
    from app.services import queue

    with patch(
        "app.services.storage_client.upload_temp", AsyncMock(return_value=IMAGE_URL)
    ) as upload:
        r = await client.post("/ocr/scan", files=_files(), data={"doc_type": "receipt"})

    assert r.status_code == 202
    upload.assert_awaited_once()

    task = await queue.dequeue(timeout=1)
    assert task["image_url"] == IMAGE_URL
    assert task["doc_type"] == DocType.RECEIPT.value
    # La imagen temporal debe descartarse al finalizar el procesamiento.
    assert task["discard_after"] is True


async def test_scan_discards_temp_image_after_processing(client, fake_redis, fake_engine):
    from app.services import queue
    from app.worker import process_job

    fake_engine(LABEL_LINES, confidence=0.99, heights=LABEL_HEIGHTS)

    with patch("app.services.storage_client.upload_temp", AsyncMock(return_value=IMAGE_URL)):
        await client.post("/ocr/scan", files=_files())

    task = await queue.dequeue(timeout=1)

    with patch("app.services.image_loader.from_url", AsyncMock(return_value=_image_bytes())), \
         patch("app.services.storage_client.discard_temp", AsyncMock(return_value=True)) as discard:
        await process_job(task)

    discard.assert_awaited_once_with(IMAGE_URL, task["token"])


async def test_scan_discards_temp_image_even_when_processing_fails(client, fake_redis):
    from app.services import image_loader, queue
    from app.worker import process_job

    with patch("app.services.storage_client.upload_temp", AsyncMock(return_value=IMAGE_URL)):
        await client.post("/ocr/scan", files=_files())

    task = await queue.dequeue(timeout=1)

    with patch(
        "app.services.image_loader.from_url",
        AsyncMock(side_effect=image_loader.ImageLoadError("boom")),
    ), patch("app.services.storage_client.discard_temp", AsyncMock(return_value=True)) as discard:
        await process_job(task)

    discard.assert_awaited_once()


async def test_scan_discards_temp_image_when_queue_is_down(client, no_redis):
    """Si no se puede encolar, la imagen temporal no debe quedar huérfana."""
    with patch("app.services.storage_client.upload_temp", AsyncMock(return_value=IMAGE_URL)), \
         patch("app.services.storage_client.discard_temp", AsyncMock(return_value=True)) as discard:
        r = await client.post("/ocr/scan", files=_files())

    assert r.status_code == 503
    discard.assert_awaited_once()


async def test_scan_reports_storage_failure_as_502(client, fake_redis):
    from app.services import storage_client

    with patch(
        "app.services.storage_client.upload_temp",
        AsyncMock(side_effect=storage_client.StorageError("storage respondió 500")),
    ):
        r = await client.post("/ocr/scan", files=_files())

    assert r.status_code == 502


async def test_scan_rejects_unsupported_format(client, fake_redis):
    r = await client.post(
        "/ocr/scan", files={"file": ("doc.pdf", b"%PDF-1.4", "application/pdf")}
    )

    assert r.status_code == 400
    assert "application/pdf" in r.json()["detail"]


async def test_scan_rejects_empty_file(client, fake_redis):
    r = await client.post("/ocr/scan", files={"file": ("empty.jpg", b"", "image/jpeg")})

    assert r.status_code == 400


# ---------------------------------------------------------------------------
# Vía sincrónica
# ---------------------------------------------------------------------------

async def test_sync_upload_returns_result_directly(client, fake_redis, fake_engine):
    fake_engine(LABEL_LINES, confidence=0.99, heights=LABEL_HEIGHTS)

    r = await client.post(
        "/ocr/sync/upload", files=_files(), data={"doc_type": "product_label"}
    )

    assert r.status_code == 200
    result = r.json()["result"]
    assert result["fields"]["nombre_producto"]["value"] == "LECHE ENTERA"
    assert result["engine"] == "fake"
    assert result["processing_ms"] >= 0


async def test_sync_upload_works_without_redis(no_redis, client, fake_engine):
    """La caché es opcional: sin Redis el pipeline sincrónico sigue funcionando."""
    fake_engine(LABEL_LINES, confidence=0.99, heights=LABEL_HEIGHTS)

    r = await client.post("/ocr/sync/upload", files=_files())

    assert r.status_code == 200


async def test_sync_from_url_downloads_and_processes(client, fake_redis, fake_engine):
    fake_engine(LABEL_LINES, confidence=0.99, heights=LABEL_HEIGHTS)

    with patch("app.services.image_loader.from_url", AsyncMock(return_value=_image_bytes())):
        r = await client.post("/ocr/sync", json={"image_url": IMAGE_URL})

    assert r.status_code == 200
    assert r.json()["result"]["fields"]["nombre_producto"]["value"] == "LECHE ENTERA"


async def test_sync_from_url_reports_download_failure_as_400(client, fake_redis):
    from app.services import image_loader

    with patch(
        "app.services.image_loader.from_url",
        AsyncMock(side_effect=image_loader.ImageLoadError("no se pudo descargar")),
    ):
        r = await client.post("/ocr/sync", json={"image_url": IMAGE_URL})

    assert r.status_code == 400


async def test_sync_discards_temp_image_when_requested(client, fake_redis, fake_engine):
    fake_engine(LABEL_LINES, confidence=0.99, heights=LABEL_HEIGHTS)

    with patch("app.services.image_loader.from_url", AsyncMock(return_value=_image_bytes())), \
         patch("app.services.storage_client.discard_temp", AsyncMock(return_value=True)) as discard:
        r = await client.post(
            "/ocr/sync", json={"image_url": IMAGE_URL, "discard_after": True}
        )

    assert r.status_code == 200
    discard.assert_awaited_once()


async def test_sync_rejects_corrupt_image(client, fake_redis, fake_engine):
    fake_engine(LABEL_LINES)

    r = await client.post(
        "/ocr/sync/upload",
        files={"file": ("fake.jpg", b"esto no es una imagen", "image/jpeg")},
    )

    assert r.status_code == 400


# ---------------------------------------------------------------------------
# Selección de motor por petición
# ---------------------------------------------------------------------------

async def test_engine_can_be_chosen_per_request(client, fake_redis, no_retry):
    """El playground compara motores mandando `engine` en cada petición."""
    captured = {}

    async def fake_process(content, doc_type, lang=None, today=None, engine=None):
        from app.schemas.ocr import OCRResult

        captured["engine"] = engine
        return OCRResult(engine=engine or "default", document_confidence=0.9)

    with patch("app.services.ocr_service.process", fake_process):
        r = await client.post(
            "/ocr/sync/upload", files=_files(), data={"engine": "easyocr"}
        )

    assert r.status_code == 200
    assert captured["engine"] == "easyocr"


async def test_unknown_engine_is_rejected(client, fake_redis):
    r = await client.post("/ocr/sync/upload", files=_files(), data={"engine": "inventado"})

    assert r.status_code == 400
    assert "desconocido" in r.json()["detail"]


async def test_engine_is_forwarded_to_the_queue(client, fake_redis):
    from app.services import queue

    with patch("app.services.storage_client.upload_temp", AsyncMock(return_value=IMAGE_URL)):
        await client.post("/ocr/scan", files=_files(), data={"engine": "tesseract"})

    task = await queue.dequeue(timeout=1)

    assert task["engine"] == "tesseract"


async def test_cache_distinguishes_engine(client, fake_redis, fake_engine, no_retry):
    """Comparar motores sobre la misma foto no debe devolver el resultado cacheado."""
    engine = fake_engine(LABEL_LINES, confidence=0.99, heights=LABEL_HEIGHTS)

    await client.post("/ocr/sync/upload", files=_files(), data={"engine": "tesseract"})
    await client.post("/ocr/sync/upload", files=_files(), data={"engine": "easyocr"})

    assert len(engine.calls) == 2


# ---------------------------------------------------------------------------
# Caché por hash de imagen
# ---------------------------------------------------------------------------

async def test_identical_image_is_not_reprocessed(client, fake_redis, fake_engine):
    engine = fake_engine(LABEL_LINES, confidence=0.99, heights=LABEL_HEIGHTS)

    first = await client.post("/ocr/sync/upload", files=_files())
    second = await client.post("/ocr/sync/upload", files=_files())

    assert first.json()["result"] == second.json()["result"]
    assert len(engine.calls) == 1  # la segunda vez salió de la caché


async def test_cache_distinguishes_doc_type(client, fake_redis, fake_engine, no_retry):
    engine = fake_engine(LABEL_LINES, confidence=0.99, heights=LABEL_HEIGHTS)

    await client.post("/ocr/sync/upload", files=_files(), data={"doc_type": "product_label"})
    await client.post("/ocr/sync/upload", files=_files(), data={"doc_type": "receipt"})

    assert len(engine.calls) == 2


async def test_layout_hint_depends_on_doc_type(client, fake_redis, fake_engine, no_retry):
    engine = fake_engine(LABEL_LINES, confidence=0.99, heights=LABEL_HEIGHTS)

    await client.post("/ocr/sync/upload", files=_files(), data={"doc_type": "product_label"})
    await client.post("/ocr/sync/upload", files=_files(), data={"doc_type": "receipt"})

    assert [call["layout"] for call in engine.calls] == ["sparse", "block"]


async def test_low_confidence_triggers_more_attempts(client, fake_redis, fake_engine):
    """Una lectura pobre reintenta con otras variantes de preparación."""
    engine = fake_engine(LABEL_LINES, confidence=0.30, heights=LABEL_HEIGHTS)

    await client.post("/ocr/sync/upload", files=_files())

    assert len(engine.calls) > 1


async def test_good_reading_does_not_retry(client, fake_redis, fake_engine):
    """Si la primera lectura alcanza el umbral, no se gasta tiempo de más."""
    engine = fake_engine(LABEL_LINES, confidence=0.99, heights=LABEL_HEIGHTS)

    await client.post("/ocr/sync/upload", files=_files())

    assert len(engine.calls) == 1
