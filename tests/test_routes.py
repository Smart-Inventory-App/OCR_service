"""Endpoints HTTP: autenticación, flujo asincrónico y vía sincrónica."""

from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.core.config import settings
from app.main import app
from app.schemas.ocr import DocType, JobStatus
from tests.conftest import barcode_photo, render_image

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
    # `barcode` viaja en el mismo diccionario pero es informativo: no decide el
    # readiness. Ver `test_barcode_decoder_does_not_break_readiness`.
    assert r.json()["checks"] == {"redis": True, "engine": True, "barcode": True}


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


async def test_the_quota_stops_a_client_that_floods(client, fake_redis, monkeypatch):
    """Un tope por usuario y por minuto, distinto según lo que cuesta cada tipo.

    Un ticket ocupa un slot del worker entre 40 y 140 s y un código de barras se
    resuelve en 0,2 s sin tocar un motor: un límite único estrangularía al
    comercio que escanea una entrega o dejaría que un solo usuario sature todo.
    """
    monkeypatch.setattr(
        settings, "RATE_LIMIT_PER_MINUTE_STR", "receipt:2,product_label:10"
    )

    for _ in range(2):
        aceptado = await client.post(
            "/ocr/jobs", json={"image_url": IMAGE_URL, "doc_type": "receipt"}
        )
        assert aceptado.status_code == 202

    rechazado = await client.post(
        "/ocr/jobs", json={"image_url": IMAGE_URL, "doc_type": "receipt"}
    )

    assert rechazado.status_code == 429
    assert rechazado.headers["retry-after"] == "60"

    # El cupo es por tipo: el de etiquetas sigue intacto.
    otra = await client.post(
        "/ocr/jobs", json={"image_url": IMAGE_URL, "doc_type": "product_label"}
    )
    assert otra.status_code == 202


async def test_a_full_queue_is_refused_instead_of_promising_a_long_wait(
    client, fake_redis, monkeypatch
):
    """Aceptar el número 16 sería mentirle: son siete minutos de espera.

    Mejor un 429 con `Retry-After` que un `processing` que no termina nunca.
    """
    monkeypatch.setattr(settings, "MAX_QUEUE_DEPTH", 2)

    for _ in range(2):
        assert (
            await client.post("/ocr/jobs", json={"image_url": IMAGE_URL})
        ).status_code == 202

    lleno = await client.post("/ocr/jobs", json={"image_url": IMAGE_URL})

    assert lleno.status_code == 429
    assert int(lleno.headers["retry-after"]) > 0


async def test_a_barcode_does_not_count_against_the_queue_depth(
    client, fake_redis, monkeypatch
):
    """No pasa por la cola ni por un motor: no tiene por qué esperar a nadie."""
    monkeypatch.setattr(settings, "MAX_QUEUE_DEPTH", 1)

    assert (await client.post("/ocr/jobs", json={"image_url": IMAGE_URL})).status_code == 202

    codigo = await client.post(
        "/ocr/jobs", json={"image_url": IMAGE_URL, "doc_type": "barcode"}
    )

    assert codigo.status_code == 202


async def test_a_queued_job_reports_its_place_in_line(client, fake_redis):
    """La App necesita poder decir «3º de 7, ~2 min» en vez de un spinner."""
    ids = []
    for _ in range(3):
        created = await client.post("/ocr/jobs", json={"image_url": IMAGE_URL})
        ids.append(created.json()["job_id"])

    tercero = (await client.get(f"/ocr/jobs/{ids[2]}")).json()

    assert tercero["queue_position"] == 3
    assert tercero["estimated_wait_seconds"] == int(
        3 * settings.QUEUE_DRAIN_SECONDS_PER_JOB
    )


async def test_a_job_of_another_user_is_not_visible(client, fake_redis):
    """La lectura de una foto es un dato del usuario, no del portador del token.

    Un `job_id` puede filtrarse por un log, una URL o el historial del cliente;
    que quien lo tenga pueda leer el ticket de otro sería peor que perderlo. Se
    responde 404 y no 403 para no confirmar siquiera que existe.
    """
    from app.core.security import get_current_user
    from app.main import app

    created = await client.post("/ocr/jobs", json={"image_url": IMAGE_URL})
    job_id = created.json()["job_id"]

    app.dependency_overrides[get_current_user] = lambda: {"sub": "otro-usuario"}
    ajeno = await client.get(f"/ocr/jobs/{job_id}")

    app.dependency_overrides[get_current_user] = lambda: {"sub": "test-user"}
    propio = await client.get(f"/ocr/jobs/{job_id}")

    assert ajeno.status_code == 404
    assert propio.status_code == 200


async def test_the_scan_also_records_its_owner(client, fake_redis):
    """`/ocr/scan` es la vía de la App: el dueño se registra igual que en `/jobs`."""
    from app.core.security import get_current_user
    from app.main import app
    from app.services import queue

    with patch(
        "app.services.storage_client.upload_temp",
        AsyncMock(return_value="https://cloudinary/tmp/foto.jpg"),
    ):
        created = await client.post(
            "/ocr/scan",
            files={"file": ("foto.jpg", _image_bytes(), "image/jpeg")},
            data={"doc_type": "product_label"},
        )
    job_id = created.json()["job_id"]

    assert await queue.owner_of(job_id) == "test-user"

    app.dependency_overrides[get_current_user] = lambda: {"sub": "otro-usuario"}
    assert (await client.get(f"/ocr/jobs/{job_id}")).status_code == 404

    app.dependency_overrides[get_current_user] = lambda: {"sub": "test-user"}


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
            "/ocr/sync/upload", files=_files(), data={"engine": "paddleocr"}
        )

    assert r.status_code == 200
    assert captured["engine"] == "paddleocr"


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
    await client.post("/ocr/sync/upload", files=_files(), data={"engine": "paddleocr"})

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


# ---------------------------------------------------------------------------
# Códigos de barras
# ---------------------------------------------------------------------------

def _barcode_files():
    return {"file": ("codigo.png", barcode_photo("779123456789"), "image/png")}


async def test_barcode_sync_upload(client, fake_redis):
    """El tipo `barcode` atraviesa la vía sincrónica sin ningún motor arriba: no
    se usa `fake_engine` porque no hay motor al que reemplazar."""
    r = await client.post(
        "/ocr/sync/upload", files=_barcode_files(), data={"doc_type": "barcode"}
    )

    assert r.status_code == 200
    body = r.json()
    assert body["doc_type"] == "barcode"
    assert body["result"]["fields"]["codigo"]["value"] == "7791234567898"
    assert body["result"]["fields"]["pais_registro"]["value"] == "Argentina"
    assert body["result"]["engine"] == "zbar"
    assert body["result"]["needs_review"] is False


async def test_barcode_async_flow(client, fake_redis):
    """Flujo completo de la App: encolar, procesar en el worker y leer el job."""
    from app.services import queue
    from app.worker import process_job

    with patch("app.services.storage_client.upload_temp", new=AsyncMock(return_value=IMAGE_URL)):
        r = await client.post(
            "/ocr/scan", files=_barcode_files(), data={"doc_type": "barcode"}
        )

    assert r.status_code == 202
    job_id = r.json()["job_id"]

    task = await queue.dequeue(timeout=1)
    with patch(
        "app.services.image_loader.from_url",
        new=AsyncMock(return_value=barcode_photo("779123456789")),
    ), patch("app.services.storage_client.discard_temp", new=AsyncMock(return_value=True)):
        await process_job(task)

    state = await client.get(f"/ocr/jobs/{job_id}")

    assert state.json()["status"] == JobStatus.DONE
    assert state.json()["result"]["fields"]["codigo"]["value"] == "7791234567898"


async def test_barcode_without_decoder_returns_503(client, fake_redis, monkeypatch):
    """Falta `libzbar` en la imagen: es una falla local, y un 503 la distingue
    del 502 con el que se reporta un servicio de motor remoto caído."""
    from app.decoders import barcode as decoder

    def explota(*_args, **_kwargs):
        raise decoder.DecoderUnavailable("falta libzbar")

    monkeypatch.setattr(decoder, "_pyzbar", explota)

    r = await client.post(
        "/ocr/sync/upload", files=_barcode_files(), data={"doc_type": "barcode"}
    )

    assert r.status_code == 503
    assert "libzbar" in r.json()["detail"]


async def test_readyz_reports_the_barcode_decoder(client, fake_redis, fake_engine):
    fake_engine(LABEL_LINES)

    r = await client.get("/ocr/readyz")

    assert r.json()["checks"]["barcode"] is True


async def test_barcode_decoder_does_not_break_readiness(client, fake_redis, fake_engine, monkeypatch):
    """Sin decodificador el servicio sigue resolviendo etiquetas, sellos y
    tickets: no puede quedar fuera de rotación por una dependencia de un solo
    tipo de documento."""
    from app.decoders import barcode as decoder

    monkeypatch.setattr(decoder, "is_available", lambda: False)
    fake_engine(LABEL_LINES)

    r = await client.get("/ocr/readyz")

    assert r.status_code == 200
    assert r.json()["checks"]["barcode"] is False


async def test_truncated_image_is_rejected_with_400(client, fake_redis):
    """Una imagen truncada tiene la cabecera bien y los datos incompletos.

    Antes pasaba la validación —que sólo miraba la cabecera— y reventaba más
    adelante: 500 con `barcode` y 502 con `product_label`. Ninguno de los dos
    describe lo que pasó, que es que el archivo llegó mal.
    """
    entera = barcode_photo("779123456789")
    truncada = entera[: len(entera) // 4]

    for doc_type in ("barcode", "product_label", "receipt"):
        r = await client.post(
            "/ocr/sync/upload",
            files={"file": ("rota.png", truncada, "image/png")},
            data={"doc_type": doc_type},
        )

        assert r.status_code == 400, doc_type
        assert "no es una imagen válida" in r.json()["detail"]


async def test_a_whole_image_is_still_accepted(client, fake_redis):
    """La contraparte: verificar el archivo entero no rechaza uno bueno."""
    r = await client.post(
        "/ocr/sync/upload", files=_barcode_files(), data={"doc_type": "barcode"}
    )

    assert r.status_code == 200
