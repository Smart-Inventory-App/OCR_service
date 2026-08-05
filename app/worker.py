"""Worker que consume la cola OCR de Redis.

Se ejecuta como proceso aparte (`python -m app.worker`, servicio `worker` del
docker-compose). Cada job: descarga la imagen → corre el pipeline → guarda el
resultado → descarta la imagen temporal si corresponde.
"""

import asyncio
import logging
import signal

from app.core.config import settings
from app.schemas.ocr import DocType
from app.services import image_loader, ocr_service, queue, storage_client

logger = logging.getLogger("ocr.worker")

BLPOP_TIMEOUT_SECONDS = 5
RETRY_DELAY_SECONDS = 5

_shutdown = asyncio.Event()


async def process_job(task: dict) -> None:
    """Procesa un job de la cola y persiste su resultado."""
    job_id = task["job_id"]
    image_url = task["image_url"]
    doc_type = DocType(task["doc_type"])
    token = task["token"]

    await queue.mark_processing(job_id)

    try:
        content = await image_loader.from_url(image_url)
        fingerprint = image_loader.fingerprint(content)

        result = await queue.get_cached(fingerprint, doc_type, task["lang"])
        if result is None:
            result = await ocr_service.process(content, doc_type, task["lang"])
            await queue.set_cached(fingerprint, doc_type, task["lang"], result)

        await queue.mark_done(job_id, result)
        logger.info("Job %s completado (confianza=%.2f)", job_id, result.document_confidence)

    except image_loader.ImageLoadError as exc:
        logger.error("Job %s falló al cargar la imagen: %s", job_id, exc)
        await queue.mark_failed(job_id, str(exc))

    except Exception as exc:
        logger.exception("Job %s falló durante el procesamiento", job_id)
        await queue.mark_failed(job_id, f"Error al procesar la imagen: {exc}")

    finally:
        # La imagen temporal se descarta siempre: haya salido bien o mal.
        if task.get("discard_after"):
            await storage_client.discard_temp(image_url, token)


async def run() -> None:
    logger.info(
        "Worker OCR iniciado (cola=%s, motor=%s)", settings.QUEUE_KEY, settings.OCR_ENGINE
    )

    while not _shutdown.is_set():
        try:
            task = await queue.dequeue(timeout=BLPOP_TIMEOUT_SECONDS)
        except queue.QueueUnavailable as exc:
            logger.error("Redis no disponible (%s). Reintentando en %ds…", exc, RETRY_DELAY_SECONDS)
            await asyncio.sleep(RETRY_DELAY_SECONDS)
            continue

        if task is not None:
            await process_job(task)

    await queue.close()
    logger.info("Worker OCR detenido")


def _install_signal_handlers(loop: asyncio.AbstractEventLoop) -> None:
    """Apagado ordenado: termina el job en curso antes de salir."""
    for name in ("SIGINT", "SIGTERM"):
        sig = getattr(signal, name, None)
        if sig is None:
            continue
        try:
            loop.add_signal_handler(sig, _shutdown.set)
        except NotImplementedError:
            # Windows no soporta add_signal_handler: se usa el KeyboardInterrupt.
            signal.signal(sig, lambda *_: _shutdown.set())


def main() -> None:
    logging.basicConfig(
        level=logging.DEBUG if settings.DEBUG else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    _install_signal_handlers(loop)

    try:
        loop.run_until_complete(run())
    except KeyboardInterrupt:
        logger.info("Interrumpido por el usuario")
    finally:
        loop.close()


if __name__ == "__main__":
    main()
