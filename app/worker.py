"""Worker que consume la cola OCR de Redis.

Se ejecuta como proceso aparte (`python -m app.worker`, servicio `worker` del
docker-compose). Cada job: descarga la imagen → corre el pipeline → guarda el
resultado → descarta la imagen temporal si corresponde.

Los jobs se procesan de a `WORKER_CONCURRENCY` a la vez. No es para exprimir esta
máquina —el reconocimiento corre en los servicios de motor, no acá— sino para que
un escaneo no espere a que terminen los que llegaron antes: en serie, cuatro
clientes simultáneos hacían que el último terminara a los 168 s teniendo cada uno
menos de 65 s de trabajo propio.
"""

import asyncio
import logging
import signal
from datetime import datetime, timedelta, timezone

from jose import jwt

from app.core.config import settings
from app.engines import factory
from app.schemas.ocr import DocType
from app.services import image_loader, ocr_service, queue, storage_client

logger = logging.getLogger("ocr.worker")

BLPOP_TIMEOUT_SECONDS = 5
RETRY_DELAY_SECONDS = 5
PURGE_INTERVAL_SECONDS = 3600

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
        image_loader.validate(content)
        fingerprint = image_loader.fingerprint(content)

        engine = task.get("engine")
        # El motor forma parte de la clave: comparar motores no debe dar caché.
        cache_lang = f"{task['lang']}:{engine}" if engine else task["lang"]

        result = await queue.get_cached(fingerprint, doc_type, cache_lang)
        if result is None:
            result = await ocr_service.process(
                content, doc_type, task["lang"], engine=engine
            )
            await queue.set_cached(fingerprint, doc_type, cache_lang, result)

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


async def _consume(numero: int) -> None:
    """Toma jobs de la cola y los procesa hasta que se pide el apagado.

    Cada consumidor hace su propio `BLPOP`: Redis entrega cada elemento a un solo
    cliente bloqueado, así que dos consumidores nunca reciben el mismo job y no
    hace falta coordinarlos.
    """
    while not _shutdown.is_set():
        try:
            task = await queue.dequeue(timeout=BLPOP_TIMEOUT_SECONDS)
        except queue.QueueUnavailable as exc:
            logger.error("Redis no disponible (%s). Reintentando en %ds…", exc, RETRY_DELAY_SECONDS)
            await asyncio.sleep(RETRY_DELAY_SECONDS)
            continue

        if task is not None:
            logger.debug("Consumidor %d tomó el job %s", numero, task["job_id"])
            await process_job(task)


async def _purge_loop() -> None:
    """Limpia cada hora las imágenes temporales que quedaron sin descartar.

    `process_job` descarta la suya en un `finally`, así que el caso normal ya
    está cubierto. Lo que no cubre es que el proceso muera en medio de un job —un
    OOM, un redeploy—: ahí la foto queda en Cloudinary hasta que alguien la
    limpie. Vive en el worker porque es quien tiene el ciclo de vida de esas
    imágenes, y porque ya es un proceso de larga duración.
    """
    while not _shutdown.is_set():
        try:
            await asyncio.wait_for(_shutdown.wait(), timeout=PURGE_INTERVAL_SECONDS)
            return  # Se pidió el apagado durante la espera.
        except asyncio.TimeoutError:
            pass

        try:
            purged = await storage_client.purge_temp(_service_token())
            if purged:
                logger.info("Purgadas %d imágenes temporales vencidas", purged)
        except storage_client.StorageError as exc:
            logger.warning("No se pudo purgar las imágenes temporales: %s", exc)


def _service_token() -> str:
    """Token propio del worker para hablar con el storage.

    Los tokens de los usuarios viven en el payload de cada job y se borran al
    terminarlo, así que para una tarea periódica no hay ninguno a mano: se firma
    uno de vida corta con el secreto compartido de la plataforma.
    """
    return jwt.encode(
        {
            "sub": "ocr-worker",
            "email": "ocr-worker@smartinventory",
            "exp": datetime.now(timezone.utc) + timedelta(minutes=5),
        },
        settings.JWT_SECRET,
        algorithm=settings.JWT_ALGORITHM,
    )


async def run() -> None:
    concurrency = max(1, settings.WORKER_CONCURRENCY)

    logger.info(
        "Worker OCR iniciado (cola=%s, motor=%s, concurrencia=%d)",
        settings.QUEUE_KEY,
        settings.OCR_ENGINE,
        concurrency,
    )

    # `return_exceptions` para que un consumidor que muera por algo que
    # `process_job` no atrapó no cancele a los demás ni tumbe el proceso: la cola
    # sigue atendida con uno menos y el error queda registrado.
    results = await asyncio.gather(
        *(_consume(i) for i in range(concurrency)), _purge_loop(), return_exceptions=True
    )

    for result in results:
        if isinstance(result, BaseException):
            logger.error("Una tarea del worker terminó con error: %s", result)

    await queue.close()
    # Cierra las conexiones abiertas hacia los servicios de motor.
    factory.reset_cache()
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
