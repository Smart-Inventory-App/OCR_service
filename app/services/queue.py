"""Cola de trabajos y caché de resultados sobre Redis.

La propuesta pide "considerar procesamiento asincrónico para evitar bloqueos en
la experiencia de usuario" (sección *Riesgos Técnicos*) y ya incluye Redis en el
stack para colas y caché.

`POST /ocr/jobs` crea el registro y encola el id; `worker.py` lo consume con
`BLPOP`; `GET /ocr/jobs/{id}` lee el estado. Se usan listas de Redis en lugar de
RQ para no depender de `fork()` (no disponible en Windows) y para que el mismo
código funcione igual en desarrollo y en Docker.

Cada job se guarda en dos claves separadas:

- `ocr:job:{id}`  — estado **público** que devuelve la API.
- `ocr:task:{id}` — payload **interno** (incluye el token del llamador, que se
  necesita para descartar la imagen temporal y nunca debe salir por la API).
"""

import json
import logging
import uuid
from datetime import datetime, timezone

import redis.asyncio as redis
from redis.exceptions import RedisError

from app.core.config import settings
from app.schemas.ocr import DocType, JobStatus, OCRJobState, OCRResult

logger = logging.getLogger(__name__)

_JOB_PREFIX = "ocr:job:"
_TASK_PREFIX = "ocr:task:"
_CACHE_PREFIX = "ocr:cache:"

_client: redis.Redis | None = None


class QueueUnavailable(Exception):
    """Redis no está accesible: el modo asincrónico no puede operar."""


def get_client() -> redis.Redis:
    """Cliente Redis memoizado (la conexión es lazy: no falla al importar)."""
    global _client

    if _client is None:
        _client = redis.from_url(settings.REDIS_URL, decode_responses=True)

    return _client


async def ping() -> bool:
    try:
        return bool(await get_client().ping())
    except RedisError as exc:
        logger.warning("Redis no disponible: %s", exc)
        return False


async def close() -> None:
    global _client

    if _client is not None:
        await _client.aclose()
        _client = None


async def enqueue(
    image_url: str,
    doc_type: DocType,
    lang: str,
    token: str,
    discard_after: bool,
) -> str:
    """Registra un job en estado `queued` y lo encola. Devuelve el `job_id`."""
    job_id = uuid.uuid4().hex
    now = _now()

    state = OCRJobState(
        job_id=job_id,
        status=JobStatus.QUEUED,
        doc_type=doc_type,
        created_at=now,
        updated_at=now,
    )

    task = {
        "job_id": job_id,
        "image_url": image_url,
        "doc_type": doc_type.value,
        "lang": lang,
        "token": token,
        "discard_after": discard_after,
    }

    try:
        client = get_client()
        async with client.pipeline(transaction=True) as pipe:
            pipe.set(_JOB_PREFIX + job_id, state.model_dump_json(), ex=settings.JOB_TTL_SECONDS)
            pipe.set(_TASK_PREFIX + job_id, json.dumps(task), ex=settings.JOB_TTL_SECONDS)
            pipe.rpush(settings.QUEUE_KEY, job_id)
            await pipe.execute()
    except RedisError as exc:
        raise QueueUnavailable(str(exc)) from exc

    logger.info("Job encolado: %s (doc_type=%s)", job_id, doc_type.value)
    return job_id


async def dequeue(timeout: int = 5) -> dict | None:
    """Espera un job de la cola. Devuelve su payload interno o None por timeout."""
    try:
        popped = await get_client().blpop([settings.QUEUE_KEY], timeout=timeout)
    except RedisError as exc:
        raise QueueUnavailable(str(exc)) from exc

    if popped is None:
        return None

    _, job_id = popped
    raw = await get_client().get(_TASK_PREFIX + job_id)

    if raw is None:
        # El TTL venció antes de que el worker llegara al job.
        logger.warning("Job %s expiró antes de ser procesado", job_id)
        return None

    return json.loads(raw)


async def get_job(job_id: str) -> OCRJobState | None:
    try:
        raw = await get_client().get(_JOB_PREFIX + job_id)
    except RedisError as exc:
        raise QueueUnavailable(str(exc)) from exc

    return OCRJobState.model_validate_json(raw) if raw else None


async def mark_processing(job_id: str) -> None:
    await _patch(job_id, status=JobStatus.PROCESSING)


async def mark_done(job_id: str, result: OCRResult) -> None:
    await _patch(job_id, status=JobStatus.DONE, result=result)


async def mark_failed(job_id: str, error: str) -> None:
    await _patch(job_id, status=JobStatus.FAILED, error=error)


async def _patch(job_id: str, **changes) -> None:
    """Actualiza el estado público del job preservando su TTL."""
    state = await get_job(job_id)
    if state is None:
        logger.warning("No se pudo actualizar el job %s: no existe", job_id)
        return

    updated = state.model_copy(update={**changes, "updated_at": _now()})

    client = get_client()
    await client.set(
        _JOB_PREFIX + job_id,
        updated.model_dump_json(),
        keepttl=True,
    )

    if updated.status in (JobStatus.DONE, JobStatus.FAILED):
        # El payload interno (con el token) se descarta en cuanto deja de usarse.
        await client.delete(_TASK_PREFIX + job_id)


async def get_cached(fingerprint: str, doc_type: DocType, lang: str) -> OCRResult | None:
    """Resultado ya calculado para la misma imagen, tipo de documento e idioma."""
    try:
        raw = await get_client().get(_cache_key(fingerprint, doc_type, lang))
    except RedisError:
        return None  # La caché es una optimización: su ausencia no es un error.

    if raw is None:
        return None

    logger.info("Cache hit para la imagen %s…", fingerprint[:12])
    return OCRResult.model_validate_json(raw)


async def set_cached(
    fingerprint: str,
    doc_type: DocType,
    lang: str,
    result: OCRResult,
) -> None:
    try:
        await get_client().set(
            _cache_key(fingerprint, doc_type, lang),
            result.model_dump_json(),
            ex=settings.CACHE_TTL_SECONDS,
        )
    except RedisError as exc:
        logger.warning("No se pudo cachear el resultado: %s", exc)


def _cache_key(fingerprint: str, doc_type: DocType, lang: str) -> str:
    return f"{_CACHE_PREFIX}{doc_type.value}:{lang}:{fingerprint}"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
