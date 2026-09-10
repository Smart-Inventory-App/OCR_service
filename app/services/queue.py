"""Cola de trabajos y caché de resultados sobre Redis.

La propuesta pide "considerar procesamiento asincrónico para evitar bloqueos en
la experiencia de usuario" (sección *Riesgos Técnicos*) y ya incluye Redis en el
stack para colas y caché.

`POST /ocr/jobs` crea el registro y encola el id; `worker.py` lo consume con
`BLPOP`; `GET /ocr/jobs/{id}` lee el estado. Se usan listas de Redis en lugar de
RQ para no depender de `fork()` (no disponible en Windows) y para que el mismo
código funcione igual en desarrollo y en Docker.

Cada job se guarda en dos claves separadas:

- `ocr:job:{id}`   — estado **público** que devuelve la API.
- `ocr:task:{id}`  — payload **interno** (incluye el token del llamador, que se
  necesita para descartar la imagen temporal y nunca debe salir por la API).
- `ocr:owner:{id}` — quién lo encoló. Sobrevive al `task`, que se borra al
  terminar, porque hay que poder decidir si quien consulta el resultado es el
  mismo que sacó la foto.
"""

import json
import logging
import uuid
from datetime import datetime, timezone

import redis.asyncio as redis
from redis.exceptions import RedisError
from redis.exceptions import TimeoutError as RedisTimeout

from app.core.config import settings
from app.schemas.ocr import DocType, JobStatus, OCRJobState, OCRResult

logger = logging.getLogger(__name__)

_JOB_PREFIX = "ocr:job:"
_TASK_PREFIX = "ocr:task:"
_OWNER_PREFIX = "ocr:owner:"
_RATE_PREFIX = "ocr:rate:"
_CACHE_PREFIX = "ocr:cache:"

_client: redis.Redis | None = None


class QueueUnavailable(Exception):
    """Redis no está accesible: el modo asincrónico no puede operar."""


def get_client() -> redis.Redis:
    """Cliente Redis memoizado (la conexión es lazy: no falla al importar)."""
    global _client

    if _client is None:
        _client = redis.from_url(
            settings.REDIS_URL,
            decode_responses=True,
            # El `socket_timeout` explícito no es cosmético: sin él, redis-py
            # deriva el plazo de lectura del propio timeout de `BLPOP`, sin
            # margen. Con una sola espera bloqueante la respuesta llega justo a
            # tiempo, pero con varias en paralelo —que es lo que hace el worker—
            # la latencia del loop empuja todas las lecturas más allá del plazo y
            # las tres fallan con `TimeoutError` a la vez. Medido y reproducido.
            socket_timeout=settings.REDIS_SOCKET_TIMEOUT_SECONDS,
            socket_keepalive=True,
        )

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
    engine: str | None = None,
    owner: str = "",
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
        "engine": engine,
    }

    try:
        client = get_client()
        async with client.pipeline(transaction=True) as pipe:
            pipe.set(_JOB_PREFIX + job_id, state.model_dump_json(), ex=settings.JOB_TTL_SECONDS)
            pipe.set(_TASK_PREFIX + job_id, json.dumps(task), ex=settings.JOB_TTL_SECONDS)
            pipe.set(_OWNER_PREFIX + job_id, owner, ex=settings.JOB_TTL_SECONDS)
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
    except RedisTimeout:
        # Una espera bloqueante que vence no es una caída: es que no había
        # trabajo. Tratarla como caída haría que el worker durmiera y dejara de
        # consumir la cola justo cuando llegan jobs.
        return None
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


async def depth() -> int:
    """Cuántos jobs esperan en la cola (los que ya se están procesando no cuentan)."""
    try:
        return int(await get_client().llen(settings.QUEUE_KEY))
    except RedisError as exc:
        raise QueueUnavailable(str(exc)) from exc


async def position_of(job_id: str) -> int | None:
    """Lugar en la fila, empezando en 1. `None` si ya salió de la cola."""
    try:
        index = await get_client().lpos(settings.QUEUE_KEY, job_id)
    except RedisError:
        # Saber el lugar es una cortesía para la UI, no un dato del que dependa
        # el procesamiento: si Redis no contesta esto, el job sigue igual.
        return None

    return None if index is None else int(index) + 1


async def consume_quota(owner: str, doc_type: DocType, limit: int) -> bool:
    """Descuenta un escaneo del cupo del usuario. False si ya lo agotó.

    Ventana fija de un minuto: la clave lleva el minuto adentro y expira sola.
    Permite gastar todo el cupo de golpe, que es lo que necesita una ráfaga
    legítima —un comercio escaneando una entrega— y sigue acotando el total.
    """
    if limit <= 0:
        return True

    minute = datetime.now(timezone.utc).strftime("%Y%m%d%H%M")
    key = f"{_RATE_PREFIX}{owner}:{doc_type.value}:{minute}"

    try:
        client = get_client()
        async with client.pipeline(transaction=True) as pipe:
            pipe.incr(key)
            pipe.expire(key, 120)
            used, _ = await pipe.execute()
    except RedisError as exc:
        # Sin Redis no hay forma de contar. Dejar pasar es preferible a bloquear
        # a todo el mundo por una falla de la caché de cuotas.
        logger.warning("No se pudo contabilizar el cupo de %s: %s", owner, exc)
        return True

    return int(used) <= limit


async def owner_of(job_id: str) -> str | None:
    """Quién encoló el job. `None` si no quedó registrado o ya expiró."""
    try:
        return await get_client().get(_OWNER_PREFIX + job_id)
    except RedisError as exc:
        raise QueueUnavailable(str(exc)) from exc


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
