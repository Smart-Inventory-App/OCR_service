"""Endpoints del servicio OCR.

Flujo previsto para la App (asincrónico, para no bloquear la UX):

    POST /ocr/scan       (multipart)  → sube la imagen a storage/temp y encola
    GET  /ocr/jobs/{id}               → polling hasta status=done
"""

import logging

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from app.core.config import settings
from app.core.security import get_bearer_token, get_current_user
from app.decoders import barcode as barcode_decoder
from app.schemas.ocr import (
    DocType,
    OCRJobCreated,
    OCRJobRequest,
    OCRJobState,
    OCRSyncResponse,
)
from app.engines.factory import build_engine
from app.engines.remote import EngineUnavailable
from app.services import image_loader, ocr_service, queue, storage_client

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ocr", tags=["ocr"])

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}


@router.post("/jobs", status_code=status.HTTP_202_ACCEPTED, response_model=OCRJobCreated)
async def create_job(
    body: OCRJobRequest,
    token: str = Depends(get_bearer_token),
    _user: dict = Depends(get_current_user),
):
    """Encola el procesamiento de una imagen ya subida (URL de Cloudinary)."""
    try:
        job_id = await queue.enqueue(
            image_url=str(body.image_url),
            doc_type=body.doc_type,
            lang=body.lang or settings.OCR_LANG,
            token=token,
            discard_after=body.discard_after,
            engine=_validate_engine(body.engine),
        )
    except queue.QueueUnavailable as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Cola de trabajos no disponible: {exc}",
        )

    return OCRJobCreated(job_id=job_id)


@router.post("/scan", status_code=status.HTTP_202_ACCEPTED, response_model=OCRJobCreated)
async def scan(
    file: UploadFile = File(...),
    doc_type: DocType = Form(default=DocType.PRODUCT_LABEL),
    lang: str | None = Form(default=None),
    engine: str | None = Form(default=None),
    token: str = Depends(get_bearer_token),
    _user: dict = Depends(get_current_user),
):
    """Punto de entrada de la App: sube la foto como temporal y encola el job.

    La imagen vive en `bazaar/ocr-temp` sólo mientras se la procesa: el worker la
    descarta al terminar (`discard_after=True`).
    """
    engine = _validate_engine(engine)
    content = await _read_image(file)

    try:
        image_url = await storage_client.upload_temp(
            content=content,
            filename=file.filename or "scan.jpg",
            content_type=file.content_type or "image/jpeg",
            token=token,
        )
    except storage_client.StorageError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"No se pudo subir la imagen temporal: {exc}",
        )

    try:
        job_id = await queue.enqueue(
            image_url=image_url,
            doc_type=doc_type,
            lang=lang or settings.OCR_LANG,
            token=token,
            discard_after=True,
            engine=engine,
        )
    except queue.QueueUnavailable as exc:
        # Sin cola el job no se procesará: la imagen temporal se descarta ya.
        await storage_client.discard_temp(image_url, token)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Cola de trabajos no disponible: {exc}",
        )

    return OCRJobCreated(job_id=job_id)


@router.get("/jobs/{job_id}", response_model=OCRJobState)
async def get_job(
    job_id: str,
    _user: dict = Depends(get_current_user),
):
    """Estado y resultado del job. La App hace polling contra este endpoint."""
    try:
        state = await queue.get_job(job_id)
    except queue.QueueUnavailable as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Cola de trabajos no disponible: {exc}",
        )

    if state is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job no encontrado o expirado",
        )

    return state


@router.post("/sync", response_model=OCRSyncResponse)
async def sync_from_url(
    body: OCRJobRequest,
    token: str = Depends(get_bearer_token),
    _user: dict = Depends(get_current_user),
):
    """Procesa la imagen y devuelve el resultado en la misma respuesta.

    Pensado para el benchmark, el backoffice y debugging. **No usar desde la App**:
    bloquea la UX mientras corre el OCR.
    """
    image_url = str(body.image_url)

    try:
        content = await image_loader.from_url(image_url)
        image_loader.validate(content)
    except image_loader.ImageLoadError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))

    result = await _process_cached(
        content, body.doc_type, body.lang or settings.OCR_LANG,
        _validate_engine(body.engine),
    )

    if body.discard_after:
        await storage_client.discard_temp(image_url, token)

    return OCRSyncResponse(doc_type=body.doc_type, result=result)


@router.post("/sync/upload", response_model=OCRSyncResponse)
async def sync_from_upload(
    file: UploadFile = File(...),
    doc_type: DocType = Form(default=DocType.PRODUCT_LABEL),
    lang: str | None = Form(default=None),
    engine: str | None = Form(default=None),
    _user: dict = Depends(get_current_user),
):
    """Procesa un binario subido directamente, sin pasar por el storage.

    Es la vía standalone: permite ejercitar el pipeline completo sin Cloudinary
    ni Redis (la caché se saltea si Redis no está disponible).
    """
    engine = _validate_engine(engine)
    content = await _read_image(file)
    result = await _process_cached(content, doc_type, lang or settings.OCR_LANG, engine)

    return OCRSyncResponse(doc_type=doc_type, result=result)


async def _process_cached(
    content: bytes,
    doc_type: DocType,
    lang: str,
    engine: str | None = None,
):
    """Procesa la imagen reutilizando el resultado cacheado por hash si existe."""
    fingerprint = image_loader.fingerprint(content)
    # El motor forma parte de la clave: comparar motores no debe devolver caché.
    cache_lang = f"{lang}:{engine}" if engine else lang

    cached = await queue.get_cached(fingerprint, doc_type, cache_lang)
    if cached is not None:
        return cached

    try:
        result = await ocr_service.process(content, doc_type, lang, engine=engine)
    except EngineUnavailable as exc:
        # Ningún intento llegó a correr: es una falla de infraestructura, no una
        # lectura pobre. Devolver un resultado vacío la disfrazaría.
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"El motor OCR no está disponible: {exc}",
        )
    except barcode_decoder.DecoderUnavailable as exc:
        # Mismo criterio que arriba, pero la falla es local: falta `libzbar` en
        # esta imagen. Un 503 lo distingue del 502 de un motor remoto caído.
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        )

    await queue.set_cached(fingerprint, doc_type, cache_lang, result)
    return result


def _validate_engine(name: str | None) -> str | None:
    """Rechaza un motor desconocido con 400 en vez de fallar al procesar."""
    if name is None:
        return None

    try:
        build_engine(name)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))

    return name


async def _read_image(file: UploadFile) -> bytes:
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        accepted = ", ".join(sorted(ALLOWED_CONTENT_TYPES))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Formato '{file.content_type}' no permitido. Formatos aceptados: {accepted}",
        )

    content = await file.read()
    if not content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="El archivo está vacío",
        )

    # Se valida acá y no en el motor para devolver un 400 claro antes de encolar
    # un trabajo que iba a fallar igual.
    try:
        image_loader.validate(content)
    except image_loader.ImageLoadError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))

    return content
