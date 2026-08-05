"""Orquestación del pipeline OCR.

Orden definido en la guía de implementación:

    cargar → preprocesar (OpenCV) → OCR (motor) → parsear → puntuar confianza

Es el único punto donde se combinan los cuatro módulos, y lo comparten tanto la
vía sincrónica (`POST /ocr/sync`) como el worker asincrónico.
"""

import asyncio
import logging
import time
from datetime import date

from app.core.config import settings
from app.engines.base import RawOCROutput
from app.engines.factory import build_engine
from app.parsers import product_label, receipt, scoring
from app.pipeline import postprocess, preprocess
from app.schemas.ocr import DocType, ExtractedField, OCRResult, ReceiptItem
from app.services.image_loader import decode

logger = logging.getLogger(__name__)

# Pista de layout para el motor, según el tipo de documento (ver engines/base).
_LAYOUT_BY_DOC_TYPE = {
    DocType.PRODUCT_LABEL: "sparse",  # etiquetas: texto disperso
    DocType.RECEIPT: "block",         # tickets: columna única de texto
}


def layout_for(doc_type: DocType) -> str:
    """Pista de layout que se le pasa al motor para este tipo de documento."""
    return _LAYOUT_BY_DOC_TYPE[doc_type]


async def process(
    content: bytes,
    doc_type: DocType,
    lang: str | None = None,
    today: date | None = None,
) -> OCRResult:
    """Procesa una imagen y devuelve el resultado ya puntuado.

    El OCR es trabajo CPU-bound (OpenCV + Tesseract): se ejecuta en un thread
    aparte para no bloquear el event loop de FastAPI ni del worker.
    """
    return await asyncio.to_thread(process_sync, content, doc_type, lang, today)


def process_sync(
    content: bytes,
    doc_type: DocType,
    lang: str | None = None,
    today: date | None = None,
) -> OCRResult:
    """Versión bloqueante del pipeline (usada por `process` y por el benchmark)."""
    started = time.perf_counter()
    language = lang or settings.OCR_LANG

    image = decode(content)
    processed = preprocess.run(image)

    engine = build_engine()
    output = engine.run(processed, lang=language, layout=layout_for(doc_type))

    text = postprocess.clean(output.text)
    result = build_result(output, text, doc_type, today)
    result.processing_ms = int((time.perf_counter() - started) * 1000)

    logger.info(
        "OCR completado: doc_type=%s engine=%s confianza=%.2f needs_review=%s (%d ms)",
        doc_type.value,
        result.engine,
        result.document_confidence,
        result.needs_review,
        result.processing_ms,
    )

    return result


def build_result(
    output: RawOCROutput,
    text: str,
    doc_type: DocType,
    today: date | None = None,
) -> OCRResult:
    """Aplica el parser del tipo de documento y calcula las métricas de confianza."""
    fields, items, key_fields = _parse(output, text, doc_type, today)

    return OCRResult(
        fields=fields,
        items=items,
        document_confidence=scoring.document_confidence(fields),
        needs_review=scoring.needs_review(fields, key_fields) or _items_need_review(items, doc_type),
        engine=output.engine,
        raw_text=text,
    )


def _parse(
    output: RawOCROutput,
    text: str,
    doc_type: DocType,
    today: date | None,
) -> tuple[dict[str, ExtractedField], list[ReceiptItem], tuple[str, ...]]:
    if doc_type is DocType.RECEIPT:
        fields, items = receipt.parse(output, text, today)
        return fields, items, receipt.KEY_FIELDS

    return product_label.parse(output, text, today), [], product_label.KEY_FIELDS


def _items_need_review(items: list[ReceiptItem], doc_type: DocType) -> bool:
    """En un ticket, la revisión también se dispara por los ítems.

    Un comprobante del que no se extrajo ningún ítem, o alguno de cuyos ítems no
    alcanza el umbral por campo, no sirve para cargar inventario sin confirmación.
    """
    if doc_type is not DocType.RECEIPT:
        return False

    if not items:
        return True

    threshold = settings.FIELD_CONFIDENCE_THRESHOLD

    return any(
        item.nombre_producto.confidence < threshold or item.cantidad.confidence < threshold
        for item in items
    )


def engine_available() -> bool:
    """Para `readyz`: el motor configurado puede ejecutarse."""
    try:
        return build_engine().is_available()
    except Exception as exc:
        logger.error("No se pudo construir el motor OCR: %s", exc)
        return False
