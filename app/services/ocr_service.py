"""Orquestación del pipeline OCR.

Es el módulo que justifica que este servicio exista aparte de los motores. Acá
vive **toda la estrategia**:

    validar → decidir qué intentos hacer → pedirle cada uno a un motor →
    limpiar → parsear → puntuar → quedarse con la mejor lectura

Los servicios de motor no saben qué documento están mirando, ni qué campos hay
que sacar, ni cuándo conviene reintentar. Reciben una imagen y una
transformación, devuelven palabras. Todo lo demás se decide desde acá.

Lo comparten la vía sincrónica (`POST /ocr/sync`) y el worker asincrónico.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import date

from app.core.config import settings
from app.decoders import barcode as barcode_decoder
from app.engines.base import RawOCROutput
from app.engines.factory import HybridEngine, build_engine
from app.engines.remote import EngineUnavailable
from app.parsers import barcode as barcode_parser
from app.parsers import expiry, product_label, receipt, scoring
from app.pipeline import postprocess, transform
from app.pipeline.transform import Transform
from app.schemas.ocr import DocType, ExtractedField, OCRResult, ReceiptItem

logger = logging.getLogger(__name__)

# Orden en el que se prueban las orientaciones. Primero la de la foto tal como
# vino; después el cuarto de vuelta en cada sentido, que es como quedan las fotos
# del sello de vencimiento cuando se apoya el envase de costado; y último el
# revés, que es el caso más raro.
ROTATIONS = (0, 90, 270, 180)


@dataclass(frozen=True)
class Attempt:
    """Una combinación concreta de preparación y segmentación a probar."""

    transform: Transform
    layout: str

    @property
    def rotation(self) -> int:
        return self.transform.rotate


def attempt_plan(doc_type: DocType, supports_preprocessing: bool = True) -> list[Attempt]:
    """Intentos a realizar, de mejor a peor apuesta.

    Ninguna combinación gana en todas las imágenes: medido sobre fotos reales,
    una etiqueta oscura y brillante, un envase mate y un sello de matriz de
    puntos favorecen tratamientos y segmentaciones distintas. En vez de elegir
    una a ciegas se prueban unas pocas en orden y se conserva la de mayor
    `document_confidence` — el camino de "reconfiguración del pipeline de
    procesamiento" que prevé la propuesta.

    `supports_preprocessing` lo declara cada motor: al que prepara su propia
    entrada no se le gastan variantes de filtros que va a ignorar.
    """
    if doc_type is DocType.BARCODE:
        # Un código de barras no se reconoce con un motor OCR, así que no hay
        # giro ni filtros que planificar. Su escalera de variantes es otra y vive
        # en `decoders/barcode.py`. Se falla explícito en vez de devolver el plan
        # de etiqueta, que correría el motor sobre una foto donde no hay texto.
        raise ValueError(
            "`barcode` no usa motor OCR: su lectura la resuelve decoders/barcode.py"
        )

    if doc_type is DocType.EXPIRY_DATE:
        return _expiry_plan(supports_preprocessing)

    if doc_type is DocType.RECEIPT:
        return _single_page_plan(supports_preprocessing, primary_layout="block")

    return _single_page_plan(supports_preprocessing, primary_layout="sparse")


def _single_page_plan(supports_preprocessing: bool, primary_layout: str) -> list[Attempt]:
    """Etiquetas y tickets: la foto viene derecha, lo que se varía son filtros."""
    base = transform.label()

    if not supports_preprocessing:
        return [Attempt(transform.geometric(), primary_layout)]

    if not settings.RETRY_ENABLED:
        return [Attempt(base, primary_layout)]

    return [
        Attempt(base, primary_layout),
        # Segmentación automática: rinde mejor en fotos con fondo y objetos.
        Attempt(base, "auto"),
        # Binarizada con Otsu: ayuda en envases de alto contraste.
        Attempt(base.binarized(transform.OTSU), "block"),
    ]


def _expiry_plan(supports_preprocessing: bool) -> list[Attempt]:
    """Sello de vencimiento: además de filtros, hay que buscar la orientación.

    Las fotos del sello llegan giradas un cuarto de vuelta porque el dato está
    impreso en un canto del envase y no hay forma cómoda de fotografiarlo
    derecho. Ningún motor lee texto girado 90°, y el detector de orientación de
    Tesseract (OSD) no sirve acá: sobre estas mismas fotos falló en la mitad de
    los casos y acertó con confianza menor a 3. Probar las cuatro y quedarse con
    la que lee sale más barato que confiar en él.
    """
    rotations = ROTATIONS if settings.ROTATION_SEARCH_ENABLED else (0,)

    if not supports_preprocessing:
        stamp_res = transform.geometric(settings.ENGINE_STAMP_MAX_WIDTH)
        return [Attempt(stamp_res.turned(deg), "block") for deg in rotations]

    plan: list[Attempt] = []
    for degrees in rotations:
        # Primero el umbral adaptativo, que es el que levanta los sellos de
        # matriz de puntos — el caso difícil y el más frecuente en un envase…
        plan.append(Attempt(transform.stamp_adaptive().turned(degrees), "block"))
        # …y después la cadena sin binarizar, que gana en los sellos impresos con
        # tinta continua y en las etiquetas adhesivas.
        plan.append(Attempt(transform.stamp().turned(degrees), "block"))

    return plan


async def process(
    content: bytes,
    doc_type: DocType,
    lang: str | None = None,
    today: date | None = None,
    engine: str | None = None,
) -> OCRResult:
    """Procesa una imagen y devuelve el resultado ya puntuado.

    La orquestación es una secuencia de llamadas HTTP bloqueantes: se ejecuta en
    un thread aparte para no trabar el event loop de FastAPI ni el del worker.
    """
    return await asyncio.to_thread(process_sync, content, doc_type, lang, today, engine)


def process_sync(
    content: bytes,
    doc_type: DocType,
    lang: str | None = None,
    today: date | None = None,
    engine_name: str | None = None,
) -> OCRResult:
    """Versión bloqueante del pipeline (usada por `process` y por el benchmark)."""
    started = time.perf_counter()
    language = lang or settings.OCR_LANG

    if doc_type is DocType.BARCODE:
        # No pasa por ningún motor: se decodifica acá. `engine_name` se ignora a
        # propósito y el resultado reporta `zbar`, para que en el banco de
        # pruebas se vea que la lectura no la hizo el motor que se pidió.
        best, attempts = _run_barcode(content, today)
    else:
        engine = build_engine(engine_name)

        if isinstance(engine, HybridEngine):
            best, attempts = _run_hybrid(engine, content, doc_type, language, today)
        else:
            best, attempts = _run_plan(
                engine,
                attempt_plan(doc_type, _supports_preprocessing(engine)),
                content,
                doc_type,
                language,
                today,
            )

    best.attempts = attempts
    best.processing_ms = int((time.perf_counter() - started) * 1000)

    logger.info(
        "OCR completado: doc_type=%s engine=%s confianza=%.2f rot=%d needs_review=%s "
        "(%d intento(s), %d ms)",
        doc_type.value,
        best.engine,
        best.document_confidence,
        best.rotation,
        best.needs_review,
        attempts,
        best.processing_ms,
    )

    return best


def _run_barcode(content: bytes, today: date | None) -> tuple[OCRResult, int]:
    """Lectura de código de barras: decodificar y puntuar, sin motor OCR.

    No hay estrategia de giro que decidir —ZBar resuelve la orientación por su
    cuenta— ni mejor lectura que elegir entre varios intentos: un código, o
    decodifica o no. La escalera de variantes vive en el decodificador y corta
    en la primera que encuentra algo.
    """
    reading = barcode_decoder.decode(content)
    fields = barcode_parser.parse(reading, today)

    result = OCRResult(
        fields=fields,
        items=[],
        document_confidence=scoring.document_confidence(fields),
        needs_review=scoring.needs_review(fields, barcode_parser.KEY_FIELDS),
        engine=barcode_decoder.NAME,
        # El texto crudo son los códigos tal como los devolvió el decodificador,
        # antes de normalizarlos: es lo que permite auditar desde el backoffice
        # que un `0`+UPC-A o un ITF-14 convertido salieron de lo que había.
        raw_text="\n".join(symbol.data for symbol in reading.symbols),
    )

    return result, max(reading.attempts, 1)


def _run_hybrid(
    hybrid: HybridEngine,
    content: bytes,
    doc_type: DocType,
    lang: str,
    today: date | None,
) -> tuple[OCRResult, int]:
    """El enfoque híbrido, orquestado con el contexto que el coordinador tiene.

    Correr los dos motores en cada intento sería la implementación obvia y la
    peor: sobre una foto girada multiplicaría por cuatro las corridas del motor
    caro. En cambio se busca la orientación con el motor barato —que cuesta
    medio segundo por intento— y al caro se le pide **una sola** corrida, ya en
    la orientación que se sabe buena.
    """
    primary_result, primary_attempts = _run_plan(
        hybrid.primary,
        attempt_plan(doc_type, _supports_preprocessing(hybrid.primary)),
        content,
        doc_type,
        lang,
        today,
    )

    # El criterio para no pagar el motor caro es el criterio de aceptación del
    # módulo, no un número aparte: si la lectura del motor barato ya sirve para
    # dar de alta el producto sin que el usuario confirme nada, no hay nada que
    # ganar. Mirar sólo la confianza global no alcanza — sobre una etiqueta,
    # Tesseract lee la marca con total nitidez y la ofrece como nombre del
    # producto: confianza alta y campo equivocado.
    if not primary_result.needs_review and primary_result.document_confidence >= hybrid.threshold:
        return primary_result, primary_attempts

    if not hybrid.fallback.is_available():
        logger.warning(
            "Confianza %.2f bajo el umbral pero el respaldo '%s' no está disponible",
            primary_result.document_confidence,
            hybrid.fallback.name,
        )
        return primary_result, primary_attempts

    fallback_plan = attempt_plan(doc_type, _supports_preprocessing(hybrid.fallback))

    # Si el principal reconoció algo, su orientación ya es información buena y le
    # ahorra al respaldo probar las otras tres. Si no reconoció nada, esa
    # información no existe y hay que dejarlo buscar.
    if primary_result.document_confidence > 0:
        fallback_plan = [
            Attempt(attempt.transform.turned(primary_result.rotation), attempt.layout)
            for attempt in fallback_plan[:1]
        ]

    fallback_result, fallback_attempts = _run_plan(
        hybrid.fallback, fallback_plan, content, doc_type, lang, today
    )

    attempts = primary_attempts + fallback_attempts

    if fallback_result.document_confidence > primary_result.document_confidence:
        logger.info(
            "Híbrido: gana '%s' (%.2f) sobre '%s' (%.2f)",
            hybrid.fallback.name,
            fallback_result.document_confidence,
            hybrid.primary.name,
            primary_result.document_confidence,
        )
        return fallback_result, attempts

    return primary_result, attempts


def _run_plan(
    engine,
    plan: list[Attempt],
    content: bytes,
    doc_type: DocType,
    lang: str,
    today: date | None,
) -> tuple[OCRResult, int]:
    """Ejecuta los intentos en orden y devuelve la mejor lectura."""
    best: OCRResult | None = None
    attempts = 0
    last_error: EngineUnavailable | None = None

    for attempt in plan:
        attempts += 1

        try:
            output = engine.run(
                content, lang=lang, layout=attempt.layout, transform=attempt.transform
            )
        except EngineUnavailable as exc:
            # Un motor caído no debe tumbar los intentos que aún pueden servir,
            # pero si fallan todos hay que decirlo y no devolver un vacío que
            # parezca una lectura legítima.
            logger.warning("Intento %d falló: %s", attempts, exc)
            last_error = exc
            continue

        result = build_result(output, postprocess.clean(output.text), doc_type, today)
        result.rotation = attempt.rotation

        if best is None or result.document_confidence > best.document_confidence:
            best = result

        # Ya alcanza para no molestar al usuario: no se gasta tiempo en más.
        if best.document_confidence >= settings.RETRY_THRESHOLD:
            break

    if best is None:
        if last_error is not None:
            raise last_error
        # Plan vacío: no debería ocurrir, pero devolver un resultado en revisión
        # es preferible a propagar un None.
        best = build_result(RawOCROutput(text="", engine=""), "", doc_type, today)

    return best, attempts


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
        needs_review=(
            scoring.needs_review(fields, key_fields) or _items_need_review(items, doc_type)
        ),
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

    if doc_type is DocType.EXPIRY_DATE:
        return expiry.parse(output, text, today), [], expiry.KEY_FIELDS

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


def _supports_preprocessing(engine) -> bool:
    """Si al motor le sirve que el coordinador le varíe la cadena de filtros.

    La respuesta la da el motor en `/engine/info`, no este servicio: es el único
    que sabe cómo prepara su entrada. El valor con el que se construye el
    `RemoteEngine` es sólo el default hasta que el servicio conteste, y la
    consulta es barata porque `info()` está cacheada.

    Un motor sin `refresh_capabilities` —el híbrido, o un doble de tests— se
    queda con el atributo que declara, y si el motor no contesta se conserva el
    default en vez de asumir nada.
    """
    refresh = getattr(engine, "refresh_capabilities", None)
    if refresh is not None:
        refresh()

    return bool(getattr(engine, "supports_preprocessing", True))


def engine_available(name: str | None = None) -> bool:
    """Para `readyz`: el motor configurado responde y puede reconocer."""
    try:
        return build_engine(name).is_available()
    except Exception as exc:
        logger.error("No se pudo construir el motor OCR: %s", exc)
        return False
