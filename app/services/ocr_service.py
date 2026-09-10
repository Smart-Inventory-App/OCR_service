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
from app.parsers.barcode import SYMBOLOGY_NAMES
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
        # El ticket tampoco viene siempre derecho. Se suponía que sí —"se
        # fotografía apoyado y de frente"— y el conjunto de prueba mostró lo
        # contrario: `dia_2.jpg` es el mismo comprobante que `dia_1.jpg` sacado
        # al revés, y salía con los renglones invertidos y `SUBTOTAL` leído
        # `SUBTAL`. Era el único comprobante de 73 que fallaba.
        #
        # Como en la etiqueta, los giros van al final del plan: el ticket derecho
        # corta antes y no los paga.
        rotations = ROTATIONS if settings.RECEIPT_ROTATION_SEARCH_ENABLED else (0,)

        return _single_page_plan(supports_preprocessing, "block", rotations)

    # La etiqueta sí puede venir girada, al contrario de lo que se suponía. Sobre
    # el conjunto de prueba, las mismas 16 fotos giradas un cuarto de vuelta
    # daban un nombre distinto en los 16 casos, y el alta automática caía de 9 a
    # 4. Los giros van al final del plan, así que la foto derecha no los paga.
    rotations = ROTATIONS if settings.LABEL_ROTATION_SEARCH_ENABLED else (0,)

    return _single_page_plan(supports_preprocessing, "sparse", rotations)


def _single_page_plan(
    supports_preprocessing: bool,
    primary_layout: str,
    rotations: tuple[int, ...] = (0,),
) -> list[Attempt]:
    """Etiquetas y tickets: se varían los filtros y, al final, la orientación.

    `rotations` va **después** de todas las variantes de filtros y no antes: el
    plan corta en cuanto una lectura alcanza `RETRY_THRESHOLD`, así que una foto
    derecha —el caso normal— nunca llega a pagar los giros. Sólo cuando la
    lectura a 0° no sirve se prueba girar, que es exactamente cuando conviene.
    """
    base = transform.label()

    if not supports_preprocessing:
        return [Attempt(transform.geometric().turned(deg), primary_layout) for deg in rotations]

    if not settings.RETRY_ENABLED:
        return [Attempt(base, primary_layout)]

    plan = [
        Attempt(base, primary_layout),
        # Segmentación automática: rinde mejor en fotos con fondo y objetos.
        Attempt(base, "auto"),
        # Binarizada con Otsu: ayuda en envases de alto contraste.
        Attempt(base.binarized(transform.OTSU), "block"),
    ]

    plan += [
        Attempt(base.turned(degrees), primary_layout)
        for degrees in rotations
        if degrees != 0
    ]

    return plan


def _band_attempts(base: Transform, rotations: tuple[int, ...]) -> list[Attempt]:
    """Intentos que recortan a las primeras bandas de texto de la imagen.

    Se prueban varias bandas porque el motor no puede saber cuál es el sello: la
    etiqueta impresa también es texto oscuro sobre claro, y medido sobre las
    fotos reales la franja mejor puntuada resultó ser el teléfono de atención al
    consumidor y no la fecha. Probar las primeras y dejar que el scoring elija es
    más barato y más confiable que afinar una heurística de discriminación.

    El giro se acota a las dos orientaciones más frecuentes: el recorte busca
    franjas horizontales, así que en una foto muy girada no encuentra la del
    sello y probar las cuatro sería gastar de más. Las otras dos las cubre igual
    el plan sobre la imagen completa, que va después.

    Sólo se le ofrecen al motor que prepara su propia entrada. Medido sobre los
    94 sellos del conjunto: los recortes del motor que depende de los filtros
    acertaron 0, y los del motor de aprendizaje profundo resolvieron 13 de los
    37 vencimientos correctos. Aislar el sello ayuda a quien puede leerlo.
    """
    if settings.EXPIRY_CROP_BANDS <= 0:
        return []

    return [
        Attempt(base.turned(degrees).cropped(banda), "block")
        for degrees in rotations[:2]
        for banda in range(settings.EXPIRY_CROP_BANDS)
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
        return _band_attempts(stamp_res, rotations) + [
            Attempt(stamp_res.turned(deg), "block") for deg in rotations
        ]

    # Al motor que depende de la cadena de filtros **no se le recorta**, y eso
    # está medido: sobre los 94 sellos del conjunto, sus intentos con recorte
    # acertaron 0. Un sello de matriz de puntos no se le vuelve legible por
    # aislarlo; le falta capacidad de reconocimiento, no encuadre. Eran seis
    # intentos por documento que nunca ganaban.
    #
    # El recorte sí paga en los motores de aprendizaje profundo —resolvió 13 de
    # los 37 vencimientos acertados— y por eso vive en la otra rama.
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
        # Un código de barras no tiene "texto"; lo legible es qué se decodificó y
        # en qué simbología, que es lo que una persona puede contrastar con la
        # foto mirando los dígitos impresos debajo de las barras.
        texto_legible=" · ".join(
            f"{symbol.data} ({SYMBOLOGY_NAMES.get(symbol.symbology, symbol.symbology)})"
            for symbol in reading.symbols
        ),
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
        # Degradar en silencio es lo peor que puede hacer acá: el resultado sale
        # con el nombre del motor barato y nada indica que el bueno no corrió.
        # Sobre el conjunto de prueba eso pasó en 24 documentos, que parecían
        # medidos y no lo estaban. La advertencia viaja en la respuesta.
        logger.warning(
            "Confianza %.2f bajo el umbral pero el respaldo '%s' no está disponible",
            primary_result.document_confidence,
            hybrid.fallback.name,
        )
        primary_result.warnings.append(
            f"respaldo_no_disponible:{hybrid.fallback.name} — la lectura se resolvió "
            f"sólo con '{hybrid.primary.name}'; el resultado puede mejorar cuando el "
            f"motor de respaldo vuelva a estar en línea"
        )
        return primary_result, primary_attempts

    fallback_plan = attempt_plan(doc_type, _supports_preprocessing(hybrid.fallback))

    # La orientación del principal se hereda **sólo si su lectura fue creíble**.
    #
    # Antes alcanzaba con que la confianza fuera mayor que cero, y eso resultó
    # ser casi cualquier cosa: sobre un ticket fotografiado al revés, Tesseract
    # devuelve basura a 0° que igual arroja una confianza chica pero positiva, el
    # coordinador daba esa orientación por buena y le concedía al respaldo un
    # único intento… a 0°. PaddleOCR lee ese mismo ticket sin problemas a 180°,
    # pero nunca llegaba a probarlo. Era el único comprobante de 73 que fallaba.
    #
    # Con el umbral por documento como piso, una lectura que no llega a ser
    # plausible deja de contar como pista de orientación y el respaldo busca.
    if primary_result.document_confidence >= settings.DOCUMENT_CONFIDENCE_THRESHOLD:
        fallback_plan = [
            Attempt(attempt.transform.turned(primary_result.rotation), attempt.layout)
            for attempt in fallback_plan[:1]
        ]
    else:
        # El principal no reconoció nada, así que no hay orientación que
        # aprovechar y habría que dejar buscar al respaldo. Pero es el motor más
        # caro y este es el peor caso: una inferencia por orientación sobre la
        # imagen que ya venció al motor barato. Medido, es el camino que agota la
        # memoria del contenedor.
        #
        # Se acota por ORIENTACIÓN y no por cantidad de intentos: cortar la lista
        # a secas dejaría dos variantes de filtro del mismo giro, que es
        # justamente lo que no hace falta. `ROTATIONS` está ordenado por
        # rendimiento, así que las primeras son la foto tal como vino y el cuarto
        # de vuelta con que llega el sello.
        permitidas = set(ROTATIONS[: settings.HYBRID_FALLBACK_MAX_ROTATIONS])
        fallback_plan = [a for a in fallback_plan if a.rotation in permitidas]

    try:
        fallback_result, fallback_attempts = _run_plan(
            hybrid.fallback, fallback_plan, content, doc_type, lang, today
        )
    except EngineUnavailable as exc:
        # El respaldo se cayó **mientras** corría —típicamente se queda sin
        # memoria a mitad de la inferencia—. Antes eso propagaba el error y se
        # perdía también la lectura del principal, que estaba hecha y era
        # utilizable: sobre el conjunto de prueba, 21 documentos terminaron en
        # 502 teniendo una lectura válida en la mano. Se conserva lo que hay y se
        # deja dicho que el resultado está incompleto.
        logger.warning("El respaldo '%s' falló durante la corrida: %s", hybrid.fallback.name, exc)
        primary_result.warnings.append(
            f"respaldo_fallo:{hybrid.fallback.name} — se cayó durante el procesamiento; "
            f"la lectura es sólo de '{hybrid.primary.name}' y puede mejorar al reintentar"
        )
        return primary_result, primary_attempts

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


def _good_enough_to_stop(result: OCRResult, doc_type: DocType) -> bool:
    """Si la lectura ya alcanza y no vale la pena gastar más intentos.

    Para el sello y el ticket alcanza con superar el umbral de confianza. Para la
    etiqueta se pide además que no quede en revisión, porque ahí el umbral solo
    se satisface con lecturas equivocadas: sobre una foto girada el motor
    devuelve texto de todos modos, el parser le encuentra una categoría y el plan
    se corta antes de probar la orientación que sí servía.
    """
    if result.document_confidence < settings.RETRY_THRESHOLD:
        return False

    if doc_type in (DocType.PRODUCT_LABEL, DocType.RECEIPT):
        # Mismo motivo en los dos: sobre una foto girada el motor devuelve texto
        # igual, el parser le encuentra algo y el plan se corta antes de probar la
        # orientación que servía.
        return not result.needs_review

    return True


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
        #
        # Con una excepción, y está medida: en una etiqueta este corte impedía
        # que se llegara a probar el giro. Una foto rotada devuelve texto igual
        # —el motor lee *algo* en cualquier orientación— y si ese algo pasa el
        # umbral, el plan se detiene y nunca prueba la orientación correcta. El
        # resultado era que la misma imagen derecha y girada daba un nombre
        # distinto en los 16 pares del conjunto. Para la etiqueta se exige, para
        # cortar, que la lectura además **no necesite revisión**: es el mismo
        # criterio con el que el híbrido decide si vale la pena seguir gastando.
        if _good_enough_to_stop(best, doc_type):
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
        texto_legible=postprocess.readable(text, doc_type.value),
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

    Un comprobante del que no se extrajo ningún ítem no sirve para cargar
    inventario sin confirmación. Lo mismo un ítem cuyo **nombre** no se leyó con
    confianza: sin saber qué producto es, la línea no se puede dar de alta.

    La **cantidad no se exige**, y es un cambio que salió de medir. Un ticket
    argentino no imprime la cantidad cuando es 1: el parser la infiere y le pone
    0,6 de confianza justamente para declarar que la infirió. Exigirle 0,85
    mandaba a revisión manual el 100 % de los tickets reales del conjunto de
    prueba —los tres, con los nombres leídos entre 0,98 y 0,99 y el total al
    centavo— por un dato que el parser acierta casi siempre y que el usuario
    corrige de un toque si hace falta. Bloquear el alta por eso no protegía de
    nada; sólo garantizaba que la carga por ticket nunca fuera automática.
    """
    if doc_type is not DocType.RECEIPT:
        return False

    if not items:
        return True

    threshold = settings.FIELD_CONFIDENCE_THRESHOLD

    return any(item.nombre_producto.confidence < threshold for item in items)


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
