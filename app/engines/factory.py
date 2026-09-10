"""Selección del motor OCR.

`OCR_ENGINE` en `.env` define el motor por defecto: `tesseract`, `paddleocr` o
`hybrid`. Los dos primeros son servicios aparte a los que se llega por HTTP; el
híbrido no lo es: es una **estrategia de este servicio** que combina un motor
principal con uno de respaldo, y por eso vive acá y no en ningún repositorio de
motor.

PaddleOCR está replicado y su URL admite varias separadas por coma: el
`RemoteEngine` las reparte por turno. Para el resto del servicio sigue siendo un
solo motor.
"""

import logging

from app.core.config import settings
from app.engines.base import OCREngine, RawOCROutput
from app.engines.remote import EngineUnavailable, RemoteEngine
from app.pipeline.transform import Transform

logger = logging.getLogger(__name__)

TESSERACT = "tesseract"
PADDLEOCR = "paddleocr"
HYBRID = "hybrid"

CONCRETE_ENGINES = (TESSERACT, PADDLEOCR)
VALID_ENGINES = (*CONCRETE_ENGINES, HYBRID)

_cache: dict[str, OCREngine] = {}


class HybridEngine:
    """El "enfoque híbrido" que la propuesta no descarta.

    Corre el motor principal y, si su lectura no alcanza el umbral, reintenta con
    el de respaldo y conserva la mejor de las dos. No es un motor: no reconoce
    nada por sí mismo. Es la decisión de cuándo vale la pena pagar el motor caro,
    y esa decisión es del coordinador.

    Expone `primary` y `fallback` porque `ocr_service` necesita orquestarlos con
    más contexto que el que tiene un `run()` suelto: sobre una foto girada, por
    ejemplo, conviene buscar la orientación con el motor barato y recién después
    pasarle al caro la orientación que ya se sabe buena.
    """

    name = HYBRID
    # El híbrido ya ES una estrategia de reintento: variarle los filtros por
    # fuera multiplicaría las corridas del motor caro.
    supports_preprocessing = False

    def __init__(self, primary: OCREngine, fallback: OCREngine, threshold: float) -> None:
        self.primary = primary
        self.fallback = fallback
        self.threshold = threshold

    def is_available(self) -> bool:
        """Alcanza con el principal: sin respaldo el híbrido degrada, no falla."""
        return self.primary.is_available()

    def run(
        self,
        content: bytes,
        lang: str = "spa",
        layout: str = "block",
        transform: Transform | None = None,
    ) -> RawOCROutput:
        """Un intento del híbrido: principal y, si hace falta, respaldo."""
        result = self.primary.run(content, lang, layout, transform)

        if result.mean_confidence >= self.threshold:
            return result

        try:
            retry = self.fallback.run(content, lang, layout, transform)
        except EngineUnavailable as exc:
            logger.warning(
                "Confianza baja (%.2f) pero el respaldo '%s' no respondió: %s",
                result.mean_confidence,
                self.fallback.name,
                exc,
            )
            return result

        logger.info(
            "Híbrido: %s dio %.2f, %s dio %.2f",
            self.primary.name,
            result.mean_confidence,
            self.fallback.name,
            retry.mean_confidence,
        )

        return retry if retry.mean_confidence > result.mean_confidence else result


def build_engine(name: str | None = None) -> OCREngine:
    """Devuelve el motor pedido (memoizado: guarda el estado del cliente HTTP)."""
    key = (name or settings.OCR_ENGINE).strip().lower()

    if key not in _cache:
        _cache[key] = _build(key)

    return _cache[key]


def _build(key: str) -> OCREngine:
    if key == HYBRID:
        return HybridEngine(
            primary=build_engine(settings.HYBRID_PRIMARY_ENGINE),
            fallback=build_engine(settings.HYBRID_FALLBACK_ENGINE),
            threshold=settings.HYBRID_FALLBACK_THRESHOLD,
        )

    if key == TESSERACT:
        return RemoteEngine(TESSERACT, settings.TESSERACT_SERVICE_URL, supports_preprocessing=True)

    if key == PADDLEOCR:
        return RemoteEngine(
            PADDLEOCR, settings.PADDLEOCR_SERVICE_URL, supports_preprocessing=False
        )

    raise ValueError(
        f"Motor OCR '{key}' desconocido. Motores válidos: {', '.join(VALID_ENGINES)}"
    )


def reset_cache() -> None:
    """Descarta los motores memoizados y cierra sus conexiones.

    La usan los tests, el benchmark (que reconstruye los motores entre corridas)
    y el apagado del servicio: un `RemoteEngine` mantiene viva la conexión con su
    motor, así que descartarlo sin cerrarla dejaría el socket colgado.
    """
    for engine in _cache.values():
        close = getattr(engine, "close", None)
        if close is not None:
            close()

    _cache.clear()
