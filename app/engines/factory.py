"""Selección del motor OCR según configuración.

`OCR_ENGINE` en `.env` define el motor: `tesseract` (default del MVP), `easyocr`
o `hybrid`. El modo híbrido implementa el "enfoque híbrido" que la propuesta no
descarta: motor principal + motor de respaldo cuando la confianza no alcanza el
umbral.
"""

import logging

import numpy as np

from app.core.config import settings
from app.engines.base import OCREngine, RawOCROutput
from app.engines.easyocr_engine import EasyOCREngine
from app.engines.tesseract_engine import TesseractEngine

logger = logging.getLogger(__name__)

_BUILDERS = {
    TesseractEngine.name: TesseractEngine,
    EasyOCREngine.name: EasyOCREngine,
}

_cache: dict[str, OCREngine] = {}


class HybridEngine:
    """Corre el motor principal y reintenta con el de respaldo si hace falta.

    El criterio de reintento es la confianza media del motor: si queda por
    debajo de `HYBRID_FALLBACK_THRESHOLD`, se prueba el respaldo y se conserva
    la salida más confiable de las dos.
    """

    def __init__(
        self,
        primary: OCREngine,
        fallback: OCREngine,
        threshold: float,
    ) -> None:
        self.primary = primary
        self.fallback = fallback
        self.threshold = threshold
        self.name = f"hybrid({primary.name}+{fallback.name})"

    def is_available(self) -> bool:
        return self.primary.is_available()

    def run(
        self,
        image: np.ndarray,
        lang: str = "spa",
        layout: str = "block",
    ) -> RawOCROutput:
        result = self.primary.run(image, lang, layout)

        if result.mean_confidence >= self.threshold:
            return result

        if not self.fallback.is_available():
            logger.warning(
                "Confianza baja (%.2f) pero el motor de respaldo '%s' no está disponible",
                result.mean_confidence,
                self.fallback.name,
            )
            return result

        logger.info(
            "Confianza %.2f < %.2f: reintentando con '%s'",
            result.mean_confidence,
            self.threshold,
            self.fallback.name,
        )
        retry = self.fallback.run(image, lang, layout)

        return retry if retry.mean_confidence > result.mean_confidence else result


def build_engine(name: str | None = None) -> OCREngine:
    """Devuelve el motor pedido (memoizado: cargar modelos es costoso)."""
    key = (name or settings.OCR_ENGINE).strip().lower()

    if key not in _cache:
        _cache[key] = _build(key)

    return _cache[key]


def _build(key: str) -> OCREngine:
    if key == "hybrid":
        fallback = settings.HYBRID_FALLBACK_ENGINE.strip().lower()
        if fallback not in _BUILDERS:
            raise ValueError(
                f"HYBRID_FALLBACK_ENGINE='{fallback}' debe ser un motor concreto: "
                f"{', '.join(sorted(_BUILDERS))}"
            )
        return HybridEngine(
            primary=build_engine(TesseractEngine.name),
            fallback=build_engine(fallback),
            threshold=settings.HYBRID_FALLBACK_THRESHOLD,
        )

    builder = _BUILDERS.get(key)
    if builder is None:
        valid = ", ".join([*sorted(_BUILDERS), "hybrid"])
        raise ValueError(f"Motor OCR '{key}' desconocido. Motores válidos: {valid}")

    return builder()


def reset_cache() -> None:
    """Descarta los motores memoizados (usado en tests y en el benchmark)."""
    _cache.clear()
