"""Cálculo de confianza por campo y por documento.

Implementa los criterios de aceptación de la propuesta:

- `field.confidence` combina la confianza del motor (por palabra) con la del
  parser (¿matcheó el patrón esperado?, ¿la fecha es válida y coherente?).
- `document_confidence` es el promedio ponderado de los campos clave y debe
  quedar ≥ `DOCUMENT_CONFIDENCE_THRESHOLD` (0.70).
- `needs_review` es True si algún campo clave queda por debajo de
  `FIELD_CONFIDENCE_THRESHOLD` (0.85), lo que dispara la corrección manual
  asistida en la App.
"""

import re

from app.core.config import settings
from app.engines.base import OCRWord, RawOCROutput
from app.schemas.ocr import ExtractedField

# Peso relativo de cada campo en `document_confidence`. Los campos que la
# propuesta nombra como clave pesan el doble que los que enriquecen el alta pero
# que el usuario puede completar de memoria.
FIELD_WEIGHTS: dict[str, float] = {
    "nombre_producto": 1.0,
    "fecha_vencimiento": 1.0,
    "fecha_compra": 1.0,
    "marca": 0.5,
    "detalles": 0.25,
    "comercio": 0.5,
    "total": 0.5,
    # Código de barras: el número es el campo clave y lo demás lo describe.
    # `simbologia` y `pais_registro` se derivan del mismo número ya verificado,
    # así que no aportan información independiente: pesan poco a propósito, para
    # que no inflen la confianza del documento repitiendo lo que ya dijo `codigo`.
    "codigo": 1.0,
    "simbologia": 0.25,
    "pais_registro": 0.25,
}

DEFAULT_WEIGHT = 0.5


def engine_confidence(words: list[OCRWord]) -> float:
    """Confianza del motor para un conjunto de palabras, ponderada por longitud."""
    scored = [w for w in words if w.text.strip()]
    if not scored:
        return 0.0

    weight = sum(len(w.text) for w in scored)
    if weight == 0:
        return 0.0

    return sum(w.confidence * len(w.text) for w in scored) / weight


def confidence_of_snippet(output: RawOCROutput, snippet: str) -> float:
    """Confianza del motor para las palabras que componen un fragmento de texto.

    Permite puntuar un campo con la confianza de las palabras que realmente lo
    formaron, en lugar de la confianza global del documento.
    """
    wanted = {piece for piece in re.split(r"\W+", snippet) if piece}
    if not wanted:
        return output.mean_confidence

    matching = [
        word for word in output.words if any(piece in word.text for piece in wanted)
    ]

    return engine_confidence(matching) if matching else output.mean_confidence


def combine(engine_conf: float, parser_conf: float) -> float:
    """Combina la confianza del motor con la del parser.

    Se usa el producto (y no un promedio) porque ambas condiciones son
    necesarias: un texto leído con nitidez pero que no matchea el patrón
    esperado no es un campo confiable, y viceversa.
    """
    return round(_clamp(engine_conf) * _clamp(parser_conf), 4)


def field(value, engine_conf: float, parser_conf: float) -> ExtractedField:
    """Construye un `ExtractedField` con la confianza ya combinada."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return ExtractedField(value=None, confidence=0.0)

    return ExtractedField(value=value, confidence=combine(engine_conf, parser_conf))


def missing() -> ExtractedField:
    return ExtractedField(value=None, confidence=0.0)


def document_confidence(fields: dict[str, ExtractedField]) -> float:
    """Promedio ponderado de los campos clave del documento."""
    if not fields:
        return 0.0

    total_weight = 0.0
    accumulated = 0.0

    for name, extracted in fields.items():
        weight = FIELD_WEIGHTS.get(name, DEFAULT_WEIGHT)
        total_weight += weight
        accumulated += extracted.confidence * weight

    if total_weight == 0:
        return 0.0

    return round(accumulated / total_weight, 4)


def needs_review(fields: dict[str, ExtractedField], key_fields: tuple[str, ...]) -> bool:
    """True si algún campo clave no alcanza el umbral por campo (0.85)."""
    threshold = settings.FIELD_CONFIDENCE_THRESHOLD

    return any(
        fields.get(name, missing()).confidence < threshold for name in key_fields
    )


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))
