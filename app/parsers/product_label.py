"""Caso A de la propuesta: etiqueta de producto.

Campos clave: `nombre_producto` y `fecha_vencimiento`.
"""

import re
from datetime import date

from app.engines.base import OCRWord, RawOCROutput
from app.parsers import dates, scoring
from app.parsers.text import fold, line_text
from app.schemas.ocr import ExtractedField

KEY_FIELDS = ("nombre_producto", "fecha_vencimiento")

# Líneas que nunca son el nombre del producto.
_NOISE_TOKENS = (
    "vto", "vence", "vencimiento", "cons pref", "lote", "lot", "ind arg",
    "industria argentina", "sin tacc", "cuit", "ean", "peso neto", "cont neto",
)

_DATE_LIKE = re.compile(r"\d{1,4}[/\-.]\d{1,2}")
_MOSTLY_DIGITS = re.compile(r"^[\W\d]+$")

# Un vencimiento razonable cae dentro de estos años respecto de hoy.
_MAX_YEARS_AHEAD = 10


def parse(output: RawOCROutput, text: str, today: date | None = None) -> dict[str, ExtractedField]:
    """Extrae los campos clave de una etiqueta de producto."""
    today = today or date.today()

    return {
        "nombre_producto": _parse_name(output, text),
        "fecha_vencimiento": _parse_expiry(output, text, today),
    }


def _parse_name(output: RawOCROutput, text: str) -> ExtractedField:
    """Heurística de nombre: la línea con mayor altura de caja (tipografía más
    grande), que es la que las etiquetas usan para la denominación del producto.
    """
    candidates = [
        line for line in output.lines() if _is_name_candidate(line_text(line))
    ]

    if not candidates:
        return scoring.missing()

    best = max(candidates, key=_name_score)
    value = line_text(best)

    return scoring.field(
        value=value,
        engine_conf=scoring.engine_confidence(best),
        parser_conf=_name_parser_confidence(value, best, candidates),
    )


def _parse_expiry(output: RawOCROutput, text: str, today: date) -> ExtractedField:
    parsed = dates.find_expiry_date(text, today)
    if parsed is None:
        return scoring.missing()

    return scoring.field(
        value=parsed.iso,
        engine_conf=scoring.confidence_of_snippet(output, parsed.matched_text),
        parser_conf=_expiry_parser_confidence(parsed, text, today),
    )


def _expiry_parser_confidence(parsed: dates.ParsedDate, text: str, today: date) -> float:
    """Puntúa qué tan verosímil es la fecha como vencimiento."""
    confidence = 1.0

    # Una fecha anunciada por "VTO"/"VENCE" es mucho más segura que una suelta.
    if not dates.has_keyword_before(text, parsed, dates.EXPIRY_KEYWORDS):
        confidence *= 0.85

    # El día inferido (formato MM/AAAA) es una suposición del parser, pero es la
    # convención correcta para ese formato de etiqueta: la penalización es leve,
    # para no mandar a revisión manual una etiqueta bien leída.
    if not parsed.day_precision:
        confidence *= 0.95

    # Un vencimiento en el pasado suele ser en realidad la fecha de elaboración.
    if parsed.value < today:
        confidence *= 0.4

    # Fechas absurdamente lejanas indican dígitos mal leídos.
    if parsed.value.year > today.year + _MAX_YEARS_AHEAD:
        confidence *= 0.5

    return confidence


def _name_parser_confidence(
    value: str,
    line: list[OCRWord],
    candidates: list[list[OCRWord]],
) -> float:
    confidence = 1.0

    # Nombres muy cortos suelen ser fragmentos ("Le", "1L").
    if len(value) < 4:
        confidence *= 0.5
    elif len(value) < 8:
        confidence *= 0.85

    # Una sola palabra es un nombre plausible pero pobre.
    if len(value.split()) == 1:
        confidence *= 0.9

    # Si la mejor candidata no se despega del resto, la heurística de "línea más
    # grande" no está discriminando y el nombre elegido es menos confiable.
    if len(candidates) > 1:
        scores = sorted((_name_score(c) for c in candidates), reverse=True)
        if scores[0] > 0 and (scores[0] - scores[1]) / scores[0] < 0.15:
            confidence *= 0.9

    return confidence


def _name_score(line: list[OCRWord]) -> float:
    """Altura media de las cajas: aproxima el tamaño de la tipografía."""
    if not line:
        return 0.0
    return sum(w.height for w in line) / len(line)


def _is_name_candidate(value: str) -> bool:
    # Los puntos se colapsan a espacios para que "CONS. PREF" matchee "cons pref".
    normalized = fold(value).replace(".", " ")

    if len(normalized) < 3:
        return False
    if _MOSTLY_DIGITS.match(normalized):
        return False
    if _DATE_LIKE.search(normalized):
        return False

    return not any(token in normalized for token in _NOISE_TOKENS)

