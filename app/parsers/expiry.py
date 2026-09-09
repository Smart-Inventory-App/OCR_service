"""Segundo caso de entrada: la foto del sello de vencimiento sobre el envase.

Extrae un único campo, `fecha_vencimiento`, y por eso puede permitirse ser mucho
más exigente que cuando ese dato competía con el nombre del producto: acá no hay
nada más que buscar en la imagen, así que toda la confianza del documento
depende de esa fecha y una lectura dudosa manda directo a revisión manual.
"""

from datetime import date

from app.engines.base import RawOCROutput
from app.parsers import dates, scoring
from app.schemas.ocr import ExtractedField

KEY_FIELDS = ("fecha_vencimiento",)

# Un vencimiento razonable cae dentro de estos años respecto de hoy.
_MAX_YEARS_AHEAD = 10


def parse(
    output: RawOCROutput, text: str, today: date | None = None
) -> dict[str, ExtractedField]:
    """Extrae la fecha de vencimiento del sello."""
    today = today or date.today()

    return {"fecha_vencimiento": _parse_expiry(output, text, today)}


def _parse_expiry(output: RawOCROutput, text: str, today: date) -> ExtractedField:
    parsed = dates.find_expiry_date(text, today)
    if parsed is None:
        return scoring.missing()

    return scoring.field(
        value=parsed.iso,
        engine_conf=scoring.confidence_of_snippet(output, parsed.matched_text),
        parser_conf=parser_confidence(parsed, text, today),
    )


def parser_confidence(parsed: dates.ParsedDate, text: str, today: date) -> float:
    """Puntúa qué tan verosímil es la fecha como vencimiento."""
    confidence = 1.0

    # Una fecha anunciada por "VTO"/"CONSUMIR PREFERENTEMENTE" es mucho más
    # segura que una suelta: en un envase hay números por todos lados (lote,
    # hora, código de barras) y sin el anuncio no hay forma de saber cuál es.
    if not dates.announced_as_expiry(text, parsed):
        confidence *= 0.85

    # El día inferido (formato MM/AAAA) es una suposición del parser, pero es la
    # convención correcta para ese formato: la penalización es leve.
    if not parsed.day_precision:
        confidence *= 0.95

    # El mes se resolvió por similitud: la lectura está reparada, no leída.
    if parsed.fuzzy_month:
        confidence *= 0.9

    # Un vencimiento en el pasado suele ser en realidad la fecha de elaboración.
    if parsed.value < today:
        confidence *= 0.4

    # Fechas absurdamente lejanas indican dígitos mal leídos.
    if parsed.value.year > today.year + _MAX_YEARS_AHEAD:
        confidence *= 0.5

    return confidence
