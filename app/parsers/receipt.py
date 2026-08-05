"""Caso B de la propuesta: ticket, remito o factura.

Del documento se extraen `fecha_compra`, `total` y `comercio`; por ítem,
`nombre_producto` y `cantidad`. Habilita la "carga masiva y automatizada de
inventario" que pide la propuesta.
"""

import re
from datetime import date

from app.engines.base import OCRWord, RawOCROutput
from app.parsers import dates, scoring
from app.parsers.text import fold, line_text
from app.schemas.ocr import ExtractedField, ReceiptItem

KEY_FIELDS = ("fecha_compra",)

# Líneas de encabezado/pie que no son ítems.
_SKIP_TOKENS = (
    "total", "subtotal", "sub total", "iva", "descuento", "efectivo", "vuelto",
    "cambio", "tarjeta", "debito", "credito", "cuit", "ingresos brutos",
    "consumidor final", "factura", "ticket", "remito", "comprobante", "caja",
    "cajero", "gracias", "resp inscripto", "n de", "nro", "cae", "importe",
    "fecha", "hora", "emision",
)

# Una línea que contiene una fecha es metadata del comprobante, no un ítem.
_DATE_LIKE = re.compile(r"\d{1,4}[/\-.]\d{1,2}[/\-.]\d{2,4}")

_TOTAL_LINE = re.compile(r"\b(total|importe\s+total)\b", re.IGNORECASE)

# Números con formato monetario argentino: 1.234,56 · 1234,56 · 1234.56 · 1234
_AMOUNT = re.compile(r"\d{1,3}(?:\.\d{3})*(?:,\d{1,2})?|\d+(?:[.,]\d{1,2})?")

# «2 x Leche» · «2x Leche» · «2 Leche»
_QTY_PREFIX = re.compile(r"^(\d{1,3})\s*(?:x|X|\*|u|un|unid\.?)?\s+(.{3,})$")
# «Leche x 2» · «Leche 2 un»
_QTY_SUFFIX = re.compile(r"^(.{3,}?)\s+(?:x|X)\s*(\d{1,3})$")

_MIN_NAME_LENGTH = 3
_MAX_NAME_LENGTH = 60


def parse(
    output: RawOCROutput,
    text: str,
    today: date | None = None,
) -> tuple[dict[str, ExtractedField], list[ReceiptItem]]:
    """Extrae los campos del documento y la lista de ítems."""
    today = today or date.today()
    lines = output.lines()

    fields = {
        "fecha_compra": _parse_purchase_date(output, text, today),
        "total": _parse_total(lines),
        "comercio": _parse_merchant(lines),
    }

    return fields, _parse_items(lines)


def _parse_purchase_date(output: RawOCROutput, text: str, today: date) -> ExtractedField:
    parsed = dates.find_purchase_date(text, today)
    if parsed is None:
        return scoring.missing()

    confidence = 1.0
    if not dates.has_keyword_before(text, parsed, dates.PURCHASE_KEYWORDS):
        confidence *= 0.85
    if not parsed.day_precision:
        confidence *= 0.8
    # Una compra no puede ser futura: si lo es, se leyó mal.
    if parsed.value > today:
        confidence *= 0.4

    return scoring.field(
        value=parsed.iso,
        engine_conf=scoring.confidence_of_snippet(output, parsed.matched_text),
        parser_conf=confidence,
    )


def _parse_total(lines: list[list[OCRWord]]) -> ExtractedField:
    """Último importe de la línea que dice TOTAL (los tickets lo repiten al pie)."""
    for line in reversed(lines):
        text = line_text(line)
        if not _TOTAL_LINE.search(fold(text)):
            continue

        amounts = _AMOUNT.findall(text)
        if not amounts:
            continue

        value = _to_amount(amounts[-1])
        if value is None:
            continue

        # "SUBTOTAL" es un total parcial: menos confiable como total del documento.
        parser_conf = 0.7 if "subtotal" in fold(text).replace(" ", "") else 1.0

        return scoring.field(
            value=value,
            engine_conf=scoring.engine_confidence(line),
            parser_conf=parser_conf,
        )

    return scoring.missing()


def _parse_merchant(lines: list[list[OCRWord]]) -> ExtractedField:
    """El comercio suele ser la primera línea alfabética del encabezado."""
    for line in lines[:5]:
        text = line_text(line)
        folded = fold(text)

        if len(text) < 4 or not re.search(r"[a-z]{3}", folded):
            continue
        if any(token in folded for token in _SKIP_TOKENS):
            continue

        return scoring.field(
            value=text,
            engine_conf=scoring.engine_confidence(line),
            # Es una heurística posicional: nunca se afirma con total certeza.
            parser_conf=0.8,
        )

    return scoring.missing()


def _parse_items(lines: list[list[OCRWord]]) -> list[ReceiptItem]:
    items: list[ReceiptItem] = []

    for line in lines:
        item = _parse_item_line(line)
        if item is not None:
            items.append(item)

    return items


def _parse_item_line(line: list[OCRWord]) -> ReceiptItem | None:
    text = line_text(line)
    folded = fold(text)

    if len(text) < _MIN_NAME_LENGTH or any(token in folded for token in _SKIP_TOKENS):
        return None
    if _DATE_LIKE.search(text):
        return None

    # Se quita el precio del final para no confundirlo con la cantidad.
    body, had_price = _strip_trailing_price(text)

    quantity, name = _split_quantity(body)
    if name is None or not _is_valid_name(name):
        return None

    # Un ítem real trae precio (ticket/factura) o cantidad explícita (remito).
    # Sin ninguno de los dos la línea es encabezado o pie, no un producto.
    if not had_price and quantity is None:
        return None

    engine_conf = scoring.engine_confidence(line)

    # Sin cantidad explícita se asume 1: es correcto en la mayoría de los tickets,
    # pero el parser lo declara como una inferencia y baja la confianza.
    quantity_conf = 1.0 if quantity is not None else 0.6
    # Una línea sin precio (remito) es un ítem algo menos seguro.
    name_conf = 1.0 if had_price else 0.85

    return ReceiptItem(
        nombre_producto=scoring.field(name, engine_conf, name_conf),
        cantidad=scoring.field(quantity if quantity is not None else 1, engine_conf, quantity_conf),
    )


def _strip_trailing_price(text: str) -> tuple[str, bool]:
    """Separa el importe final de la línea del resto de la descripción."""
    match = re.search(r"[\s$]\$?\s*(\d{1,3}(?:\.\d{3})*(?:,\d{1,2})|\d+[.,]\d{2})\s*$", text)
    if match is None:
        return text.strip(), False

    return text[: match.start()].strip(), True


def _split_quantity(body: str) -> tuple[int | None, str | None]:
    """Devuelve `(cantidad, nombre)` según el formato de la línea."""
    body = body.strip(" -_.·*")

    match = _QTY_PREFIX.match(body)
    if match:
        return int(match.group(1)), _clean_name(match.group(2))

    match = _QTY_SUFFIX.match(body)
    if match:
        return int(match.group(2)), _clean_name(match.group(1))

    return None, _clean_name(body)


def _clean_name(value: str) -> str | None:
    cleaned = re.sub(r"\s+", " ", value).strip(" -_.·*x×")
    return cleaned or None


def _is_valid_name(name: str) -> bool:
    if not (_MIN_NAME_LENGTH <= len(name) <= _MAX_NAME_LENGTH):
        return False

    # Debe tener al menos una secuencia de letras: descarta códigos y precios.
    return bool(re.search(r"[a-z]{3}", fold(name)))


def _to_amount(raw: str) -> float | None:
    """`1.234,56` → `1234.56`; `1234.56` → `1234.56`."""
    value = raw
    if "," in value:
        value = value.replace(".", "").replace(",", ".")

    try:
        return round(float(value), 2)
    except ValueError:
        return None

