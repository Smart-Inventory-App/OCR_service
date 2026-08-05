"""Extracción y normalización de fechas a ISO `YYYY-MM-DD`.

Cubre los formatos habituales en etiquetas y comprobantes argentinos, con las
etiquetas que suelen precederlas (`VTO`, `VENCE`, `CONS. PREF.`, `FECHA`).
"""

import re
from dataclasses import dataclass
from datetime import date

from dateutil.relativedelta import relativedelta

from app.parsers.text import fold

# Palabras que anuncian una fecha de vencimiento en una etiqueta.
EXPIRY_KEYWORDS = (
    "vto",
    "vence",
    "vencimiento",
    "cons pref",
    "cons. pref",
    "consumir preferentemente",
    "consumir antes de",
    "best before",
    "exp",
)

# Palabras que anuncian la fecha de emisión de un comprobante.
PURCHASE_KEYWORDS = ("fecha", "emision", "emisión", "compra", "f. emision")

_MONTHS_ES = {
    "ene": 1, "feb": 2, "mar": 3, "abr": 4, "may": 5, "jun": 6,
    "jul": 7, "ago": 8, "sep": 9, "set": 9, "oct": 10, "nov": 11, "dic": 12,
}

_SEP = r"[/\-.\s]"

# Caracteres previos a la fecha en los que se busca la palabra clave: alcanza
# para "Consumir preferentemente antes de", la etiqueta más larga que se maneja.
_KEYWORD_WINDOW = 40

# Cada patrón declara qué grupos representan día, mes y año.
_PATTERNS: list[tuple[re.Pattern, tuple[str, ...]]] = [
    # 15/08/2026 · 15-08-26 · 15.08.2026
    (re.compile(rf"\b(\d{{1,2}}){_SEP}(\d{{1,2}}){_SEP}(\d{{2,4}})\b"), ("d", "m", "y")),
    # 2026-08-15 (ISO)
    (re.compile(rf"\b(\d{{4}}){_SEP}(\d{{1,2}}){_SEP}(\d{{1,2}})\b"), ("y", "m", "d")),
    # 15 AGO 2026 · 15-ago-26
    (
        re.compile(
            rf"\b(\d{{1,2}}){_SEP}({'|'.join(_MONTHS_ES)})\w*{_SEP}(\d{{2,4}})\b",
            re.IGNORECASE,
        ),
        ("d", "m_name", "y"),
    ),
    # AGO 2026
    (
        re.compile(rf"\b({'|'.join(_MONTHS_ES)})\w*{_SEP}(\d{{4}})\b", re.IGNORECASE),
        ("m_name", "y"),
    ),
    # 08/2026 — sólo mes y año: se asume el último día del mes
    (re.compile(rf"\b(\d{{1,2}}){_SEP}(\d{{4}})\b"), ("m", "y")),
]


@dataclass(frozen=True)
class ParsedDate:
    value: date
    """Fecha normalizada."""

    matched_text: str
    """Texto crudo del que se extrajo, para auditoría."""

    start: int
    """Posición de la coincidencia en el texto original."""

    day_precision: bool
    """False si el formato sólo traía mes/año (día inferido)."""

    @property
    def iso(self) -> str:
        return self.value.isoformat()


def find_dates(text: str) -> list[ParsedDate]:
    """Todas las fechas válidas del texto, en orden de aparición."""
    found: list[ParsedDate] = []

    for pattern, groups in _PATTERNS:
        for match in pattern.finditer(text):
            parsed = _build_date(match, groups)
            if parsed is None:
                continue

            # Un patrón anterior (más específico) ya cubrió este tramo del texto.
            if any(
                prev.start <= parsed.start < prev.start + len(prev.matched_text)
                for prev in found
            ):
                continue

            found.append(parsed)

    return sorted(found, key=lambda parsed: parsed.start)


def find_expiry_date(text: str, today: date | None = None) -> ParsedDate | None:
    """Fecha de vencimiento de una etiqueta de producto.

    Prioriza las fechas precedidas por una palabra clave (`VTO`, `VENCE`…) y,
    entre las demás, la primera que no esté en el pasado — un vencimiento ya
    cumplido casi siempre es en realidad la fecha de elaboración.
    """
    candidates = find_dates(text)
    if not candidates:
        return None

    today = today or date.today()

    keyed = [c for c in candidates if has_keyword_before(text, c, EXPIRY_KEYWORDS)]
    if keyed:
        return keyed[0]

    future = [c for c in candidates if c.value >= today]
    if future:
        return future[0]

    # Sin ninguna futura: se devuelve la más lejana y el scoring penaliza.
    return max(candidates, key=lambda c: c.value)


def find_purchase_date(text: str, today: date | None = None) -> ParsedDate | None:
    """Fecha de compra de un ticket: la primera fecha no futura del documento."""
    candidates = find_dates(text)
    if not candidates:
        return None

    today = today or date.today()

    keyed = [c for c in candidates if has_keyword_before(text, c, PURCHASE_KEYWORDS)]
    if keyed:
        return keyed[0]

    past = [c for c in candidates if c.value <= today]
    return past[0] if past else candidates[0]


def has_keyword_before(text: str, parsed: ParsedDate, keywords: tuple[str, ...]) -> bool:
    """True si alguna keyword antecede a la fecha en su misma línea.

    Se acota a la línea porque es la unidad en la que las etiquetas anuncian el
    dato ("VTO 15/08/2027"), y se limita su largo para que en un texto sin saltos
    de línea una keyword lejana no cuente como si estuviera al lado.
    """
    line_start = text.rfind("\n", 0, parsed.start) + 1
    window_start = max(line_start, parsed.start - _KEYWORD_WINDOW)

    return any(keyword in fold(text[window_start : parsed.start]) for keyword in keywords)


def _build_date(match: re.Match, groups: tuple[str, ...]) -> ParsedDate | None:
    parts = dict(zip(groups, match.groups()))

    year = _expand_year(parts["y"])
    month = _resolve_month(parts)
    if month is None:
        return None

    if "d" in parts:
        day = int(parts["d"])
        day_precision = True
    else:
        # Formato mes/año: el producto vence al cierre del mes.
        day = _last_day_of_month(year, month)
        day_precision = False

    try:
        value = date(year, month, day)
    except ValueError:
        return None

    return ParsedDate(
        value=value,
        matched_text=match.group(0),
        start=match.start(),
        day_precision=day_precision,
    )


def _resolve_month(parts: dict[str, str]) -> int | None:
    if "m_name" in parts:
        return _MONTHS_ES.get(parts["m_name"][:3].lower())

    month = int(parts["m"])
    return month if 1 <= month <= 12 else None


def _expand_year(raw: str) -> int:
    """`26` → `2026`. Dos dígitos se interpretan siempre en el siglo XXI."""
    year = int(raw)
    return year if year >= 100 else 2000 + year


def _last_day_of_month(year: int, month: int) -> int:
    return (date(year, month, 1) + relativedelta(months=1, days=-1)).day
