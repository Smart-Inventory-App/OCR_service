"""Extracción y normalización de fechas a ISO `YYYY-MM-DD`.

Cubre los formatos habituales en etiquetas y comprobantes argentinos, con las
palabras que suelen precederlas (`VTO`, `VENCE`, `CONS. PREF.`, `FECHA`), y las
dos trampas que aparecieron al medir sobre fotos reales de envase:

- El sello del envase trae **dos fechas**: la de elaboración y la de vencimiento
  (`ENV.: 05/01/26  VTO.: 05/01/27`). Devolver la primera que aparece da la
  fecha equivocada, así que las anunciadas como elaboración se descartan.
- Está impreso en **matriz de puntos**, y el motor confunde letras con dígitos
  dentro del mes (`24 GCT 27` por `24 OCT 27`, `27FEBZ7` por `27FEB27`). Un
  patrón tolerante recupera esos casos y el scoring los penaliza por ser una
  lectura reparada y no una lectura limpia.
"""

import re
from dataclasses import dataclass, replace
from datetime import date

from dateutil.relativedelta import relativedelta
from rapidfuzz import process

from app.parsers.text import fold

# Palabras que anuncian una fecha de vencimiento.
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
    # Cómo lee el motor `VTO` cuando el sello está impreso con trazo fino: la `T`
    # y la `I` son dos palos verticales y se confunden sistemáticamente. Se
    # listan explícitamente en vez de comparar por similitud, para no aceptar
    # cualquier palabra de tres letras que caiga antes de una fecha.
    "vio",
    "v1o",
    "vt0",
)

# Palabras que anuncian la fecha de **elaboración**: la fecha que las sigue nunca
# es el vencimiento.
PRODUCTION_KEYWORDS = ("elab", "elaborado", "env", "envasado", "fab", "fabricacion", "prod")

# Palabras que anuncian la fecha de emisión de un comprobante.
PURCHASE_KEYWORDS = ("fecha", "emision", "emisión", "compra", "f. emision")

_MONTHS_ES = {
    "ene": 1, "feb": 2, "mar": 3, "abr": 4, "may": 5, "jun": 6,
    "jul": 7, "ago": 8, "sep": 9, "set": 9, "oct": 10, "nov": 11, "dic": 12,
}

_MONTH_ALT = "|".join(_MONTHS_ES)

_SEP = r"[/\-.\s]"

# Caracteres previos a la fecha en los que se busca la palabra clave: alcanza
# para "Consumir preferentemente antes de", la etiqueta más larga que se maneja.
_KEYWORD_WINDOW = 40

# Dígitos que el OCR devuelve como letras dentro de un mes de tres caracteres.
# Se aplican sólo a ese token, nunca al texto general.
_LETTER_LOOKALIKES = str.maketrans({"0": "O", "1": "I", "5": "S", "8": "B", "6": "G", "2": "Z"})

# Similitud mínima entre el token leído y un nombre de mes. Con tres caracteres,
# 60 tolera exactamente una confusión.
_MONTH_SIMILARITY = 60


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

    fuzzy_month: bool = False
    """True si el mes se resolvió por similitud y no por coincidencia exacta."""

    @property
    def iso(self) -> str:
        return self.value.isoformat()


# Cada patrón declara qué grupos representan día, mes y año. El orden importa:
# los más específicos van primero y los siguientes no pisan lo ya reconocido.
_PATTERNS: list[tuple[re.Pattern, tuple[str, ...]]] = [
    # 15/08/2026 · 15-08-26 · 15.08.2026
    (re.compile(rf"\b(\d{{1,2}}){_SEP}(\d{{1,2}}){_SEP}(\d{{2,4}})\b"), ("d", "m", "y")),
    # 2026-08-15 (ISO)
    (re.compile(rf"\b(\d{{4}}){_SEP}(\d{{1,2}}){_SEP}(\d{{1,2}})\b"), ("y", "m", "d")),
    # 15 AGO 2026 · 15-ago-26
    (
        re.compile(rf"\b(\d{{1,2}}){_SEP}({_MONTH_ALT})\w*{_SEP}(\d{{2,4}})\b", re.IGNORECASE),
        ("d", "m_name", "y"),
    ),
    # 27FEB27 · 25AGO2026 — sin separadores. Es el formato habitual de las
    # impresoras de lote y vencimiento sobre el envase.
    (
        re.compile(rf"\b(\d{{1,2}})({_MONTH_ALT})\w*?(\d{{2,4}})\b", re.IGNORECASE),
        ("d", "m_name", "y"),
    ),
    # AGO 2026
    (re.compile(rf"\b({_MONTH_ALT})\w*{_SEP}(\d{{4}})\b", re.IGNORECASE), ("m_name", "y")),
    # 08/2026 — sólo mes y año: se asume el último día del mes
    (re.compile(rf"\b(\d{{1,2}}){_SEP}(\d{{4}})\b"), ("m", "y")),
]

# Último recurso: un mes de tres caracteres que no matchea ninguno de los
# nombres, encajonado entre un día y un año. Es el caso del sello de matriz de
# puntos leído con una letra mal.
_FUZZY_MONTH_PATTERN = re.compile(r"\b(\d{1,2})[\s/.\-]*([A-Za-z0-9]{3})[\s/.\-]*(\d{2,4})\b")

# Mes/año con año de DOS dígitos: `VTO 08/28`, `VTO08/28`, `CONS.PREF. 12/27`.
#
# Es el formato más frecuente en el envase argentino y estaba sin cubrir: los
# patrones de arriba sólo aceptan mes/año con año de cuatro dígitos. La razón de
# la exclusión era buena —`08/28` suelto es indistinguible de un día/mes— y por
# eso acá se exige que **lo anuncie una palabra de vencimiento**. Con `VTO`
# adelante no hay ambigüedad posible: nadie imprime un día y un mes después de
# «vence».
#
# La palabra clave queda fuera del tramo que se reporta como `matched_text`
# (ver `_build_keyed_month_year`), para que el resto del scoring siga viendo la
# fecha y no la etiqueta.
_KEYWORD_ALT = r"vto|vence|vencimiento|cons\.?\s*pref\.?|consumir\s+preferentemente|best\s+before|v[t1i]0|vio"

_KEYED_MONTH_YEAR = re.compile(
    rf"\b(?:{_KEYWORD_ALT})[^0-9A-Za-z]{{0,4}}(\d{{1,2}})\s*[/\-.]\s*(\d{{2}})(?!\d)",
    re.IGNORECASE,
)


def find_dates(text: str) -> list[ParsedDate]:
    """Todas las fechas válidas del texto, en orden de aparición."""
    found: list[ParsedDate] = []

    for pattern, groups in _PATTERNS:
        for match in pattern.finditer(text):
            _append(found, _build_date(match, groups))

    for match in _KEYED_MONTH_YEAR.finditer(text):
        _append(found, _build_keyed_month_year(match))

    for match in _FUZZY_MONTH_PATTERN.finditer(text):
        _append(found, _build_fuzzy_date(match))

    return sorted(found, key=lambda parsed: parsed.start)


def _append(found: list[ParsedDate], parsed: ParsedDate | None) -> None:
    """Agrega la fecha salvo que un patrón anterior ya cubriera ese tramo."""
    if parsed is None:
        return

    if any(
        prev.start <= parsed.start < prev.start + len(prev.matched_text) for prev in found
    ):
        return

    found.append(parsed)


# Ventanas de plausibilidad. Una fecha fuera de rango no es una lectura dudosa
# sino ruido interpretado como dígitos: se descarta en vez de reportarse con
# poca confianza, porque un dato falso que el usuario podría aceptar es peor
# que no devolver nada.
MAX_YEARS_PAST_EXPIRY = 2
MAX_YEARS_AHEAD_EXPIRY = 15
MAX_YEARS_PAST_PURCHASE = 10


def is_plausible_expiry(value: date, today: date) -> bool:
    return (
        date(today.year - MAX_YEARS_PAST_EXPIRY, 1, 1)
        <= value
        <= date(today.year + MAX_YEARS_AHEAD_EXPIRY, 12, 31)
    )


def is_plausible_purchase(value: date, today: date) -> bool:
    return date(today.year - MAX_YEARS_PAST_PURCHASE, 1, 1) <= value <= today


def find_expiry_date(text: str, today: date | None = None) -> ParsedDate | None:
    """Fecha de vencimiento.

    Prioriza las anunciadas por una palabra clave (`VTO`, `VENCE`…), descarta las
    anunciadas como elaboración y, entre las que quedan sin anuncio, elige la
    primera que no esté en el pasado — un vencimiento ya cumplido casi siempre es
    en realidad la fecha de elaboración.
    """
    today = today or date.today()

    candidates = [
        candidate
        for candidate in find_dates(text)
        if is_plausible_expiry(candidate.value, today)
        and not _announced_as_production(text, candidate)
    ]
    if not candidates:
        return None

    keyed = [c for c in candidates if has_keyword_before(text, c, EXPIRY_KEYWORDS)]
    if keyed:
        # Entre varias anunciadas, la más lejana: en un sello con elaboración y
        # vencimiento juntos, el vencimiento es siempre la posterior.
        return max(keyed, key=lambda c: c.value)

    future = [c for c in candidates if c.value >= today]
    if future:
        return future[0]

    # Sin ninguna futura: se devuelve la más lejana y el scoring penaliza.
    return max(candidates, key=lambda c: c.value)


def find_purchase_date(text: str, today: date | None = None) -> ParsedDate | None:
    """Fecha de compra de un ticket: la primera fecha no futura del documento."""
    today = today or date.today()

    candidates = [c for c in find_dates(text) if is_plausible_purchase(c.value, today)]
    if not candidates:
        return None

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
    window = _window_before(text, parsed)

    return any(keyword in window for keyword in keywords)


def _announced_as_production(text: str, parsed: ParsedDate) -> bool:
    """True si la fecha viene anunciada como elaboración y no como vencimiento.

    Las keywords de producción son cortas (`env`, `elab`) y aparecen dentro de
    otras palabras, así que acá sí se exige límite de palabra.
    """
    window = _window_before(text, parsed)

    if any(keyword in window for keyword in EXPIRY_KEYWORDS):
        return False

    return any(
        re.search(rf"\b{re.escape(keyword)}", window) for keyword in PRODUCTION_KEYWORDS
    )


def _window_before(text: str, parsed: ParsedDate) -> str:
    line_start = text.rfind("\n", 0, parsed.start) + 1
    window_start = max(line_start, parsed.start - _KEYWORD_WINDOW)

    return fold(text[window_start : parsed.start])


def _build_date(match: re.Match, groups: tuple[str, ...]) -> ParsedDate | None:
    parts = dict(zip(groups, match.groups()))

    month = _resolve_month(parts)
    if month is None:
        return None

    return _assemble(match, _expand_year(parts["y"]), month, parts.get("d"))


def _build_keyed_month_year(match: re.Match) -> ParsedDate | None:
    """`VTO 08/28` → 2028-08-31, con el día inferido al cierre del mes.

    `start` y `matched_text` apuntan a la fecha y no al match completo, que
    incluye la palabra clave. Es lo que permite que `has_keyword_before` la
    encuentre mirando hacia atrás y que la confianza no se penalice como si la
    fecha hubiera aparecido suelta — que es justamente lo contrario de lo que
    pasó acá.
    """
    month = int(match.group(1))
    if not 1 <= month <= 12:
        return None

    year = _expand_year(match.group(2))

    try:
        value = date(year, month, _last_day_of_month(year, month))
    except ValueError:
        return None

    return ParsedDate(
        value=value,
        matched_text=match.string[match.start(1) : match.end(2)],
        start=match.start(1),
        day_precision=False,
    )


def _build_fuzzy_date(match: re.Match) -> ParsedDate | None:
    """Interpreta un mes de tres caracteres leído con alguna letra mal."""
    month = _month_by_similarity(match.group(2))
    if month is None:
        return None

    parsed = _assemble(match, _expand_year(match.group(3)), month, match.group(1))

    return None if parsed is None else replace(parsed, fuzzy_month=True)


def _assemble(match: re.Match, year: int, month: int, raw_day: str | None) -> ParsedDate | None:
    if raw_day is not None:
        day = int(raw_day)
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


def _month_by_similarity(token: str) -> int | None:
    """Mes más parecido al token, tolerando una confusión de carácter.

    Antes de comparar, los dígitos vuelven a su letra parecida: la impresora usa
    tipografía de puntos y el motor devuelve `0` donde hay una `O`, o `Z` donde
    hay un `2`.
    """
    normalized = token.upper().translate(_LETTER_LOOKALIKES).lower()

    if normalized in _MONTHS_ES:
        # Coincidencia exacta tras normalizar: no es un caso difuso de verdad,
        # pero tampoco lo cubrió el patrón estricto.
        return _MONTHS_ES[normalized]

    best = process.extractOne(normalized, list(_MONTHS_ES), score_cutoff=_MONTH_SIMILARITY)

    return _MONTHS_ES[best[0]] if best else None


def _expand_year(raw: str) -> int:
    """`26` → `2026`. Dos dígitos se interpretan siempre en el siglo XXI."""
    year = int(raw)
    return year if year >= 100 else 2000 + year


def _last_day_of_month(year: int, month: int) -> int:
    return (date(year, month, 1) + relativedelta(months=1, days=-1)).day
