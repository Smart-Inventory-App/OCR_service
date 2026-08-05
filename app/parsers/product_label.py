"""Primer caso de entrada: la etiqueta del producto en primer plano.

Extrae `nombre_producto`, `marca` y `detalles`. La fecha de vencimiento ya no se
busca acá: tiene su propio tipo de documento y su propia foto.

La heurística que se usaba antes —"el nombre es la línea con la tipografía más
grande"— resultó estar equivocada al medirla sobre envases reales. En un envase
argentino la tipografía más grande es casi siempre **la marca**: `LaVirginia`
sobre `CAFÉ INSTANTÁNEO`, `PUREZA` sobre `HARINA`, `Cabrales` sobre `CAFÉ
TOSTADO MOLIDO`. Lo que identifica al producto es la categoría, no el tamaño.

Por eso la extracción va al revés de lo que parece intuitivo:

1. el **nombre** es el bloque que nombra una categoría de producto,
2. la **marca** es el bloque grande que queda cuando se saca el nombre,
3. los **detalles** son el contenido y los calificativos (`1 litro`, `250 g`,
   `Larga Vida`, `Sabor Naranja`).
"""

import re
from dataclasses import dataclass
from datetime import date

from app.engines.base import OCRWord, RawOCROutput
from app.parsers import scoring
from app.parsers.text import fold, line_text
from app.schemas.ocr import ExtractedField

KEY_FIELDS = ("nombre_producto",)
"""El campo clave que exige la propuesta. `marca` y `detalles` enriquecen el alta
pero no bloquean: son datos que el usuario puede completar de memoria, y exigir
la marca mandaría a revisión manual etiquetas perfectamente leídas."""

# Categorías de producto habituales en un inventario doméstico. Es lo que
# convierte una línea cualquiera en el nombre del producto.
CATEGORIES = (
    "cafe", "leche", "yogur", "queso", "manteca", "crema", "dulce de leche",
    "harina", "arroz", "fideos", "polenta", "avena", "azucar", "sal", "yerba",
    "te", "mate cocido", "cacao", "chocolate", "galletitas", "galletas", "pan",
    "budin", "aceite", "vinagre", "mayonesa", "mostaza", "ketchup", "salsa",
    "pure de tomate", "tomate", "atun", "caballa", "arvejas", "lentejas",
    "porotos", "garbanzos", "choclo", "mermelada", "miel", "gelatina", "flan",
    "postre", "helado", "agua", "agua saborizada", "gaseosa", "jugo", "soda",
    "cerveza", "vino", "fernet", "aperitivo", "caldo", "sopa", "conserva",
    "snack", "papas fritas", "mani", "detergente", "jabon", "lavandina",
    "suavizante", "shampoo", "acondicionador", "papel higienico", "servilletas",
)

# Calificativos: describen al producto pero no lo nombran ni son su marca.
DESCRIPTORS = (
    "clasico", "clasica", "tradicional", "original", "especial", "premium",
    "light", "cero", "zero", "diet", "integral", "entera", "entero",
    "descremada", "descremado", "semidescremada", "larga vida", "instantaneo",
    "instantanea", "tostado", "molido", "en grano", "en sobres", "en polvo",
    "sin sal", "sin azucar", "sin gluten", "libre de gluten", "sin tacc",
    "natural", "casero", "casera", "espuma", "sabor", "listo", "fortificada",
    "fortificado", "con vitaminas", "masa madre", "para pizzas",
)

# Líneas que no son ningún campo: texto legal, de trazabilidad o de servicio.
NOISE_TOKENS = (
    "vto", "vence", "vencimiento", "cons pref", "consumir preferentemente",
    "lote", "lot", "ind arg", "industria argentina", "cuit", "ean", "rnpa",
    "rne", "elaborado", "envasado", "www", "http", "elab", "origen",
    "conservar", "una vez abierto", "informacion nutricional",
    "valor energetico", "no contiene", "codigo de barras",
)

# Cantidad de producto: `1 kg`, `250 g`, `1 litro`, `36 sobres`, `900 ml`.
CONTENT = re.compile(
    r"\b\d+(?:[.,]\d+)?\s*"
    r"(?:kg|kgs|g|gr|grs|mg|ml|cc|lts?|l|litros?|unidades?|un|u|sobres?|rollos?|"
    r"cap(?:sulas)?|saquitos?)\b",
    re.IGNORECASE,
)

_MOSTLY_DIGITS = re.compile(r"^[\W\d]+$")
_DATE_LIKE = re.compile(r"\d{1,4}[/\-.]\d{1,2}")
_HAS_DIGIT = re.compile(r"\d")

# Proporción mínima de letras para considerar que una línea es texto y no ruido.
_MIN_LETTER_RATIO = 0.6

# Confianza mínima del motor para que un bloque pueda ser nombre o marca.
MIN_ENGINE_CONFIDENCE = 0.5

# Dos bloques se funden si sus alturas no difieren más que esto…
_MERGE_HEIGHT_TOLERANCE = 0.35
# …y el hueco vertical entre ellos no supera este múltiplo de la altura. Los
# envases separan bastante los renglones de un mismo título, así que el margen
# es generoso; lo que evita que se pegue cualquier cosa es el solapamiento
# horizontal, que es la condición realmente discriminante.
_MERGE_GAP_RATIO = 1.2

# Una marca es un rótulo corto: más palabras que esto ya es una frase descriptiva.
_MAX_BRAND_WORDS = 3

# …y está impresa en grande: al menos esta fracción de la tipografía más grande.
_MIN_BRAND_HEIGHT_RATIO = 0.6

# Cuántos detalles se conservan, en orden de aparición.
_MAX_DETAILS = 4


@dataclass
class Block:
    """Un renglón o grupo de renglones contiguos con la misma tipografía."""

    words: list[OCRWord]

    @property
    def text(self) -> str:
        return line_text(self.words)

    @property
    def height(self) -> float:
        """Altura media de las cajas: aproxima el tamaño de la tipografía."""
        return sum(w.height for w in self.words) / len(self.words) if self.words else 0.0

    @property
    def confidence(self) -> float:
        return scoring.engine_confidence(self.words)


def parse(
    output: RawOCROutput, text: str, today: date | None = None
) -> dict[str, ExtractedField]:
    """Extrae los campos de una etiqueta de producto.

    `today` se acepta por simetría con los otros parsers; una etiqueta no tiene
    campos que dependan de la fecha actual.
    """
    blocks = build_blocks(output)

    name_block = _pick_name(blocks)
    brand_block = _pick_brand(blocks, name_block)

    return {
        "nombre_producto": _as_field(name_block, _name_parser_confidence),
        "marca": _as_field(brand_block, _brand_parser_confidence),
        "detalles": _details(blocks, name_block, brand_block),
    }


# ── Agrupamiento ──────────────────────────────────────────────────────────────


def build_blocks(output: RawOCROutput) -> list[Block]:
    """Funde renglones contiguos de tipografía parecida en un solo bloque.

    Sin esto `Leche` y `Entera` quedan como dos lecturas sueltas y ninguna de las
    dos es el nombre del producto: los envases parten el nombre en varios
    renglones porque queda mejor en el diseño, no porque sean datos distintos.

    El agrupamiento es **geométrico y no por orden de lectura**. En una foto de
    envase el motor intercala renglones de basura —fragmentos del fondo, texto de
    la foto del producto— entre las líneas que sí forman el nombre, y siguiendo
    el orden de lectura esa basura corta el grupo. Ordenando por posición
    vertical, en cambio, cada renglón se compara con el que tiene efectivamente
    encima, que es la relación que importa.
    """
    lines = [Block(words=line) for line in output.lines() if line]
    if not lines:
        return []

    lines.sort(key=lambda block: (min(w.top for w in block.words), min(w.left for w in block.words)))

    groups: list[Block] = []

    for block in lines:
        # Se busca entre **todos** los grupos abiertos y no sólo en el último:
        # entre dos renglones del nombre suele colarse un renglón de basura de
        # otra zona de la foto, y comparando sólo con el anterior ese intruso
        # corta el grupo y parte el nombre en pedazos.
        target = _closest_mergeable(groups, block)

        if target is None:
            groups.append(block)
        else:
            groups[groups.index(target)] = Block(words=target.words + block.words)

    return groups


def _closest_mergeable(groups: list[Block], block: Block) -> Block | None:
    """El grupo fusionable cuyo último renglón queda más cerca de `block`."""
    top = min(w.top for w in block.words)

    mergeable = [group for group in groups if _should_merge(group, block)]
    if not mergeable:
        return None

    return min(mergeable, key=lambda group: abs(top - max(w.bottom for w in group.words)))


def _should_merge(previous: Block, current: Block) -> bool:
    tallest = max(previous.height, current.height)
    if tallest == 0:
        return False

    if abs(previous.height - current.height) / tallest > _MERGE_HEIGHT_TOLERANCE:
        return False

    gap = min(w.top for w in current.words) - max(w.bottom for w in previous.words)
    if gap > tallest * _MERGE_GAP_RATIO or gap < -tallest:
        return False

    # Deben solaparse horizontalmente: si no, son dos columnas distintas.
    return _overlaps_horizontally(previous, current)


def _overlaps_horizontally(previous: Block, current: Block) -> bool:
    left = max(min(w.left for w in previous.words), min(w.left for w in current.words))
    right = min(
        max(w.left + w.width for w in previous.words),
        max(w.left + w.width for w in current.words),
    )
    return right > left


# ── Selección de campos ───────────────────────────────────────────────────────


def _pick_name(blocks: list[Block]) -> Block | None:
    """El bloque que nombra una categoría de producto; si no hay, el más grande."""
    readable = [b for b in blocks if _is_candidate(b)]
    if not readable:
        return None

    named = [b for b in readable if _has_category(b.text)]
    if named:
        return max(named, key=lambda b: b.height)

    # Sin categoría reconocible (marcas de fantasía, envases importados) se
    # vuelve a la tipografía más grande, que sigue siendo la mejor apuesta.
    return max(
        (b for b in readable if not _is_content(b.text)),
        key=lambda b: b.height,
        default=None,
    )


def _pick_brand(blocks: list[Block], name: Block | None) -> Block | None:
    """El rótulo grande que queda al sacar el nombre y los calificativos.

    Se exige que sea **tipográficamente prominente**, y no sólo el más grande de
    lo que quedó: en un envase la marca siempre está impresa en grande, así que
    un renglón de letra chica nunca lo es. Sin este piso, cuando la marca real no
    se lee —porque está girada, en relieve o fuera de foco— el campo se llena con
    el primer fragmento de la letra chica legal (`do Nero:` de "Contenido Neto"),
    que es exactamente el tipo de dato falso que no hay que ofrecerle al usuario.
    """
    tallest = max((b.height for b in blocks if _is_candidate(b)), default=0.0)
    floor = tallest * _MIN_BRAND_HEIGHT_RATIO

    candidates = [
        block
        for block in blocks
        if block is not name
        and _is_candidate(block)
        and block.height >= floor
        and not _is_content(block.text)
        and not _is_descriptor(block.text)
        and not _has_category(block.text)
        and len(block.text.split()) <= _MAX_BRAND_WORDS
        and not _HAS_DIGIT.search(block.text)
    ]

    return max(candidates, key=lambda b: b.height, default=None)


def _details(blocks: list[Block], name: Block | None, brand: Block | None) -> ExtractedField:
    """Contenido y calificativos, en orden de aparición en la etiqueta.

    A diferencia del nombre y de la marca, acá **no** se exige que la línea
    parezca texto: un detalle legítimo como `1 kg` o `900 ml` es casi todo
    dígitos y el filtro de legibilidad —que existe para no ofrecer ruido como
    nombre de producto— lo descartaría.
    """
    picked = [
        block
        for block in blocks
        if block is not name
        and block is not brand
        and (_is_content(block.text) or _is_descriptor(block.text))
        and not _is_noise(block.text)
    ][:_MAX_DETAILS]

    if not picked:
        return scoring.missing()

    return scoring.field(
        value=" · ".join(block.text for block in picked),
        engine_conf=scoring.engine_confidence([w for b in picked for w in b.words]),
        # Los detalles son informativos: no hay un patrón que validar más allá de
        # haberlos reconocido como contenido o calificativo.
        parser_conf=1.0,
    )


def _as_field(block: Block | None, parser_confidence) -> ExtractedField:
    if block is None:
        return scoring.missing()

    return scoring.field(
        value=block.text,
        engine_conf=block.confidence,
        parser_conf=parser_confidence(block.text),
    )


# ── Confianza del parser ──────────────────────────────────────────────────────


def _name_parser_confidence(value: str) -> float:
    confidence = 1.0

    # Un nombre que nombra su categoría es un nombre reconocido; uno que no, es
    # la apuesta de reserva por la tipografía más grande.
    if not _has_category(value):
        confidence *= 0.8

    # Nombres muy cortos suelen ser fragmentos ("Le", "1L").
    if len(value) < 4:
        confidence *= 0.5
    elif len(value) < 8:
        confidence *= 0.9

    return confidence


def _brand_parser_confidence(value: str) -> float:
    confidence = 1.0

    if len(value) < 3:
        confidence *= 0.5
    elif len(value) < 5:
        confidence *= 0.9

    # Una marca es un rótulo, no una frase.
    if len(value.split()) > 2:
        confidence *= 0.9

    return confidence


# ── Clasificación de líneas ───────────────────────────────────────────────────


def _has_category(value: str) -> bool:
    normalized = _normalize(value)
    return any(re.search(rf"\b{re.escape(term)}\b", normalized) for term in CATEGORIES)


def _is_descriptor(value: str) -> bool:
    normalized = _normalize(value)
    return any(term in normalized for term in DESCRIPTORS)


def _is_content(value: str) -> bool:
    return bool(CONTENT.search(value))


def _is_noise(value: str) -> bool:
    normalized = _normalize(value)
    return any(token in normalized for token in NOISE_TOKENS)


def _is_candidate(block: Block) -> bool:
    """Si el bloque puede ofrecerse como nombre o como marca.

    Además de parecer texto, se le exige que el motor lo haya leído con una
    confianza mínima. Es una decisión de producto: sobre un envase con fotos y
    fondo, el OCR devuelve lecturas como `AÚ RARA,` o `MAXanm) sibok` con
    confianza casi nula, y ofrecerle eso al usuario como nombre del producto es
    peor que no ofrecerle nada. Sin candidato, el campo queda vacío y
    `needs_review` dispara la validación manual asistida.
    """
    return block.confidence >= MIN_ENGINE_CONFIDENCE and _is_readable(block.text)


def _is_readable(value: str) -> bool:
    """Descarta la basura que devuelve el OCR sobre fondos e imágenes.

    Una lectura fallida produce líneas como `( E Y e L` o `AR es`: caracteres
    sueltos y símbolos. Devolver `None` es preferible a devolver ruido, porque
    dispara la revisión manual en lugar de ofrecerle al usuario un valor sin
    sentido.
    """
    normalized = _normalize(value)

    if len(normalized) < 3:
        return False
    if _MOSTLY_DIGITS.match(normalized):
        return False
    if _DATE_LIKE.search(normalized):
        return False
    if _is_noise(value):
        return False

    dense = normalized.replace(" ", "")
    letters = sum(1 for char in dense if char.isalpha())
    if letters / len(dense) < _MIN_LETTER_RATIO:
        return False

    # Estructura de palabras: una palabra larga ("Fargo", "Leche") o al menos dos
    # medianas. El ruido del OCR produce letras sueltas separadas por símbolos.
    words = re.findall(r"[a-z]{3,}", normalized)

    return any(len(word) >= 4 for word in words) or len(words) >= 2


def _normalize(value: str) -> str:
    """Forma canónica para clasificar una línea (no para mostrarla).

    Además de bajar a minúsculas, quitar acentos y colapsar la puntuación, pega
    la letra suelta a la palabra siguiente. El motor parte con frecuencia la
    primera letra de una palabra en versalitas —`C AFÉ` por `CAFÉ`, `L ECHE` por
    `LECHE`— y sin pegarla la categoría del producto no se reconoce, que es
    justo el dato del que cuelga toda la extracción.

    Se exige que lo que sigue sea una palabra de dos letras o más: si no, una
    lectura fallida como `( E Y e L` se pegaría en `eyel` y pasaría por texto.
    """
    collapsed = re.sub(r"[^\w\s]+", " ", fold(value))
    collapsed = re.sub(r"\s+", " ", collapsed).strip()

    return re.sub(r"\b([a-z])\s+(?=[a-z]{2,})", r"\1", collapsed)
