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
    # Condimentos y especias. Faltaban enteros, y sin la categoría el nombre
    # cae en la marca: sobre el perejil de Alicante el campo devolvía
    # "Alicante Perejil" en vez de "Perejil deshidratado".
    "perejil", "oregano", "albahaca", "comino", "pimienta", "pimenton",
    "laurel", "romero", "tomillo", "ajo", "curry", "canela", "provenzal",
    "chimichurri", "condimento", "condimentos", "especias", "nuez moscada",
    # Farmacia y perfumería. Faltaban enteras: las etiquetas de farmacia daban 2
    # altas automáticas sobre 27 (7 %) contra 59 % en almacén.
    #
    # Lo que entra acá es lo que **nombra** el producto, y en una caja argentina
    # eso es la monodroga (`Ibuprofeno 400 mg`) o el artículo en sí (`gasa`,
    # `protector solar`). Medido sobre las fotos: el nombre correcto ya estaba
    # siendo leído, pero caía en `detalles` porque la línea trae la dosis y el
    # filtro de contenido la reclamaba antes.
    #
    # Lo que describe el EFECTO —analgésico, antiinflamatorio— va a DESCRIPTORS y
    # no acá: si compitiera por el nombre, en `Sertal Compuesto` ganaría
    # «Antiespasmódico y analgésico», que no es el nombre de nada.
    # Sólo las monodrogas que el envase imprime en grande como nombre del
    # producto. Las que aparecen únicamente en la línea de composición —en letra
    # chica: `Propinox Clonixinato de lisina`, `Hioscina butilbromuro`— quedaron
    # afuera a propósito: como categoría harían que esa línea le gane al nombre,
    # porque entre los bloques con categoría se elige el más grande y un renglón
    # de composición sin competencia gana aunque sea diminuto.
    "ibuprofeno", "paracetamol", "aspirina", "diclofenac", "naproxeno",
    "amoxicilina", "azitromicina", "omeprazol", "ranitidina", "loratadina",
    "cetirizina", "dipirona",
    "jarabe", "suspension", "supositorios", "pomada", "unguento", "colirio",
    "alcohol en gel", "agua oxigenada", "iodopovidona", "curitas", "gasa",
    "venda", "algodon", "barbijo", "termometro", "suero fisiologico",
    "protector solar", "repelente", "crema dental", "pasta dental", "enjuague",
    "hilo dental", "desodorante", "talco", "vaselina", "jabon", "jabon liquido",
    "suplemento", "magnesio", "calcio", "hierro",
    # `gotas`, `polvo` y `vitaminas` quedaron deliberadamente afuera: aparecen en
    # el texto de uso del envase («8 gotas = 1 cucharadita») o ya son
    # descriptores («en polvo», «con vitaminas»), y como categoría harían que una
    # línea accesoria le gane al nombre real del producto.
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
    "deshidratado", "deshidratada", "sin gas", "con gas", "mineralizada",
    "baja en sodio", "descafeinado", "en hebras", "molida",
    # Farmacia: lo que describe el efecto o la presentación, no el producto.
    # Sobre las fotos del conjunto, estas frases estaban GANANDO el campo
    # `nombre_producto` —«RÁPIDA ACCIÓN», «RÁPIDA ACCIÓN NUEVA FÓRMULA»— porque
    # nada las clasificaba y quedaban como el bloque legible más grande.
    "analgesico", "antiinflamatorio", "antiacido", "antialergico",
    "antibiotico", "antiespasmodico", "antihemorroidal", "antigripal",
    "antitusivo", "descongestivo", "laxante", "flebotonico", "antibacterial",
    "rapida accion", "accion rapida", "nueva formula", "venta libre",
    "uso externo", "via oral", "recubiertos", "recubierto", "efervescente",
    "forte", "duo", "plus", "extra fuerte",
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
    r"cap(?:sulas)?|saquitos?|comprimidos?|comp|ampollas?|pastillas?)\b",
    re.IGNORECASE,
)
"""Cantidad de producto. Las presentaciones farmacéuticas contables
—`10 comprimidos`, `20 cápsulas`— entran acá y no en CATEGORIES: son el
contenido del envase, igual que `500 g`, y no el nombre de lo que hay adentro."""

_MOSTLY_DIGITS = re.compile(r"^[\W\d]+$")
_DATE_LIKE = re.compile(r"\d{1,4}[/\-.]\d{1,2}")
_HAS_DIGIT = re.compile(r"\d")

# Proporción mínima de letras para considerar que una línea es texto y no ruido.
_MIN_LETTER_RATIO = 0.6

# Confianza mínima del motor para que un bloque pueda ser nombre o marca.
MIN_ENGINE_CONFIDENCE = 0.5

# Vocabulario con el que se decide si una palabra pegada por el motor
# (`CAFÉINSTANTÁNEO`) esconde una categoría. Ver `_has_category`.
_KNOWN_TERMS = frozenset(CATEGORIES) | frozenset(DESCRIPTORS)

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

    name, glued_brand = _split_leading_brand(name_block.text if name_block else "")

    brand = _as_field(brand_block, _brand_parser_confidence)
    # Si no había un bloque de marca aparte, la marca puede ser lo que se acaba
    # de recortar del nombre: ahí el agrupamiento la había pegado, no perdido.
    if brand_block is None and glued_brand:
        brand = _as_field(name_block, _brand_parser_confidence, text=glued_brand)

    return {
        "nombre_producto": _as_field(name_block, _name_parser_confidence, text=name),
        "marca": brand,
        "detalles": _details(blocks, name_block, brand_block),
    }


def _split_leading_brand(text: str) -> tuple[str, str | None]:
    """Separa la marca que el agrupamiento haya pegado delante del nombre.

    Cuando la marca y el nombre están impresos en cuerpos parecidos y uno
    encima del otro —`Alicante` sobre `Perejil`—, el agrupamiento por geometría
    los une en un solo bloque, y el nombre del producto sale con la marca
    adelante: `Alicante Perejil`. Separarlos por geometría no se puede; el
    vocabulario sí sabe cuál es cuál.

    Se recorta **sólo** cuando la categoría aparece después de la primera
    palabra y nada de lo que la precede nombra una categoría. Eso deja intactos
    los nombres que empiezan por su propia categoría, que son la mayoría:
    `Leche Entera`, `YOGUR FIRME FRUTILLA` y `CAFE TOSTADO MOLIDO` no se tocan,
    porque ahí el recorte se comería el nombre en vez de la marca.
    """
    words = text.split()

    # Se recorre de atrás para adelante para quedarse con el corte MÁS TARDÍO que
    # cumple, es decir, con la cola más corta que todavía nombra la categoría.
    # Cortando en el primero que cumple se recortaba de menos: sobre
    # «RAPIDA ACCION Ibuprofeno 400 mg» el primer corte válido deja
    # «ACCION Ibuprofeno 400 mg», con la mitad del descriptor pegada al nombre.
    for index in range(len(words) - 1, 0, -1):
        head, tail = " ".join(words[:index]), " ".join(words[index:])
        if _has_category(tail) and not _has_category(head):
            return tail, head if _is_readable(head) else None

    return text, None


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


def _as_field(block: Block | None, parser_confidence, text: str | None = None) -> ExtractedField:
    """Campo puntuado con la confianza del bloque que lo aportó.

    `text` permite ofrecer un recorte del bloque —la marca y el nombre que el
    agrupamiento pegó— conservando la confianza del motor sobre esas palabras.
    """
    if block is None:
        return scoring.missing()

    value = block.text if text is None else text
    if not value:
        return scoring.missing()

    return scoring.field(
        value=value,
        engine_conf=block.confidence,
        parser_conf=parser_confidence(value),
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
    """Si la línea nombra una categoría de producto.

    El nombre del producto cuelga entero de esta decisión, y los motores no
    respetan los límites de palabra: sobre las mismas fotos aparecen las dos
    formas de romperlos.

    - **De más.** `C AFÉ` por `CAFÉ`: la versalita inicial sale como token
      suelto. Lo repara `_normalize`, que pega la letra suelta a la palabra que
      sigue.
    - **De menos.** `O CAFÉ`, donde la `O` no es una letra partida sino una
      marca del logo. Ahí la misma reparación estorba —deja `ocafe`— así que
      también se prueba la forma sin pegar.
    - **Al revés.** `CAFÉINSTANTÁNEO`, dos palabras que el motor devolvió como
      una sola región. Para esas, la categoría se busca como prefijo de palabra
      en vez de como palabra entera.

    Para el último caso **no alcanza con buscar la categoría como prefijo**: eso
    convertiría en nombre a media góndola de marcas derivadas de su propio
    producto (`Yogurisimo` empieza con `yogur`, `Manisero` con `mani`), que es
    exactamente la confusión entre marca y nombre que este parser existe para
    evitar. Una palabra pegada cuenta sólo si **lo que sigue a la categoría es
    otro término conocido** —`cafeinstantaneo` es `cafe` + `instantaneo`, un
    calificativo del vocabulario—, y `yogurisimo` no lo es porque `isimo` no
    nombra nada.
    """
    forms = {_normalize(value), _collapse(value)}

    return any(_mentions_category(form) for form in forms)


def _mentions_category(normalized: str) -> bool:
    if any(re.search(rf"\b{re.escape(term)}\b", normalized) for term in CATEGORIES):
        return True

    return any(_is_glued_category(word) for word in normalized.split())


def _is_glued_category(word: str) -> bool:
    """`cafeinstantaneo` → `cafe` + `instantaneo`, dos términos del vocabulario."""
    return any(
        word.startswith(term) and word[len(term):] in _KNOWN_TERMS
        for term in CATEGORIES
    )


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
    return re.sub(r"\b([a-z])\s+(?=[a-z]{2,})", r"\1", _collapse(value))


def _collapse(value: str) -> str:
    """Minúsculas sin acentos y con la puntuación vuelta espacio, sin pegar nada.

    Es `_normalize` sin su reparación de la letra suelta. La usa `_has_category`
    para poder mirar la línea también como vino, porque esa reparación acierta
    en `C AFÉ` pero estorba en `O CAFÉ`.
    """
    collapsed = re.sub(r"[^\w\s]+", " ", fold(value))

    return re.sub(r"\s+", " ", collapsed).strip()
