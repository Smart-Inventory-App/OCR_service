"""Cuarto caso de entrada: el código de barras del envase.

Extrae `codigo`, `simbologia` y `pais_registro`. Es el caso más confiable de los
cuatro y también el más fácil de interpretar mal, y las dos cosas tienen el mismo
origen: el número viene con un dígito verificador, así que o se leyó bien o no se
leyó — pero que el número sea correcto no significa que sirva como identificador
del producto.

Tres situaciones, todas frecuentes en el circuito argentino, en las que el código
es válido y aun así no identifica un producto de góndola:

- **Códigos de circulación restringida** (`02`, `20`–`29`). Es lo que imprime la
  balanza del supermercado sobre la fruta, la fiambrería y la carne. El número
  codifica precio o peso y **sólo significa algo dentro de ese comercio**: el
  mismo `2001234500009` es otra cosa en otra cadena. Cargarlo en el inventario
  como si fuera el identificador global del producto es cargar basura.
- **Etiquetas logísticas** (CODE-128, CODE-39). Un encargo llega con la etiqueta
  del correo pegada al lado del envase, y es un número de seguimiento, no un
  producto.
- **ITF-14**. Es el código de la caja de un pedido mayorista. Sí identifica al
  producto —el GTIN-13 del envase individual va adentro— pero no dice cuántas
  unidades trae la caja.

En los tres el campo se devuelve con la confianza baja que corresponde, para que
`needs_review` mande a la validación manual asistida en vez de dar el alta sola.
"""

from datetime import date

from app.decoders.barcode import BarcodeReading, Symbol
from app.parsers import scoring
from app.schemas.ocr import ExtractedField

KEY_FIELDS = ("codigo",)
"""El identificador del producto. `simbologia` y `pais_registro` describen la
lectura pero no bloquean el alta."""

# Nombre legible de cada simbología que devuelve ZBar.
SYMBOLOGY_NAMES = {
    "EAN13": "EAN-13",
    "EAN8": "EAN-8",
    "UPCA": "UPC-A",
    "UPCE": "UPC-E",
    "I25": "ITF",
    "CODE128": "CODE-128",
    "CODE39": "CODE-39",
    "QRCODE": "QR",
}

# Simbologías cuyo dígito verificador valida el propio decodificador: ZBar no
# emite un EAN-13 cuyo check no cierre, así que la lectura se verifica a sí
# misma. Es lo que hace que este tipo de documento no necesite un motor OCR.
SELF_VERIFYING = frozenset({"EAN13", "EAN8", "UPCA", "UPCE", "CODE128", "QRCODE"})

# Simbologías que codifican un GTIN, es decir, el identificador global de un
# producto. El resto puede ser cualquier cosa.
RETAIL = frozenset({"EAN13", "EAN8", "UPCA", "UPCE"})

# Prefijos GS1 → país de la organización donde se registró la empresa.
#
# **No es el país de fabricación**, que es el malentendido habitual con este
# dato: un fideo con prefijo 779 lo registró una empresa argentina, y puede
# estar elaborado en cualquier lado. Por eso el campo se llama `pais_registro` y
# no `pais_origen`.
#
# La tabla es deliberadamente parcial: cubre el Mercosur, el resto de
# Latinoamérica y los orígenes de importación que efectivamente aparecen en una
# alacena argentina. Un prefijo que no está devuelve `None`, que es preferible a
# adivinar.
GS1_PREFIXES: tuple[tuple[range, str], ...] = (
    (range(0, 20), "Estados Unidos / Canadá"),
    (range(30, 40), "Estados Unidos"),
    (range(60, 140), "Estados Unidos / Canadá"),
    (range(300, 380), "Francia"),
    (range(400, 441), "Alemania"),
    (range(450, 460), "Japón"),
    (range(460, 470), "Rusia"),
    (range(500, 510), "Reino Unido"),
    (range(539, 540), "Irlanda"),
    (range(540, 550), "Bélgica / Luxemburgo"),
    (range(560, 561), "Portugal"),
    (range(569, 570), "Islandia"),
    (range(570, 580), "Dinamarca"),
    (range(590, 591), "Polonia"),
    (range(640, 650), "Finlandia"),
    (range(690, 700), "China"),
    (range(700, 710), "Noruega"),
    (range(729, 730), "Israel"),
    (range(730, 740), "Suecia"),
    (range(740, 741), "Guatemala"),
    (range(741, 742), "El Salvador"),
    (range(742, 743), "Honduras"),
    (range(743, 744), "Nicaragua"),
    (range(744, 745), "Costa Rica"),
    (range(745, 746), "Panamá"),
    (range(746, 747), "República Dominicana"),
    (range(750, 751), "México"),
    (range(754, 756), "Canadá"),
    (range(759, 760), "Venezuela"),
    (range(760, 770), "Suiza"),
    (range(770, 772), "Colombia"),
    (range(773, 774), "Uruguay"),
    (range(775, 776), "Perú"),
    (range(777, 778), "Bolivia"),
    (range(778, 780), "Argentina"),
    (range(780, 781), "Chile"),
    (range(784, 785), "Paraguay"),
    (range(786, 787), "Ecuador"),
    (range(789, 791), "Brasil"),
    (range(800, 840), "Italia"),
    (range(840, 850), "España"),
    (range(858, 859), "Eslovaquia"),
    (range(859, 860), "Chequia"),
    (range(867, 868), "Corea del Norte"),
    (range(868, 870), "Turquía"),
    (range(870, 880), "Países Bajos"),
    (range(880, 881), "Corea del Sur"),
    (range(885, 886), "Tailandia"),
    (range(888, 889), "Singapur"),
    (range(890, 891), "India"),
    (range(893, 894), "Vietnam"),
    (range(899, 900), "Indonesia"),
    (range(900, 920), "Austria"),
    (range(930, 940), "Australia"),
    (range(940, 950), "Nueva Zelanda"),
    (range(955, 956), "Malasia"),
)

# Prefijos que no son geográficos: no hay país que reportar aunque el código sea
# perfectamente válido.
RESTRICTED_PREFIXES = (range(20, 30), range(40, 50), range(200, 300))
"""Circulación restringida: el número lo asigna el propio comercio."""

COUPON_PREFIXES = (range(50, 60), range(980, 1000))
PUBLICATION_PREFIXES = (range(977, 980),)

# Penalizaciones del parser. Cada una responde a "¿esto identifica al producto
# que el usuario quiere cargar?", que es una pregunta distinta de "¿se leyó bien?".
CONF_RESTRICTED = 0.5
"""Código interno del comercio: fuera de ese comercio no identifica nada."""

CONF_NON_RETAIL = 0.6
"""CODE-128, CODE-39 o QR: casi siempre una etiqueta de envío o una URL."""

CONF_CARTON = 0.9
"""ITF-14: identifica el producto, pero no cuántas unidades trae la caja."""

CONF_AMBIGUOUS = 0.7
"""Más de un código distinto en la foto: no está claro cuál es el producto."""

CONF_UNVERIFIED = 0.9
"""Simbología sin dígito verificador obligatorio (ITF suelto, CODE-39)."""


def parse(
    reading: BarcodeReading, today: date | None = None
) -> dict[str, ExtractedField]:
    """Extrae los campos de una lectura de códigos de barras.

    `today` se acepta por simetría con los otros parsers; un código de barras no
    tiene campos que dependan de la fecha actual.
    """
    symbol = _pick(reading.symbols)
    if symbol is None:
        return {
            "codigo": scoring.missing(),
            "simbologia": scoring.missing(),
            "pais_registro": scoring.missing(),
        }

    code, symbology = _normalize(symbol)
    if code is None:
        # El dígito verificador no cierra: el número existe pero no es un GTIN
        # válido. Ofrecerlo sería peor que no ofrecer nada, porque el usuario no
        # tiene cómo notar que está mal.
        return {
            "codigo": scoring.missing(),
            "simbologia": _field(SYMBOLOGY_NAMES.get(symbol.symbology, symbol.symbology), 1.0),
            "pais_registro": scoring.missing(),
        }

    ambiguous = _distinct_codes(reading.symbols) > 1

    return {
        "codigo": scoring.field(
            value=code,
            engine_conf=decoder_confidence(symbol),
            parser_conf=parser_confidence(symbol, code, ambiguous),
        ),
        "simbologia": _field(symbology, 1.0),
        "pais_registro": _country_field(symbol, code),
    }


def _pick(symbols: list[Symbol]) -> Symbol | None:
    """Cuál de los códigos de la foto es el del producto.

    Un envase fotografiado de cerca suele traer un solo código, pero un encargo
    llega con la etiqueta del correo al lado y una caja mayorista tiene el ITF-14
    y el EAN-13 del envase impresos juntos. El orden de preferencia es:

    1. un código de góndola (EAN/UPC), que es lo que el usuario quiere cargar;
    2. si no hay, el más grande de los que haya — el que domina el encuadre es
       el que el usuario apuntó con la cámara.
    """
    if not symbols:
        return None

    retail = [symbol for symbol in symbols if symbol.symbology in RETAIL]

    return max(retail or symbols, key=lambda symbol: symbol.width * symbol.height)


def _distinct_codes(symbols: list[Symbol]) -> int:
    """Cuántos códigos **distintos** hay en la foto.

    Se cuentan los valores y no los símbolos: ZBar puede devolver el mismo código
    dos veces si la imagen lo contiene repetido, y eso no es una ambigüedad.
    """
    return len({symbol.data for symbol in symbols})


# ── Normalización a GTIN ──────────────────────────────────────────────────────


def _normalize(symbol: Symbol) -> tuple[str | None, str]:
    """Lleva el código a su forma canónica y devuelve `(código, simbología)`.

    Todo lo que sea un GTIN se normaliza a **EAN-13**, que es la forma con la que
    conviene indexar el inventario: así el mismo producto comprado en la góndola
    local y encargado a un importador —que trae UPC-A— cae en la misma clave en
    vez de generar dos productos distintos.
    """
    data = symbol.data
    kind = symbol.symbology
    name = SYMBOLOGY_NAMES.get(kind, kind)

    if kind == "UPCE":
        expanded = expand_upce(data)
        if expanded is None:
            return None, name
        data, kind = expanded, "UPCA"

    if kind == "UPCA":
        # Un UPC-A es un EAN-13 con un cero adelante. Es la misma numeración.
        return _validated("0" + data), f"{name} → EAN-13"

    if kind in ("EAN13", "EAN8"):
        return _validated(data), name

    if kind == "I25":
        return _normalize_itf(data, name)

    # CODE-128, CODE-39, QR: no son GTIN. Se devuelven tal cual, y la confianza
    # del parser es la que dice que no hay que confiarles el alta.
    return data, name


def _normalize_itf(data: str, name: str) -> tuple[str | None, str]:
    """Un ITF de 14 dígitos con check válido es la caja de un producto.

    El GTIN-14 lleva un dígito indicador adelante, los 12 dígitos del GTIN-13 del
    envase individual en el medio y su propio verificador al final. Sacando el
    indicador y recalculando el verificador se recupera el código del producto
    que hay adentro, que es el que el usuario tiene en la alacena.
    """
    if len(data) == 14 and _check_digit(data[:-1]) == int(data[-1]):
        body = data[1:13]
        return body + str(_check_digit(body)), "ITF-14 → EAN-13"

    # Cualquier otro ITF es un código de uso interno de una cadena logística.
    return data, name


def _validated(code: str) -> str | None:
    """Devuelve el código sólo si su dígito verificador cierra."""
    if not code.isdigit() or len(code) < 8:
        return None

    return code if _check_digit(code[:-1]) == int(code[-1]) else None


def _check_digit(body: str) -> int:
    """Dígito verificador GS1 (módulo 10).

    Sirve igual para EAN-8, EAN-13, UPC-A y GTIN-14: los pesos 3 y 1 se alternan
    **desde la derecha**, así que la fórmula no depende del largo del código.
    """
    total = sum(
        int(digit) * (3 if index % 2 == 0 else 1)
        for index, digit in enumerate(reversed(body))
    )

    return (10 - total % 10) % 10


def expand_upce(code: str) -> str | None:
    """Expande un UPC-E de 8 dígitos al UPC-A de 12 que representa.

    UPC-E es un UPC-A comprimido, quitándole ceros según el último dígito del
    cuerpo. Se expande en vez de devolverlo tal cual porque comprimido no se
    puede comparar con nada: el mismo producto en su presentación grande trae el
    UPC-A entero, y sin expandir serían dos productos distintos en el inventario.
    """
    if not code.isdigit() or len(code) != 8:
        return None

    system, body, check = code[0], code[1:7], code[7]
    if system not in ("0", "1"):
        return None

    last = body[5]
    if last in ("0", "1", "2"):
        expanded = f"{body[:2]}{last}0000{body[2:5]}"
    elif last == "3":
        expanded = f"{body[:3]}00000{body[3:5]}"
    elif last == "4":
        expanded = f"{body[:4]}00000{body[4]}"
    else:
        expanded = f"{body[:5]}0000{last}"

    upca = f"{system}{expanded}"

    return upca + check if _check_digit(upca) == int(check) else None


# ── Confianza ─────────────────────────────────────────────────────────────────


def decoder_confidence(symbol: Symbol) -> float:
    """Qué tan segura es la lectura en sí misma.

    Acá no hay una confianza por palabra como la que da un motor OCR, y tampoco
    hace falta inventarla: una simbología con dígito verificador obligatorio se
    valida sola, y ZBar no emite el símbolo si no cierra. Una lectura que llegó
    hasta acá es correcta, y decir otra cosa sería fingir una duda que no existe.

    Las que no tienen verificador obligatorio —el ITF suelto, CODE-39 sin check—
    sí pueden devolver un número mal leído, y ahí la confianza baja. La excepción
    es el ITF de 14 dígitos: la simbología no obliga a llevar verificador, pero
    el GTIN-14 sí lo lleva, y este parser lo recalcula. Un ITF-14 cuyo check
    cierra quedó verificado igual que un EAN-13, sólo que por nosotros.
    """
    if symbol.symbology in SELF_VERIFYING:
        return 1.0

    if _is_gtin14(symbol.data):
        return 1.0

    return CONF_UNVERIFIED


def _is_gtin14(data: str) -> bool:
    return len(data) == 14 and data.isdigit() and _check_digit(data[:-1]) == int(data[-1])


def parser_confidence(symbol: Symbol, code: str, ambiguous: bool) -> float:
    """Qué tan seguro es que este código identifique al producto del usuario.

    Es una pregunta distinta de si se leyó bien, y es la que decide si el alta
    puede darse sola. Un código de balanza se lee perfecto y no sirve para
    identificar nada fuera de ese supermercado.
    """
    confidence = 1.0

    if symbol.symbology not in RETAIL and symbol.symbology != "I25":
        confidence *= CONF_NON_RETAIL
    elif is_restricted(code):
        confidence *= CONF_RESTRICTED
    elif symbol.symbology == "I25":
        confidence *= CONF_CARTON if _is_gtin14(symbol.data) else CONF_NON_RETAIL

    if ambiguous:
        confidence *= CONF_AMBIGUOUS

    return confidence


def is_restricted(code: str) -> bool:
    """True si el código es de circulación restringida (interno de un comercio).

    Son los que imprime la balanza del súper sobre lo que se vende por peso. El
    prefijo se mira sobre el EAN-13 ya normalizado; un EAN-8 usa dos dígitos.
    """
    if not code.isdigit():
        return False

    if len(code) == 8:
        return code[0] in ("0", "2")

    return any(int(code[:3]) in prefixes for prefixes in RESTRICTED_PREFIXES)


def country_of(code: str) -> str | None:
    """País de la organización GS1 en la que se registró el código.

    Devuelve `None` cuando el prefijo no es geográfico —circulación restringida,
    cupones, ISBN/ISSN— o cuando no está en la tabla. Nunca adivina.
    """
    if not code.isdigit() or len(code) != 13:
        return None

    prefix = int(code[:3])

    for prefixes in (*RESTRICTED_PREFIXES, *COUPON_PREFIXES, *PUBLICATION_PREFIXES):
        if prefix in prefixes:
            return None

    for prefixes, country in GS1_PREFIXES:
        if prefix in prefixes:
            return country

    return None


# ── Construcción de campos ────────────────────────────────────────────────────


def _country_field(symbol: Symbol, code: str) -> ExtractedField:
    if symbol.symbology not in RETAIL and symbol.symbology != "I25":
        return scoring.missing()

    country = country_of(code)
    if country is None:
        return scoring.missing()

    # El prefijo se lee del mismo número ya verificado: si el código es correcto,
    # el país que sale de su prefijo también lo es.
    return _field(country, decoder_confidence(symbol))


def _field(value: str, confidence: float) -> ExtractedField:
    return scoring.field(value=value, engine_conf=confidence, parser_conf=1.0)
