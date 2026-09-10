"""Limpieza del texto crudo devuelto por el motor OCR.

Corrige los errores sistemáticos del reconocimiento sobre etiquetas, sellos y
tickets antes de que los parsers apliquen sus expresiones regulares.
"""

import re

# Confusiones habituales del OCR dentro de secuencias numéricas (fechas,
# cantidades, precios). Sólo se aplican cuando el contexto es numérico.
_DIGIT_CONFUSIONS = str.maketrans(
    {"O": "0", "o": "0", "l": "1", "I": "1", "S": "5", "B": "8", "Z": "2"}
)

_NUMERIC_CONTEXT = re.compile(r"\b(?=[0-9OoIlSBZ]*[0-9])[0-9OoIlSBZ]{2,}\b")

# Sello de vencimiento sin separadores: `27FEB27`. El motor confunde los dígitos
# de los extremos porque están impresos en matriz de puntos, pero el mes del
# medio los delimita sin ambigüedad, así que ahí sí se puede reparar sin riesgo.
_STAMP_TOKEN = re.compile(r"\b([0-9OoIlSBZ]{1,2})([A-Za-z]{3})([0-9OoIlSBZ]{2,4})\b")

_MULTI_SPACE = re.compile(r"[ \t]+")
_BLANK_LINES = re.compile(r"\n{3,}")
# Ruido típico de bordes de etiqueta: líneas de 1-3 símbolos sin letras ni dígitos.
_NOISE_LINE = re.compile(r"^[^\w]{1,3}$")


def fix_numeric_confusions(text: str) -> str:
    """Convierte O→0, l→1, S→5, Z→2… dentro de tokens predominantemente numéricos."""
    return _NUMERIC_CONTEXT.sub(lambda m: m.group(0).translate(_DIGIT_CONFUSIONS), text)


def fix_stamp_tokens(text: str) -> str:
    """Repara los extremos numéricos de un sello tipo `27FEBZ7` → `27FEB27`.

    Se limita a este patrón —dígitos, tres letras, dígitos, sin espacios— porque
    fuera de él la misma sustitución rompería palabras legítimas.
    """
    return _STAMP_TOKEN.sub(
        lambda m: (
            m.group(1).translate(_DIGIT_CONFUSIONS)
            + m.group(2)
            + m.group(3).translate(_DIGIT_CONFUSIONS)
        ),
        text,
    )


def normalize_whitespace(text: str) -> str:
    lines = [_MULTI_SPACE.sub(" ", line).strip() for line in text.splitlines()]
    return _BLANK_LINES.sub("\n\n", "\n".join(lines)).strip()


def drop_noise_lines(text: str) -> str:
    return "\n".join(
        line for line in text.splitlines() if not _NOISE_LINE.match(line.strip())
    )


def clean(text: str) -> str:
    """Cadena completa de limpieza aplicada al texto crudo."""
    return normalize_whitespace(
        drop_noise_lines(fix_stamp_tokens(fix_numeric_confusions(text)))
    )


# Fragmentos que el motor devuelve cuando mira una zona sin texto: letras
# sueltas, símbolos, restos de un logo. No aportan nada a quien revisa.
_GARBAGE_LINE = re.compile(r"^(?:[^\w\s]|\w(?=\s|$)){1,4}$")


def readable(text: str, doc_type: str | None = None) -> str:
    """El texto reconocido en una forma que una persona pueda contrastar.

    `raw_text` conserva la salida cruda con sus saltos de línea, que sirve para
    depurar la estrategia pero es incómoda de leer: el motor parte los renglones
    donde se le da la gana y mete líneas de basura entre medio. Esto devuelve lo
    mismo sin el ruido y en la forma que corresponde a cada documento.

    En una etiqueta el orden de los renglones no significa nada, así que se
    juntan en un solo párrafo —igual que como está escrito el ground truth—. En
    un comprobante cada renglón **es** un dato, así que se conservan.
    """
    lineas = [
        " ".join(linea.split())
        for linea in text.splitlines()
        if linea.strip() and not _GARBAGE_LINE.match(linea.strip())
    ]

    if not lineas:
        return ""

    if doc_type == "receipt":
        return "\n".join(lineas)

    return " ".join(lineas)
