"""Limpieza del texto crudo devuelto por el motor OCR.

Corrige los errores sistemáticos de Tesseract sobre etiquetas y tickets antes
de que los parsers apliquen sus expresiones regulares.
"""

import re

# Confusiones habituales del OCR dentro de secuencias numéricas (fechas,
# cantidades, precios). Sólo se aplican cuando el contexto es numérico.
_DIGIT_CONFUSIONS = str.maketrans({"O": "0", "o": "0", "l": "1", "I": "1", "S": "5", "B": "8"})

_NUMERIC_CONTEXT = re.compile(r"\b(?=[0-9OoIlSB]*[0-9])[0-9OoIlSB]{2,}\b")
_MULTI_SPACE = re.compile(r"[ \t]+")
_BLANK_LINES = re.compile(r"\n{3,}")
# Ruido típico de bordes de etiqueta: líneas de 1-3 símbolos sin letras ni dígitos.
_NOISE_LINE = re.compile(r"^[^\w]{1,3}$")


def fix_numeric_confusions(text: str) -> str:
    """Convierte O→0, l→1, S→5… dentro de tokens predominantemente numéricos."""
    return _NUMERIC_CONTEXT.sub(lambda m: m.group(0).translate(_DIGIT_CONFUSIONS), text)


def normalize_whitespace(text: str) -> str:
    lines = [_MULTI_SPACE.sub(" ", line).strip() for line in text.splitlines()]
    return _BLANK_LINES.sub("\n\n", "\n".join(lines)).strip()


def drop_noise_lines(text: str) -> str:
    return "\n".join(
        line for line in text.splitlines() if not _NOISE_LINE.match(line.strip())
    )


def clean(text: str) -> str:
    """Cadena completa de limpieza aplicada al texto crudo."""
    return normalize_whitespace(drop_noise_lines(fix_numeric_confusions(text)))
