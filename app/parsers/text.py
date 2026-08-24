"""Utilidades de texto compartidas por los parsers."""

import unicodedata

from app.engines.base import OCRWord


def fold(value: str) -> str:
    """Minúsculas sin acentos, para comparar contra palabras clave."""
    decomposed = unicodedata.normalize("NFKD", value.lower())
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def line_text(line: list[OCRWord]) -> str:
    """Texto de una línea, en orden de lectura."""
    return " ".join(word.text for word in line).strip()
