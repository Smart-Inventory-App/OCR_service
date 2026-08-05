"""Interfaz común de los motores OCR.

La propuesta exige una evaluación comparativa entre Tesseract, PaddleOCR y
EasyOCR, y "no descarta" un enfoque híbrido. Todo el resto del servicio depende
únicamente de esta interfaz, de modo que cambiar de motor no toca el pipeline,
los parsers ni las rutas.
"""

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class OCRWord:
    """Una palabra reconocida con su caja y su confianza normalizada a 0-1."""

    text: str
    confidence: float
    left: int
    top: int
    width: int
    height: int
    line_num: int = 0

    @property
    def bottom(self) -> int:
        return self.top + self.height

    @property
    def area(self) -> int:
        return self.width * self.height


@dataclass
class RawOCROutput:
    """Salida cruda de un motor: texto plano + palabras con confianza."""

    text: str
    words: list[OCRWord] = field(default_factory=list)
    engine: str = ""

    @property
    def mean_confidence(self) -> float:
        """Confianza media ponderada por longitud de palabra."""
        scored = [w for w in self.words if w.text.strip()]
        if not scored:
            return 0.0

        total_weight = sum(len(w.text) for w in scored)
        if total_weight == 0:
            return 0.0

        return sum(w.confidence * len(w.text) for w in scored) / total_weight

    def lines(self) -> list[list[OCRWord]]:
        """Agrupa las palabras por línea, preservando el orden de lectura."""
        grouped: dict[int, list[OCRWord]] = {}
        for word in self.words:
            if word.text.strip():
                grouped.setdefault(word.line_num, []).append(word)

        return [
            sorted(words, key=lambda w: w.left)
            for _, words in sorted(grouped.items())
        ]

    def render_text(self) -> str:
        """Texto plano reconstruido a partir del agrupamiento por líneas."""
        return "\n".join(" ".join(w.text for w in line) for line in self.lines())

    @classmethod
    def from_words(cls, words: list[OCRWord], engine: str) -> "RawOCROutput":
        """Construye la salida derivando el texto de las palabras detectadas."""
        output = cls(text="", words=words, engine=engine)
        output.text = output.render_text()
        return output


@runtime_checkable
class OCREngine(Protocol):
    name: str

    def run(self, image: np.ndarray, lang: str = "spa") -> RawOCROutput: ...

    def is_available(self) -> bool:
        """True si el motor puede ejecutarse (binario/modelo presente)."""
        ...
