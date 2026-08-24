"""Interfaz común de los motores OCR.

La propuesta exige una evaluación comparativa entre Tesseract, PaddleOCR y
EasyOCR, y "no descarta" un enfoque híbrido. Todo el resto del servicio depende
únicamente de esta interfaz, de modo que agregar o cambiar un motor no toca los
parsers, el scoring ni las rutas.

Desde la separación en tres repositorios, los motores ya no corren acá: cada uno
es un servicio con su propia imagen y esta interfaz es el contrato que hablan.
Por eso `run` recibe **bytes** y no una matriz de OpenCV — este servicio no
decodifica imágenes, las reenvía.
"""

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from app.pipeline.transform import Transform


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
    elapsed_ms: int = 0

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
    def from_words(cls, words: list[OCRWord], engine: str, elapsed_ms: int = 0) -> "RawOCROutput":
        """Construye la salida derivando el texto de las palabras detectadas."""
        output = cls(text="", words=words, engine=engine, elapsed_ms=elapsed_ms)
        output.text = output.render_text()
        return output


@runtime_checkable
class OCREngine(Protocol):
    """Contrato de un motor OCR visto desde el coordinador."""

    name: str

    supports_preprocessing: bool
    """Si al motor le sirve que el coordinador varíe la cadena de filtros.

    Tesseract: sí, su precisión depende fuertemente de ellos. EasyOCR: no,
    prepara su propia entrada y cada corrida cuesta segundos. El coordinador lo
    consulta para no gastar reintentos que no van a cambiar nada.
    """

    def run(
        self,
        content: bytes,
        lang: str = "spa",
        layout: str = "block",
        transform: Transform | None = None,
    ) -> RawOCROutput:
        """Reconoce el texto de la imagen original, aplicando `transform`."""
        ...

    def is_available(self) -> bool:
        """True si el motor puede ejecutarse (servicio arriba y motor cargado)."""
        ...
