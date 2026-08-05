"""Motor EasyOCR — alternativa evaluada y motor de respaldo del modo híbrido.

La propuesta lo describe como "buena relación entre facilidad de uso y precisión",
con mayor consumo de memoria y tiempo que Tesseract en CPU. Su dependencia
(`easyocr` + PyTorch) NO se instala en el deploy del MVP: el import es diferido,
de modo que el servicio arranca igual sin ella y `is_available()` devuelve False.
"""

import logging

import numpy as np

from app.engines.base import OCRWord, RawOCROutput

logger = logging.getLogger(__name__)

# Alto en píxeles por debajo del cual dos cajas se consideran de la misma línea.
_LINE_TOLERANCE_RATIO = 0.6


class EasyOCREngine:
    name = "easyocr"

    def __init__(self) -> None:
        self._reader = None

    def is_available(self) -> bool:
        try:
            import easyocr  # noqa: F401
            return True
        except ImportError:
            return False

    def run(
        self,
        image: np.ndarray,
        lang: str = "spa",
        layout: str = "block",
    ) -> RawOCROutput:
        reader = self._get_reader(lang)
        detections = reader.readtext(image)

        words = [_to_word(box, text, conf) for box, text, conf in detections if text.strip()]

        return RawOCROutput.from_words(_assign_lines(words), self.name)

    def _get_reader(self, lang: str):
        """Carga el modelo una sola vez (es costoso) y lo memoiza."""
        if self._reader is None:
            import easyocr

            # EasyOCR usa códigos ISO-639-1: `spa` → `es`.
            code = {"spa": "es", "eng": "en"}.get(lang, lang)
            self._reader = easyocr.Reader([code], gpu=False, verbose=False)

        return self._reader


def _to_word(box, text: str, confidence: float) -> OCRWord:
    xs = [int(point[0]) for point in box]
    ys = [int(point[1]) for point in box]

    return OCRWord(
        text=text,
        confidence=float(confidence),
        left=min(xs),
        top=min(ys),
        width=max(xs) - min(xs),
        height=max(ys) - min(ys),
    )


def _assign_lines(words: list[OCRWord]) -> list[OCRWord]:
    """EasyOCR no devuelve número de línea: se agrupa por proximidad vertical."""
    if not words:
        return []

    ordered = sorted(words, key=lambda w: (w.top, w.left))
    tolerance = max(1, int(_median_height(ordered) * _LINE_TOLERANCE_RATIO))

    result: list[OCRWord] = []
    line_num = 0
    line_top = ordered[0].top

    for word in ordered:
        if word.top - line_top > tolerance:
            line_num += 1
            line_top = word.top
        result.append(
            OCRWord(
                text=word.text,
                confidence=word.confidence,
                left=word.left,
                top=word.top,
                width=word.width,
                height=word.height,
                line_num=line_num,
            )
        )

    return result


def _median_height(words: list[OCRWord]) -> float:
    heights = sorted(w.height for w in words)
    return heights[len(heights) // 2] or 1.0
