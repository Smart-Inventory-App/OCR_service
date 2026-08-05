"""Motor Tesseract — motor inicial del MVP.

La propuesta lo elige como punto de partida por ser open source, no requerir GPU
y tener bajo costo computacional ("adecuada para un MVP o entornos con recursos
limitados").
"""

import logging

import numpy as np
import pytesseract
from pytesseract import Output

from app.core.config import settings
from app.engines.base import OCRWord, RawOCROutput

logger = logging.getLogger(__name__)

# `layout` es una pista agnóstica del motor; acá se traduce al page segmentation
# mode de Tesseract.
_PSM_BY_LAYOUT = {
    "block": 6,   # bloque uniforme de texto (tickets, remitos)
    "sparse": 11,  # texto disperso sin orden garantizado (etiquetas de producto)
}

_FALLBACK_LANG = "eng"

if settings.TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = settings.TESSERACT_CMD


class TesseractEngine:
    name = "tesseract"

    def is_available(self) -> bool:
        try:
            pytesseract.get_tesseract_version()
            return True
        except Exception as exc:
            logger.warning("Tesseract no disponible: %s", exc)
            return False

    def run(
        self,
        image: np.ndarray,
        lang: str = "spa",
        layout: str = "block",
    ) -> RawOCROutput:
        psm = _PSM_BY_LAYOUT.get(layout, _PSM_BY_LAYOUT["block"])

        data = pytesseract.image_to_data(
            image,
            lang=self._resolve_lang(lang),
            config=f"--psm {psm}",
            output_type=Output.DICT,
        )

        words = [
            word for word in map(_to_word, *_columns(data)) if word is not None
        ]

        return RawOCROutput.from_words(words, self.name)

    def _resolve_lang(self, lang: str) -> str:
        """Cae a inglés si falta el paquete de idioma pedido (dev sin `spa`)."""
        try:
            available = set(pytesseract.get_languages(config=""))
        except Exception:
            return lang

        if lang in available:
            return lang

        logger.warning(
            "Idioma '%s' no instalado en Tesseract; se usa '%s'", lang, _FALLBACK_LANG
        )
        return _FALLBACK_LANG if _FALLBACK_LANG in available else lang


def _columns(data: dict) -> tuple[list, ...]:
    return (
        data["text"],
        data["conf"],
        data["left"],
        data["top"],
        data["width"],
        data["height"],
        data["block_num"],
        data["par_num"],
        data["line_num"],
    )


def _to_word(
    text: str,
    conf,
    left: int,
    top: int,
    width: int,
    height: int,
    block_num: int,
    par_num: int,
    line_num: int,
) -> OCRWord | None:
    if not str(text).strip():
        return None

    confidence = float(conf)
    if confidence < 0:  # Tesseract marca con -1 las cajas sin texto reconocido
        return None

    return OCRWord(
        text=str(text),
        confidence=confidence / 100.0,
        left=int(left),
        top=int(top),
        width=int(width),
        height=int(height),
        # Identificador de línea único a lo largo de todos los bloques y párrafos,
        # monótono en orden de lectura.
        line_num=int(block_num) * 10_000 + int(par_num) * 100 + int(line_num),
    )
