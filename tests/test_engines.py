"""Interfaz de motores, factory y modo híbrido.

El binario de Tesseract se mockea: la CI no depende de que esté instalado.
"""

from unittest.mock import patch

import numpy as np
import pytest

from app.engines import factory
from app.engines.base import OCREngine, OCRWord, RawOCROutput
from app.engines.easyocr_engine import EasyOCREngine
from app.engines.factory import HybridEngine
from app.engines.tesseract_engine import TesseractEngine
from tests.conftest import FakeEngine, make_output

IMAGE = np.zeros((40, 120), dtype=np.uint8)


@pytest.fixture(autouse=True)
def clear_engine_cache():
    factory.reset_cache()
    yield
    factory.reset_cache()


# ---------------------------------------------------------------------------
# RawOCROutput
# ---------------------------------------------------------------------------

def test_mean_confidence_is_weighted_by_word_length():
    output = RawOCROutput(
        text="",
        words=[
            OCRWord("abcdefghij", 1.0, 0, 0, 10, 10),  # 10 caracteres
            OCRWord("x", 0.0, 0, 0, 10, 10),           # 1 carácter
        ],
    )

    assert output.mean_confidence == pytest.approx(10 / 11)


def test_mean_confidence_without_words_is_zero():
    assert RawOCROutput(text="").mean_confidence == 0.0


def test_lines_groups_and_orders_words():
    output = make_output(["hola mundo", "segunda linea"])

    assert [len(line) for line in output.lines()] == [2, 2]
    assert output.text == "hola mundo\nsegunda linea"


def test_from_words_derives_text_in_reading_order():
    words = [
        OCRWord("mundo", 0.9, 60, 0, 40, 20, line_num=0),
        OCRWord("hola", 0.9, 10, 0, 40, 20, line_num=0),
        OCRWord("chau", 0.9, 10, 40, 40, 20, line_num=1),
    ]

    output = RawOCROutput.from_words(words, "test")

    assert output.text == "hola mundo\nchau"
    assert output.engine == "test"


def test_fake_engine_satisfies_the_protocol():
    assert isinstance(FakeEngine(["x"]), OCREngine)


def test_real_engines_satisfy_the_protocol():
    assert isinstance(TesseractEngine(), OCREngine)
    assert isinstance(EasyOCREngine(), OCREngine)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def test_factory_returns_tesseract_by_default():
    assert factory.build_engine().name == "tesseract"


def test_factory_selects_engine_by_name():
    assert factory.build_engine("easyocr").name == "easyocr"


def test_factory_is_case_insensitive():
    assert factory.build_engine("TESSERACT").name == "tesseract"


def test_factory_memoizes_instances():
    assert factory.build_engine("tesseract") is factory.build_engine("tesseract")


def test_factory_rejects_unknown_engine():
    with pytest.raises(ValueError, match="desconocido"):
        factory.build_engine("no-existe")


def test_factory_builds_hybrid(monkeypatch):
    from app.core.config import settings

    monkeypatch.setattr(settings, "HYBRID_FALLBACK_ENGINE", "easyocr")

    engine = factory.build_engine("hybrid")

    assert isinstance(engine, HybridEngine)
    assert engine.name == "hybrid(tesseract+easyocr)"


def test_hybrid_rejects_recursive_fallback(monkeypatch):
    from app.core.config import settings

    monkeypatch.setattr(settings, "HYBRID_FALLBACK_ENGINE", "hybrid")

    with pytest.raises(ValueError, match="motor concreto"):
        factory.build_engine("hybrid")


# ---------------------------------------------------------------------------
# Modo híbrido
# ---------------------------------------------------------------------------

def test_hybrid_keeps_primary_when_confidence_is_high():
    primary = FakeEngine(["texto nitido"], confidence=0.95)
    fallback = FakeEngine(["otro texto"], confidence=0.99)

    result = HybridEngine(primary, fallback, threshold=0.70).run(IMAGE)

    assert result.engine == "fake"
    assert fallback.calls == []  # el respaldo no se invocó


def test_hybrid_retries_with_fallback_when_confidence_is_low():
    primary = FakeEngine(["borroso"], confidence=0.30)
    fallback = FakeEngine(["nitido"], confidence=0.90)

    result = HybridEngine(primary, fallback, threshold=0.70).run(IMAGE)

    assert len(fallback.calls) == 1
    assert result.mean_confidence == pytest.approx(0.90)


def test_hybrid_keeps_primary_if_fallback_is_worse():
    primary = FakeEngine(["algo"], confidence=0.60)
    fallback = FakeEngine(["peor"], confidence=0.10)

    result = HybridEngine(primary, fallback, threshold=0.70).run(IMAGE)

    assert result.mean_confidence == pytest.approx(0.60)


def test_hybrid_survives_unavailable_fallback():
    primary = FakeEngine(["borroso"], confidence=0.30)

    class Unavailable(FakeEngine):
        def is_available(self) -> bool:
            return False

    fallback = Unavailable(["nunca"], confidence=0.99)
    result = HybridEngine(primary, fallback, threshold=0.70).run(IMAGE)

    assert fallback.calls == []
    assert result.mean_confidence == pytest.approx(0.30)


def test_hybrid_propagates_layout_hint():
    primary = FakeEngine(["x"], confidence=0.95)

    HybridEngine(primary, FakeEngine(["y"]), threshold=0.70).run(IMAGE, "spa", "sparse")

    assert primary.calls[0]["layout"] == "sparse"


# ---------------------------------------------------------------------------
# Motor Tesseract (con pytesseract mockeado)
# ---------------------------------------------------------------------------

_TESSERACT_DATA = {
    "text": ["Leche", "Entera", "", "VTO"],
    "conf": [96, 92, -1, 88],
    "left": [10, 80, 0, 10],
    "top": [10, 10, 0, 60],
    "width": [60, 70, 0, 40],
    "height": [30, 30, 0, 20],
    "block_num": [1, 1, 1, 1],
    "par_num": [0, 0, 0, 0],
    "line_num": [0, 0, 0, 1],
}


def test_tesseract_maps_data_to_words():
    with patch("pytesseract.image_to_data", return_value=_TESSERACT_DATA), \
         patch("pytesseract.get_languages", return_value=["spa", "eng"]):
        output = TesseractEngine().run(IMAGE, lang="spa")

    assert output.engine == "tesseract"
    assert output.text == "Leche Entera\nVTO"
    # Las cajas sin texto reconocido (conf = -1) se descartan.
    assert len(output.words) == 3
    assert output.words[0].confidence == pytest.approx(0.96)


def test_tesseract_maps_layout_to_psm():
    with patch("pytesseract.image_to_data", return_value=_TESSERACT_DATA) as mock, \
         patch("pytesseract.get_languages", return_value=["spa"]):
        TesseractEngine().run(IMAGE, lang="spa", layout="sparse")

    assert mock.call_args.kwargs["config"] == "--psm 11"


def test_tesseract_falls_back_to_english_when_spanish_is_missing():
    with patch("pytesseract.image_to_data", return_value=_TESSERACT_DATA) as mock, \
         patch("pytesseract.get_languages", return_value=["eng"]):
        TesseractEngine().run(IMAGE, lang="spa")

    assert mock.call_args.kwargs["lang"] == "eng"


def test_tesseract_availability_reflects_the_binary():
    with patch("pytesseract.get_tesseract_version", return_value="5.3.0"):
        assert TesseractEngine().is_available() is True

    with patch("pytesseract.get_tesseract_version", side_effect=OSError("not found")):
        assert TesseractEngine().is_available() is False


def test_tesseract_line_ids_are_unique_across_blocks():
    data = {**_TESSERACT_DATA, "block_num": [1, 2, 1, 1], "line_num": [0, 0, 0, 0]}

    with patch("pytesseract.image_to_data", return_value=data), \
         patch("pytesseract.get_languages", return_value=["spa"]):
        output = TesseractEngine().run(IMAGE)

    # Palabras en bloques distintos no se mezclan en la misma línea.
    assert len(output.lines()) == 2


# ---------------------------------------------------------------------------
# Motor EasyOCR (opcional: no se instala en el MVP)
# ---------------------------------------------------------------------------

def test_easyocr_availability_depends_on_the_package():
    engine = EasyOCREngine()

    # En el deploy del MVP la dependencia no está: debe reportarlo, no explotar.
    assert engine.is_available() in (True, False)


def test_easyocr_converts_detections_to_words():
    engine = EasyOCREngine()
    detections = [
        ([[10, 10], [70, 10], [70, 40], [10, 40]], "Leche", 0.95),
        ([[80, 12], [150, 12], [150, 40], [80, 40]], "Entera", 0.91),
        ([[10, 80], [50, 80], [50, 105], [10, 105]], "VTO", 0.88),
    ]

    class Reader:
        def readtext(self, _image):
            return detections

    engine._reader = Reader()
    output = engine.run(IMAGE)

    assert output.engine == "easyocr"
    assert output.text == "Leche Entera\nVTO"
