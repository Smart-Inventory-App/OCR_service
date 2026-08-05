"""Pipeline de preprocesamiento OpenCV."""

import cv2
import numpy as np

from app.pipeline import postprocess, preprocess
from app.pipeline.preprocess import PreprocessConfig


def test_grayscale_reduces_to_two_dimensions():
    color = np.zeros((50, 80, 3), dtype=np.uint8)

    assert preprocess.to_grayscale(color).ndim == 2


def test_grayscale_is_idempotent(gray_image):
    once = preprocess.to_grayscale(gray_image)

    assert preprocess.to_grayscale(once).shape == once.shape


def test_upscale_reaches_min_width_and_keeps_aspect_ratio():
    small = np.full((100, 200), 200, dtype=np.uint8)

    result = preprocess.upscale(small, min_width=1000)

    assert result.shape[1] == 1000
    assert result.shape[0] == 500  # se preserva la relación 1:2


def test_upscale_leaves_large_images_untouched():
    large = np.full((100, 1500), 200, dtype=np.uint8)

    assert preprocess.upscale(large, min_width=1000).shape == large.shape


def test_threshold_returns_binary_image(gray_image):
    result = preprocess.threshold(gray_image)

    assert set(np.unique(result)).issubset({0, 255})


def test_deskew_corrects_rotated_text():
    """Un bloque de texto rotado vuelve a quedar aproximadamente horizontal."""
    image = np.full((300, 600), 255, dtype=np.uint8)
    image[140:160, 100:500] = 0

    matrix = cv2.getRotationMatrix2D((300, 150), 8, 1.0)
    rotated = cv2.warpAffine(image, matrix, (600, 300), borderValue=255)

    corrected = preprocess.deskew(rotated)

    assert _skew_angle(corrected) < _skew_angle(rotated)


def test_deskew_leaves_straight_image_untouched(gray_image):
    result = preprocess.deskew(gray_image)

    assert np.array_equal(result, gray_image)


def test_deskew_handles_blank_image():
    blank = np.full((100, 100), 255, dtype=np.uint8)

    assert np.array_equal(preprocess.deskew(blank), blank)


def test_run_applies_full_chain(gray_image):
    result = preprocess.run(gray_image, PreprocessConfig(min_width=1000))

    assert result.ndim == 2
    assert result.shape[1] >= 1000
    assert set(np.unique(result)).issubset({0, 255})


def test_run_with_all_steps_disabled_is_a_noop(gray_image):
    config = PreprocessConfig(
        grayscale=False, upscale=False, denoise=False, threshold=False, deskew=False
    )

    assert np.array_equal(preprocess.run(gray_image, config), gray_image)


def test_config_reads_from_settings(monkeypatch):
    from app.core.config import settings

    monkeypatch.setattr(settings, "PREPROCESS_DENOISE", False)
    monkeypatch.setattr(settings, "PREPROCESS_MIN_WIDTH", 1234)

    config = PreprocessConfig.from_settings()

    assert config.denoise is False
    assert config.min_width == 1234


def test_run_accepts_color_input():
    color = np.full((120, 300, 3), 220, dtype=np.uint8)
    color[40:80, 20:200] = 20

    result = preprocess.run(color, PreprocessConfig(min_width=600))

    assert result.ndim == 2


def _skew_angle(image: np.ndarray) -> float:
    """Desviación respecto de la horizontal, en grados (0 = recto)."""
    binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    coords = cv2.findNonZero(binary)
    angle = cv2.minAreaRect(coords)[-1] % 90

    if angle > 45:
        angle -= 90

    return abs(angle)


# ---------------------------------------------------------------------------
# Postprocesamiento de texto
# ---------------------------------------------------------------------------

def test_numeric_confusions_are_fixed_only_in_numeric_context():
    assert postprocess.fix_numeric_confusions("VTO 15/O8/2O27") == "VTO 15/08/2027"


def test_letters_are_not_touched_outside_numeric_tokens():
    assert "LOTE" in postprocess.fix_numeric_confusions("LOTE ABC")


def test_whitespace_is_normalized():
    assert postprocess.normalize_whitespace("  Leche   Entera  \n\n\n\n  1L ") == (
        "Leche Entera\n\n1L"
    )


def test_noise_lines_are_dropped():
    assert postprocess.drop_noise_lines("Leche\n--\n1L") == "Leche\n1L"


def test_clean_runs_the_full_chain():
    raw = "  LECHE   ENTERA \n~~\n VTO  15/O8/2O27  "

    assert postprocess.clean(raw) == "LECHE ENTERA\nVTO 15/08/2027"


def test_clean_on_empty_text():
    assert postprocess.clean("") == ""
