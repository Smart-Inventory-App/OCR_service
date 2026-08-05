"""Preprocesamiento de imágenes con OpenCV.

La propuesta señala que Tesseract "requiere preprocesamiento (OpenCV) para
obtener buenos resultados" en escenarios reales (tipografías variadas, fondos
complejos, baja iluminación, imágenes inclinadas).

Cada paso es una función pura `np.ndarray -> np.ndarray` y la cadena se arma a
partir de un `PreprocessConfig`, de modo que el benchmark pueda activar y
desactivar pasos para medir cuáles mejoran la precisión.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from app.core.config import settings


@dataclass(frozen=True)
class PreprocessConfig:
    grayscale: bool = True
    upscale: bool = True
    denoise: bool = True
    threshold: bool = True
    deskew: bool = True
    min_width: int = 1000

    @classmethod
    def from_settings(cls) -> "PreprocessConfig":
        return cls(
            grayscale=settings.PREPROCESS_GRAYSCALE,
            upscale=settings.PREPROCESS_UPSCALE,
            denoise=settings.PREPROCESS_DENOISE,
            threshold=settings.PREPROCESS_THRESHOLD,
            deskew=settings.PREPROCESS_DESKEW,
            min_width=settings.PREPROCESS_MIN_WIDTH,
        )


def to_grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def upscale(image: np.ndarray, min_width: int = 1000) -> np.ndarray:
    """Reescala hacia arriba: Tesseract rinde mejor con ~300 DPI equivalentes."""
    height, width = image.shape[:2]

    if width == 0 or width >= min_width:
        return image

    factor = min_width / width
    return cv2.resize(
        image,
        (int(width * factor), int(height * factor)),
        interpolation=cv2.INTER_CUBIC,
    )


def denoise(image: np.ndarray) -> np.ndarray:
    """Reduce ruido preservando los bordes de los caracteres."""
    if image.ndim == 2:
        return cv2.bilateralFilter(image, d=7, sigmaColor=50, sigmaSpace=50)
    return cv2.fastNlMeansDenoisingColored(image, None, 7, 7, 7, 21)


def threshold(image: np.ndarray) -> np.ndarray:
    """Binarización adaptativa: robusta ante iluminación desigual."""
    gray = to_grayscale(image)
    return cv2.adaptiveThreshold(
        gray,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=31,
        C=15,
    )


def deskew(image: np.ndarray, max_angle: float = 20.0) -> np.ndarray:
    """Corrige la inclinación estimándola con `minAreaRect` sobre el texto."""
    gray = to_grayscale(image)

    # El texto debe quedar en blanco sobre fondo negro para `minAreaRect`.
    binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    coords = cv2.findNonZero(binary)

    if coords is None:
        return image

    # `minAreaRect` devuelve el ángulo en (-90, 0] o en (0, 90] según la versión
    # de OpenCV: se normaliza a la desviación respecto de la horizontal, en
    # [-45, 45), para que la corrección sea la misma en ambas convenciones.
    angle = cv2.minAreaRect(coords)[-1] % 90
    if angle > 45:
        angle -= 90

    # Rotaciones grandes casi siempre son un error de estimación (o la foto está
    # de costado, caso que este paso no pretende resolver).
    if abs(angle) < 0.25 or abs(angle) > max_angle:
        return image

    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)

    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def run(image: np.ndarray, config: PreprocessConfig | None = None) -> np.ndarray:
    """Aplica la cadena de preprocesamiento configurada."""
    config = config or PreprocessConfig.from_settings()
    result = image

    if config.grayscale:
        result = to_grayscale(result)
    if config.upscale:
        result = upscale(result, config.min_width)
    if config.denoise:
        result = denoise(result)
    if config.deskew:
        result = deskew(result)
    if config.threshold:
        result = threshold(result)

    return result
