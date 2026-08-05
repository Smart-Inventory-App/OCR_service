"""Carga de imágenes desde URL (Cloudinary) o bytes.

El servicio recibe la **URL** de la imagen, respetando el patrón de
`image_storage_service` ("los servicios consumidores solo operan con URLs").
También se soporta el binario directo para tests y uso standalone.
"""

import hashlib
import logging

import cv2
import httpx
import numpy as np

from app.core.config import settings

logger = logging.getLogger(__name__)

MAX_DOWNLOAD_BYTES = 15 * 1024 * 1024


class ImageLoadError(Exception):
    """La imagen no pudo descargarse o decodificarse."""


async def from_url(url: str) -> bytes:
    """Descarga la imagen. No sigue redirecciones a hosts arbitrarios."""
    try:
        async with httpx.AsyncClient(timeout=settings.HTTP_TIMEOUT_SECONDS) as client:
            response = await client.get(url, follow_redirects=True)
            response.raise_for_status()
    except httpx.HTTPError as exc:
        raise ImageLoadError(f"No se pudo descargar la imagen: {exc}") from exc

    content = response.content
    if not content:
        raise ImageLoadError("La imagen descargada está vacía")
    if len(content) > MAX_DOWNLOAD_BYTES:
        raise ImageLoadError(
            f"La imagen supera el límite de {MAX_DOWNLOAD_BYTES // (1024 * 1024)} MB"
        )

    logger.info("Imagen descargada: %s (%d bytes)", url, len(content))
    return content


def decode(content: bytes) -> np.ndarray:
    """Decodifica bytes a una matriz BGR de OpenCV."""
    buffer = np.frombuffer(content, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)

    if image is None:
        raise ImageLoadError("El archivo no es una imagen válida")

    return image


def fingerprint(content: bytes) -> str:
    """Hash del contenido, usado como clave de caché para no reprocesar."""
    return hashlib.sha256(content).hexdigest()
