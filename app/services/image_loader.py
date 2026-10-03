"""Carga y validación de imágenes.

El servicio recibe la **URL** de la imagen, respetando el patrón de
`image_storage_service` ("los servicios consumidores solo operan con URLs").
También se soporta el binario directo para tests y uso standalone.

Desde que los motores viven en servicios aparte, este servicio ya no decodifica
la imagen: la reenvía tal cual y quien la decodifica es el motor, que es el que
tiene OpenCV. Lo único que se hace acá es comprobar que el binario **sea** una
imagen, para devolver un 400 claro en vez de propagar un error del motor.
"""

import hashlib
import io
import logging

import httpx
from PIL import Image, UnidentifiedImageError

from app.core.config import settings

logger = logging.getLogger(__name__)

MAX_DOWNLOAD_BYTES = 15 * 1024 * 1024


class ImageLoadError(Exception):
    """La imagen no pudo descargarse o no es una imagen válida."""


async def from_url(url: str) -> bytes:
    """Descarga la imagen."""
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


def validate(content: bytes) -> tuple[int, int]:
    """Comprueba que el binario sea una imagen decodificable y devuelve su tamaño.

    `Image.open` sólo lee la cabecera, así que por sí solo deja pasar un archivo
    **truncado**: la cabecera está bien y el resto de los bytes no llegaron. Eso
    no se notaba hasta que alguien intentaba decodificar la imagen de verdad, y
    entonces la falla salía como un 500 en vez del 400 que corresponde a un
    archivo mal subido.

    Por eso además de abrirla se fuerza a Pillow a recorrer los datos con
    `verify()`, que detecta el truncamiento sin descomprimir la imagen a memoria.
    Como `verify()` deja el archivo inutilizable, se abre dos veces: una para
    medir y otra para verificar. Las dos son operaciones de cabecera y flujo, no
    de decodificación completa.
    """
    try:
        with Image.open(io.BytesIO(content)) as image:
            size = image.size
            image_format = image.format

        with Image.open(io.BytesIO(content)) as image:
            image.verify()

        if image_format == "JPEG":
            # `verify()` no mira los datos de un JPEG, así que uno cortado a la
            # mitad pasaba, se subía al storage, esperaba su turno en la cola y
            # recién el motor lo rechazaba. Decodificarlo en modo `draft` recorre
            # el archivo a una fracción del tamaño —sin pagar la imagen completa 
            # en memoria— y falla enseguida si faltan bytes.
            with Image.open(io.BytesIO(content)) as image:
                image.draft("L", (64, 64))
                image.load()
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as exc:
        raise ImageLoadError("El archivo no es una imagen válida") from exc

    return size


def fingerprint(content: bytes) -> str:
    """Hash del contenido, usado como clave de caché para no reprocesar."""
    return hashlib.sha256(content).hexdigest()
