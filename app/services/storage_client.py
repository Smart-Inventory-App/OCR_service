"""Cliente de `image_storage_service`.

El OCR necesita una imagen que exista sólo mientras dura el procesamiento: se
sube a `POST /storage/temp`, se le aplican los filtros del pipeline y se
descarta con `DELETE /storage/temp`. El TTL del storage es la red de seguridad
si este servicio no llega a descartarla.
"""

import logging

import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)

_TEMP_ENDPOINT = "/storage/temp"


class StorageError(Exception):
    """El servicio de storage no pudo completar la operación."""


async def upload_temp(
    content: bytes,
    filename: str,
    content_type: str,
    token: str,
) -> str:
    """Sube la imagen a la carpeta temporal y devuelve su URL."""
    url = f"{settings.STORAGE_SERVICE_URL}{_TEMP_ENDPOINT}"

    try:
        async with httpx.AsyncClient(timeout=settings.HTTP_TIMEOUT_SECONDS) as client:
            response = await client.post(
                url,
                headers={"Authorization": f"Bearer {token}"},
                files={"file": (filename, content, content_type)},
            )
            response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise StorageError(
            f"storage respondió {exc.response.status_code}: {exc.response.text}"
        ) from exc
    except httpx.HTTPError as exc:
        raise StorageError(f"No se pudo contactar al storage: {exc}") from exc

    body = response.json()
    logger.info("Imagen temporal subida: %s (expira %s)", body["url"], body.get("expires_at"))

    return body["url"]


async def discard_temp(image_url: str, token: str) -> bool:
    """Descarta la imagen temporal. Best-effort: no interrumpe el flujo del OCR."""
    url = f"{settings.STORAGE_SERVICE_URL}{_TEMP_ENDPOINT}"

    try:
        async with httpx.AsyncClient(timeout=settings.HTTP_TIMEOUT_SECONDS) as client:
            response = await client.request(
                "DELETE",
                url,
                headers={"Authorization": f"Bearer {token}"},
                json={"url": image_url},
            )
            response.raise_for_status()
    except httpx.HTTPError as exc:
        # El storage purga las temporales vencidas por su cuenta: se registra y sigue.
        logger.warning("No se pudo descartar la imagen temporal %s: %s", image_url, exc)
        return False

    logger.info("Imagen temporal descartada: %s", image_url)
    return True
