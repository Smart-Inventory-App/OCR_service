"""Banco de pruebas del OCR — herramienta de desarrollo.

Página estática que replica el patrón de autenticación del backoffice: se inicia
sesión una sola vez contra el servicio Auth, el token queda en `localStorage` y
se adjunta automáticamente en cada llamada a `/ocr/*`. Permite validar el
servicio subiendo imágenes desde el navegador, sin usar la terminal.

**Sólo se sirve con `DEBUG=true`.** En producción devuelve 404: no es parte del
contrato del servicio, es instrumental para pruebas manuales y demos.
"""

from pathlib import Path

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import HTMLResponse

from app.core.config import settings

router = APIRouter(prefix="/ocr", tags=["playground"])

_PAGE = Path(__file__).resolve().parent.parent / "static" / "playground.html"
_AUTH_URL_PLACEHOLDER = "__AUTH_PUBLIC_URL__"


@router.get("/playground", response_class=HTMLResponse, include_in_schema=False)
async def playground() -> HTMLResponse:
    if not settings.DEBUG:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)

    html = _PAGE.read_text(encoding="utf-8")

    # La página corre en el navegador: necesita la URL pública de Auth
    # (`localhost:8001`), no el hostname interno de la red de Docker.
    return HTMLResponse(html.replace(_AUTH_URL_PLACEHOLDER, settings.AUTH_PUBLIC_URL))
