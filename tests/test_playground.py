"""Banco de pruebas: se sirve sólo en desarrollo y con la URL de Auth resuelta."""

from httpx import ASGITransport, AsyncClient

from app.core.config import settings
from app.main import app
from app.routes.playground import _AUTH_URL_PLACEHOLDER


async def _get(path: str = "/ocr/playground"):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        return await c.get(path)


async def test_served_in_debug_mode(monkeypatch):
    monkeypatch.setattr(settings, "DEBUG", True)

    r = await _get()

    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    assert "Banco de pruebas del servicio OCR" in r.text


async def test_hidden_when_debug_is_off(monkeypatch):
    monkeypatch.setattr(settings, "DEBUG", False)

    assert (await _get()).status_code == 404


async def test_auth_url_is_injected(monkeypatch):
    monkeypatch.setattr(settings, "DEBUG", True)
    monkeypatch.setattr(settings, "AUTH_PUBLIC_URL", "http://mi-auth:9999")

    body = (await _get()).text

    assert "http://mi-auth:9999" in body
    assert _AUTH_URL_PLACEHOLDER not in body


async def test_needs_no_token(monkeypatch):
    """La página es pública; los endpoints que consume siguen protegidos."""
    monkeypatch.setattr(settings, "DEBUG", True)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        r = await c.get("/ocr/playground")  # sin header Authorization

    assert r.status_code == 200


async def test_not_published_in_openapi():
    """Es instrumental: no debe figurar en el contrato del servicio."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        schema = (await c.get("/openapi.json")).json()

    assert "/ocr/playground" not in schema["paths"]
