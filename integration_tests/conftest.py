"""Configuración de las pruebas de integración con los servicios de motor.

Vive fuera de `tests/` a propósito. El CI corre `pytest tests/` con los motores
**deliberadamente no instalados**, así que cualquier archivo que cuelgue de ahí o
se saltea (y no prueba nada donde importa) o rompe el build. Estas pruebas son lo
contrario: sólo tienen sentido con los tres servicios levantados, y por eso se
piden explícitamente.

    docker compose up -d
    pytest -c pytest-integration.ini

El directorio es un paquete (`__init__.py`) por la misma razón que `tests/`: sin
él pytest importa los módulos de test como módulos sueltos de primer nivel y dos
archivos con el mismo nombre en carpetas distintas chocan. Lo que **no** hace
falta es importar nada de acá desde los tests: lo compartido viaja como fixture,
que además centraliza el `skip` en un solo lugar.

Las URLs de los motores se resuelven por defecto a `localhost`, que es como se
los ve desde la máquina de quien corre esto; dentro de la red de Docker los
nombres son otros y los toma `.env`. Cualquiera de las tres se puede pisar por
entorno si los puertos publicados son distintos.
"""

import os
from pathlib import Path

import pytest

# Antes de importar nada de `app`: `settings` se construye en el import.
os.environ.setdefault("JWT_SECRET", "test-secret")
os.environ.setdefault("OCR_ENGINE", "hybrid")
os.environ.setdefault("TESSERACT_SERVICE_URL", "http://localhost:8005")
os.environ.setdefault(
    "PADDLEOCR_SERVICE_URL", "http://localhost:8007,http://localhost:8017"
)

from app.engines.factory import CONCRETE_ENGINES, build_engine  # noqa: E402

# La carpeta con las fotos del conjunto de prueba. **No vive en el repositorio**:
# es material local, así que las pruebas que la necesitan se saltean si no está.
FOTOS_ENV = "OCR_FOTOS"


def pytest_collection_modifyitems(items):
    """Marca todo lo que hay acá, para poder filtrarlo desde afuera."""
    for item in items:
        item.add_marker(pytest.mark.integracion)


@pytest.fixture(scope="session", autouse=True)
def motores_arriba():
    """Corta la sesión entera si los motores no están, en vez de fallar test a test.

    Un `skip` con el motivo puesto vale más que veinte errores de conexión: dice
    qué falta hacer (`docker compose up -d`) en lugar de dejar el diagnóstico por
    cuenta de quien lee el traceback.
    """
    caidos = [nombre for nombre in CONCRETE_ENGINES if not build_engine(nombre).is_available()]

    if caidos:
        pytest.skip(
            f"servicios de motor no disponibles: {', '.join(caidos)}. "
            "Levantalos con `docker compose up -d`.",
            allow_module_level=True,
        )


@pytest.fixture(scope="session")
def fotos() -> Path:
    """Carpeta raíz del conjunto de prueba, o `skip` si no se dijo dónde está.

    Es lo que separa las pruebas que se sostienen solas —con las dos fotos
    versionadas del repositorio— de las que necesitan el material local.
    """
    raiz = os.environ.get(FOTOS_ENV)
    if not raiz:
        pytest.skip(
            f"sin fotos del conjunto de prueba: definí {FOTOS_ENV} con la carpeta raíz"
        )

    return Path(raiz)
