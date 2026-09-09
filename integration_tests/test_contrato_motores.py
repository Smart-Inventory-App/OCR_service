"""El contrato con los tres servicios de motor, contra los servicios de verdad.

`tests/` sustituye los motores por dobles, y eso es lo correcto para probar la
estrategia de intentos, los parsers y el scoring sin depender de nada externo.
Pero un doble contesta lo que uno le dijo que conteste: no puede desmentir una
suposición equivocada sobre el otro lado. Estas pruebas verifican justamente las
suposiciones que el coordinador da por ciertas cada vez que arma un plan:

1. que los tres motores están y se identifican,
2. que devuelven palabras con caja y confianza sobre una foto real,
3. que la `Transform` viaja por HTTP y **se aplica**,
4. que cada uno declara si prepara su propia entrada — el dato del que depende
   que el plan lleve filtros o sólo geometría.

La foto es la del repositorio (`tests/fixtures/imagenes/codigo_ean13.jpg`): una
caja de farmacia con texto impreso, que ya está versionada para las pruebas de
código de barras y acá sirve igual como texto real.
"""

from pathlib import Path

import pytest

from app.engines.factory import CONCRETE_ENGINES, build_engine
from app.pipeline.transform import Transform

FOTO = Path(__file__).parent.parent / "tests" / "fixtures" / "imagenes" / "codigo_ean13.jpg"

# Qué motor prepara su propia entrada. No es un detalle interno: es lo que decide
# si `attempt_plan` prueba variantes de filtros (Tesseract depende de ellas) o
# sólo giros y recortes (los de aprendizaje profundo binarizan por su cuenta y
# variarles el contraste no cambia nada).
PREPARA_SU_ENTRADA = {"tesseract": True, "easyocr": False, "paddleocr": False}


@pytest.fixture(scope="module")
def imagen() -> bytes:
    return FOTO.read_bytes()


@pytest.mark.parametrize("nombre", CONCRETE_ENGINES)
def test_el_motor_esta_arriba_y_dice_quien_es(nombre):
    motor = build_engine(nombre)

    assert motor.is_available() is True

    info = motor.info()

    assert info.get("engine") == nombre
    assert info.get("version")


@pytest.mark.parametrize("nombre", CONCRETE_ENGINES)
def test_el_motor_devuelve_palabras_con_caja_y_confianza(nombre, imagen):
    """Lo mínimo que promete el contrato, sobre una foto y no sobre un dibujo."""
    salida = build_engine(nombre).run(imagen, lang="spa", layout="block")

    assert salida.engine == nombre
    assert salida.words, "no reconoció una sola palabra"
    assert salida.text.strip()
    assert all(0.0 <= palabra.confidence <= 1.0 for palabra in salida.words)
    assert all(palabra.width > 0 and palabra.height > 0 for palabra in salida.words)
    assert 0.0 < salida.mean_confidence <= 1.0


@pytest.mark.parametrize("nombre", CONCRETE_ENGINES)
def test_cada_motor_declara_si_prepara_su_propia_entrada(nombre):
    """La capacidad se pregunta, no se supone.

    `attempt_plan` arma un plan distinto según este dato, así que si el motor
    cambiara de opinión —o si el default del coordinador quedara viejo— los
    planes se armarían mal y ningún test con dobles lo notaría.
    """
    motor = build_engine(nombre)
    motor.refresh_capabilities()

    assert motor.supports_preprocessing is PREPARA_SU_ENTRADA[nombre]


def test_la_transformacion_viaja_por_http_y_el_motor_la_aplica(imagen):
    """El coordinador decide el giro y el motor lo ejecuta.

    Si la `Transform` no llegara —o llegara y se ignorara— las dos lecturas
    serían idénticas, porque es la misma imagen. Se usa Tesseract porque es el
    más barato de los tres y la pregunta no es cuál lee mejor sino si el pedido
    cruzó el cable.
    """
    motor = build_engine("tesseract")

    derecha = motor.run(imagen, transform=Transform(rotate=0))
    invertida = motor.run(imagen, transform=Transform(rotate=180))

    assert derecha.text != invertida.text
