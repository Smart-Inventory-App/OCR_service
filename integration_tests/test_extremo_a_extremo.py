"""El módulo entero sobre fotos reales, con los motores levantados.

Dos preguntas que sólo se pueden contestar acá:

- **Que el código de barras no dependa del motor configurado.** Es la razón por
  la que ZBar corre dentro del coordinador y no repartido entre los tres
  servicios: un EAN-13 leído con `engine=tesseract` y con `engine=paddleocr`
  tiene que dar el mismo número o algo está mal. Con dobles la afirmación es
  vacía —los dobles no decodifican nada—; con los motores arriba es una
  verificación de verdad.

- **Que las lecturas grabadas sigan representando al módulo.**
  `tests/test_fotos_reales.py` corre los parsers contra la salida que los motores
  devolvieron *el día que se grabó*, y esa es exactamente su virtud (40 KB y
  milisegundos, sin motores) y su riesgo: si un motor mejora, si cambia la
  estrategia de intentos o si se toca la preparación de imagen, el fixture puede
  quedar viejo sin que nada lo diga. Esta prueba vuelve a procesar las fotos de
  verdad y comprueba que el campo clave sigue saliendo igual. Es el puente entre
  el test barato del CI y la realidad, y la señal de cuándo hay que correr
  `python -m bench.grabar_lecturas`.

Las fotos **no viven en el repositorio** —son material de prueba local—, así que
la segunda parte se saltea salvo que se diga dónde están:

    OCR_FOTOS="D:/.../ImagenesOCR" pytest -c pytest-integration.ini
"""

from datetime import date
from pathlib import Path

import pytest

from app.schemas.ocr import DocType
from app.services import ocr_service
from bench.grabar_lecturas import CASOS, HOY

FOTO_CODIGO = (
    Path(__file__).parent.parent / "tests" / "fixtures" / "imagenes" / "codigo_ean13.jpg"
)

# El campo clave de cada lectura grabada y el valor que `tests/test_fotos_reales.py`
# afirma sobre ella. Si acá sale otra cosa, el fixture quedó viejo.
ESPERADO: dict[str, tuple[str, object]] = {
    "etiqueta_cafe": ("marca", "LaVirginia"),
    "etiqueta_farmacia": ("nombre_producto", "Ibuprofeno 400 mg"),
    "vencimiento_sello": ("fecha_vencimiento", "2028-08-31"),
    "vencimiento_girado": ("fecha_vencimiento", "2027-02-27"),
    "ticket_dia": ("total", 16015.25),
    "remito": ("fecha_compra", "2026-08-29"),
}


def test_el_codigo_de_barras_no_depende_del_motor_configurado():
    """Misma foto, dos motores, un solo número.

    Ninguno de los dos participa: el decodificador vive en el coordinador. Eso es
    precisamente lo que se comprueba — que la elección de motor no se filtre a un
    resultado que no debería depender de ella.
    """
    contenido = FOTO_CODIGO.read_bytes()

    con_tesseract = ocr_service.process_sync(
        contenido, DocType.BARCODE, today=date(2026, 9, 8), engine_name="tesseract"
    )
    con_paddle = ocr_service.process_sync(
        contenido, DocType.BARCODE, today=date(2026, 9, 8), engine_name="paddleocr"
    )

    assert con_tesseract.fields["codigo"].value == "7794640172021"
    assert con_paddle.fields["codigo"].value == con_tesseract.fields["codigo"].value
    assert con_paddle.engine == con_tesseract.engine == "zbar"


@pytest.mark.parametrize("nombre,relativa,doc_type", CASOS, ids=[caso[0] for caso in CASOS])
def test_la_lectura_grabada_sigue_representando_al_modulo(nombre, relativa, doc_type, fotos):
    """Reprocesa la foto real y compara con lo que el fixture afirma.

    Falla si el fixture quedó viejo, no si el módulo empeoró: puede fallar porque
    un motor mejoró y ahora lee algo distinto. En cualquiera de los dos casos hay
    que mirar el diff de `bench.grabar_lecturas` antes de tocar nada.
    """
    foto = fotos / relativa
    if not foto.exists():
        pytest.skip(f"falta {relativa} en {fotos}")

    resultado = ocr_service.process_sync(
        foto.read_bytes(), doc_type, today=HOY, engine_name="hybrid"
    )

    campo, valor = ESPERADO[nombre]

    assert resultado.fields[campo].value == valor, (
        f"{relativa}: {campo} salió {resultado.fields[campo].value!r} y el fixture "
        f"de tests/fixtures/lecturas/{nombre}.json dice {valor!r}. "
        "Si el cambio es una mejora, regrabá con `python -m bench.grabar_lecturas`."
    )
