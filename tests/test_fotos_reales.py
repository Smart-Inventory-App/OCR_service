"""Regresión sobre las fotos del conjunto de prueba, sin levantar un motor.

El resto de la suite prueba parsers y scoring con texto escrito a mano, que es
lo correcto para cubrir cada rama. Lo que no puede decir es si el módulo resuelve
**una foto de verdad**, con su ortografía rota, sus renglones fuera de orden y
sus caracteres confundidos. Y esa es la pregunta del producto.

La forma barata de contestarla es partirla en dos, siguiendo la misma división
que tiene el módulo:

- **El código de barras se decodifica acá**, con ZBar, que es una librería del
  sistema y no un modelo. Así que la foto entra entera y se comprueba el
  resultado final. Son las dos únicas imágenes que quedan en el repositorio
  (`Farmacia/8C.jpg` tal cual, y `Códigos de Barras Impresos/1A.jpg`
  recomprimida de 2,5 MB a 395 KB, que es todo lo que se le puede sacar sin
  quitarle el desenfoque que la hace interesante).
- **El texto lo reconocen los motores, que viven en otros servicios.** Levantarlos
  para un test cuesta minutos por documento y el CI corre a propósito sin ellos
  —ni PyTorch, ni Paddle, ni el binario de Tesseract—, así que un test que los
  necesite no prueba nada donde importa. Lo que se guarda es **la salida que los
  motores devolvieron de verdad** (`tests/fixtures/lecturas/*.json`: texto y
  palabras con caja y confianza) y contra eso corren los parsers y el scoring,
  que es exactamente lo que este servicio aporta. Son 40 KB y milisegundos, en
  lugar de 1,7 MB de fotos y siete minutos.

Para regrabar las lecturas cuando cambie un motor o la estrategia de intentos,
`bench/grabar_lecturas.py` con los servicios arriba.
"""

import json
from datetime import date
from pathlib import Path

import pytest

from app.decoders import barcode as decoder
from app.engines.base import OCRWord, RawOCROutput
from app.schemas.ocr import DocType
from app.services import ocr_service

FIXTURES = Path(__file__).parent / "fixtures"
IMAGENES = FIXTURES / "imagenes"
LECTURAS = FIXTURES / "lecturas"

# Las lecturas se grabaron con este "hoy". Fijarlo evita que un vencimiento
# plausible hoy deje de serlo dentro de dos años y el test empiece a fallar solo.
HOY = date(2026, 9, 8)


def leer_lectura(nombre: str) -> tuple[RawOCROutput, str, DocType]:
    """Reconstruye la salida cruda que el motor devolvió para esa foto."""
    grabada = json.loads((LECTURAS / f"{nombre}.json").read_text(encoding="utf-8"))

    output = RawOCROutput(
        text=grabada["text"],
        words=[OCRWord(**palabra) for palabra in grabada["words"]],
        engine=grabada["motor"],
    )

    return output, grabada["text"], DocType(grabada["doc_type"])


def procesar(nombre: str):
    output, texto, doc_type = leer_lectura(nombre)

    return ocr_service.build_result(output, texto, doc_type, HOY)


# ---------------------------------------------------------------------------
# Códigos de barras: la foto entra entera, porque se decodifica en este servicio
# ---------------------------------------------------------------------------

def test_ean13_argentino_de_una_caja_de_farmacia():
    contenido = (IMAGENES / "codigo_ean13.jpg").read_bytes()

    result, _ = ocr_service._run_barcode(contenido, HOY)

    assert result.fields["codigo"].value == "7794640172021"
    assert result.fields["pais_registro"].value == "Argentina"
    assert result.needs_review is False


def test_el_codigo_impreso_grande_solo_se_lee_reduciendo():
    """La foto que motivó el cambio con más impacto medido del módulo.

    El código ocupa casi todo un cuadro de 4080 px y está algo fuera de foco:
    ampliar no aporta y enfocar tampoco, porque el desenfoque está repartido en
    más píxeles de los que cubre la máscara. Sólo lo levanta achicar.

    Es la única foto pesada que queda en el repositorio (395 KB) y no se puede
    recomprimir: bajarle la resolución o la calidad le quita justamente el
    desenfoque que la hace interesante, y entonces decodifica al primer intento
    sin ejercitar la escalera.
    """
    lectura = decoder.decode((IMAGENES / "codigo_impreso.jpg").read_bytes())

    assert [simbolo.data for simbolo in lectura.symbols] == ["2000003141593"]
    assert lectura.variant.startswith("reducida")


def test_un_codigo_de_circulacion_restringida_no_carga_solo():
    """Prefijo 200: lo emitió un comercio, no identifica un producto."""
    contenido = (IMAGENES / "codigo_impreso.jpg").read_bytes()

    result, _ = ocr_service._run_barcode(contenido, HOY)

    assert result.fields["codigo"].value == "2000003141593"
    assert result.needs_review is True


# ---------------------------------------------------------------------------
# Texto: contra la salida que los motores devolvieron de verdad
# ---------------------------------------------------------------------------

def test_la_marca_grande_no_gana_el_nombre_del_producto():
    """En el envase, `LaVirginia` está impreso mucho más grande que el nombre."""
    result = procesar("etiqueta_cafe")

    assert "CAF" in result.fields["nombre_producto"].value.upper()
    assert result.fields["marca"].value == "LaVirginia"
    assert result.needs_review is False


def test_en_farmacia_el_nombre_es_la_monodroga():
    """`IbuEvanol` es la marca; lo que identifica al producto es `Ibuprofeno`.

    El motor ni siquiera devuelve la marca legible (`Ibu&l`), así que el nombre
    tiene que salir de la monodroga y no del rótulo más grande.
    """
    result = procesar("etiqueta_farmacia")

    assert result.fields["nombre_producto"].value == "Ibuprofeno 400 mg"


def test_sello_en_matriz_de_puntos_con_formato_mes_ano():
    """`1L6217 07:45-VTO08/28` sobre el canto de una botella.

    No se leía en ninguna corrida previa: hacía falta el formato MM/AA anunciado
    por palabra clave, que hasta entonces sólo se aceptaba con año de cuatro
    dígitos. El día se infiere al cierre del mes.
    """
    result = procesar("vencimiento_sello")

    assert result.fields["fecha_vencimiento"].value == "2028-08-31"
    assert result.needs_review is False


def test_sello_sin_separadores_con_una_letra_mal_leida():
    """`27FEB27`: día, mes literal y año pegados, como sale de la impresora."""
    result = procesar("vencimiento_girado")

    assert result.fields["fecha_vencimiento"].value == "2027-02-27"


def test_factura_b_real():
    result = procesar("ticket_dia")

    assert result.fields["fecha_compra"].value == "2026-08-22"
    assert result.fields["total"].value == 16015.25
    assert "DIA" in result.fields["comercio"].value.upper()
    assert result.items
    assert all(item.nombre_producto.value for item in result.items)


def test_un_remito_sin_precios_extrae_sus_items():
    """Sus renglones no traen importe: se reconocen por la cantidad explícita.

    Y `TOTAL DE BULTOS` es un conteo de paquetes, no un importe: un remito no
    tiene total monetario y devolver 17 sería peor que no devolver nada.
    """
    result = procesar("remito")

    assert len(result.items) == 6
    assert [item.cantidad.value for item in result.items] == [6, 1, 6, 1, 1, 2]
    assert result.fields["total"].value is None


def test_el_texto_legible_sale_en_una_forma_revisable():
    """El usuario tiene que poder contrastar lo leído con la foto."""
    etiqueta = procesar("etiqueta_cafe")

    # En una etiqueta el orden de los renglones no significa nada: va en un solo
    # bloque, como está escrito el ground truth.
    assert etiqueta.texto_legible
    assert "\n" not in etiqueta.texto_legible

    # En un comprobante cada renglón es un dato: se conservan.
    assert "\n" in procesar("ticket_dia").texto_legible


@pytest.mark.parametrize(
    "nombre", [ruta.stem for ruta in sorted(LECTURAS.glob("*.json"))]
)
def test_ninguna_foto_real_queda_sin_campos(nombre):
    """Barrido sobre todas las lecturas grabadas.

    No comprueba qué dice cada campo —eso lo hacen los tests de arriba— sino que
    ninguna foto del conjunto termine con el documento vacío, que es la falla
    que el usuario ve como "no leyó nada".
    """
    result = procesar(nombre)

    assert any(campo.value is not None for campo in result.fields.values())
    assert result.document_confidence > 0.0
