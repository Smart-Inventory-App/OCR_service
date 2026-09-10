"""Lectura de códigos de barras: decodificador, parser y confianza.

A diferencia del resto de la suite, acá **no hay doble del reconocedor**: los
códigos se generan con `python-barcode` —la misma codificación que imprime una
etiquetadora— y los decodifica ZBar de verdad. Es lo que corresponde, porque en
este tipo de documento el reconocimiento corre dentro de este servicio: si se lo
sustituyera por un doble, los tests no probarían nada de lo que se agregó.

Los umbrales de `TestEscalera` son mediciones, no números elegidos: fijan en qué
condiciones cada variante de la escalera se gana su lugar, y si una versión de
ZBar los mueve, la suite lo reporta.
"""

import io

import pytest
from PIL import Image

from app.core.config import settings
from app.decoders import barcode as decoder
from app.parsers import barcode as parser
from app.parsers.barcode import expand_upce, is_restricted
from app.schemas.ocr import DocType
from app.services import ocr_service
from tests.conftest import barcode_photo, render_barcode

# Un EAN-13 argentino real en su forma completa: 779 + empresa + producto + check.
EAN13_AR = "7791234567898"
# Los 12 dígitos sin el verificador, que es lo que recibe el generador.
EAN13_AR_BODY = "779123456789"


def decode_fields(image: bytes) -> dict:
    """Corre el pipeline completo y devuelve los campos ya puntuados."""
    return ocr_service.process_sync(image, DocType.BARCODE).fields


def symbol(data: str, symbology: str = "EAN13", width: int = 300, height: int = 150):
    return decoder.Symbol(data=data, symbology=symbology, width=width, height=height)


def reading(*symbols, corroborated: bool = True) -> decoder.BarcodeReading:
    """Lectura de prueba. Se marca corroborada salvo que el test diga lo
    contrario: es el caso normal, y así los tests miden lo que quieren medir en
    vez de arrastrar la penalización por falta de segunda opinión."""
    return decoder.BarcodeReading(
        symbols=list(symbols), attempts=1, variant="directo",
        corroborated=corroborated,
    )


# ---------------------------------------------------------------------------
# La escalera de intentos — los umbrales medidos
# ---------------------------------------------------------------------------

class TestEscalera:
    """Cada variante tiene que ganarse su lugar resolviendo algo que la anterior
    no resuelve. Un filtro que nunca cambia el resultado es tiempo regalado."""

    def test_codigo_grande_se_lee_en_el_primer_intento(self):
        result = decoder.decode(barcode_photo(EAN13_AR_BODY, width=420))

        assert [s.data for s in result.symbols] == [EAN13_AR]
        assert result.variant == "directo"
        assert result.attempts == 1

    def test_codigo_chico_lo_recupera_la_ampliacion(self):
        """A 180 px el intento directo no lee y la ampliación sí: es la medición
        que justifica que la segunda variante exista."""
        result = decoder.decode(barcode_photo(EAN13_AR_BODY, width=180))

        assert [s.data for s in result.symbols] == [EAN13_AR]
        assert result.variant == "ampliada"
        # No es el primer intento: el directo falló y las reducciones tampoco
        # sirven acá, porque el problema es que falta resolución, no que sobre.
        assert result.attempts > 1

    def test_codigo_grande_y_desenfocado_lo_recupera_la_reduccion(self):
        """El caso que hizo fallar 100 de 100 fotos de códigos impresos.

        Un código enorme dentro de un cuadro enorme y algo fuera de foco: ampliar
        no aporta y la máscara de enfoque tampoco, porque el desenfoque está
        repartido en muchos más píxeles de los que cubre su radio. Reducir vuelve
        a definir el borde entre barra y espacio.
        """
        photo = barcode_photo(EAN13_AR_BODY, width=3000, blur=12, canvas=(4000, 2400))

        result = decoder.decode(photo)

        assert [s.data for s in result.symbols] == [EAN13_AR]
        assert result.variant.startswith("reducida")

    def test_sin_reducciones_ese_codigo_no_se_lee(self, monkeypatch):
        """La contraparte del test anterior: es la reducción y no otra cosa.

        Sin ella, la misma foto agota la escalera sin leer nada — que es
        exactamente lo que hacía el servicio antes de agregarla.
        """
        monkeypatch.setattr(settings, "BARCODE_DOWNSCALE_WIDTHS_STR", "")

        photo = barcode_photo(EAN13_AR_BODY, width=3000, blur=12, canvas=(4000, 2400))

        assert decoder.decode(photo).found is False

    def test_las_reducciones_se_prueban_antes_que_la_ampliacion(self):
        """Orden de la escalera: primero lo que más recupera y menos cuesta.

        Las reducciones resuelven 82 de los 116 fallos medidos contra 11 de la
        ampliación, y además decodificar una imagen chica cuesta menos que una
        del doble de lado.
        """
        nombres = [name for name, _ in decoder._variants()]

        from app.core.config import settings

        assert nombres[0] == "directo"
        assert nombres.index("reducida 2048px") < nombres.index("ampliada")
        # Las reducciones simples, de mayor a menor, antes de cualquier combinación.
        simples = [n for n in nombres if n.startswith("reducida") and "+" not in n]
        assert simples == [f"reducida {w}px" for w in settings.barcode_downscale_widths]

    def test_una_reduccion_no_agranda_una_imagen_chica(self):
        """Reducir a 2048 px una foto de 1200 no tiene sentido: se saltea."""
        reducir = decoder._downscaler(2048)

        assert reducir(Image.new("L", (1200, 800))) is None
        assert reducir(Image.new("L", (4000, 2400))).width == 2048

    def test_codigo_desenfocado_lo_recupera_el_enfoque(self):
        """Mismo criterio para la máscara de enfoque: con desenfoque 2,0 es la
        única variante que lo rescata."""
        result = decoder.decode(barcode_photo(EAN13_AR_BODY, width=420, blur=2.0))

        assert [s.data for s in result.symbols] == [EAN13_AR]
        assert result.variant == "enfocada"

    def test_codigo_demasiado_chico_no_se_inventa(self):
        """Por debajo del umbral la información no está en la foto. Se agotan las
        variantes y se devuelve vacío, que es lo que dispara la revisión manual."""
        result = decoder.decode(barcode_photo(EAN13_AR_BODY, width=120))

        assert result.symbols == []
        assert result.found is False
        # Se agotó la escalera: todas las variantes que aplican a esta imagen.
        aplicables = sum(
            1 for _n, build in decoder._variants()
            if build(Image.open(io.BytesIO(barcode_photo(EAN13_AR_BODY, width=120)))
                     .convert("L")) is not None
        )
        assert result.attempts == aplicables

    def test_desenfoque_severo_no_se_inventa(self):
        result = decoder.decode(barcode_photo(EAN13_AR_BODY, width=420, blur=3.2))

        assert result.found is False

    @pytest.mark.parametrize("degrees", [0, 90, 180, 270])
    def test_no_hace_falta_buscar_la_orientacion(self, degrees):
        """Al revés que el sello de vencimiento, que necesita las cuatro
        rotaciones: ZBar recorre la imagen en varias direcciones por su cuenta,
        así que las cuatro se resuelven en el **primer** intento."""
        photo = Image.open(io.BytesIO(barcode_photo(EAN13_AR_BODY)))
        buffer = io.BytesIO()
        photo.rotate(degrees, expand=True).save(buffer, format="PNG")

        result = decoder.decode(buffer.getvalue())

        assert [s.data for s in result.symbols] == [EAN13_AR]
        assert result.attempts == 1

    def test_la_ampliacion_se_saltea_si_se_pasa_del_tope_de_pixeles(self, monkeypatch):
        """El tope acota el pico de memoria. Cuando aplica, la variante no corre
        y tampoco se cuenta como intento fallido: no se intentó.

        Se apagan las reducciones y el enfoque para dejar sola a la ampliación:
        si no, no se podría distinguir "no se intentó" de "se intentó otra cosa".
        """
        monkeypatch.setattr(settings, "BARCODE_MAX_PIXELS", 1000)
        monkeypatch.setattr(settings, "BARCODE_DOWNSCALE_WIDTHS_STR", "")
        monkeypatch.setattr(settings, "BARCODE_SHARPEN_ENABLED", False)

        result = decoder.decode(barcode_photo(EAN13_AR_BODY, width=120))

        assert result.attempts == 1

    def test_imagen_invalida(self):
        with pytest.raises(ValueError):
            decoder.decode(b"esto no es una imagen")

    def test_ampliacion_desactivada(self, monkeypatch):
        """Con factor 1 la variante no aporta nada y no se corre."""
        monkeypatch.setattr(settings, "BARCODE_UPSCALE_FACTOR", 1)
        monkeypatch.setattr(settings, "BARCODE_DOWNSCALE_WIDTHS_STR", "")
        monkeypatch.setattr(settings, "BARCODE_SHARPEN_ENABLED", False)

        result = decoder.decode(barcode_photo(EAN13_AR_BODY, width=180))

        assert result.attempts == 1
        assert result.found is False

    def test_simbolo_con_contenido_no_textual_se_descarta(self, monkeypatch):
        """Un QR puede traer bytes que no son texto. No es un código de producto
        y ofrecerlo como tal sería peor que no devolver nada."""

        class Rect:
            left = top = width = height = 10

        class Binario:
            data = b"\xff\xfe\x00"
            type = "QRCODE"
            rect = Rect()
            quality = 1

        class FakeZbar:
            ZBarSymbol = decoder._pyzbar().ZBarSymbol

            @staticmethod
            def decode(_image, symbols=None):
                return [Binario()]

        monkeypatch.setattr(decoder, "_pyzbar", lambda: FakeZbar)

        result = decoder.decode(barcode_photo(EAN13_AR_BODY))

        assert result.symbols == []


# ---------------------------------------------------------------------------
# Dígito verificador y normalización a GTIN
# ---------------------------------------------------------------------------

class TestNormalizacion:

    def test_ean13_argentino(self):
        fields = decode_fields(barcode_photo(EAN13_AR_BODY))

        assert fields["codigo"].value == EAN13_AR
        assert fields["simbologia"].value == "EAN-13"
        assert fields["pais_registro"].value == "Argentina"

    def test_ean8(self):
        fields = decode_fields(barcode_photo("7791234", kind="ean8"))

        assert fields["codigo"].value == "77912343"
        assert fields["simbologia"].value == "EAN-8"

    def test_upca_se_normaliza_a_ean13(self):
        """Un UPC-A es un EAN-13 con un cero adelante. Normalizarlo es lo que
        evita que el mismo producto comprado local y encargado afuera entre dos
        veces al inventario."""
        fields = decode_fields(barcode_photo("01234567890", kind="upca"))

        assert fields["codigo"].value == "0012345678905"
        assert fields["simbologia"].value == "UPC-A → EAN-13"

    def test_itf14_devuelve_el_ean13_del_envase(self):
        """La caja de un pedido mayorista lleva el GTIN-13 del envase individual
        adentro. Se extrae, porque es el producto que el usuario tiene."""
        gtin14 = "1" + EAN13_AR_BODY
        gtin14 += str(parser._check_digit(gtin14))

        fields = decode_fields(barcode_photo(gtin14, kind="itf"))

        assert fields["codigo"].value == EAN13_AR
        assert fields["simbologia"].value == "ITF-14 → EAN-13"

    def test_digito_verificador_invalido_no_se_ofrece(self):
        """Un GTIN cuyo check no cierra no es un código: es un número mal leído.
        Se devuelve vacío en vez de un dato que el usuario no puede auditar."""
        fields = parser.parse(reading(symbol("7791234567891")))

        assert fields["codigo"].value is None
        assert fields["codigo"].confidence == 0.0
        # La simbología sí se sabe: lo que falló es el número, no la lectura.
        assert fields["simbologia"].value == "EAN-13"

    @pytest.mark.parametrize(
        "upce, upca",
        [
            # Un caso por cada forma de compresión, que es donde está la lógica:
            # el último dígito del cuerpo dice dónde van los ceros.
            ("04252614", "042100005264"),  # termina en 0, 1 o 2
            ("01234531", "012300000451"),  # termina en 3
            ("01234543", "012340000053"),  # termina en 4
            ("02345673", "023456000073"),  # termina en 5 a 9
        ],
    )
    def test_expansion_de_upce(self, upce, upca):
        """Los pares salieron del generador de UPC-A, no de calcularlos a mano:
        el verificador de un UPC-E es el mismo que el de su UPC-A."""
        assert expand_upce(upce) == upca

    @pytest.mark.parametrize(
        "invalido",
        [
            "1234567",   # largo incorrecto
            "abcdefgh",  # no son dígitos
            "34252614",  # el sistema numérico sólo puede ser 0 o 1
            "04252610",  # el verificador no cierra
        ],
    )
    def test_upce_invalido(self, invalido):
        assert expand_upce(invalido) is None

    def test_upce_que_no_expande_no_se_ofrece(self):
        fields = parser.parse(reading(symbol("34252614", symbology="UPCE")))

        assert fields["codigo"].value is None
        assert fields["simbologia"].value == "UPC-E"

    @pytest.mark.parametrize("basura", ["1234567", "no-es-un-numero", ""])
    def test_un_gtin_mal_formado_no_pasa_la_validacion(self, basura):
        assert parser._validated(basura) is None

    def test_pais_de_un_prefijo_desconocido(self):
        """Un prefijo que no está en la tabla devuelve None en vez de adivinar."""
        body = "999123456789"
        assert parser.country_of(body + str(parser._check_digit(body))) is None

    @pytest.mark.parametrize("valor", ["AR847362910", ""])
    def test_restringido_solo_aplica_a_numeros(self, valor):
        assert is_restricted(valor) is False

    def test_check_digit_gs1(self):
        assert parser._check_digit(EAN13_AR_BODY) == 8
        assert parser._check_digit("7791234") == 3

    @pytest.mark.parametrize(
        "body",
        [
            "779123456789",
            "779555000123",
            "780123456789",
            "789012345678",
            "750000000000",
            "000000000001",
            "999999999999",
        ],
    )
    def test_el_digito_verificador_coincide_con_una_implementacion_independiente(self, body):
        """Contrasta nuestro cálculo contra el del generador de códigos.

        Escribir a mano el verificador esperado en cada caso es justo el tipo de
        dato que se equivoca en silencio —y de hecho se equivocó al armar el
        conjunto de aceptación—, así que acá se compara contra `python-barcode`,
        que lo implementa por su cuenta. Que dos implementaciones distintas
        coincidan es una garantía que un literal escrito a mano no da.
        """
        import barcode as barcode_lib

        expected = barcode_lib.get_barcode_class("ean13")(body).get_fullcode()

        assert body + str(parser._check_digit(body)) == expected


# ---------------------------------------------------------------------------
# Dominio argentino
# ---------------------------------------------------------------------------

class TestCircuitoArgentino:

    @pytest.mark.parametrize(
        "prefijo, pais",
        [
            ("779", "Argentina"),
            ("778", "Argentina"),
            ("789", "Brasil"),
            ("773", "Uruguay"),
            ("780", "Chile"),
            ("784", "Paraguay"),
            ("750", "México"),
            ("775", "Perú"),
            ("840", "España"),
            ("690", "China"),
        ],
    )
    def test_pais_de_registro_por_prefijo(self, prefijo, pais):
        body = prefijo + "123456789"[: 12 - len(prefijo)]
        code = body + str(parser._check_digit(body))

        assert parser.country_of(code) == pais

    def test_el_prefijo_no_dice_donde_se_fabrico(self):
        """El campo es `pais_registro` y no `pais_origen` a propósito: GS1
        identifica dónde se registró la empresa, no dónde se produjo la cosa.
        Este test fija el nombre para que nadie lo 'corrija' al equivocado."""
        fields = decode_fields(barcode_photo(EAN13_AR_BODY))

        assert "pais_registro" in fields
        assert "pais_origen" not in fields

    @pytest.mark.parametrize("prefijo", ["200", "290", "020", "040"])
    def test_codigos_de_balanza_son_de_circulacion_restringida(self, prefijo):
        """Lo que imprime la balanza del súper sobre la fruta y la fiambrería.
        El número es válido pero sólo significa algo dentro de ese comercio."""
        body = prefijo + "123456789"[:9]
        code = body + str(parser._check_digit(body))

        assert is_restricted(code) is True
        assert parser.country_of(code) is None

    def test_codigo_de_balanza_va_a_revision(self):
        body = "200123450000"
        code = body + str(parser._check_digit(body))

        result = ocr_service.process_sync(barcode_photo(body), DocType.BARCODE)

        assert result.fields["codigo"].value == code
        assert result.fields["codigo"].confidence == pytest.approx(0.5)
        assert result.needs_review is True

    def test_etiqueta_logistica_no_es_un_producto(self):
        """Un encargo llega con la etiqueta del correo pegada al envase. Se lee
        perfecto y no es el producto, así que no puede dar el alta sola."""
        fields = decode_fields(barcode_photo("AR847362910", kind="code128"))

        assert fields["codigo"].value == "AR847362910"
        assert fields["codigo"].confidence == pytest.approx(0.6)
        assert fields["pais_registro"].value is None

    def test_ean13_valido_no_es_restringido(self):
        assert is_restricted(EAN13_AR) is False


# ---------------------------------------------------------------------------
# Confianza y needs_review
# ---------------------------------------------------------------------------

class TestConfianza:

    def test_lectura_limpia_no_necesita_revision(self):
        result = ocr_service.process_sync(barcode_photo(EAN13_AR_BODY), DocType.BARCODE)

        assert result.fields["codigo"].confidence == 1.0
        assert result.document_confidence >= settings.DOCUMENT_CONFIDENCE_THRESHOLD
        assert result.needs_review is False

    def test_sin_codigo_el_documento_va_a_revision(self):
        result = ocr_service.process_sync(
            barcode_photo(EAN13_AR_BODY, width=120), DocType.BARCODE
        )

        assert result.fields["codigo"].value is None
        assert result.needs_review is True
        assert result.document_confidence == 0.0

    def test_dos_codigos_distintos_bajan_la_confianza(self):
        """El envase y la etiqueta de envío en la misma foto: no está claro cuál
        es el producto, y la ambigüedad tiene que verse en la confianza."""
        fields = parser.parse(
            reading(symbol(EAN13_AR), symbol("7891234567895", width=200, height=100))
        )

        assert fields["codigo"].confidence == pytest.approx(parser.CONF_AMBIGUOUS)

    def test_el_mismo_codigo_repetido_no_es_ambiguo(self):
        """ZBar puede devolver dos veces el mismo símbolo. Eso no es ambigüedad y
        no tiene por qué penalizarse."""
        fields = parser.parse(reading(symbol(EAN13_AR), symbol(EAN13_AR)))

        assert fields["codigo"].confidence == 1.0

    def test_se_prefiere_el_codigo_de_gondola_sobre_el_logistico(self):
        """Aunque la etiqueta de envío sea más grande en la foto, el que el
        usuario quiere cargar es el del producto."""
        fields = parser.parse(
            reading(
                symbol("AR847362910", symbology="CODE128", width=900, height=300),
                symbol(EAN13_AR, width=200, height=100),
            )
        )

        assert fields["codigo"].value == EAN13_AR

    def test_entre_varios_gana_el_mas_grande(self):
        fields = parser.parse(
            reading(
                symbol("7891234567895", width=200, height=100),
                symbol(EAN13_AR, width=800, height=400),
            )
        )

        assert fields["codigo"].value == EAN13_AR

    def test_caja_mayorista_identifica_el_producto_pero_no_la_cantidad(self):
        gtin14 = "1" + EAN13_AR_BODY
        gtin14 += str(parser._check_digit(gtin14))

        fields = parser.parse(reading(symbol(gtin14, symbology="I25")))

        assert fields["codigo"].value == EAN13_AR
        assert fields["codigo"].confidence == pytest.approx(parser.CONF_CARTON)

    def test_itf_sin_check_valido_es_codigo_interno(self):
        fields = parser.parse(reading(symbol("01779123456789", symbology="I25")))

        assert fields["codigo"].confidence == pytest.approx(
            parser.CONF_UNVERIFIED * parser.CONF_NON_RETAIL
        )


# ---------------------------------------------------------------------------
# Integración con el coordinador
# ---------------------------------------------------------------------------

class TestCoordinador:

    def test_no_pasa_por_ningun_motor(self, fake_engine):
        """El motor no se toca: un código de barras no se lee reconociendo
        caracteres. Si el coordinador lo llamara, `calls` no estaría vacío."""
        engine = fake_engine(["lo que sea"])

        result = ocr_service.process_sync(barcode_photo(EAN13_AR_BODY), DocType.BARCODE)

        assert engine.calls == []
        assert result.engine == "zbar"

    def test_el_motor_pedido_se_ignora(self):
        """`engine=paddleocr` sobre un código de barras no cambia el número. El
        resultado reporta `zbar` para que en el banco de pruebas se vea quién
        leyó de verdad."""
        result = ocr_service.process_sync(
            barcode_photo(EAN13_AR_BODY), DocType.BARCODE, engine_name="paddleocr"
        )

        assert result.engine == "zbar"
        assert result.fields["codigo"].value == EAN13_AR

    def test_no_hay_plan_de_intentos_ocr(self):
        """`attempt_plan` falla explícito en vez de devolver el plan de etiqueta,
        que correría el motor sobre una foto donde no hay texto que leer."""
        with pytest.raises(ValueError, match="barcode"):
            ocr_service.attempt_plan(DocType.BARCODE)

    def test_el_texto_crudo_conserva_lo_decodificado(self):
        """`raw_text` guarda el código tal como vino, antes de normalizarlo: es
        lo que permite auditar desde el backoffice de dónde salió el EAN-13."""
        gtin14 = "1" + EAN13_AR_BODY
        gtin14 += str(parser._check_digit(gtin14))

        result = ocr_service.process_sync(barcode_photo(gtin14, kind="itf"), DocType.BARCODE)

        assert result.raw_text == gtin14
        assert result.fields["codigo"].value == EAN13_AR

    def test_sin_decodificador_se_reporta_como_no_disponible(self, monkeypatch):
        def explota():
            raise decoder.DecoderUnavailable("falta libzbar")

        monkeypatch.setattr(decoder, "_pyzbar", explota)

        assert decoder.is_available() is False

        with pytest.raises(decoder.DecoderUnavailable):
            decoder.decode(barcode_photo(EAN13_AR_BODY))

    def test_el_decodificador_esta_disponible(self):
        assert decoder.is_available() is True
        assert decoder.version() != "no instalado"


# ---------------------------------------------------------------------------
# Corroboración entre variantes
# ---------------------------------------------------------------------------

class TestCorroboracion:
    """Una segunda variante que lee lo mismo es la única evidencia independiente.

    El dígito verificador atrapa todos los errores de un dígito pero no todas las
    transposiciones: sobre el conjunto de prueba aparecieron imágenes donde dos
    variantes daban dos códigos **distintos y los dos válidos**, y se elegía por
    el tamaño del símbolo, es decir por nada.
    """

    def test_una_lectura_corroborada_no_se_penaliza(self):
        fields = parser.parse(reading(symbol(EAN13_AR), corroborated=True))

        assert fields["codigo"].confidence == 1.0

    def test_una_lectura_sin_corroborar_baja_la_confianza(self):
        fields = parser.parse(reading(symbol(EAN13_AR), corroborated=False))

        assert fields["codigo"].confidence == pytest.approx(parser.CONF_SIN_CORROBORAR)
        assert fields["codigo"].value == EAN13_AR, "se sigue ofreciendo el código"

    def test_una_foto_nitida_se_corrobora_sola(self):
        """Un código legible lo lee más de una variante: es el caso normal y no
        tiene que pagar la penalización."""
        result = decoder.decode(barcode_photo(EAN13_AR_BODY, width=420))

        assert result.corroborated is True

    def test_no_hay_corroboracion_si_las_demas_variantes_no_leen(self):
        """Sin una segunda opinión, la lectura se marca sin corroborar.

        Se prueba sobre `_corroborate` y no sobre una foto: con 22 variantes en
        la escalera, encontrar una imagen que lea en exactamente una es un
        equilibrio frágil, y el test mediría la suerte del caso en vez de la
        regla.
        """
        pyzbar = decoder._pyzbar()
        simbolos = [decoder.Symbol(data=EAN13_AR, symbology="EAN13")]
        blanco = Image.new("L", (400, 200), 255)

        # Variantes posteriores que no leen nada: nadie confirma.
        variantes = [("ganadora", lambda im: im), ("otra", lambda im: im)]

        assert decoder._corroborate(pyzbar, simbolos, iter(variantes), blanco, "ganadora") is False

    def test_hay_corroboracion_si_otra_variante_lee_lo_mismo(self):
        pyzbar = decoder._pyzbar()
        photo = Image.open(io.BytesIO(barcode_photo(EAN13_AR_BODY, width=420))).convert("L")
        simbolos = decoder._scan(pyzbar, photo)
        variantes = [("ganadora", lambda im: im), ("otra", lambda im: im)]

        assert decoder._corroborate(pyzbar, simbolos, iter(variantes), photo, "ganadora") is True

    def test_una_lectura_recuperada_por_reduccion_sigue_corroborada(self):
        """Varias reducciones leen el mismo código: se confirman entre ellas."""
        photo = barcode_photo(EAN13_AR_BODY, width=3000, blur=12, canvas=(4000, 2400))

        assert decoder.decode(photo).corroborated is True
