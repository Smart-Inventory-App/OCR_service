"""Parsers de etiqueta y ticket, y cálculo de confianza."""

from datetime import date

import pytest

from app.core.config import settings
from app.parsers import expiry, product_label, receipt, scoring
from app.schemas.ocr import DocType, ExtractedField
from app.services import ocr_service
from tests.conftest import make_output

TODAY = date(2026, 7, 29)


# ---------------------------------------------------------------------------
# Caso A — etiqueta de producto: nombre, marca y detalles
# ---------------------------------------------------------------------------

def test_label_extracts_name_brand_and_details():
    output = make_output(
        ["La Serenisima", "LECHE ENTERA", "1 litro"], heights=[60, 30, 18]
    )

    fields = product_label.parse(output, output.text, TODAY)

    assert fields["nombre_producto"].value == "LECHE ENTERA"
    assert fields["marca"].value == "La Serenisima"
    assert "1 litro" in fields["detalles"].value


def test_label_name_is_the_category_not_the_biggest_line():
    """La tipografía más grande de un envase es la marca, no el producto.

    Es la regresión de la heurística anterior: `PUREZA` está impreso mucho más
    grande que `HARINA`, y el nombre del producto es `HARINA`.
    """
    output = make_output(["HARINA", "PUREZA", "1 kg"], heights=[34, 80, 16])

    fields = product_label.parse(output, output.text, TODAY)

    assert fields["nombre_producto"].value == "HARINA"
    assert fields["marca"].value == "PUREZA"


def test_label_merges_a_name_split_across_lines():
    """`Leche` y `Entera` en dos renglones son un solo nombre."""
    output = make_output(["Leche", "Entera", "1 litro"], heights=[40, 40, 16])

    fields = product_label.parse(output, output.text, TODAY)

    assert fields["nombre_producto"].value == "Leche Entera"


def test_label_recovers_a_split_first_letter():
    """El motor parte la versalita inicial: `C AFÉ` sigue siendo la categoría."""
    output = make_output(["LaVirginia", "C AFE INSTANTANEO", "en sobres"], heights=[70, 34, 18])

    fields = product_label.parse(output, output.text, TODAY)

    assert fields["nombre_producto"].value == "C AFE INSTANTANEO"
    assert fields["marca"].value == "LaVirginia"


def test_label_recovers_a_category_glued_to_its_qualifier():
    """PaddleOCR devuelve regiones, no palabras: `CAFEINSTANTANEO` sale pegado."""
    output = make_output(["LaVirginia", "CAFEINSTANTANEO", "en sobres"], heights=[70, 34, 18])

    fields = product_label.parse(output, output.text, TODAY)

    assert fields["nombre_producto"].value == "CAFEINSTANTANEO"
    assert fields["marca"].value == "LaVirginia"


def test_a_brand_derived_from_its_category_is_not_the_name():
    """`Yogurisimo` empieza con `yogur` y aun así es la marca, no el producto.

    Es el límite de la reparación anterior: se acepta una categoría pegada sólo
    cuando lo que sigue es otro término conocido. `isimo` no nombra nada.
    """
    output = make_output(
        ["Yogurisimo", "YOGUR FIRME FRUTILLA", "190 g"], heights=[70, 34, 16]
    )

    fields = product_label.parse(output, output.text, TODAY)

    assert fields["nombre_producto"].value == "YOGUR FIRME FRUTILLA"
    assert fields["marca"].value == "Yogurisimo"


def test_a_condiment_is_a_product_category():
    """Faltaban enteras: sin `perejil`, el nombre caía en la marca.

    Las alturas son parecidas a propósito: así el agrupamiento une la marca con
    el nombre en un solo bloque, que es lo que pasa en la foto real del envase.
    """
    output = make_output(
        ["Alicante", "Perejil", "deshidratado", "25 g"], heights=[46, 60, 30, 14]
    )

    fields = product_label.parse(output, output.text, TODAY)

    assert fields["nombre_producto"].value == "Perejil"
    assert fields["marca"].value == "Alicante"


def test_a_name_that_starts_with_its_category_is_not_trimmed():
    """El recorte de marca pegada no debe comerse la primera palabra del nombre."""
    for lines, heights in (
        (["Leche", "Entera", "1 litro"], [40, 40, 16]),
        (["Yogurisimo", "YOGUR FIRME FRUTILLA", "190 g"], [70, 34, 16]),
    ):
        output = make_output(lines, heights=heights)

        nombre = product_label.parse(output, output.text, TODAY)["nombre_producto"].value

        assert nombre in ("Leche Entera", "YOGUR FIRME FRUTILLA")


def test_label_survives_a_logo_mark_before_the_category():
    """La `O` del logo no es una versalita partida: pegarla borraría `CAFE`.

    Además se recorta del nombre, por la misma regla que separa la marca que el
    agrupamiento pega adelante: lo que precede a la categoría no es el producto.
    """
    output = make_output(["Cabrales", "O CAFE TOSTADO", "250 g"], heights=[80, 40, 16])

    fields = product_label.parse(output, output.text, TODAY)

    assert fields["nombre_producto"].value == "CAFE TOSTADO"
    assert fields["marca"].value == "Cabrales"


def test_label_does_not_extract_expiry_date():
    """El vencimiento tiene su propio tipo de documento y su propia foto."""
    output = make_output(["LECHE ENTERA", "VTO 15/08/2027"], heights=[40, 20])

    fields = product_label.parse(output, output.text, TODAY)

    assert "fecha_vencimiento" not in fields


def test_label_ignores_noise_lines_as_name():
    output = make_output(["VTO 15/08/2027", "LOTE 99321", "Arroz Largo Fino"], heights=[50, 50, 30])

    fields = product_label.parse(output, output.text, TODAY)

    assert fields["nombre_producto"].value == "Arroz Largo Fino"


def test_label_low_engine_confidence_propagates_to_fields():
    clear = make_output(["Marca Uno", "Leche Entera"], confidence=0.98, heights=[50, 30])
    blurry = make_output(["Marca Uno", "Leche Entera"], confidence=0.55, heights=[50, 30])

    clear_fields = product_label.parse(clear, clear.text, TODAY)
    blurry_fields = product_label.parse(blurry, blurry.text, TODAY)

    assert blurry_fields["nombre_producto"].confidence < clear_fields["nombre_producto"].confidence


@pytest.mark.parametrize("garbage", ["( E Y e L", "AR es", "*/ -- ~~", "1 2 3 4 5"])
def test_label_rejects_ocr_garbage_as_name(garbage):
    """Ruido de una lectura fallida no debe ofrecerse como nombre de producto."""
    output = make_output([garbage], heights=[40])

    fields = product_label.parse(output, output.text, TODAY)

    assert fields["nombre_producto"].value is None


def test_label_rejects_a_name_read_with_low_confidence():
    """Un valor sin sentido es peor que un campo vacío: vacío pide revisión."""
    output = make_output(["AU RARA"], confidence=0.2, heights=[40])

    fields = product_label.parse(output, output.text, TODAY)

    assert fields["nombre_producto"].value is None


def test_label_brand_must_be_typographically_prominent():
    """La letra chica legal nunca es la marca, aunque sea lo único que quedó."""
    output = make_output(["Leche Entera", "do Nero"], confidence=0.9, heights=[60, 14])

    fields = product_label.parse(output, output.text, TODAY)

    assert fields["nombre_producto"].value == "Leche Entera"
    assert fields["marca"].value is None


def test_label_accepts_real_name_over_garbage():
    output = make_output(["( E Y e L", "Leche Entera"], heights=[50, 30])

    fields = product_label.parse(output, output.text, TODAY)

    assert fields["nombre_producto"].value == "Leche Entera"


def test_label_empty_output_returns_empty_fields():
    output = make_output([])

    fields = product_label.parse(output, output.text, TODAY)

    assert fields["nombre_producto"].value is None
    assert fields["marca"].value is None
    assert fields["detalles"].value is None


# ---------------------------------------------------------------------------
# Caso B — sello de fecha de vencimiento
# ---------------------------------------------------------------------------

def test_expiry_extracts_the_announced_date():
    output = make_output(["CONSUMIR PREFERENTEMENTE ANTES DE:", "24/10/2027"])

    fields = expiry.parse(output, output.text, TODAY)

    assert fields["fecha_vencimiento"].value == "2027-10-24"


def test_expiry_ignores_the_production_date():
    """`ENV.: 05/01/26  VTO.: 05/01/27` — la de envasado no es el vencimiento."""
    output = make_output(["ENV.: 05/01/26", "VTO.: 05/01/27", "L:2601 B 09:15"])

    fields = expiry.parse(output, output.text, TODAY)

    assert fields["fecha_vencimiento"].value == "2027-01-05"


def test_expiry_without_a_date_reports_zero_confidence():
    output = make_output(["LOTE A2291", "B 09:15"])

    fields = expiry.parse(output, output.text, TODAY)

    assert fields["fecha_vencimiento"].value is None
    assert fields["fecha_vencimiento"].confidence == 0.0


def test_expiry_past_date_is_penalized():
    """Una fecha ya vencida suele ser la de elaboración: baja la confianza."""
    past = make_output(["VTO 01/01/2025"])
    future = make_output(["VTO 01/01/2028"])

    past_conf = expiry.parse(past, past.text, TODAY)["fecha_vencimiento"].confidence
    future_conf = expiry.parse(future, future.text, TODAY)["fecha_vencimiento"].confidence

    assert past_conf < future_conf


def test_expiry_without_keyword_is_less_confident():
    keyed = make_output(["VTO 15/08/2027"])
    loose = make_output(["15/08/2027"])

    keyed_conf = expiry.parse(keyed, keyed.text, TODAY)["fecha_vencimiento"].confidence
    loose_conf = expiry.parse(loose, loose.text, TODAY)["fecha_vencimiento"].confidence

    assert loose_conf < keyed_conf


def test_expiry_repaired_month_is_less_confident():
    """`24 GCT 27` se recupera, pero es una lectura reparada y se nota."""
    clean = make_output(["VTO 24 OCT 2027"])
    repaired = make_output(["VTO 24 GCT 2027"])

    clean_fields = expiry.parse(clean, clean.text, TODAY)
    repaired_fields = expiry.parse(repaired, repaired.text, TODAY)

    assert repaired_fields["fecha_vencimiento"].value == "2027-10-24"
    assert (
        repaired_fields["fecha_vencimiento"].confidence
        < clean_fields["fecha_vencimiento"].confidence
    )


# ---------------------------------------------------------------------------
# Caso B — ticket / remito / factura
# ---------------------------------------------------------------------------

RECEIPT_LINES = [
    "SUPERMERCADO LA ESQUINA",
    "CUIT 30-12345678-9",
    "FECHA 20/07/2026",
    "2 x Leche Entera 1L 2.400,00",
    "1 x Pan Lactal 1.850,50",
    "3 Yogur Frutilla 990,00",
    "TOTAL 5.240,50",
]


def test_receipt_extracts_document_fields():
    output = make_output(RECEIPT_LINES)

    fields, _ = receipt.parse(output, output.text, TODAY)

    assert fields["fecha_compra"].value == "2026-07-20"
    assert fields["total"].value == 5240.50
    assert fields["comercio"].value == "SUPERMERCADO LA ESQUINA"


def test_receipt_extracts_items_with_quantities():
    output = make_output(RECEIPT_LINES)

    _, items = receipt.parse(output, output.text, TODAY)

    extracted = {item.nombre_producto.value: item.cantidad.value for item in items}

    assert extracted == {
        "Leche Entera 1L": 2,
        "Pan Lactal": 1,
        "Yogur Frutilla": 3,
    }


def test_receipt_skips_header_and_footer_lines():
    output = make_output(RECEIPT_LINES)

    _, items = receipt.parse(output, output.text, TODAY)
    names = [item.nombre_producto.value for item in items]

    assert not any("TOTAL" in name or "CUIT" in name for name in names)
    assert "SUPERMERCADO LA ESQUINA" not in names


# ---------------------------------------------------------------------------
# Factura fiscal argentina — los tres casos salieron de un ticket real de DIA
# ---------------------------------------------------------------------------

FACTURA_B_LINES = [
    "DIA ARGENTINA SA",
    "CUIT:30-63584975-1",
    "FACTURA B ORIGINAL (COD. 006)",
    "NUM.08028-00545295 22-08-2026 17:04:50",
    "Descripcion (%IVA) [%BI ] IMPORTE",
    "1,00X 8909,00",
    "PAN ARTESANO BIMBO (21,00) 8909,00",
    "1,00X 4475,00",
    "YOG FRUT 0% LACTOSA (21,00) 4475,00",
    "SUBTOTAL: 17134,00",
    "DESCUENTOS: -1118,75",
    "TOTAL $ 16015.25",
    "IVA CONTENIDO 2779,51",
    "OTROS TRIBUTOS NACIONALES",
    "INDIRECTOS 0,00",
    "TARJETA $16015.25",
    "Suma de sus pagos: $16015.25",
    "AHORRO OFERTAS 1118,75",
    "TOT AHORRO 1118,75",
]


def test_receipt_total_with_a_dot_decimal_among_comma_decimals():
    """El mismo ticket lista los ítems con coma y el total con punto.

    Antes el importe se partía en `160`, `15` y `25`, y el total del documento
    quedaba en 25,00 pesos.
    """
    output = make_output(FACTURA_B_LINES)

    fields, _ = receipt.parse(output, output.text, TODAY)

    assert fields["total"].value == 16015.25


def test_receipt_drops_the_iva_column_from_the_item_name():
    """`(21,00)` es la alícuota de la factura, no parte del producto."""
    output = make_output(FACTURA_B_LINES)

    _, items = receipt.parse(output, output.text, TODAY)
    names = [item.nombre_producto.value for item in items]

    assert "PAN ARTESANO BIMBO" in names
    assert "YOG FRUT 0% LACTOSA" in names


def test_receipt_ignores_the_fiscal_footer_of_a_real_invoice():
    """Debajo del total hay media docena de renglones con forma de ítem."""
    output = make_output(FACTURA_B_LINES)

    _, items = receipt.parse(output, output.text, TODAY)
    names = [item.nombre_producto.value for item in items]

    assert names == ["PAN ARTESANO BIMBO", "YOG FRUT 0% LACTOSA"]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("16015.25", 16015.25),   # punto decimal
        ("8909,00", 8909.00),     # coma decimal
        ("1.234,56", 1234.56),    # punto de miles + coma decimal
        ("1.234", 1234.0),        # punto de miles solo: no es 1,234
        ("1.234.567,89", 1234567.89),
        ("1234", 1234.0),
    ],
)
def test_amounts_are_read_under_both_conventions(raw, expected):
    assert receipt._to_amount(raw) == expected


def test_receipt_quantity_suffix_format():
    output = make_output(["KIOSCO", "FECHA 20/07/2026", "Alfajor Triple x 4 1.200,00"])

    _, items = receipt.parse(output, output.text, TODAY)

    assert items[0].nombre_producto.value == "Alfajor Triple"
    assert items[0].cantidad.value == 4


def test_receipt_quantity_glued_to_name():
    """Tesseract suele leer «2 x Leche» como «2xLeche», sin espacios."""
    output = make_output(["ALMACEN", "FECHA 20/07/2026", "2xLeche Entera 1L 2.400,00"])

    _, items = receipt.parse(output, output.text, TODAY)

    assert items[0].nombre_producto.value == "Leche Entera 1L"
    assert items[0].cantidad.value == 2


def test_receipt_unit_marker_does_not_split_a_word():
    """«2 unidades Leche» no debe partirse en «idades Leche»."""
    output = make_output(["ALMACEN", "FECHA 20/07/2026", "2 unidades Leche 2.400,00"])

    _, items = receipt.parse(output, output.text, TODAY)

    assert items[0].nombre_producto.value == "unidades Leche"
    assert items[0].cantidad.value == 2


def test_receipt_name_keeps_trailing_x():
    """Recortar la «x» rompería productos que se llaman «Max» o «Box»."""
    output = make_output(["ALMACEN", "FECHA 20/07/2026", "1 x Jugo Max 1.200,00"])

    _, items = receipt.parse(output, output.text, TODAY)

    assert items[0].nombre_producto.value == "Jugo Max"


def test_receipt_implicit_quantity_is_one_with_lower_confidence():
    output = make_output(["ALMACEN", "FECHA 20/07/2026", "Aceite Girasol 900ml 3.100,00"])

    _, items = receipt.parse(output, output.text, TODAY)

    assert items[0].cantidad.value == 1
    assert items[0].cantidad.confidence < items[0].nombre_producto.confidence


def test_receipt_total_prefers_total_over_subtotal():
    output = make_output(
        ["ALMACEN", "FECHA 20/07/2026", "Leche 1L 2.400,00", "SUBTOTAL 2.400,00", "TOTAL 2.900,00"]
    )

    fields, _ = receipt.parse(output, output.text, TODAY)

    assert fields["total"].value == 2900.00


def test_receipt_without_items_returns_empty_list():
    output = make_output(["SUPERMERCADO", "FECHA 20/07/2026", "TOTAL 0,00"])

    _, items = receipt.parse(output, output.text, TODAY)

    assert items == []


def test_receipt_amount_without_thousands_separator():
    output = make_output(["ALMACEN", "FECHA 20/07/2026", "TOTAL 999,90"])

    fields, _ = receipt.parse(output, output.text, TODAY)

    assert fields["total"].value == 999.90


def test_receipt_future_purchase_date_is_penalized():
    future = make_output(["ALMACEN", "FECHA 20/07/2030", "TOTAL 100,00"])
    past = make_output(["ALMACEN", "FECHA 20/07/2026", "TOTAL 100,00"])

    future_conf = receipt.parse(future, future.text, TODAY)[0]["fecha_compra"].confidence
    past_conf = receipt.parse(past, past.text, TODAY)[0]["fecha_compra"].confidence

    assert future_conf < past_conf


# ---------------------------------------------------------------------------
# Scoring y umbrales de la propuesta
# ---------------------------------------------------------------------------

def test_combine_multiplies_engine_and_parser_confidence():
    assert scoring.combine(0.9, 0.5) == 0.45


def test_combine_clamps_out_of_range_values():
    assert scoring.combine(1.5, 1.0) == 1.0
    assert scoring.combine(-0.2, 1.0) == 0.0


def test_field_without_value_has_zero_confidence():
    assert scoring.field(None, 0.99, 0.99).confidence == 0.0
    assert scoring.field("   ", 0.99, 0.99).confidence == 0.0


def test_document_confidence_is_weighted_average():
    fields = {
        "nombre_producto": ExtractedField(value="Leche", confidence=1.0),   # peso 1.0
        "fecha_vencimiento": ExtractedField(value="2027-01-01", confidence=0.0),  # peso 1.0
        "comercio": ExtractedField(value="X", confidence=1.0),              # peso 0.5
    }

    # (1.0 + 0.0 + 0.5) / 2.5 = 0.6
    assert scoring.document_confidence(fields) == 0.6


def test_document_confidence_of_empty_fields_is_zero():
    assert scoring.document_confidence({}) == 0.0


KEY_FIELDS_BOTH = ("nombre_producto", "fecha_vencimiento")


def test_needs_review_when_a_key_field_is_below_threshold():
    fields = {
        "nombre_producto": ExtractedField(value="Leche", confidence=0.99),
        "fecha_vencimiento": ExtractedField(value="2027-01-01", confidence=0.60),
    }

    assert scoring.needs_review(fields, KEY_FIELDS_BOTH) is True


def test_no_review_needed_when_all_key_fields_are_confident():
    fields = {
        "nombre_producto": ExtractedField(value="Leche", confidence=0.99),
        "fecha_vencimiento": ExtractedField(value="2027-01-01", confidence=0.90),
    }

    assert scoring.needs_review(fields, KEY_FIELDS_BOTH) is False


def test_missing_key_field_forces_review():
    fields = {"nombre_producto": ExtractedField(value="Leche", confidence=0.99)}

    assert scoring.needs_review(fields, KEY_FIELDS_BOTH) is True


def test_thresholds_match_the_proposal():
    assert settings.FIELD_CONFIDENCE_THRESHOLD == 0.85
    assert settings.DOCUMENT_CONFIDENCE_THRESHOLD == 0.70


# ---------------------------------------------------------------------------
# build_result — integración parser + scoring
# ---------------------------------------------------------------------------

def test_build_result_for_clean_label_does_not_need_review():
    output = make_output(
        ["La Serenisima", "LECHE ENTERA", "1 litro"], confidence=0.99, heights=[60, 30, 16]
    )

    result = ocr_service.build_result(output, output.text, DocType.PRODUCT_LABEL, TODAY)

    assert result.needs_review is False
    assert result.document_confidence >= settings.DOCUMENT_CONFIDENCE_THRESHOLD
    assert result.engine == "fake"


def test_build_result_for_blurry_label_needs_review():
    output = make_output(
        ["La Serenisima", "LECHE ENTERA", "1 litro"], confidence=0.45, heights=[60, 30, 16]
    )

    result = ocr_service.build_result(output, output.text, DocType.PRODUCT_LABEL, TODAY)

    assert result.needs_review is True


def test_build_result_for_receipt_populates_items():
    output = make_output(RECEIPT_LINES, confidence=0.99)

    result = ocr_service.build_result(output, output.text, DocType.RECEIPT, TODAY)

    assert len(result.items) == 3
    assert "fecha_compra" in result.fields


def test_receipt_without_items_needs_review():
    """Un comprobante sin ítems no sirve para cargar inventario sin confirmar."""
    output = make_output(["SUPERMERCADO", "FECHA 20/07/2026", "TOTAL 0,00"], confidence=0.99)

    result = ocr_service.build_result(output, output.text, DocType.RECEIPT, TODAY)

    assert result.items == []
    assert result.needs_review is True


def test_receipt_with_inferred_quantities_does_not_need_review():
    """Una cantidad inferida no bloquea el alta.

    Un ticket argentino no imprime la cantidad cuando es 1: el parser la infiere
    y le pone 0,6 para declarar que la infirió. Exigirle el umbral por campo
    mandaba a revisión el 100 % de los tickets reales del conjunto de prueba, con
    los nombres leídos a 0,99 y el total al centavo. Lo que sí sigue bloqueando
    es un nombre de ítem dudoso: sin saber qué producto es, no hay alta posible.
    """
    output = make_output(
        ["ALMACEN", "FECHA 20/07/2026", "Aceite Girasol 3.100,00", "TOTAL 3.100,00"],
        confidence=0.99,
    )

    result = ocr_service.build_result(output, output.text, DocType.RECEIPT, TODAY)

    assert result.items
    assert result.items[0].cantidad.value == 1
    assert result.items[0].cantidad.confidence < 0.85
    assert result.needs_review is False


def test_receipt_with_an_unreadable_item_name_still_needs_review():
    """La contraparte: el nombre sí es condición para el alta automática."""
    output = make_output(
        ["ALMACEN", "FECHA 20/07/2026", "Aceite Girasol 3.100,00", "TOTAL 3.100,00"],
        confidence=0.45,
    )

    result = ocr_service.build_result(output, output.text, DocType.RECEIPT, TODAY)

    assert result.needs_review is True


def test_build_result_keeps_raw_text_for_audit():
    output = make_output(["Leche Entera", "1 litro"], heights=[40, 20])

    result = ocr_service.build_result(output, output.text, DocType.PRODUCT_LABEL, TODAY)

    assert "Leche Entera" in result.raw_text


def test_build_result_for_expiry_scores_only_the_date():
    output = make_output(["VTO: 15/08/2027"], confidence=0.99)

    result = ocr_service.build_result(output, output.text, DocType.EXPIRY_DATE, TODAY)

    assert set(result.fields) == {"fecha_vencimiento"}
    assert result.fields["fecha_vencimiento"].value == "2027-08-15"
    assert result.needs_review is False


def test_build_result_for_unreadable_stamp_needs_review():
    output = make_output(["LOTE A2291", "B 09:15"], confidence=0.99)

    result = ocr_service.build_result(output, output.text, DocType.EXPIRY_DATE, TODAY)

    assert result.fields["fecha_vencimiento"].value is None
    assert result.needs_review is True


# ---------------------------------------------------------------------------
# Vocabulario de farmacia
# ---------------------------------------------------------------------------
#
# Faltaba entero. Sobre el conjunto de prueba las etiquetas de farmacia daban 2
# altas automáticas sobre 27 (7 %) contra 59 % en almacén: sin una categoría
# reconocible el parser cae a la heurística de reserva —la tipografía más
# grande— que es justo la que confunde la marca con el nombre.

@pytest.mark.parametrize(
    "lineas, alturas, nombre, marca",
    [
        (["BAYASPIRINA", "IBUPROFENO 400 mg", "20 comprimidos"], [60, 30, 16],
         "IBUPROFENO 400 mg", "BAYASPIRINA"),
        (["GENIOL", "PARACETAMOL", "500 mg"], [58, 28, 14], "PARACETAMOL", "GENIOL"),
        (["DERMAGLOS", "PROTECTOR SOLAR FPS 50", "200 ml"], [60, 26, 14],
         "PROTECTOR SOLAR FPS 50", "DERMAGLOS"),
        (["COLGATE", "CREMA DENTAL TRIPLE ACCION", "90 g"], [60, 26, 14],
         "CREMA DENTAL TRIPLE ACCION", "COLGATE"),
    ],
)
def test_pharmacy_labels_name_the_product_and_not_the_brand(lineas, alturas, nombre, marca):
    output = make_output(lineas, confidence=0.97, heights=alturas)

    fields = product_label.parse(output, output.text)

    assert fields["nombre_producto"].value == nombre
    assert fields["marca"].value == marca
    assert fields["nombre_producto"].confidence >= 0.85


def test_the_dosage_line_of_a_package_is_not_a_product_name():
    """`gotas`, `polvo` y `vitaminas` quedaron fuera del vocabulario a propósito.

    Aparecen en el texto de uso del envase o ya son descriptores, y como
    categoría harían que una línea accesoria le ganara al nombre real.
    """
    output = make_output(
        ["HILERET", "EDULCORANTE LIQUIDO", "8 gotas = 1 cucharadita"],
        confidence=0.96,
        heights=[60, 28, 12],
    )

    fields = product_label.parse(output, output.text)

    assert "gotas" not in (fields["nombre_producto"].value or "").lower()


@pytest.mark.parametrize(
    "caso, lineas, alturas, nombre, marca",
    [
        # Los tres salieron de fotos reales del conjunto de prueba, con la
        # geometría que tienen en el envase.
        (
            "IbuEvanol Forte",
            ["ANALGESICO Y ANTIINFLAMATORIO HALEON", "IbuEvanol", "FORTE",
             "RAPIDA ACCION", "Ibuprofeno 400 mg", "20 capsulas blandas"],
            [12, 60, 34, 20, 30, 14],
            "Ibuprofeno 400 mg", "IbuEvanol",
        ),
        (
            "Sertal Compuesto",
            ["Sertal", "Compuesto", "Antiespasmodico y analgesico",
             "10 Comprimidos recubiertos", "Propinox Clonixinato de lisina"],
            [60, 36, 22, 16, 14],
            "Sertal", None,
        ),
        (
            "Contractil",
            ["Efecto FLEBOCONSTRICTOR", "Contractil", "ANTIHEMORROIDAL",
             "CONT. 30 Capsulas", "Elea"],
            [16, 66, 20, 12, 24],
            "Contractil", None,
        ),
    ],
)
def test_real_pharmacy_boxes(caso, lineas, alturas, nombre, marca):
    output = make_output(lineas, confidence=0.96, heights=alturas)

    fields = product_label.parse(output, output.text)

    assert fields["nombre_producto"].value == nombre, caso
    if marca is not None:
        assert fields["marca"].value == marca, caso


@pytest.mark.parametrize(
    "frase",
    ["RAPIDA ACCION", "NUEVA FORMULA", "ANTIINFLAMATORIO", "VENTA LIBRE",
     "ANTIESPASMODICO Y ANALGESICO"],
)
def test_marketing_and_effect_lines_never_become_the_product_name(frase):
    """Sobre las fotos de farmacia estas frases estaban ganando el campo.

    No nombran un producto: describen su efecto o su presentación. Van a
    DESCRIPTORS justamente para que no compitan por el nombre.
    """
    output = make_output(
        ["MARCAX", frase, "Ibuprofeno 400 mg"], confidence=0.96, heights=[60, 40, 26]
    )

    fields = product_label.parse(output, output.text)

    assert fields["nombre_producto"].value == "Ibuprofeno 400 mg"


def test_the_composition_line_does_not_beat_the_product_name():
    """La letra chica de composición no puede ganarle al nombre.

    Entre los bloques que nombran una categoría se elige el más grande, así que
    una monodroga que sólo aparece en la línea de composición ganaría sin
    competencia aunque midiera la mitad. Por eso esas monodrogas no están en el
    vocabulario.
    """
    output = make_output(
        ["Sertal", "Compuesto", "Propinox Clonixinato de lisina"],
        confidence=0.96,
        heights=[60, 36, 12],
    )

    fields = product_label.parse(output, output.text)

    assert fields["nombre_producto"].value == "Sertal"


def test_glued_descriptor_is_trimmed_from_the_name():
    """`RAPIDA ACCION Ibuprofeno 400 mg` tiene que dar sólo la parte del nombre.

    El agrupamiento por geometría pega el descriptor adelante; el recorte busca
    el corte más tardío que todavía nombra la categoría, no el primero.
    """
    assert product_label._split_leading_brand("RAPIDA ACCION Ibuprofeno 400 mg") == (
        "Ibuprofeno 400 mg",
        "RAPIDA ACCION",
    )


def test_a_fiscal_receipt_date_without_a_label_is_still_trusted():
    """El controlador fiscal imprime la fecha pegada al número de comprobante.

    Sin rótulo `FECHA`, pero completa: día, mes y año de cuatro dígitos. Esa
    forma no aparece por casualidad en un ticket, así que la penalización por
    falta de palabra clave se afloja y el comprobante puede cargarse solo.
    """
    output = make_output(
        ["DIA ARGENTINA SA",
         "NUM.08028-00545295 22-08-2026 17:04:50 A CONSUMIDOR FINAL",
         "PAN ARTESANO BIMBO 8909,00",
         "TOTAL 16015.25"],
        confidence=0.98,
    )

    result = ocr_service.build_result(output, output.text, DocType.RECEIPT, date(2026, 9, 7))

    assert result.fields["fecha_compra"].value == "2026-08-22"
    assert result.fields["fecha_compra"].confidence >= settings.FIELD_CONFIDENCE_THRESHOLD
    assert result.needs_review is False


def test_an_incomplete_receipt_date_keeps_the_full_penalty():
    """La contraparte: si la fecha viene sin año de cuatro dígitos, la duda sigue.

    Sin rótulo y sin fecha completa puede ser cualquier número del ticket, así
    que la penalización se mantiene y el documento va a revisión.
    """
    output = make_output(
        ["ALMACEN", "08/2026", "Aceite Girasol 3.100,00", "TOTAL 3.100,00"],
        confidence=0.98,
    )

    result = ocr_service.build_result(output, output.text, DocType.RECEIPT, date(2026, 9, 7))

    assert result.fields["fecha_compra"].confidence < settings.FIELD_CONFIDENCE_THRESHOLD


def test_a_package_count_is_not_the_total():
    """`TOTAL DE BULTOS: 17` es un conteo, no plata.

    Aparece al pie de cualquier remito —que además no lleva importes— y sin
    filtrarlo el documento salía con `total = 17`: un número correcto puesto en
    el campo equivocado, que es peor que no tener el campo.
    """
    output = make_output(
        ["LOGISTICA SUR S.R.L.", "Fecha: 29/08/2026",
         "6 x DIA LOMITOS DE ATUN", "2 x HILERET LIGHT",
         "TOTAL DE BULTOS: 17"],
        confidence=0.97,
    )

    fields, items = receipt.parse(output, output.text, date(2026, 9, 8))

    assert fields["total"].value is None
    assert len(items) == 2


def test_a_real_total_is_still_read():
    output = make_output(
        ["ALMACEN", "Fecha: 29/08/2026", "Aceite Girasol 3.100,00", "TOTAL: 3.100,00"],
        confidence=0.97,
    )

    fields, _ = receipt.parse(output, output.text, date(2026, 9, 8))

    assert fields["total"].value == 3100.00


def test_a_delivery_note_without_prices_is_not_penalised():
    """Un remito no lleva importes: no le falta nada.

    Penalizar sus ítems por no traer precio —correcto en un ticket, donde una
    línea sin importe suele ser un encabezado— castiga al documento por su
    formato. Medido: los 20 remitos generados extraían bien sus ítems y los 20
    caían igual en revisión, con la confianza justo abajo del umbral.
    """
    output = make_output(
        ["LOGISTICA SUR S.R.L.", "Fecha: 29/08/2026",
         "6 x DIA LOMITOS DE ATUN", "2 x HILERET LIGHT", "1 x ALICANTE PEREJIL"],
        confidence=0.97,
    )

    result = ocr_service.build_result(output, output.text, DocType.RECEIPT, date(2026, 9, 8))

    assert len(result.items) == 3
    assert all(i.nombre_producto.confidence >= 0.85 for i in result.items)
    assert result.needs_review is False


def test_a_ticket_line_without_a_price_is_still_less_certain():
    """La contraparte: si el documento SÍ tiene precios, la línea sin importe
    sigue siendo un ítem menos seguro. La penalización no desapareció."""
    output = make_output(
        ["ALMACEN", "Fecha: 29/08/2026",
         "2 x YERBA PLAYADITO", "Aceite Girasol 3.100,00", "TOTAL: 3.100,00"],
        confidence=0.97,
    )

    _fields, items = receipt.parse(output, output.text, date(2026, 9, 8))
    sin_precio = [i for i in items if "YERBA" in (i.nombre_producto.value or "")]

    assert sin_precio and sin_precio[0].nombre_producto.confidence < 0.97


# ---------------------------------------------------------------------------
# Cantidad anunciada en el renglón anterior
# ---------------------------------------------------------------------------

def test_the_quantity_announced_on_the_previous_line_is_used():
    """Un fiscal argentino imprime `2,00 x 8.128,62` ARRIBA de la descripción.

    Leyendo renglón por renglón la descripción no trae cantidad y el parser
    asumía 1: correcto de casualidad en la mayoría de los ítems, equivocado en
    todos los demás.
    """
    output = make_output(
        ["DISTRIBUIDORA NORTE", "Fecha: 22/08/2026",
         "2,00 x 8.128,62", "PANTENE PRO-V MIRACLES $ 16.257,24",
         "3,00 x 534,71", "ELITE ULTRA ABSORBENTE $ 1.604,13",
         "TOTAL: 17.861,37"],
        confidence=0.98,
    )

    _fields, items = receipt.parse(output, output.text, date(2026, 9, 8))
    cantidades = {i.nombre_producto.value.split()[0]: i.cantidad.value for i in items}

    assert cantidades.get("PANTENE") == 2
    assert cantidades.get("ELITE") == 3


def test_the_announcement_only_applies_to_the_next_line():
    """Arrastrar el anuncio convertiría un descuido del OCR en cantidades
    inventadas: vale para el renglón siguiente y se descarta."""
    output = make_output(
        ["ALMACEN", "Fecha: 22/08/2026",
         "3,00 x 500,00", "YERBA PLAYADITO $ 1.500,00",
         "ACEITE NATURA $ 900,00",
         "TOTAL: 2.400,00"],
        confidence=0.98,
    )

    _fields, items = receipt.parse(output, output.text, date(2026, 9, 8))
    cantidades = {i.nombre_producto.value.split()[0]: i.cantidad.value for i in items}

    assert cantidades.get("YERBA") == 3
    assert cantidades.get("ACEITE") == 1


def test_a_fractional_announcement_is_ignored():
    """`0,25 x 12.000,01` es un cuarto de kilo, no una unidad contable."""
    output = make_output(
        ["CASA RAMOS", "Fecha: 04/09/2026",
         "0,25 x 12.000,01", "TALLARINES AL HUEVO FINOS $ 3.000,00",
         "TOTAL: 3.000,00"],
        confidence=0.98,
    )

    _fields, items = receipt.parse(output, output.text, date(2026, 9, 8))

    assert items and items[0].cantidad.value == 1
    # Y se declara como inferencia, no como dato leído.
    assert items[0].cantidad.confidence < 0.85


def test_a_quantity_written_on_the_line_wins_over_the_announcement():
    output = make_output(
        ["ALMACEN", "Fecha: 22/08/2026",
         "3,00 x 500,00", "2 x YERBA PLAYADITO $ 1.000,00",
         "TOTAL: 1.000,00"],
        confidence=0.98,
    )

    _fields, items = receipt.parse(output, output.text, date(2026, 9, 8))

    assert items and items[0].cantidad.value == 2
