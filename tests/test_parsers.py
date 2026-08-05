"""Parsers de etiqueta y ticket, y cálculo de confianza."""

from datetime import date

from app.core.config import settings
from app.parsers import product_label, receipt, scoring
from app.schemas.ocr import DocType, ExtractedField
from app.services import ocr_service
from tests.conftest import make_output

TODAY = date(2026, 7, 29)


# ---------------------------------------------------------------------------
# Caso A — etiqueta de producto
# ---------------------------------------------------------------------------

def test_label_extracts_name_and_expiry():
    output = make_output(
        ["Leche La Serenisima 1L", "VTO 15/08/2027", "LOTE A2291"],
        heights=[44, 22, 18],
    )

    fields = product_label.parse(output, output.text, TODAY)

    assert fields["nombre_producto"].value == "Leche La Serenisima 1L"
    assert fields["fecha_vencimiento"].value == "2027-08-15"


def test_label_name_uses_largest_typography():
    """El nombre es la línea de tipografía más grande, no la primera."""
    output = make_output(
        ["Industria Argentina", "Yogur Firme Frutilla", "Peso neto 190g"],
        heights=[16, 48, 16],
    )

    fields = product_label.parse(output, output.text, TODAY)

    assert fields["nombre_producto"].value == "Yogur Firme Frutilla"


def test_label_ignores_noise_lines_as_name():
    output = make_output(["VTO 15/08/2027", "LOTE 99321", "Arroz Largo Fino"], heights=[50, 50, 30])

    fields = product_label.parse(output, output.text, TODAY)

    assert fields["nombre_producto"].value == "Arroz Largo Fino"


def test_label_without_date_reports_zero_confidence():
    output = make_output(["Fideos Tirabuzon"], heights=[40])

    fields = product_label.parse(output, output.text, TODAY)

    assert fields["fecha_vencimiento"].value is None
    assert fields["fecha_vencimiento"].confidence == 0.0


def test_label_past_expiry_is_penalized():
    """Una fecha ya vencida suele ser la de elaboración: baja la confianza."""
    past = make_output(["Leche Entera 1L", "VTO 01/01/2020"], heights=[40, 20])
    future = make_output(["Leche Entera 1L", "VTO 01/01/2028"], heights=[40, 20])

    past_conf = product_label.parse(past, past.text, TODAY)["fecha_vencimiento"].confidence
    future_conf = product_label.parse(future, future.text, TODAY)["fecha_vencimiento"].confidence

    assert past_conf < future_conf


def test_label_low_engine_confidence_propagates_to_fields():
    clear = make_output(["Leche Entera 1L", "VTO 01/01/2028"], confidence=0.98, heights=[40, 20])
    blurry = make_output(["Leche Entera 1L", "VTO 01/01/2028"], confidence=0.42, heights=[40, 20])

    clear_fields = product_label.parse(clear, clear.text, TODAY)
    blurry_fields = product_label.parse(blurry, blurry.text, TODAY)

    assert blurry_fields["nombre_producto"].confidence < clear_fields["nombre_producto"].confidence


def test_label_empty_output_returns_empty_fields():
    output = make_output([])

    fields = product_label.parse(output, output.text, TODAY)

    assert fields["nombre_producto"].value is None
    assert fields["fecha_vencimiento"].value is None


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


def test_receipt_quantity_suffix_format():
    output = make_output(["KIOSCO", "FECHA 20/07/2026", "Alfajor Triple x 4 1.200,00"])

    _, items = receipt.parse(output, output.text, TODAY)

    assert items[0].nombre_producto.value == "Alfajor Triple"
    assert items[0].cantidad.value == 4


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


def test_needs_review_when_a_key_field_is_below_threshold():
    fields = {
        "nombre_producto": ExtractedField(value="Leche", confidence=0.99),
        "fecha_vencimiento": ExtractedField(value="2027-01-01", confidence=0.60),
    }

    assert scoring.needs_review(fields, product_label.KEY_FIELDS) is True


def test_no_review_needed_when_all_key_fields_are_confident():
    fields = {
        "nombre_producto": ExtractedField(value="Leche", confidence=0.99),
        "fecha_vencimiento": ExtractedField(value="2027-01-01", confidence=0.90),
    }

    assert scoring.needs_review(fields, product_label.KEY_FIELDS) is False


def test_missing_key_field_forces_review():
    fields = {"nombre_producto": ExtractedField(value="Leche", confidence=0.99)}

    assert scoring.needs_review(fields, product_label.KEY_FIELDS) is True


def test_thresholds_match_the_proposal():
    assert settings.FIELD_CONFIDENCE_THRESHOLD == 0.85
    assert settings.DOCUMENT_CONFIDENCE_THRESHOLD == 0.70


# ---------------------------------------------------------------------------
# build_result — integración parser + scoring
# ---------------------------------------------------------------------------

def test_build_result_for_clean_label_does_not_need_review():
    output = make_output(
        ["Leche La Serenisima 1L", "VTO 15/08/2027"], confidence=0.99, heights=[44, 20]
    )

    result = ocr_service.build_result(output, output.text, DocType.PRODUCT_LABEL, TODAY)

    assert result.needs_review is False
    assert result.document_confidence >= settings.DOCUMENT_CONFIDENCE_THRESHOLD
    assert result.engine == "fake"


def test_build_result_for_blurry_label_needs_review():
    output = make_output(
        ["Leche La Serenisima 1L", "VTO 15/08/2027"], confidence=0.45, heights=[44, 20]
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


def test_receipt_with_inferred_quantities_needs_review():
    """Cantidad inferida (conf 0.6 < 0.85) obliga a revisión manual asistida."""
    output = make_output(
        ["ALMACEN", "FECHA 20/07/2026", "Aceite Girasol 3.100,00", "TOTAL 3.100,00"],
        confidence=0.99,
    )

    result = ocr_service.build_result(output, output.text, DocType.RECEIPT, TODAY)

    assert result.items
    assert result.needs_review is True


def test_build_result_keeps_raw_text_for_audit():
    output = make_output(["Leche Entera 1L", "VTO 15/08/2027"], heights=[40, 20])

    result = ocr_service.build_result(output, output.text, DocType.PRODUCT_LABEL, TODAY)

    assert "Leche Entera 1L" in result.raw_text
