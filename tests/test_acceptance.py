"""Tests de aceptación del módulo OCR — el *gate* de la propuesta.

Verifican los dos criterios de aceptación:

- **Precisión ≥ 85%** en la extracción de campos clave sobre el conjunto de validación.
- **Precisión global por documento ≥ 70%**.

El conjunto de validación de este prototipo son documentos sintéticos con su
ground truth: el motor se sustituye por un doble cuya confianza y cuyas líneas
reproducen escenarios realistas. Es un *gate* de regresión sobre la extracción y
el scoring, no una medición de precisión de los motores — para eso está
`run_benchmark.py`, que corre sobre el dataset que se le indique.

Los tres tipos de documento se cubren por separado, que es justamente el cambio
que introdujo la división del escaneo: la etiqueta aporta nombre, marca y
detalles; el sello aporta la fecha.
"""

from datetime import date

import pytest

from app.core.config import settings
from app.schemas.ocr import DocType
from app.services import ocr_service
from tests.conftest import barcode_photo, make_output

TODAY = date(2026, 7, 29)

# ---------------------------------------------------------------------------
# Conjunto de validación: (líneas leídas, alturas, confianza, ground truth)
# ---------------------------------------------------------------------------

LABEL_CASES = [
    (
        ["La Serenisima", "LECHE ENTERA", "1 litro"],
        [60, 30, 16],
        0.97,
        {"nombre_producto": "LECHE ENTERA", "marca": "La Serenisima"},
    ),
    (
        ["Yogurisimo", "YOGUR FIRME FRUTILLA", "190 g"],
        [60, 28, 14],
        0.95,
        {"nombre_producto": "YOGUR FIRME FRUTILLA", "marca": "Yogurisimo"},
    ),
    (
        ["Gallo", "ARROZ LARGO FINO", "1 kg"],
        [60, 28, 14],
        0.93,
        {"nombre_producto": "ARROZ LARGO FINO", "marca": "Gallo"},
    ),
    (
        ["Natura", "ACEITE DE GIRASOL", "900 ml"],
        [60, 28, 14],
        0.96,
        {"nombre_producto": "ACEITE DE GIRASOL", "marca": "Natura"},
    ),
    (
        ["Matarazzo", "FIDEOS TIRABUZON", "500 g"],
        [60, 28, 14],
        0.94,
        {"nombre_producto": "FIDEOS TIRABUZON", "marca": "Matarazzo"},
    ),
    (
        ["La Virginia", "CAFE MOLIDO", "250 g"],
        [60, 28, 14],
        0.95,
        {"nombre_producto": "CAFE MOLIDO", "marca": "La Virginia"},
    ),
]

# Sellos de vencimiento: los formatos que aparecen impresos sobre el envase.
EXPIRY_CASES = [
    (["VTO 15/08/2027"], 0.96, "2027-08-15"),
    (["CONS. PREF. 01/12/2026"], 0.95, "2026-12-01"),
    (["VENCE 20/03/2028"], 0.94, "2028-03-20"),
    (["Consumir preferentemente antes de 10/11/2027"], 0.95, "2027-11-10"),
    (["VTO 15-ago-27"], 0.93, "2027-08-15"),
    (["VTO.: 05/01/27 ENV.: 05/01/26"], 0.95, "2027-01-05"),
]

RECEIPT_CASES = [
    (
        [
            "SUPERMERCADO LA ESQUINA",
            "CUIT 30-12345678-9",
            "FECHA 20/07/2026",
            "2 x Leche Entera 1L 2.400,00",
            "1 x Pan Lactal 1.850,50",
            "3 Yogur Frutilla 990,00",
            "TOTAL 5.240,50",
        ],
        0.96,
        {
            "fecha_compra": "2026-07-20",
            "total": 5240.50,
            "items": {"Leche Entera 1L": 2, "Pan Lactal": 1, "Yogur Frutilla": 3},
        },
    ),
    (
        [
            "ALMACEN DON JOSE",
            "FECHA 18/07/2026",
            "1 Aceite Girasol 900ml 3.100,00",
            "2 x Arroz Largo Fino 2.500,00",
            "TOTAL 5.600,00",
        ],
        0.94,
        {
            "fecha_compra": "2026-07-18",
            "total": 5600.00,
            "items": {"Aceite Girasol 900ml": 1, "Arroz Largo Fino": 2},
        },
    ),
    (
        [
            "KIOSCO CENTRAL",
            "FECHA 15/07/2026",
            "Alfajor Triple x 4 1.200,00",
            "Gaseosa 2L x 2 3.000,00",
            "TOTAL 4.200,00",
        ],
        0.95,
        {
            "fecha_compra": "2026-07-15",
            "total": 4200.00,
            "items": {"Alfajor Triple": 4, "Gaseosa 2L": 2},
        },
    ),
]


def _label_result(case):
    lines, heights, confidence, _ = case
    output = make_output(lines, confidence=confidence, heights=heights)
    return ocr_service.build_result(output, output.text, DocType.PRODUCT_LABEL, TODAY)


def _expiry_result(case):
    lines, confidence, _ = case
    output = make_output(lines, confidence=confidence)
    return ocr_service.build_result(output, output.text, DocType.EXPIRY_DATE, TODAY)


def _receipt_result(case):
    lines, confidence, _ = case
    output = make_output(lines, confidence=confidence)
    return ocr_service.build_result(output, output.text, DocType.RECEIPT, TODAY)


# ---------------------------------------------------------------------------
# Criterio 1 — precisión ≥ 85% en campos clave
# ---------------------------------------------------------------------------

def test_key_field_accuracy_meets_85_percent():
    correct = 0
    total = 0

    for case in LABEL_CASES:
        result = _label_result(case)
        expected = case[3]

        for name, value in expected.items():
            total += 1
            correct += result.fields[name].value == value

    for case in EXPIRY_CASES:
        total += 1
        correct += _expiry_result(case).fields["fecha_vencimiento"].value == case[2]

    for case in RECEIPT_CASES:
        result = _receipt_result(case)
        expected = case[2]

        for name in ("fecha_compra", "total"):
            total += 1
            correct += result.fields[name].value == expected[name]

        extracted = {i.nombre_producto.value: i.cantidad.value for i in result.items}
        for name, quantity in expected["items"].items():
            total += 1
            correct += extracted.get(name) == quantity

    accuracy = correct / total

    assert accuracy >= 0.85, f"Precisión de campos clave {accuracy:.1%} < 85% ({correct}/{total})"


# ---------------------------------------------------------------------------
# Criterio 2 — precisión global por documento ≥ 70%
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("case", LABEL_CASES, ids=lambda c: c[0][0][:24])
def test_label_document_confidence_meets_threshold(case):
    result = _label_result(case)

    assert result.document_confidence >= settings.DOCUMENT_CONFIDENCE_THRESHOLD


@pytest.mark.parametrize("case", EXPIRY_CASES, ids=lambda c: c[0][0][:24])
def test_expiry_document_confidence_meets_threshold(case):
    result = _expiry_result(case)

    assert result.document_confidence >= settings.DOCUMENT_CONFIDENCE_THRESHOLD


@pytest.mark.parametrize("case", RECEIPT_CASES, ids=lambda c: c[0][0][:24])
def test_receipt_document_confidence_meets_threshold(case):
    result = _receipt_result(case)

    assert result.document_confidence >= settings.DOCUMENT_CONFIDENCE_THRESHOLD


def test_clean_documents_do_not_require_manual_review():
    """Con lectura nítida y campos completos, no debe pedirse corrección manual."""
    flagged = [case[3]["nombre_producto"] for case in LABEL_CASES if _label_result(case).needs_review]
    flagged += [case[2] for case in EXPIRY_CASES if _expiry_result(case).needs_review]

    assert flagged == []


# ---------------------------------------------------------------------------
# Contraparte: los documentos malos SÍ deben pedir revisión
# ---------------------------------------------------------------------------

def test_blurry_documents_are_flagged_for_review():
    """Confianza del motor baja ⇒ needs_review, que dispara la corrección asistida."""
    for lines, heights, _, _ in LABEL_CASES:
        output = make_output(lines, confidence=0.40, heights=heights)
        result = ocr_service.build_result(output, output.text, DocType.PRODUCT_LABEL, TODAY)

        assert result.needs_review is True
        assert result.document_confidence < settings.DOCUMENT_CONFIDENCE_THRESHOLD


def test_illegible_label_is_flagged_for_review():
    output = make_output(["####", "??"], confidence=0.20, heights=[20, 20])
    result = ocr_service.build_result(output, output.text, DocType.PRODUCT_LABEL, TODAY)

    assert result.needs_review is True
    assert result.fields["nombre_producto"].value is None


def test_unreadable_stamp_is_flagged_for_review():
    """Sin fecha legible el servicio no inventa un valor: lo deja en None."""
    output = make_output(["L: 2601 B 09:15"], confidence=0.90)
    result = ocr_service.build_result(output, output.text, DocType.EXPIRY_DATE, TODAY)

    assert result.needs_review is True
    assert result.fields["fecha_vencimiento"].value is None


# ---------------------------------------------------------------------------
# Códigos de barras — el mismo gate, medido aparte
# ---------------------------------------------------------------------------
#
# Se mide en su propio bloque y **no** se promedia con los casos de OCR, por la
# misma razón por la que el benchmark separa los campos clave de los accesorios:
# mezclar dos números esconde el que hay que mirar. Un código de barras trae
# dígito verificador y acierta el 100 % cuando decodifica, así que sumarlo al
# promedio de OCR subiría esa cifra sin que ningún motor haya mejorado, y una
# regresión en las etiquetas quedaría tapada.
#
# Estos casos son además la parte más exigente de la suite: son los únicos que
# **no usan un doble del reconocedor**. La imagen se genera con la codificación
# real y la decodifica ZBar de verdad.

BARCODE_CASES = [
    # (descripción, cuerpo del código, tipo, ancho en la foto, código esperado)
    ("EAN-13 argentino", "779123456789", "ean13", 420, "7791234567898"),
    ("EAN-13 argentino chico", "779555000123", "ean13", 180, "7795550001234"),
    ("EAN-13 importado de Brasil", "789123456789", "ean13", 420, "7891234567895"),
    ("EAN-13 importado de Chile", "780123456789", "ean13", 420, "7801234567894"),
    ("UPC-A de un encargo", "01234567890", "upca", 420, "0012345678905"),
    ("EAN-8 de envase chico", "7791234", "ean8", 300, "77912343"),
    ("ITF-14 de caja mayorista", "17791234567895", "itf", 520, "7791234567898"),
]


def _barcode_result(case):
    _, body, kind, width, _ = case
    return ocr_service.process_sync(
        barcode_photo(body, kind=kind, width=width), DocType.BARCODE, today=TODAY
    )


def test_barcode_key_field_accuracy_meets_85_percent():
    """Criterio 1 sobre el campo clave del tipo `barcode`: `codigo`."""
    correct = sum(
        _barcode_result(case).fields["codigo"].value == case[4] for case in BARCODE_CASES
    )
    total = len(BARCODE_CASES)
    accuracy = correct / total

    assert accuracy >= 0.85, f"Precisión de `codigo` {accuracy:.1%} < 85% ({correct}/{total})"


@pytest.mark.parametrize("case", BARCODE_CASES, ids=lambda c: c[0][:26])
def test_barcode_document_confidence_meets_threshold(case):
    """Criterio 2: confianza global por documento ≥ 70%."""
    assert _barcode_result(case).document_confidence >= settings.DOCUMENT_CONFIDENCE_THRESHOLD


@pytest.mark.parametrize("case", BARCODE_CASES, ids=lambda c: c[0][:26])
def test_readable_barcodes_do_not_require_manual_review(case):
    """Un código legible carga el producto sin que el usuario confirme nada."""
    assert _barcode_result(case).needs_review is False


def test_unreadable_barcode_is_flagged_for_review():
    """Contraparte: por debajo del umbral de resolución no se inventa un número.
    Un código inventado es peor que ninguno, porque el dígito verificador le da
    al usuario una apariencia de certeza que no tiene cómo auditar."""
    result = ocr_service.process_sync(
        barcode_photo("779123456789", width=120), DocType.BARCODE, today=TODAY
    )

    assert result.fields["codigo"].value is None
    assert result.needs_review is True
    assert result.document_confidence < settings.DOCUMENT_CONFIDENCE_THRESHOLD


def test_store_internal_barcode_is_flagged_for_review():
    """El código de la balanza del súper decodifica perfecto y aun así va a
    revisión: no identifica un producto fuera de ese comercio."""
    result = ocr_service.process_sync(
        barcode_photo("200123450000"), DocType.BARCODE, today=TODAY
    )

    assert result.fields["codigo"].value == "2001234500005"
    assert result.needs_review is True
