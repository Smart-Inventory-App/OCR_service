"""Normalización de fechas: formatos habituales en etiquetas y tickets AR."""

from datetime import date

import pytest

from app.parsers import dates

TODAY = date(2026, 7, 29)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("VTO 15/08/2027", "2027-08-15"),
        ("VTO: 15-08-27", "2027-08-15"),
        ("Vence 15.08.2027", "2027-08-15"),
        ("VENCIMIENTO 2027-08-15", "2027-08-15"),
        ("VTO 15 AGO 2027", "2027-08-15"),
        ("VTO 15-ago-27", "2027-08-15"),
        ("CONS. PREF. ANTES DE 15/08/2027", "2027-08-15"),
        ("Consumir preferentemente antes de 01/12/2027", "2027-12-01"),
        ("BEST BEFORE 15/08/2027", "2027-08-15"),
    ],
)
def test_expiry_formats(text, expected):
    parsed = dates.find_expiry_date(text, TODAY)

    assert parsed is not None
    assert parsed.iso == expected


def test_month_year_only_uses_last_day_of_month():
    parsed = dates.find_expiry_date("VTO 02/2028", TODAY)

    assert parsed is not None
    assert parsed.iso == "2028-02-29"  # 2028 es bisiesto
    assert parsed.day_precision is False


def test_month_name_and_year_only():
    parsed = dates.find_expiry_date("VTO AGO 2027", TODAY)

    assert parsed is not None
    assert parsed.iso == "2027-08-31"
    assert parsed.day_precision is False


def test_keyword_wins_over_position():
    """Aunque la elaboración aparezca primero, gana la fecha marcada con VTO."""
    text = "ELAB 10/01/2026\nVTO 10/01/2028"
    parsed = dates.find_expiry_date(text, TODAY)

    assert parsed is not None
    assert parsed.iso == "2028-01-10"


def test_without_keyword_prefers_future_date():
    text = "10/01/2025 y 10/01/2028"
    parsed = dates.find_expiry_date(text, TODAY)

    assert parsed is not None
    assert parsed.iso == "2028-01-10"


def test_invalid_dates_are_ignored():
    assert dates.find_expiry_date("VTO 32/13/2027", TODAY) is None
    assert dates.find_expiry_date("sin fecha alguna", TODAY) is None


def test_two_digit_year_expands_to_current_century():
    parsed = dates.find_expiry_date("VTO 01/01/30", TODAY)

    assert parsed is not None
    assert parsed.value.year == 2030


def test_iso_is_not_misread_as_day_month():
    parsed = dates.find_expiry_date("VTO 2027-11-03", TODAY)

    assert parsed is not None
    assert parsed.iso == "2027-11-03"


def test_purchase_date_prefers_keyword():
    text = "SUPERMERCADO\nFECHA 20/07/2026\nTOTAL 15400,00"
    parsed = dates.find_purchase_date(text, TODAY)

    assert parsed is not None
    assert parsed.iso == "2026-07-20"


def test_purchase_date_prefers_past_when_no_keyword():
    text = "10/07/2026 ... 10/07/2030"
    parsed = dates.find_purchase_date(text, TODAY)

    assert parsed is not None
    assert parsed.iso == "2026-07-10"


def test_find_dates_returns_in_reading_order():
    found = dates.find_dates("01/02/2027 luego 05/06/2028")

    assert [d.iso for d in found] == ["2027-02-01", "2028-06-05"]


def test_overlapping_patterns_do_not_duplicate():
    """`15/08/2027` no debe producir además un match de `08/2027`."""
    found = dates.find_dates("15/08/2027")

    assert len(found) == 1
    assert found[0].iso == "2027-08-15"
