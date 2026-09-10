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


def test_implausible_expiry_year_is_rejected():
    """El ruido leído como dígitos produce fechas absurdas: mejor None que un dato falso."""
    assert dates.find_expiry_date("VTO 31/03/4006", TODAY) is None
    assert dates.find_expiry_date("VTO 01/01/2010", TODAY) is None  # demasiado vieja


def test_plausible_expiry_window_is_accepted():
    assert dates.find_expiry_date("VTO 01/01/2035", TODAY) is not None
    assert dates.find_expiry_date("VTO 01/01/2025", TODAY) is not None  # vencida reciente


def test_future_purchase_date_is_rejected():
    """Un ticket no puede ser del futuro: es una lectura errónea."""
    assert dates.find_purchase_date("FECHA 20/07/2030", TODAY) is None


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


# ---------------------------------------------------------------------------
# Mes/año con año de dos dígitos — `VTO 08/28`
# ---------------------------------------------------------------------------
#
# Es el formato más frecuente en el envase argentino y no estaba cubierto: los
# patrones sólo aceptaban mes/año con año de cuatro dígitos. Se acepta **sólo**
# cuando lo anuncia una palabra de vencimiento, porque `08/28` suelto es
# indistinguible de un día y un mes.

@pytest.mark.parametrize(
    "texto, esperado",
    [
        ("VTO 08/28", "2028-08-31"),
        # Pegado, que es como sale del motor sobre un sello real.
        ("1L6217 07:45-VTO08/28", "2028-08-31"),
        ("CONS.PREF. 12/27", "2027-12-31"),
        ("VENCE 03/29", "2029-03-31"),
        ("L: 1234 VTO 11/26", "2026-11-30"),
        ("Vto: 01/30", "2030-01-31"),
        # `VIO` es como el motor lee `VTO` con trazo fino: la T y la I son dos
        # palos verticales.
        ("VIO 06/27", "2027-06-30"),
    ],
)
def test_month_year_with_two_digit_year_needs_a_keyword(texto, esperado):
    assert dates.find_expiry_date(texto, TODAY).iso == esperado


@pytest.mark.parametrize(
    "texto",
    [
        # Sin palabra clave no se acepta: sería indistinguible de un día/mes.
        "Precio 08/28",
        "Lote 4523 08/28",
        # Mes inválido: 15 no es un mes, así que esto es un día y un mes.
        "VTO 15/08",
        # Año fuera de la ventana de plausibilidad.
        "VTO 08/99",
    ],
)
def test_month_year_without_a_keyword_is_not_a_date(texto):
    assert dates.find_expiry_date(texto, TODAY) is None


def test_the_keyword_still_counts_as_announced():
    """El tramo reportado es la fecha, no la palabra clave.

    Si `matched_text` incluyera el `VTO`, la posición quedaría antes de él y
    `has_keyword_before` no lo encontraría: la fecha se penalizaría por venir
    suelta cuando en realidad es el único caso en que se exige que no lo esté.
    """
    parsed = dates.find_expiry_date("VTO 08/28", TODAY)

    assert parsed.matched_text == "08/28"
    assert dates.has_keyword_before("VTO 08/28", parsed, dates.EXPIRY_KEYWORDS) is True


def test_production_date_does_not_win_over_expiry_in_two_digit_format():
    """`ELAB 05/26  VTO 05/27` tiene que dar el vencimiento, no la elaboración."""
    assert dates.find_expiry_date("ELAB 05/26 VTO 05/27", TODAY).iso == "2027-05-31"


# ---------------------------------------------------------------------------
# La `V` sola del envase de farmacia, y el separador que tiene que repetirse
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "texto,esperado",
    [
        # El sello de la caja de farmacia: `L` es el lote y `V` el vencimiento.
        ("L 1279 V 03/28", "2028-03-31"),
        ("V 12 2027 B", "2027-12-31"),
        # Con la palabra entera sigue ganando la palabra entera.
        ("Vence: 11 2027", "2027-11-30"),
        ("Vencimiento: 09 2027", "2027-09-30"),
    ],
)
def test_la_v_sola_anuncia_el_vencimiento(texto, esperado):
    """Medido sobre las cajas de farmacia del conjunto: `V` es «vence».

    Sólo cuenta para el patrón de mes/año, que ya exige el anuncio para existir.
    Agregarla a `EXPIRY_KEYWORDS` no serviría —esa búsqueda es por subcadena y
    cualquier `v` cercana contaría como anuncio— y por eso el patrón marca la
    fecha como anunciada él mismo.
    """
    assert dates.find_expiry_date(texto, TODAY).iso == esperado


def test_la_v_anunciada_no_se_penaliza_como_fecha_suelta():
    parsed = dates.find_expiry_date("L 1279 V 03/28", TODAY)

    assert parsed.keyed is True
    assert dates.announced_as_expiry("L 1279 V 03/28", parsed) is True


@pytest.mark.parametrize(
    "texto,esperado",
    [
        ("15/08/2026", "2026-08-15"),
        ("15-08-26", "2026-08-15"),
        ("15.08.2026", "2026-08-15"),
        ("15 08 2026", "2026-08-15"),
    ],
)
def test_los_separadores_habituales_siguen_valiendo(texto, esperado):
    assert dates.find_expiry_date(texto, TODAY).iso == esperado


def test_un_digito_suelto_no_se_convierte_en_el_dia():
    """`1 09/27` es un resto del lote pegado a un mes/año, no el 1 de septiembre.

    Nadie escribe una fecha mezclando separadores, así que el patrón de día/mes/
    año exige que los dos sean el mismo. Sin anuncio de vencimiento, `09/27`
    queda sin interpretar — que es lo correcto: es indistinguible de un día y un
    mes, y devolver una fecha falsa que el usuario podría aceptar es peor que no
    devolver nada.
    """
    assert dates.find_expiry_date("1 09/27", TODAY) is None
    # Con el anuncio sí se resuelve, y como mes/año.
    assert dates.find_expiry_date("VTO 09/27", TODAY).iso == "2027-09-30"
