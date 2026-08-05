"""Estrategia de intentos del coordinador.

Es el código que justifica que este servicio exista aparte de los motores:
decidir qué giro, qué filtros y qué segmentación pedir para cada tipo de
documento, cuándo dejar de probar y cuándo vale la pena pagar el motor caro.
"""

from datetime import date

import pytest

from app.core.config import settings
from app.engines.factory import HybridEngine
from app.engines.remote import EngineUnavailable
from app.pipeline import transform
from app.schemas.ocr import DocType
from app.services import ocr_service
from tests.conftest import FakeEngine

TODAY = date(2026, 7, 29)

LABEL = ["La Serenisima", "LECHE ENTERA", "1 litro"]
LABEL_HEIGHTS = [60, 30, 16]
STAMP = ["VTO 15/08/2027"]


# ---------------------------------------------------------------------------
# Planes de intentos
# ---------------------------------------------------------------------------

def test_label_plan_varies_filters_but_not_orientation():
    """Una etiqueta se fotografía derecha: girarla sería gastar intentos."""
    plan = ocr_service.attempt_plan(DocType.PRODUCT_LABEL, supports_preprocessing=True)

    assert {attempt.rotation for attempt in plan} == {0}
    assert len(plan) > 1


def test_expiry_plan_searches_all_four_orientations():
    """El sello se fotografía en el canto del envase y llega girado."""
    plan = ocr_service.attempt_plan(DocType.EXPIRY_DATE, supports_preprocessing=True)

    assert {attempt.rotation for attempt in plan} == {0, 90, 180, 270}


def test_expiry_plan_includes_the_adaptive_threshold():
    """Es la única variante que levanta la matriz de puntos."""
    plan = ocr_service.attempt_plan(DocType.EXPIRY_DATE, supports_preprocessing=True)

    assert any(attempt.transform.binarize == transform.ADAPTIVE for attempt in plan)


def test_label_plan_never_uses_the_adaptive_threshold():
    """Sobre un logo, el umbral adaptativo vacía el interior de los trazos."""
    plan = ocr_service.attempt_plan(DocType.PRODUCT_LABEL, supports_preprocessing=True)

    assert all(attempt.transform.binarize != transform.ADAPTIVE for attempt in plan)


def test_a_self_preparing_engine_gets_one_attempt_per_orientation():
    """A EasyOCR no se le varían filtros que ignora, pero sí se le busca el giro."""
    plan = ocr_service.attempt_plan(DocType.EXPIRY_DATE, supports_preprocessing=False)

    assert len(plan) == 4
    assert all(attempt.transform.binarize == transform.NONE for attempt in plan)


def test_a_self_preparing_engine_gets_a_single_attempt_on_a_label():
    plan = ocr_service.attempt_plan(DocType.PRODUCT_LABEL, supports_preprocessing=False)

    assert len(plan) == 1


def test_the_stamp_asks_for_more_resolution_than_a_label():
    """El sello ocupa una franja diminuta del cuadro: necesita más detalle."""
    label = ocr_service.attempt_plan(DocType.PRODUCT_LABEL, supports_preprocessing=False)[0]
    stamp = ocr_service.attempt_plan(DocType.EXPIRY_DATE, supports_preprocessing=False)[0]

    assert stamp.transform.max_width > label.transform.max_width


def test_rotation_search_can_be_disabled(monkeypatch):
    monkeypatch.setattr(settings, "ROTATION_SEARCH_ENABLED", False)

    plan = ocr_service.attempt_plan(DocType.EXPIRY_DATE, supports_preprocessing=True)

    assert {attempt.rotation for attempt in plan} == {0}


def test_retries_can_be_disabled(monkeypatch):
    monkeypatch.setattr(settings, "RETRY_ENABLED", False)

    assert len(ocr_service.attempt_plan(DocType.PRODUCT_LABEL, supports_preprocessing=True)) == 1


# ---------------------------------------------------------------------------
# Ejecución del plan
# ---------------------------------------------------------------------------

def test_a_good_first_reading_stops_the_plan(fake_engine):
    engine = fake_engine(LABEL, confidence=0.99, heights=LABEL_HEIGHTS)

    result = ocr_service.process_sync(b"imagen", DocType.PRODUCT_LABEL, today=TODAY)

    assert len(engine.calls) == 1
    assert result.attempts == 1


def test_a_poor_reading_exhausts_the_plan(fake_engine):
    engine = fake_engine(["xx"], confidence=0.10)

    ocr_service.process_sync(b"imagen", DocType.PRODUCT_LABEL, today=TODAY)

    assert len(engine.calls) > 1


def test_the_layout_hint_matches_the_document_type(fake_engine, no_retry):
    engine = fake_engine(LABEL, heights=LABEL_HEIGHTS)

    ocr_service.process_sync(b"imagen", DocType.RECEIPT, today=TODAY)

    assert engine.calls[0]["layout"] == "block"


def test_the_result_reports_the_winning_rotation(fake_engine):
    """Saber con qué giro se leyó es lo que hace depurable la búsqueda."""
    engine = fake_engine(STAMP, confidence=0.99)

    result = ocr_service.process_sync(b"imagen", DocType.EXPIRY_DATE, today=TODAY)

    assert result.rotation == engine.calls[0]["rotate"] == 0


def test_the_engine_receives_the_original_bytes(fake_engine, no_retry):
    """Este servicio no decodifica ni transforma la imagen: la reenvía."""
    engine = fake_engine(LABEL, heights=LABEL_HEIGHTS)

    ocr_service.process_sync(b"12345", DocType.PRODUCT_LABEL, today=TODAY)

    assert engine.calls[0]["bytes"] == 5


def test_an_engine_that_fails_every_attempt_propagates_the_error(monkeypatch):
    class DeadEngine:
        name = "muerto"
        supports_preprocessing = True

        def is_available(self):
            return False

        def run(self, *_args, **_kwargs):
            raise EngineUnavailable("el servicio no responde")

    monkeypatch.setattr("app.services.ocr_service.build_engine", lambda *_, **__: DeadEngine())

    with pytest.raises(EngineUnavailable):
        ocr_service.process_sync(b"imagen", DocType.PRODUCT_LABEL, today=TODAY)


# ---------------------------------------------------------------------------
# Orquestación del híbrido
# ---------------------------------------------------------------------------

def hybrid_of(primary: FakeEngine, fallback: FakeEngine, monkeypatch) -> HybridEngine:
    hybrid = HybridEngine(primary, fallback, threshold=settings.HYBRID_FALLBACK_THRESHOLD)
    monkeypatch.setattr("app.services.ocr_service.build_engine", lambda *_, **__: hybrid)
    return hybrid


def test_hybrid_does_not_pay_the_expensive_engine_on_a_clean_reading(monkeypatch):
    primary = FakeEngine(LABEL, confidence=0.99, heights=LABEL_HEIGHTS, name="barato")
    fallback = FakeEngine(LABEL, confidence=0.99, heights=LABEL_HEIGHTS, name="caro")
    hybrid_of(primary, fallback, monkeypatch)

    result = ocr_service.process_sync(b"imagen", DocType.PRODUCT_LABEL, today=TODAY)

    assert fallback.calls == []
    assert result.engine == "barato"


def test_hybrid_calls_the_fallback_when_the_reading_needs_review(monkeypatch):
    """Confianza alta con campo dudoso no alcanza: el criterio es `needs_review`."""
    primary = FakeEngine(["Marca Sola"], confidence=0.99, heights=[60], name="barato")
    fallback = FakeEngine(LABEL, confidence=0.99, heights=LABEL_HEIGHTS, name="caro")
    hybrid_of(primary, fallback, monkeypatch)

    result = ocr_service.process_sync(b"imagen", DocType.PRODUCT_LABEL, today=TODAY)

    assert fallback.calls
    assert result.engine == "caro"


def test_hybrid_reuses_the_orientation_found_by_the_cheap_engine(monkeypatch):
    """El motor caro no repite la búsqueda de giro: la hereda ya resuelta."""
    primary = FakeEngine(STAMP, confidence=0.60, name="barato")
    fallback = FakeEngine(STAMP, confidence=0.99, name="caro")
    hybrid_of(primary, fallback, monkeypatch)

    ocr_service.process_sync(b"imagen", DocType.EXPIRY_DATE, today=TODAY)

    assert len(fallback.calls) == 1


def test_hybrid_lets_the_fallback_search_when_the_primary_read_nothing(monkeypatch):
    """Sin lectura del principal no hay orientación que heredar: se busca de nuevo."""
    primary = FakeEngine(["...."], confidence=0.05, name="barato")
    # También el respaldo lee mal, para que agote su plan y se vea cuántos
    # intentos le concedió el coordinador.
    fallback = FakeEngine(["...."], confidence=0.05, name="caro")
    hybrid_of(primary, fallback, monkeypatch)

    ocr_service.process_sync(b"imagen", DocType.EXPIRY_DATE, today=TODAY)

    assert {call["rotate"] for call in fallback.calls} == {0, 90, 180, 270}


def test_hybrid_counts_the_attempts_of_both_engines(monkeypatch):
    primary = FakeEngine(["xx"], confidence=0.10, name="barato")
    fallback = FakeEngine(LABEL, confidence=0.99, heights=LABEL_HEIGHTS, name="caro")
    hybrid_of(primary, fallback, monkeypatch)

    result = ocr_service.process_sync(b"imagen", DocType.PRODUCT_LABEL, today=TODAY)

    assert result.attempts == len(primary.calls) + len(fallback.calls)
