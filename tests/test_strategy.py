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

def test_label_plan_tries_filters_first_and_rotations_last():
    """La etiqueta prueba filtros y, sólo después, orientaciones.

    Se suponía que una etiqueta se fotografía siempre derecha. El conjunto de
    prueba mostró que no: las mismas 16 fotos giradas un cuarto de vuelta daban
    un nombre distinto en los 16 casos. Ahora se giran, pero al final del plan,
    así que la foto derecha —el caso normal— corta antes de pagarlos.
    """
    plan = ocr_service.attempt_plan(DocType.PRODUCT_LABEL, supports_preprocessing=True)

    rotations = [attempt.rotation for attempt in plan]

    assert rotations[0] == 0
    assert set(rotations) == {0, 90, 180, 270}
    # Ningún giro antes de haber agotado las variantes de filtros a 0°.
    assert rotations.index(90) > max(
        index for index, degrees in enumerate(rotations) if degrees == 0
    )


def test_label_rotations_can_be_switched_off():
    from app.core.config import settings

    original = settings.LABEL_ROTATION_SEARCH_ENABLED
    settings.LABEL_ROTATION_SEARCH_ENABLED = False
    try:
        plan = ocr_service.attempt_plan(DocType.PRODUCT_LABEL, supports_preprocessing=True)
    finally:
        settings.LABEL_ROTATION_SEARCH_ENABLED = original

    assert {attempt.rotation for attempt in plan} == {0}


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

    # Los recortes de banda van primero y también le corresponden: el recorte es
    # geometría, no filtrado, y un motor que prepara su entrada tampoco puede
    # inventar el sello si le llega de 90 px dentro de una foto de 4000.
    completos = [a for a in plan if a.transform.crop_band < 0]

    assert len(completos) == 4
    assert {a.rotation for a in completos} == {0, 90, 180, 270}
    assert {a.transform.binarize for a in completos} == {"none"}
    assert all(attempt.transform.binarize == transform.NONE for attempt in plan)


def test_a_self_preparing_engine_gets_one_attempt_per_orientation_on_a_label():
    """Al motor que prepara su propia entrada no se le varían los filtros.

    Lo que sí se le sigue variando es la orientación, que es geometría y no
    filtrado: ningún motor lee texto girado 90°, prepare o no su entrada.
    """
    plan = ocr_service.attempt_plan(DocType.PRODUCT_LABEL, supports_preprocessing=False)

    assert [attempt.rotation for attempt in plan] == [0, 90, 270, 180]
    assert {attempt.transform.binarize for attempt in plan} == {"none"}


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
# De dónde sale `supports_preprocessing`
# ---------------------------------------------------------------------------

def test_the_capability_is_taken_from_the_engine_service():
    """Quién sabe si le sirve el preprocesamiento es el motor, no el coordinador.

    El valor con el que se construye el `RemoteEngine` es sólo un default: si el
    servicio declara otra cosa en `/engine/info`, manda el servicio.
    """
    class Declaring:
        name = "declarante"
        supports_preprocessing = True

        def refresh_capabilities(self):
            self.supports_preprocessing = False

    assert ocr_service._supports_preprocessing(Declaring()) is False


def test_an_engine_that_cannot_declare_keeps_its_default():
    """Un doble de tests, o el híbrido, no tienen `/engine/info` que consultar."""
    assert ocr_service._supports_preprocessing(FakeEngine([], supports_preprocessing=False)) is False
    assert ocr_service._supports_preprocessing(FakeEngine([])) is True


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


def test_hybrid_caps_the_fallback_when_the_primary_read_nothing(monkeypatch):
    """Sin lectura del principal no hay orientación que heredar, pero tampoco se
    le regalan al motor caro las cuatro.

    Antes se le concedía el plan entero —una inferencia por orientación— y es el
    camino que agota la memoria: sobre los sellos del conjunto de prueba el
    contenedor de PaddleOCR acumuló más de 90 reinicios por OOM y se perdieron
    19 mediciones. Se acota a las primeras orientaciones del orden, que son la
    foto tal como vino y el cuarto de vuelta con que llega el sello.
    """
    from app.core.config import settings

    primary = FakeEngine(["...."], confidence=0.05, name="barato")
    # También el respaldo lee mal, para que agote su plan y se vea cuántos
    # intentos le concedió el coordinador.
    fallback = FakeEngine(["...."], confidence=0.05, name="caro")
    hybrid_of(primary, fallback, monkeypatch)

    ocr_service.process_sync(b"imagen", DocType.EXPIRY_DATE, today=TODAY)

    giros = {call["rotate"] for call in fallback.calls}

    assert giros == set(ocr_service.ROTATIONS[: settings.HYBRID_FALLBACK_MAX_ROTATIONS])


def test_hybrid_counts_the_attempts_of_both_engines(monkeypatch):
    primary = FakeEngine(["xx"], confidence=0.10, name="barato")
    fallback = FakeEngine(LABEL, confidence=0.99, heights=LABEL_HEIGHTS, name="caro")
    hybrid_of(primary, fallback, monkeypatch)

    result = ocr_service.process_sync(b"imagen", DocType.PRODUCT_LABEL, today=TODAY)

    assert result.attempts == len(primary.calls) + len(fallback.calls)


def test_the_fallback_engine_is_capped_when_the_primary_read_nothing():
    """El motor caro no recibe una corrida por orientación.

    Cuando el principal no reconoce nada no hay orientación conocida que
    pasarle al respaldo, y sin tope ese caso —el peor— le pide al motor más
    caro una inferencia por cada giro. Medido sobre los sellos del conjunto de
    prueba, es el camino que agota la memoria del contenedor.
    """
    from app.core.config import settings
    from app.engines.factory import HybridEngine
    from tests.conftest import FakeEngine

    primary = FakeEngine(["nada legible"], confidence=0.0)
    fallback = FakeEngine(["VTO 15/08/2027"], confidence=0.99,
                          supports_preprocessing=False)
    hybrid = HybridEngine(primary=primary, fallback=fallback, threshold=0.7)

    ocr_service._run_hybrid(hybrid, b"x", DocType.EXPIRY_DATE, "spa", None)

    assert len(fallback.calls) <= settings.HYBRID_FALLBACK_MAX_ROTATIONS


def test_the_fallback_gets_a_single_run_when_the_primary_found_the_orientation():
    """Si el principal leyó algo, su giro es información y basta una corrida."""
    from app.engines.factory import HybridEngine
    from tests.conftest import FakeEngine

    primary = FakeEngine(["VTO 15/08/2027"], confidence=0.60)
    fallback = FakeEngine(["VTO 15/08/2027"], confidence=0.99,
                          supports_preprocessing=False)
    hybrid = HybridEngine(primary=primary, fallback=fallback, threshold=0.7)

    ocr_service._run_hybrid(hybrid, b"x", DocType.EXPIRY_DATE, "spa", None)

    assert len(fallback.calls) == 1


# ---------------------------------------------------------------------------
# Recorte a la banda del sello
# ---------------------------------------------------------------------------

def test_only_the_self_preparing_engine_gets_cropped_bands():
    """El recorte se le ofrece al motor de aprendizaje profundo, no al otro.

    Está medido sobre los 94 sellos del conjunto: los intentos con recorte del
    motor que depende de la cadena de filtros acertaron **0**, y los del motor
    que prepara su propia entrada resolvieron 13 de los 37 vencimientos
    correctos. Aislar el sello ayuda a quien puede leerlo; al que no, le seguían
    costando seis intentos por documento que nunca ganaban.
    """
    from app.core.config import settings

    con_filtros = ocr_service.attempt_plan(DocType.EXPIRY_DATE, supports_preprocessing=True)
    sin_filtros = ocr_service.attempt_plan(DocType.EXPIRY_DATE, supports_preprocessing=False)

    assert all(a.transform.crop_band < 0 for a in con_filtros)

    recortes = [i for i, a in enumerate(sin_filtros) if a.transform.crop_band >= 0]
    completos = [i for i, a in enumerate(sin_filtros) if a.transform.crop_band < 0]

    assert recortes, "el motor pesado tiene que probar bandas recortadas"
    # Y primero, porque una banda de 900x90 cuesta una fracción del envase entero.
    assert max(recortes) < min(completos)
    # Varias bandas por orientación: el motor no puede saber cuál es el sello.
    assert len({sin_filtros[i].transform.crop_band for i in recortes}) == settings.EXPIRY_CROP_BANDS


def test_only_the_expiry_plan_crops():
    """Recortar una etiqueta o un ticket sería tirar el resto del documento."""
    for doc_type in (DocType.PRODUCT_LABEL, DocType.RECEIPT):
        plan = ocr_service.attempt_plan(doc_type, supports_preprocessing=True)

        assert all(a.transform.crop_band < 0 for a in plan), doc_type


def test_cropping_can_be_switched_off():
    from app.core.config import settings

    original = settings.EXPIRY_CROP_BANDS
    settings.EXPIRY_CROP_BANDS = 0
    try:
        plan = ocr_service.attempt_plan(DocType.EXPIRY_DATE, supports_preprocessing=False)
    finally:
        settings.EXPIRY_CROP_BANDS = original

    assert all(a.transform.crop_band < 0 for a in plan)


def test_the_crop_travels_in_the_engine_payload():
    """El motor tiene que recibir qué banda recortar; si no, no recorta nada."""
    from app.pipeline.transform import Transform

    assert Transform().cropped(2).as_payload()["crop_band"] == 2
    assert Transform().as_payload()["crop_band"] == -1


def test_the_receipt_plan_also_searches_orientation():
    """El ticket tampoco viene siempre derecho.

    `dia_2.jpg` es el mismo comprobante que `dia_1.jpg` sacado al revés, y era
    el único de 73 que fallaba: salía con los renglones invertidos y `SUBTOTAL`
    leído `SUBTAL`. Como en la etiqueta, los giros van al final: el ticket
    derecho corta antes y no los paga.
    """
    plan = ocr_service.attempt_plan(DocType.RECEIPT, supports_preprocessing=True)
    rotaciones = [a.rotation for a in plan]

    assert rotaciones[0] == 0
    assert set(rotaciones) == {0, 90, 180, 270}
    assert rotaciones.index(90) > max(
        i for i, deg in enumerate(rotaciones) if deg == 0
    )


def test_receipt_rotations_can_be_switched_off():
    from app.core.config import settings

    original = settings.RECEIPT_ROTATION_SEARCH_ENABLED
    settings.RECEIPT_ROTATION_SEARCH_ENABLED = False
    try:
        plan = ocr_service.attempt_plan(DocType.RECEIPT, supports_preprocessing=True)
    finally:
        settings.RECEIPT_ROTATION_SEARCH_ENABLED = original

    assert {a.rotation for a in plan} == {0}


def test_the_primary_orientation_is_inherited_only_when_its_reading_is_credible(monkeypatch):
    """Una lectura mala no es una pista de orientación.

    Antes alcanzaba con que la confianza fuera mayor que cero. Sobre un ticket
    fotografiado al revés, el motor liviano devuelve basura a 0° que igual arroja
    una confianza chica pero positiva; el coordinador daba ese giro por bueno y
    le concedía al respaldo un único intento a 0°. El motor pesado lee ese mismo
    ticket a 180°, pero nunca llegaba a probarlo.
    """
    primary = FakeEngine(["..", "??"], confidence=0.20, name="barato")
    fallback = FakeEngine(["..", "??"], confidence=0.20, name="caro",
                          supports_preprocessing=False)
    hybrid_of(primary, fallback, monkeypatch)

    ocr_service.process_sync(b"imagen", DocType.RECEIPT, today=TODAY)

    # El respaldo buscó orientación en vez de heredar una que no significaba nada.
    assert len({call["rotate"] for call in fallback.calls}) > 1


def test_a_credible_primary_reading_still_saves_the_fallback_the_search(monkeypatch):
    """La contraparte: cuando el principal sí leyó, su giro se aprovecha."""
    primary = FakeEngine(
        ["La Serenisima", "LECHE ENTERA", "1 litro"], confidence=0.99,
        heights=[60, 30, 16], name="barato",
    )
    fallback = FakeEngine(["LECHE ENTERA"], confidence=0.99, name="caro",
                          supports_preprocessing=False)
    hybrid_of(primary, fallback, monkeypatch)

    ocr_service.process_sync(b"imagen", DocType.PRODUCT_LABEL, today=TODAY)

    # Con una lectura buena el principal ni siquiera necesita el respaldo.
    assert len(fallback.calls) <= 1
