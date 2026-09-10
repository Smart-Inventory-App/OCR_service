"""Cliente de los servicios de motor, factory y modo híbrido.

Los servicios de motor no se levantan: se doblan sus respuestas HTTP con
`respx`. Lo que se verifica acá es el contrato que este servicio habla con
ellos, no que Tesseract sepa leer — eso es responsabilidad de su repositorio.
"""

import json

import httpx
import pytest
import respx

from app.core.config import settings
from app.engines import factory
from app.engines.base import OCREngine, OCRWord, RawOCROutput
from app.engines.factory import HybridEngine
from app.engines.remote import EngineUnavailable, RemoteEngine
from app.pipeline.transform import Transform
from tests.conftest import FakeEngine, make_output

IMAGE = b"\xff\xd8\xff\xe0 binario de una imagen"

TESSERACT_URL = "http://engine-tesseract:8005"


@pytest.fixture(autouse=True)
def clear_engine_cache():
    factory.reset_cache()
    yield
    factory.reset_cache()


def engine_response(words: list[dict], engine: str = "tesseract") -> dict:
    return {
        "engine": engine,
        "text": " ".join(word["text"] for word in words),
        "words": words,
        "elapsed_ms": 120,
        "image_width": 1600,
        "image_height": 900,
        "lang": "spa",
    }


WORD = {
    "text": "LECHE", "confidence": 0.93,
    "left": 10, "top": 10, "width": 90, "height": 30, "line_num": 0,
}


def remote() -> RemoteEngine:
    return RemoteEngine("tesseract", TESSERACT_URL, supports_preprocessing=True)


# ---------------------------------------------------------------------------
# RawOCROutput
# ---------------------------------------------------------------------------

def test_mean_confidence_is_weighted_by_word_length():
    output = RawOCROutput(
        text="",
        words=[
            OCRWord("abcdefghij", 1.0, 0, 0, 10, 10),  # 10 caracteres
            OCRWord("x", 0.0, 0, 0, 10, 10),           # 1 carácter
        ],
    )

    assert output.mean_confidence == pytest.approx(10 / 11)


def test_mean_confidence_without_words_is_zero():
    assert RawOCROutput(text="").mean_confidence == 0.0


def test_lines_groups_words_and_preserves_reading_order():
    output = make_output(["Leche Entera", "VTO 15/08/2027"])

    assert [" ".join(w.text for w in line) for line in output.lines()] == [
        "Leche Entera",
        "VTO 15/08/2027",
    ]


# ---------------------------------------------------------------------------
# RemoteEngine
# ---------------------------------------------------------------------------

@respx.mock
def test_remote_engine_sends_the_transform_as_json():
    """La transformación viaja junto al binario: es lo que decide el coordinador."""
    route = respx.post(f"{TESSERACT_URL}/engine/run").mock(
        return_value=httpx.Response(200, json=engine_response([WORD]))
    )

    remote().run(IMAGE, lang="spa", layout="sparse", transform=Transform(rotate=90, binarize="adaptive"))

    body = route.calls.last.request.content.decode("latin-1")
    assert "sparse" in body

    sent = json.loads(body.split('name="transform"')[1].split("\r\n\r\n")[1].split("\r\n")[0])
    assert sent["rotate"] == 90
    assert sent["binarize"] == "adaptive"


@respx.mock
def test_remote_engine_maps_words_and_confidence():
    respx.post(f"{TESSERACT_URL}/engine/run").mock(
        return_value=httpx.Response(200, json=engine_response([WORD]))
    )

    output = remote().run(IMAGE)

    assert output.engine == "tesseract"
    assert output.words[0].text == "LECHE"
    assert output.words[0].confidence == pytest.approx(0.93)
    assert output.elapsed_ms == 120


@respx.mock
def test_remote_engine_rebuilds_text_when_the_service_omits_it():
    body = engine_response([WORD])
    body["text"] = ""
    respx.post(f"{TESSERACT_URL}/engine/run").mock(return_value=httpx.Response(200, json=body))

    assert remote().run(IMAGE).text == "LECHE"


@respx.mock
def test_remote_engine_raises_when_the_service_errors():
    respx.post(f"{TESSERACT_URL}/engine/run").mock(
        return_value=httpx.Response(500, json={"detail": "el motor falló"})
    )

    with pytest.raises(EngineUnavailable, match="500"):
        remote().run(IMAGE)


@respx.mock
def test_remote_engine_raises_when_the_service_is_down():
    respx.post(f"{TESSERACT_URL}/engine/run").mock(side_effect=httpx.ConnectError("refused"))

    with pytest.raises(EngineUnavailable, match="No se pudo contactar"):
        remote().run(IMAGE)


@respx.mock
def test_a_replicated_engine_spreads_the_load_between_its_replicas():
    """Turno redondo: es lo que ensancha el cuello de botella del módulo.

    Medido, PaddleOCR con una sola instancia se clava en ~650 % de CPU y deja el
    techo del sistema en 2,2 documentos pesados por minuto, mientras el resto del
    módulo mira.
    """
    una = respx.post("http://réplica-a:8007/engine/run").mock(
        return_value=httpx.Response(200, json={"engine": "paddleocr", "words": []})
    )
    otra = respx.post("http://réplica-b:8007/engine/run").mock(
        return_value=httpx.Response(200, json={"engine": "paddleocr", "words": []})
    )

    engine = RemoteEngine(
        "paddleocr", "http://réplica-a:8007,http://réplica-b:8007",
        supports_preprocessing=False,
    )
    for _ in range(4):
        engine.run(IMAGE)

    assert una.call_count == 2
    assert otra.call_count == 2


@respx.mock
def test_a_replica_that_is_down_falls_over_to_the_other():
    """Tener dos sirve para repartir carga y para sobrevivir a que una se caiga."""
    respx.post("http://caida:8007/engine/run").mock(side_effect=httpx.ConnectError("refused"))
    viva = respx.post("http://viva:8007/engine/run").mock(
        return_value=httpx.Response(200, json={"engine": "paddleocr", "words": []})
    )

    engine = RemoteEngine(
        "paddleocr", "http://caida:8007,http://viva:8007", supports_preprocessing=False
    )

    assert engine.run(IMAGE).engine == "paddleocr"
    assert viva.call_count == 1


@respx.mock
def test_an_application_error_is_not_retried_on_the_other_replica():
    """Una imagen inválida va a dar lo mismo en la otra: reintentar sería gastar."""
    primera = respx.post("http://a:8007/engine/run").mock(
        return_value=httpx.Response(400, json={"detail": "no es una imagen"})
    )
    segunda = respx.post("http://b:8007/engine/run").mock(
        return_value=httpx.Response(200, json={"engine": "paddleocr", "words": []})
    )

    engine = RemoteEngine(
        "paddleocr", "http://a:8007,http://b:8007", supports_preprocessing=False
    )

    with pytest.raises(EngineUnavailable, match="400"):
        engine.run(IMAGE)

    assert primera.call_count == 1
    assert segunda.call_count == 0


@respx.mock
def test_a_replicated_engine_is_available_if_any_replica_answers():
    respx.get("http://caida:8007/engine/readyz").mock(side_effect=httpx.ConnectError("refused"))
    respx.get("http://viva:8007/engine/readyz").mock(
        return_value=httpx.Response(200, json={"status": "ok", "engine": "paddleocr"})
    )

    engine = RemoteEngine(
        "paddleocr", "http://caida:8007,http://viva:8007", supports_preprocessing=False
    )

    assert engine.is_available() is True


@respx.mock
def test_remote_engine_is_available_only_when_ready():
    respx.get(f"{TESSERACT_URL}/engine/readyz").mock(
        return_value=httpx.Response(200, json={"status": "ok", "engine": "tesseract"})
    )

    assert remote().is_available()


@respx.mock
def test_remote_engine_degraded_still_counts_as_available():
    """Sin el paquete de idioma el motor cae a inglés: sirve, con aviso."""
    respx.get(f"{TESSERACT_URL}/engine/readyz").mock(
        return_value=httpx.Response(
            200, json={"status": "degraded", "engine": "tesseract", "detail": "falta spa"}
        )
    )

    assert remote().is_available()


@respx.mock
def test_remote_engine_unavailable_when_the_engine_reports_error():
    respx.get(f"{TESSERACT_URL}/engine/readyz").mock(
        return_value=httpx.Response(200, json={"status": "error", "engine": "tesseract"})
    )

    assert not remote().is_available()


@respx.mock
def test_remote_engine_unavailable_when_unreachable():
    respx.get(f"{TESSERACT_URL}/engine/readyz").mock(side_effect=httpx.ConnectError("refused"))

    assert not remote().is_available()


@respx.mock
def test_remote_engine_takes_its_capabilities_from_the_service():
    """Quién sabe si le sirve el preprocesamiento es el motor, no el coordinador."""
    respx.get(f"{TESSERACT_URL}/engine/info").mock(
        return_value=httpx.Response(
            200,
            json={
                "engine": "tesseract", "version": "5.5.0",
                "languages": ["spa"], "supports_preprocessing": False,
            },
        )
    )

    engine = remote()
    engine.refresh_capabilities()

    assert engine.supports_preprocessing is False


@respx.mock
def test_remote_engine_reuses_the_connection_between_attempts():
    """Un escaneo de vencimiento son hasta ocho llamadas al mismo servicio."""
    respx.post(f"{TESSERACT_URL}/engine/run").mock(
        return_value=httpx.Response(200, json=engine_response([WORD]))
    )

    engine = remote()
    engine.run(IMAGE)
    first = engine._client()
    engine.run(IMAGE)

    assert engine._client() is first


@respx.mock
def test_closing_the_engine_releases_the_connection():
    respx.post(f"{TESSERACT_URL}/engine/run").mock(
        return_value=httpx.Response(200, json=engine_response([WORD]))
    )

    engine = remote()
    engine.run(IMAGE)
    client = engine._client()
    engine.close()

    assert client.is_closed
    # Y vuelve a servir: cerrar no lo deja inutilizable.
    engine.run(IMAGE)
    assert not engine._client().is_closed


@respx.mock
def test_reset_cache_closes_the_engines_it_discards():
    """Descartar un motor sin cerrarlo dejaría el socket colgado."""
    engine = factory.build_engine("tesseract")
    client = engine._client()

    factory.reset_cache()

    assert client.is_closed


@respx.mock
def test_remote_engine_info_is_cached():
    route = respx.get(f"{TESSERACT_URL}/engine/info").mock(
        return_value=httpx.Response(200, json={"engine": "tesseract", "version": "5.5.0"})
    )

    engine = remote()
    engine.info()
    engine.info()

    assert route.call_count == 1


@respx.mock
def test_remote_engine_survives_an_unreadable_info():
    respx.get(f"{TESSERACT_URL}/engine/info").mock(side_effect=httpx.ConnectError("refused"))

    assert remote().info()["version"] == "desconocida"


@respx.mock
def test_remote_engine_sends_the_shared_key(monkeypatch):
    monkeypatch.setattr(settings, "ENGINE_API_KEY", "s3cr3t")
    route = respx.post(f"{TESSERACT_URL}/engine/run").mock(
        return_value=httpx.Response(200, json=engine_response([WORD]))
    )

    remote().run(IMAGE)

    assert route.calls.last.request.headers["X-Engine-Key"] == "s3cr3t"


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def test_factory_builds_each_concrete_engine_from_its_service_url():
    tesseract = factory.build_engine("tesseract")
    paddleocr = factory.build_engine("paddleocr")

    assert tesseract.base_url == settings.TESSERACT_SERVICE_URL.split(",")[0].rstrip("/")
    assert paddleocr.base_url == settings.PADDLEOCR_SERVICE_URL.split(",")[0].rstrip("/")


def test_the_engines_of_the_module_are_available():
    """Los dos que quedaron en pie después de medirlos."""
    assert set(factory.CONCRETE_ENGINES) == {"tesseract", "paddleocr"}


def test_paddleocr_is_built_with_all_its_replicas():
    """Va replicado: es el respaldo del híbrido y satura sus núcleos antes que nadie."""
    replicas = factory.build_engine("paddleocr").replicas

    assert len(replicas) == len(settings.PADDLEOCR_SERVICE_URL.split(","))
    assert all(url.startswith("http") for url in replicas)


def test_factory_declares_who_benefits_from_preprocessing():
    assert factory.build_engine("tesseract").supports_preprocessing is True
    # PP-OCR normaliza y reescala dentro de la red y espera color: variarle los
    # filtros gasta corridas caras sin cambiar nada.
    assert factory.build_engine("paddleocr").supports_preprocessing is False


def test_factory_memoizes_engines():
    assert factory.build_engine("tesseract") is factory.build_engine("tesseract")


def test_factory_rejects_an_unknown_engine():
    with pytest.raises(ValueError, match="desconocido"):
        factory.build_engine("no-existe")


def test_factory_builds_the_hybrid_from_the_two_configured_engines():
    hybrid = factory.build_engine("hybrid")

    assert isinstance(hybrid, HybridEngine)
    assert hybrid.primary.name == settings.HYBRID_PRIMARY_ENGINE
    assert hybrid.fallback.name == settings.HYBRID_FALLBACK_ENGINE


def test_the_hybrid_can_take_any_concrete_engine_as_backup(monkeypatch):
    """Cuál conviene de respaldo lo decide el benchmark, no el código."""
    monkeypatch.setattr(settings, "HYBRID_FALLBACK_ENGINE", "paddleocr")
    factory.reset_cache()

    assert factory.build_engine("hybrid").fallback.name == "paddleocr"


def test_engines_satisfy_the_protocol():
    assert isinstance(factory.build_engine("tesseract"), OCREngine)
    assert isinstance(factory.build_engine("paddleocr"), OCREngine)
    assert isinstance(factory.build_engine("hybrid"), OCREngine)


# ---------------------------------------------------------------------------
# Modo híbrido
# ---------------------------------------------------------------------------

class DeadEngine:
    """Servicio de motor caído."""

    name = "respaldo"
    supports_preprocessing = False

    def is_available(self) -> bool:
        return False

    def run(self, *_args, **_kwargs):
        raise EngineUnavailable("el servicio no responde")


def test_hybrid_keeps_the_primary_when_it_reads_well():
    primary = FakeEngine(["Leche Entera"], confidence=0.95)
    fallback = FakeEngine(["Leche Entera"], confidence=0.99)

    output = HybridEngine(primary, fallback, threshold=0.70).run(IMAGE)

    assert output.engine == primary.name
    assert fallback.calls == []


def test_hybrid_falls_back_when_the_primary_reads_poorly():
    primary = FakeEngine(["ruido"], confidence=0.20, name="primario")
    fallback = FakeEngine(["Leche Entera"], confidence=0.95, name="respaldo")

    output = HybridEngine(primary, fallback, threshold=0.70).run(IMAGE)

    assert output.engine == "respaldo"
    assert len(fallback.calls) == 1


def test_hybrid_keeps_the_primary_if_the_fallback_reads_worse():
    primary = FakeEngine(["Leche"], confidence=0.60, name="primario")
    fallback = FakeEngine(["Leche"], confidence=0.30, name="respaldo")

    output = HybridEngine(primary, fallback, threshold=0.70).run(IMAGE)

    assert output.engine == "primario"


def test_hybrid_survives_a_fallback_that_is_down():
    """Sin respaldo el híbrido degrada a su motor principal, no falla."""
    primary = FakeEngine(["Leche"], confidence=0.40, name="primario")

    output = HybridEngine(primary, DeadEngine(), threshold=0.70).run(IMAGE)

    assert output.engine == "primario"


def test_hybrid_availability_depends_on_the_primary():
    assert HybridEngine(FakeEngine(["x"]), DeadEngine(), 0.7).is_available() is True
    assert HybridEngine(DeadEngine(), FakeEngine(["x"]), 0.7).is_available() is False
