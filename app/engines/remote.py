"""Cliente de un servicio de motor OCR.

Es la pieza que reemplaza a los adaptadores que antes ejecutaban los motores
dentro de este proceso. Desde afuera se comporta igual —cumple `OCREngine`— pero
por dentro delega en `ocr_service_tesseract` u `ocr_service_paddleocr`, que
hablan el mismo contrato.

Un motor puede estar replicado: `base_url` admite varias URLs separadas por coma
y se reparten por turno, con caída a la siguiente si una no responde. Para el
resto del servicio sigue siendo un solo motor.

Todo lo que decide *qué* pedirle al motor (giro, filtros, segmentación, cuántos
intentos) sigue estando de este lado. El servicio remoto sólo ejecuta.
"""

import itertools
import json
import logging
import re

import httpx

from app.core.config import settings
from app.engines.base import OCRWord, RawOCROutput
from app.pipeline.transform import Transform

logger = logging.getLogger(__name__)

_API_KEY_HEADER = "X-Engine-Key"

# Cuánto se cachea la respuesta de `/engine/info`: cambia sólo si se redespliega
# el motor, y consultarla en cada petición agregaría un round-trip inútil.
_INFO_TTL_CALLS = 50


class EngineUnavailable(RuntimeError):
    """El servicio del motor no respondió o devolvió un error."""


class RemoteEngine:
    """Motor OCR que corre en otro servicio, detrás de HTTP."""

    def __init__(self, name: str, base_url: str, supports_preprocessing: bool) -> None:
        self.name = name
        # Un motor puede estar replicado: `base_url` admite varias separadas por
        # coma. Es la forma de ensanchar el cuello de botella cuando el
        # reconocimiento es CPU y una sola instancia satura sus núcleos.
        self.replicas = _split_urls(base_url)
        # Valor por defecto hasta que `/engine/info` diga otra cosa: cada motor
        # es la autoridad sobre si le sirve que le varíen los filtros.
        self.supports_preprocessing = supports_preprocessing
        self._info: dict | None = None
        self._info_calls = 0
        self._http: httpx.Client | None = None
        self._next = itertools.count()

    @property
    def base_url(self) -> str:
        """La primera réplica. Es lo que se nombra en logs y mensajes de error."""
        return self.replicas[0]

    def _pick(self) -> str:
        """Réplica que atiende esta petición, por turno.

        Turno redondo y no "la menos ocupada": para saber cuál está menos
        ocupada habría que contabilizar peticiones en vuelo, y con dos réplicas
        idénticas y peticiones de duración parecida el turno reparte igual de
        bien sin estado compartido.
        """
        return self.replicas[next(self._next) % len(self.replicas)]

    # Cliente HTTP
    def _client(self) -> httpx.Client:
        """Cliente memoizado, para reusar la conexión con el motor.

        Un escaneo de vencimiento encadena hasta ocho llamadas al mismo servicio.
        Abriendo un cliente nuevo en cada una se pagaba un *handshake* TCP por
        intento; con uno solo, la conexión queda viva entre intentos. El timeout
        se pasa por petición porque no es el mismo para reconocer que para
        preguntar si el motor está vivo.
        """
        if self._http is None:
            self._http = httpx.Client(timeout=settings.HTTP_TIMEOUT_SECONDS)

        return self._http

    def close(self) -> None:
        """Cierra la conexión con el motor. La llama el apagado del servicio."""
        if self._http is not None:
            self._http.close()
            self._http = None

    # Contrato OCREngine
    def run(
        self,
        content: bytes,
        lang: str = "spa",
        layout: str = "block",
        transform: Transform | None = None,
    ) -> RawOCROutput:
        """Manda la imagen original al motor y traduce su respuesta."""
        spec = transform or Transform()

        payload = {
            "lang": lang,
            "layout": layout,
            # El contrato viaja como JSON en un campo del multipart: es la forma
            # de mandar una estructura junto al binario sin inventar un encoding.
            "transform": json.dumps(spec.as_payload()),
        }

        # Si la réplica que tocaba no contesta se prueba con las otras: tener dos
        # instancias sirve tanto para repartir la carga como para sobrevivir a que
        # una se caiga o se esté reiniciando.
        ultimo: Exception | None = None

        for _ in range(len(self.replicas)):
            url = self._pick()
            try:
                response = self._client().post(
                    f"{url}/engine/run",
                    files={"file": ("image.jpg", content, "image/jpeg")},
                    data=payload,
                    headers=self._headers(),
                    timeout=settings.ENGINE_TIMEOUT_SECONDS,
                )
                response.raise_for_status()
                return _to_output(response.json(), self.name)
            except httpx.HTTPStatusError as exc:
                # Un error de la aplicación (una imagen inválida, por ejemplo) va
                # a dar lo mismo en la otra réplica: no se reintenta.
                raise EngineUnavailable(
                    f"El motor '{self.name}' respondió {exc.response.status_code}: "
                    f"{_detail(exc.response)}"
                ) from exc
            except httpx.HTTPError as exc:
                logger.warning("Réplica %s de '%s' no respondió: %s", url, self.name, exc)
                ultimo = exc

        # El mensaje se adapta: con una sola instancia, hablar de "réplicas" sería
        # confuso para quien lee el log.
        donde = (
            f"en {self.base_url}"
            if len(self.replicas) == 1
            else f"en ninguna de sus {len(self.replicas)} réplicas ({', '.join(self.replicas)})"
        )
        raise EngineUnavailable(
            f"No se pudo contactar al motor '{self.name}' {donde}: {ultimo}"
        ) from ultimo

    def is_available(self) -> bool:
        """Con que responda una réplica alcanza: el motor sigue utilizable."""
        return any(self._replica_available(url) for url in self.replicas)

    def _replica_available(self, url: str) -> bool:
        try:
            response = self._client().get(
                f"{url}/engine/readyz", headers=self._headers()
            )
            # `degraded` (por ejemplo, falta el paquete de idioma) sigue siendo
            # utilizable: el motor cae a inglés y lo avisa.
            return response.status_code == 200 and response.json().get("status") != "error"
        except (httpx.HTTPError, ValueError) as exc:
            logger.warning("Réplica %s de '%s' no disponible: %s", url, self.name, exc)
            return False

    # ── Metadatos ─────────────────────────────────────────────────────────────

    def info(self) -> dict:
        """Qué motor y qué versión hay del otro lado. Se cachea."""
        if self._info is None or self._info_calls >= _INFO_TTL_CALLS:
            self._info = self._fetch_info()
            self._info_calls = 0

        self._info_calls += 1
        return self._info

    def refresh_capabilities(self) -> None:
        """Toma de `/engine/info` si al motor le sirve el preprocesamiento."""
        declared = self.info().get("supports_preprocessing")
        if isinstance(declared, bool):
            self.supports_preprocessing = declared

    def _fetch_info(self) -> dict:
        try:
            response = self._client().get(
                f"{self.base_url}/engine/info", headers=self._headers()
            )
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPError, ValueError) as exc:
            logger.warning("No se pudo leer /engine/info de '%s': %s", self.name, exc)
            return {"engine": self.name, "version": "desconocida", "languages": []}

    def _headers(self) -> dict[str, str]:
        return {_API_KEY_HEADER: settings.ENGINE_API_KEY} if settings.ENGINE_API_KEY else {}


# Escrituras que este servicio no lee: ideogramas chinos y japoneses, silabario
# coreano y sus signos. El idioma de trabajo es español.
_FOREIGN_SCRIPT = re.compile(
    "[\u1100-\u11ff\u2e80-\u2fdf\u3000-\u30ff\u3100-\u318f\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af\uf900-\ufaff]"
)


def strip_foreign_script(text: str) -> str:
    """El texto sin caracteres de escrituras que no se leen."""
    return _FOREIGN_SCRIPT.sub("", text).strip()


def _to_output(body: dict, fallback_name: str) -> RawOCROutput:
    words = []
    for word in body.get("words", []):
        text = strip_foreign_script(word["text"])
        if not text:
            continue
        words.append(
            OCRWord(
                text=text,
                confidence=float(word.get("confidence", 0.0)),
                left=int(word.get("left", 0)),
                top=int(word.get("top", 0)),
                width=int(word.get("width", 0)),
                height=int(word.get("height", 0)),
                line_num=int(word.get("line_num", 0)),
            )
        )

    raw_text = _FOREIGN_SCRIPT.sub("", body.get("text") or "")

    return RawOCROutput(
        # El texto lo arma el motor a partir de sus propias líneas; si no vino,
        # se reconstruye acá con la misma regla.
        text=raw_text if raw_text.strip() else RawOCROutput.from_words(words, fallback_name).text,
        words=words,
        engine=body.get("engine") or fallback_name,
        elapsed_ms=int(body.get("elapsed_ms", 0)),
    )


def _detail(response: httpx.Response) -> str:
    try:
        return str(response.json().get("detail", response.text))[:200]
    except ValueError:
        return response.text[:200]


def _split_urls(raw: str) -> list[str]:
    """`"http://a:8007, http://b:8007"` → las dos, sin barra final."""
    urls = [piece.strip().rstrip("/") for piece in raw.split(",") if piece.strip()]

    if not urls:
        raise ValueError("Un motor remoto necesita al menos una URL")

    return urls
