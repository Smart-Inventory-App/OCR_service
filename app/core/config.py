from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Servidor
    HOST: str = "0.0.0.0"
    PORT: int = 8003
    DEBUG: bool = False

    # JWT — mismo secret que el servicio Auth
    JWT_SECRET: str
    JWT_ALGORITHM: str = "HS256"

    # Servicios del monorepo
    STORAGE_SERVICE_URL: str = "http://storage_service:8000"
    HTTP_TIMEOUT_SECONDS: float = 30.0

    # URL del servicio Auth tal como la ve el NAVEGADOR (no la red de Docker).
    # Sólo la usa la página de prueba (`/ocr/playground`) para hacer login.
    AUTH_PUBLIC_URL: str = "http://localhost:8001"

    # ── Motores ───────────────────────────────────────────────────────────────
    # Cada motor es un servicio aparte, con su propio repositorio e imagen. Este
    # servicio no ejecuta ningún motor: los coordina.
    TESSERACT_SERVICE_URL: str = "http://tpp_ocr_tesseract:8005"
    # PaddleOCR va replicado y acepta varias URLs separadas por coma: el
    # coordinador las reparte por turno y se cae a la otra si una no responde.
    # Es el respaldo del híbrido, así que casi todo documento difícil termina
    # ahí, y una sola instancia satura sus núcleos mucho antes que el resto del
    # módulo.
    PADDLEOCR_SERVICE_URL: str = (
        "http://tpp_ocr_paddleocr:8007,http://tpp_ocr_paddleocr_2:8007"
    )

    # Clave compartida con los servicios de motor. Vacía = sin control de acceso.
    ENGINE_API_KEY: str = ""

    # PaddleOCR tarda segundos por documento: necesita más margen que Tesseract.
    ENGINE_TIMEOUT_SECONDS: float = 120.0

    OCR_ENGINE: str = "hybrid"  # tesseract | paddleocr | hybrid
    OCR_LANG: str = "spa"

    # Redis (cola de trabajos + caché de resultados)
    REDIS_URL: str = "redis://redis:6379/0"
    # Tiene que ser holgadamente mayor que el timeout del `BLPOP` del worker
    # (5 s): es el plazo de lectura del socket, no el de la espera en la cola.
    REDIS_SOCKET_TIMEOUT_SECONDS: float = 30.0
    JOB_TTL_SECONDS: int = 86400
    QUEUE_KEY: str = "ocr:queue"
    CACHE_TTL_SECONDS: int = 86400

    # Cuántos jobs procesa el worker a la vez.
    #
    # En 1 —como estaba— la cola es estrictamente serie y el costo lo paga el
    # último de la fila: medido con cuatro clientes escaneando a la vez, el
    # primero terminó a los 20 s y el cuarto a los 168 s, aunque ninguno tardaba
    # más de 65 s por su cuenta. Y el orden es de llegada, así que un código de
    # barras —que se resuelve en 200 ms sin tocar un motor— puede quedar detrás
    # de tres tickets.
    #
    # No es paralelismo gratis: los motores son servicios aparte y la inferencia
    # es CPU, así que subirlo de más los satura en vez de acelerarlos. Medido
    # sobre la misma tanda de seis documentos, con PaddleOCR replicado:
    #
    #     concurrencia 1 → 200,3 s     concurrencia 3 → 111,6 s
    #     concurrencia 4 → 121,5 s     concurrencia 6 → 167,0 s (una réplica)
    #
    # 3 es el punto óptimo, y no por casualidad: son dos réplicas de PaddleOCR
    # más Tesseract. Para más carga, otra réplica de motor rinde bastante más que
    # subir este número.
    WORKER_CONCURRENCY: int = 3

    # Cada cuánto drena la cola un documento pesado, medido: 6 documentos en
    # 111,6 s con PaddleOCR replicado, o sea uno cada ~19 s. Con esto se le
    # informa al usuario cuánto le falta en vez de dejarlo mirando un spinner.
    #
    # Si se agrega o se quita una réplica de motor hay que volver a medirlo: una
    # estimación vieja es peor que ninguna, porque el usuario la cree.
    QUEUE_DRAIN_SECONDS_PER_JOB: float = 19.0

    # Profundidad máxima de la cola antes de rechazar documentos pesados.
    #
    # Al ritmo de arriba, 15 en espera ya son siete minutos. Aceptar el número 16
    # sería mentirle: es mejor un 429 con `Retry-After` y un mensaje honesto que
    # un `processing` que no termina nunca. Los códigos de barras no cuentan para
    # este tope porque no pasan por la cola.
    MAX_QUEUE_DEPTH: int = 15

    # Tope de escaneos por usuario y por minuto, por tipo de documento.
    #
    # No cuestan lo mismo: un código de barras se decodifica en 0,2 s sin tocar
    # un motor y un ticket ocupa un slot del worker entre 40 y 140 s. Un límite
    # único estrangularía al comercio que escanea una entrega o dejaría que un
    # solo usuario con tickets sature los motores.
    #
    # Es una ventana fija de un minuto, no un token bucket: el usuario puede
    # gastar todo su cupo de golpe al empezar la ventana, que es justamente lo
    # que hace falta para una ráfaga legítima. El freno de capacidad real es
    # `MAX_QUEUE_DEPTH`; esto es contra el abuso y contra un cliente con un bug
    # de reintentos.
    RATE_LIMIT_PER_MINUTE_STR: str = "barcode:60,product_label:10,expiry_date:10,receipt:4"

    # Umbrales de aceptación (criterios de la propuesta)
    FIELD_CONFIDENCE_THRESHOLD: float = 0.85
    DOCUMENT_CONFIDENCE_THRESHOLD: float = 0.70

    # En modo híbrido, si la confianza no alcanza este valor se reintenta con el
    # motor de respaldo.
    HYBRID_FALLBACK_THRESHOLD: float = 0.70
    HYBRID_PRIMARY_ENGINE: str = "tesseract"
    # Cuál de los dos motores pesados conviene como respaldo es una pregunta
    # empírica, y la responde `python -m bench.run_benchmark`: sobre las fotos
    # reales del proyecto, PaddleOCR es el único motor que pasa los dos
    # criterios de aceptación por sí solo, y con él de respaldo el híbrido
    # también los pasa.
    HYBRID_FALLBACK_ENGINE: str = "paddleocr"

    # Tope de ORIENTACIONES que se le conceden al motor de respaldo cuando el
    # principal no reconoció nada y por lo tanto no hay un giro conocido que
    # pasarle.
    #
    # Sin tope, ese caso —el peor— le pide al motor MÁS caro una inferencia por
    # cada orientación, y es exactamente el camino que agota la memoria cuando el
    # entorno va justo: sobre los sellos del conjunto de prueba, con 1,9 GB
    # repartidos entre seis contenedores, PaddleOCR acumuló más de 90 reinicios
    # por OOM del kernel.
    #
    # **El default son las cuatro**, es decir, sin recorte: bajarlo cuesta
    # lecturas reales y eso no puede pasar en silencio. Medido sobre este mismo
    # conjunto, ponerlo en 2 perdió 6 sellos que PaddleOCR sí leía —todos en los
    # giros que quedaron afuera— a cambio de que la corrida terminara en un
    # tercio del tiempo y sin quedarse sin memoria.
    #
    # O sea: es una perilla para entornos apretados, no una mejora. Si hay que
    # bajarla, el orden de `ROTATIONS` deja primero las que más rinden —la foto
    # tal como vino y el cuarto de vuelta con que llega el sello—.
    HYBRID_FALLBACK_MAX_ROTATIONS: int = 4

    # ── Estrategia de intentos (la decide este servicio, no los motores) ───────
    # Ante una lectura pobre se prueban otras variantes de preparación y se
    # conserva la de mayor confianza.
    RETRY_ENABLED: bool = True
    RETRY_THRESHOLD: float = 0.75

    # Las fotos de la fecha de vencimiento se sacan apoyando el envase de costado
    # y llegan giradas un cuarto de vuelta. Se prueban las cuatro orientaciones y
    # se conserva la que mejor lee.
    ROTATION_SEARCH_ENABLED: bool = True

    # La etiqueta también puede venir girada. Se suponía que no —"la foto de
    # etiqueta viene derecha"— y el conjunto de prueba mostró lo contrario: las
    # mismas 16 fotos giradas un cuarto de vuelta daban un nombre distinto en los
    # 16 casos. Los giros se prueban al final del plan, después de las variantes
    # de filtros, así que la foto derecha sigue resolviéndose en un intento.
    LABEL_ROTATION_SEARCH_ENABLED: bool = True

    # Y el ticket tampoco. Misma suposición, mismo desmentido: `dia_2.jpg` es el
    # mismo comprobante que `dia_1.jpg` sacado al revés, y era el único de 73 que
    # fallaba —salía con los renglones invertidos y `SUBTOTAL` leído `SUBTAL`—.
    RECEIPT_ROTATION_SEARCH_ENABLED: bool = True

    # Preparación de la imagen para el escaneo de etiquetas y tickets.
    PREPROCESS_GRAYSCALE: bool = True
    PREPROCESS_UPSCALE: bool = True
    PREPROCESS_INVERT: bool = True
    PREPROCESS_CONTRAST: bool = True
    PREPROCESS_DENOISE: bool = False
    PREPROCESS_DESKEW: bool = True
    PREPROCESS_MIN_WIDTH: int = 1600
    PREPROCESS_MAX_WIDTH: int = 3000

    # Preparación para el sello de vencimiento: se amplía más y se usa umbral
    # adaptativo, que es lo contrario de lo que le conviene a una etiqueta.
    STAMP_MIN_WIDTH: int = 2400
    STAMP_MAX_WIDTH: int = 3600

    # Cuántas bandas de texto se recortan y prueban antes de mandar el envase
    # entero. En una foto de 4000 px el sello es un renglón de 90 px de alto:
    # recortarlo lo deja solo frente al modelo, sin la etiqueta impresa ni el
    # fondo compitiendo por la detección. Se prueban varias bandas porque el
    # motor no puede distinguir el sello de la etiqueta impresa
    # —medido, la franja mejor puntuada resultó ser el teléfono de atención al
    # consumidor— y elegir es trabajo del scoring, no de una heurística.
    EXPIRY_CROP_BANDS: int = 3

    # Ancho que se le pide a un motor que prepara su propia entrada (PaddleOCR).
    # No es un detalle de implementación del motor: es una decisión de calidad
    # contra costo, y por eso la toma el coordinador. En una etiqueta 800 px
    # alcanzan y sobran; en un sello de vencimiento son la diferencia entre leer
    # `27FEBZ7` —que el postproceso repara a `27FEB27`— y leer `27FEBZZ`, que no
    # se puede reparar sin inventar el dato.
    ENGINE_MAX_WIDTH: int = 800
    ENGINE_STAMP_MAX_WIDTH: int = 1280

    # ── Códigos de barras ─────────────────────────────────────────────────────
    # El único reconocimiento que corre dentro de este servicio. No necesita
    # búsqueda de orientación —ZBar recorre la imagen en varias direcciones por
    # su cuenta— ni realce de contraste —binariza internamente—. Lo único que
    # mueve la aguja es la resolución, y el enfoque en una banda angosta.
    BARCODE_DOWNSCALE_WIDTHS_STR: str = "2048,1600,1400,1200,1024,900,800,640,512"
    """Anchos a los que se prueba reducir la imagen, de mayor a menor.

    Es el intento que más lecturas recupera de todo el módulo: sobre el conjunto
    de prueba resuelve 82 de los 116 códigos que fallaban. Son varios anchos
    porque ninguno domina a los otros —2048 px recupera 43, 1600 recupera 60 y
    640 recupera 71, y no son subconjuntos—, así que se recorren de mayor a menor
    para descartar lo menos posible antes de tener que descartar más.

    Los anchos intermedios y el de 512 se agregaron después, midiendo sobre los
    31 códigos impresos que seguían sin leerse: 512 px recupera 6 de ellos y
    1400 px otros 4. Cada ancho es un intento más, pero sólo lo pagan las
    imágenes que fallaron en todos los anteriores.
    """

    BARCODE_UPSCALE_FACTOR: int = 2
    """Ampliación para el caso contrario: el código chico dentro del cuadro.
    Medido: recupera la banda de ~180 a ~200 px de ancho de código, que falla en
    el intento directo."""

    BARCODE_SHARPEN_ENABLED: bool = True
    """Máscara de enfoque para el desenfoque leve de una foto a pulso."""

    BARCODE_MAX_PIXELS: int = 60_000_000
    """Tope de la imagen ampliada, para acotar el pico de memoria. Ampliar al
    doble una foto de 12 MP da 48 MP, que en escala de grises son ~48 MB."""

    # CORS
    ALLOWED_ORIGINS_STR: str = (
        "http://localhost:3000,http://localhost:8081,http://localhost:19000"
    )

    @property
    def ALLOWED_ORIGINS(self) -> list[str]:
        return [o.strip() for o in self.ALLOWED_ORIGINS_STR.split(",") if o.strip()]

    @property
    def rate_limit_per_minute(self) -> dict[str, int]:
        """Los topes por tipo de documento, ya parseados."""
        limits: dict[str, int] = {}

        for piece in self.RATE_LIMIT_PER_MINUTE_STR.split(","):
            name, _, value = piece.partition(":")
            if name.strip() and value.strip().isdigit():
                limits[name.strip()] = int(value)

        return limits

    @property
    def barcode_downscale_widths(self) -> list[int]:
        """Los anchos de reducción ya parseados, ordenados de mayor a menor."""
        widths = {
            int(piece)
            for piece in self.BARCODE_DOWNSCALE_WIDTHS_STR.split(",")
            if piece.strip().isdigit() and int(piece) > 0
        }
        return sorted(widths, reverse=True)


settings = Settings()
