from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Servidor
    HOST: str = "0.0.0.0"
    PORT: int = 8003
    DEBUG: bool = False

    # JWT — mismo secret que el servicio Auth
    JWT_SECRET: str = "change-this-secret-in-production"
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
    EASYOCR_SERVICE_URL: str = "http://tpp_ocr_easyocr:8006"

    # Clave compartida con los servicios de motor. Vacía = sin control de acceso.
    ENGINE_API_KEY: str = ""

    # EasyOCR tarda segundos por documento: necesita más margen que Tesseract.
    ENGINE_TIMEOUT_SECONDS: float = 120.0

    OCR_ENGINE: str = "hybrid"  # tesseract | easyocr | hybrid
    OCR_LANG: str = "spa"

    # Redis (cola de trabajos + caché de resultados)
    REDIS_URL: str = "redis://redis:6379/0"
    JOB_TTL_SECONDS: int = 86400
    QUEUE_KEY: str = "ocr:queue"
    CACHE_TTL_SECONDS: int = 86400

    # Umbrales de aceptación (criterios de la propuesta)
    FIELD_CONFIDENCE_THRESHOLD: float = 0.85
    DOCUMENT_CONFIDENCE_THRESHOLD: float = 0.70

    # En modo híbrido, si la confianza no alcanza este valor se reintenta con el
    # motor de respaldo.
    HYBRID_FALLBACK_THRESHOLD: float = 0.70
    HYBRID_PRIMARY_ENGINE: str = "tesseract"
    HYBRID_FALLBACK_ENGINE: str = "easyocr"

    # ── Estrategia de intentos (la decide este servicio, no los motores) ───────
    # Ante una lectura pobre se prueban otras variantes de preparación y se
    # conserva la de mayor confianza.
    RETRY_ENABLED: bool = True
    RETRY_THRESHOLD: float = 0.75

    # Las fotos de la fecha de vencimiento se sacan apoyando el envase de costado
    # y llegan giradas un cuarto de vuelta. Se prueban las cuatro orientaciones y
    # se conserva la que mejor lee.
    ROTATION_SEARCH_ENABLED: bool = True

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

    # Ancho que se le pide a un motor que prepara su propia entrada (EasyOCR).
    # No es un detalle de implementación del motor: es una decisión de calidad
    # contra costo, y por eso la toma el coordinador. En una etiqueta 800 px
    # alcanzan y sobran; en un sello de vencimiento son la diferencia entre leer
    # `27FEBZ7` —que el postproceso repara a `27FEB27`— y leer `27FEBZZ`, que no
    # se puede reparar sin inventar el dato.
    ENGINE_MAX_WIDTH: int = 800
    ENGINE_STAMP_MAX_WIDTH: int = 1280

    # CORS
    ALLOWED_ORIGINS_STR: str = (
        "http://localhost:3000,http://localhost:8081,http://localhost:19000"
    )

    @property
    def ALLOWED_ORIGINS(self) -> list[str]:
        return [o.strip() for o in self.ALLOWED_ORIGINS_STR.split(",") if o.strip()]


settings = Settings()
