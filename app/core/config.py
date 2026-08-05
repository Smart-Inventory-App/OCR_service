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

    # Redis (cola de trabajos + caché de resultados)
    REDIS_URL: str = "redis://redis:6379/0"
    JOB_TTL_SECONDS: int = 86400
    QUEUE_KEY: str = "ocr:queue"
    CACHE_TTL_SECONDS: int = 86400

    # Motor OCR
    OCR_ENGINE: str = "tesseract"  # tesseract | easyocr | hybrid
    OCR_LANG: str = "spa"
    TESSERACT_CMD: str = ""  # vacío = usar el binario del PATH

    # Umbrales de aceptación (criterios de la propuesta)
    FIELD_CONFIDENCE_THRESHOLD: float = 0.85
    DOCUMENT_CONFIDENCE_THRESHOLD: float = 0.70

    # En modo híbrido, si la confianza global no alcanza este valor se reintenta
    # con el motor de respaldo.
    HYBRID_FALLBACK_THRESHOLD: float = 0.70
    HYBRID_FALLBACK_ENGINE: str = "easyocr"

    # Preprocesamiento OpenCV — cada paso se puede desactivar para el benchmark
    PREPROCESS_GRAYSCALE: bool = True
    PREPROCESS_UPSCALE: bool = True
    PREPROCESS_DENOISE: bool = True
    PREPROCESS_THRESHOLD: bool = True
    PREPROCESS_DESKEW: bool = True
    PREPROCESS_MIN_WIDTH: int = 1000

    # CORS
    ALLOWED_ORIGINS_STR: str = (
        "http://localhost:3000,http://localhost:8081,http://localhost:19000"
    )

    @property
    def ALLOWED_ORIGINS(self) -> list[str]:
        return [o.strip() for o in self.ALLOWED_ORIGINS_STR.split(",") if o.strip()]


settings = Settings()
