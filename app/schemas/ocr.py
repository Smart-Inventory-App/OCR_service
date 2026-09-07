from enum import Enum
from typing import Union

from pydantic import BaseModel, Field, HttpUrl

FieldValue = Union[str, int, float, None]


class DocType(str, Enum):
    """Los tres tipos de entrada que acepta el servicio.

    La etiqueta y la fecha de vencimiento eran un solo caso y ahora son dos. El
    motivo es medido: en una foto de la etiqueta la fecha casi nunca entra (está
    impresa en otra cara del envase), y cuando entra, la preparación de imagen
    que hace legible el logo es exactamente la que borra el sello de matriz de
    puntos. Pidiendo dos fotos, cada una se procesa con la cadena que le sirve.
    """

    PRODUCT_LABEL = "product_label"
    """Etiqueta en primer plano: nombre, marca y detalles del producto."""

    EXPIRY_DATE = "expiry_date"
    """Sello de vencimiento sobre el envase: sólo la fecha."""

    RECEIPT = "receipt"
    """Ticket, remito o factura: ítems con cantidad, fecha de compra y total."""

    BARCODE = "barcode"
    """Código de barras del envase: el identificador global del producto.

    Es el único tipo que **no pasa por un motor OCR**. Un código de barras no se
    lee reconociendo caracteres: se decodifica la simbología, y el resultado trae
    su propio dígito verificador. Eso lo vuelve la entrada más confiable de las
    cuatro —o falla, o acierta— y por eso el coordinador lo resuelve por su
    cuenta en vez de delegarlo.
    """


class JobStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    DONE = "done"
    FAILED = "failed"


class ExtractedField(BaseModel):
    """Un campo clave extraído, con su confianza combinada motor+parser."""

    value: FieldValue = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class ReceiptItem(BaseModel):
    nombre_producto: ExtractedField
    cantidad: ExtractedField


class OCRResult(BaseModel):
    fields: dict[str, ExtractedField] = Field(default_factory=dict)
    items: list[ReceiptItem] = Field(default_factory=list)
    document_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    needs_review: bool = True
    engine: str = ""
    raw_text: str = ""
    processing_ms: int = 0

    rotation: int = Field(
        default=0,
        description="Giro con el que se obtuvo esta lectura (0, 90, 180 o 270).",
    )
    attempts: int = Field(
        default=1,
        description="Cuántas combinaciones de giro y filtros hizo falta probar.",
    )


class OCRJobRequest(BaseModel):
    image_url: HttpUrl
    doc_type: DocType = DocType.PRODUCT_LABEL
    lang: str | None = None
    engine: str | None = Field(
        default=None,
        description=(
            "Motor a usar: `tesseract`, `easyocr`, `paddleocr` o `hybrid`. "
            "Si se omite, se usa el configurado en `OCR_ENGINE`."
        ),
    )
    discard_after: bool = Field(
        default=False,
        description=(
            "Si es true, al terminar el procesamiento se descarta la imagen en "
            "image_storage_service (DELETE /storage/temp)."
        ),
    )


class OCRJobCreated(BaseModel):
    job_id: str
    status: JobStatus = JobStatus.QUEUED


class OCRJobState(BaseModel):
    job_id: str
    status: JobStatus
    doc_type: DocType
    result: OCRResult | None = None
    error: str | None = None
    created_at: str
    updated_at: str


class OCRSyncResponse(BaseModel):
    doc_type: DocType
    result: OCRResult


class EngineStatus(BaseModel):
    """Estado de un servicio de motor, tal como lo ve el coordinador."""

    name: str
    url: str
    available: bool
    version: str = "desconocida"
    languages: list[str] = Field(default_factory=list)
    supports_preprocessing: bool = False
