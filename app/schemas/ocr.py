from enum import Enum
from typing import Union

from pydantic import BaseModel, Field, HttpUrl

FieldValue = Union[str, int, float, None]


class DocType(str, Enum):
    """Los dos casos de entrada que define la propuesta."""

    PRODUCT_LABEL = "product_label"
    RECEIPT = "receipt"


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


class OCRJobRequest(BaseModel):
    image_url: HttpUrl
    doc_type: DocType = DocType.PRODUCT_LABEL
    lang: str | None = None
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
