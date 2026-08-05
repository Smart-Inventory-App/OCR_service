"""Evaluación comparativa de motores OCR.

La propuesta lo pide explícitamente:

    "Durante el desarrollo del proyecto se realizará una evaluación comparativa
    entre estas herramientas mediante pruebas sobre datos reales, considerando
    métricas como precisión de reconocimiento, tiempo de respuesta y consumo de
    recursos."

Corre cada motor sobre las mismas imágenes con el mismo preprocesamiento y emite
una tabla comparativa: precisión por campo, precisión por documento, latencia
media y memoria. Es el insumo para la decisión final del motor.

Uso:

    python -m bench.run_benchmark                          # todos los motores disponibles
    python -m bench.run_benchmark --engines tesseract       # sólo uno
    python -m bench.run_benchmark --no-preprocess           # medir el aporte del preprocesamiento
    python -m bench.run_benchmark --dataset bench/dataset   # otro dataset

Este script NO forma parte del deploy del MVP: EasyOCR y PaddleOCR están
comentados en `requirements.txt` y se instalan sólo para correr el benchmark.
"""

import argparse
import json
import statistics
import sys
import time
import tracemalloc
from dataclasses import dataclass, field
from pathlib import Path

from app.core.config import settings
from app.engines import factory
from app.pipeline import postprocess, preprocess
from app.pipeline.preprocess import PreprocessConfig
from app.schemas.ocr import DocType, OCRResult
from app.services import image_loader, ocr_service

DEFAULT_DATASET = Path("bench/dataset")
GROUND_TRUTH_FILE = "ground_truth.json"

CANDIDATE_ENGINES = ("tesseract", "easyocr", "hybrid")


@dataclass
class DocumentScore:
    name: str
    doc_type: str
    correct_fields: int = 0
    total_fields: int = 0
    document_confidence: float = 0.0
    elapsed_ms: float = 0.0
    error: str | None = None

    @property
    def field_accuracy(self) -> float:
        return self.correct_fields / self.total_fields if self.total_fields else 0.0


@dataclass
class EngineReport:
    engine: str
    documents: list[DocumentScore] = field(default_factory=list)
    peak_memory_mb: float = 0.0

    @property
    def field_accuracy(self) -> float:
        correct = sum(d.correct_fields for d in self.documents)
        total = sum(d.total_fields for d in self.documents)
        return correct / total if total else 0.0

    @property
    def document_accuracy(self) -> float:
        """Fracción de documentos que superan el umbral por documento (≥ 0.70)."""
        if not self.documents:
            return 0.0

        passing = sum(
            1
            for d in self.documents
            if d.document_confidence >= settings.DOCUMENT_CONFIDENCE_THRESHOLD
        )
        return passing / len(self.documents)

    @property
    def mean_latency_ms(self) -> float:
        times = [d.elapsed_ms for d in self.documents if d.error is None]
        return statistics.mean(times) if times else 0.0

    @property
    def failures(self) -> int:
        return sum(1 for d in self.documents if d.error is not None)


def load_ground_truth(dataset: Path) -> dict:
    path = dataset / GROUND_TRUTH_FILE
    if not path.exists():
        raise SystemExit(
            f"No se encontró {path}.\n"
            f"Colocá las imágenes reales en {dataset}/ junto a un {GROUND_TRUTH_FILE} "
            f"(ver bench/README.md para el formato)."
        )

    return json.loads(path.read_text(encoding="utf-8"))


def score_document(result: OCRResult, expected: dict) -> tuple[int, int]:
    """Compara el resultado contra el ground truth. Devuelve (aciertos, total)."""
    correct = 0
    total = 0

    for name, value in expected.get("fields", {}).items():
        total += 1
        extracted = result.fields.get(name)
        if extracted is not None and _matches(extracted.value, value):
            correct += 1

    expected_items = expected.get("items", [])
    if expected_items:
        extracted_items = {
            item.nombre_producto.value: item.cantidad.value for item in result.items
        }
        for item in expected_items:
            total += 1
            name = item["nombre_producto"]
            match = next(
                (
                    quantity
                    for extracted_name, quantity in extracted_items.items()
                    if _matches(extracted_name, name)
                ),
                None,
            )
            if match is not None and match == item.get("cantidad"):
                correct += 1

    return correct, total


def _matches(extracted, expected) -> bool:
    """Compara campos tolerando diferencias de caja, acentos y espacios."""
    if extracted is None:
        return False

    if isinstance(expected, str) and isinstance(extracted, str):
        from rapidfuzz import fuzz

        from app.parsers.text import fold

        # Los nombres de producto se comparan con similitud: el OCR puede
        # equivocarse en un carácter sin que el campo deje de ser correcto.
        return fuzz.ratio(fold(extracted), fold(expected)) >= 90

    return extracted == expected


def run_engine(
    engine_name: str,
    dataset: Path,
    ground_truth: dict,
    config: PreprocessConfig,
) -> EngineReport:
    report = EngineReport(engine=engine_name)

    factory.reset_cache()
    engine = factory.build_engine(engine_name)

    if not engine.is_available():
        print(f"  ! motor '{engine_name}' no disponible: se omite")
        return report

    tracemalloc.start()

    for filename, expected in sorted(ground_truth.items()):
        path = dataset / filename
        score = DocumentScore(name=filename, doc_type=expected.get("doc_type", "product_label"))

        if not path.exists():
            score.error = "imagen faltante"
            report.documents.append(score)
            continue

        doc_type = DocType(score.doc_type)

        try:
            started = time.perf_counter()

            image = image_loader.decode(path.read_bytes())
            processed = preprocess.run(image, config)
            output = engine.run(
                processed,
                lang=settings.OCR_LANG,
                layout=ocr_service.layout_for(doc_type),
            )
            text = postprocess.clean(output.text)
            result = ocr_service.build_result(output, text, doc_type)

            score.elapsed_ms = (time.perf_counter() - started) * 1000
            score.correct_fields, score.total_fields = score_document(result, expected)
            score.document_confidence = result.document_confidence

        except Exception as exc:  # el benchmark no debe abortar por un documento
            score.error = str(exc)

        report.documents.append(score)
        _print_document(score)

    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    report.peak_memory_mb = peak / (1024 * 1024)

    return report


def _print_document(score: DocumentScore) -> None:
    if score.error:
        print(f"  {score.name:<32} ERROR: {score.error}")
        return

    print(
        f"  {score.name:<32} campos {score.correct_fields}/{score.total_fields} "
        f"({score.field_accuracy:>5.1%})  conf {score.document_confidence:.2f}  "
        f"{score.elapsed_ms:>7.0f} ms"
    )


def print_summary(reports: list[EngineReport]) -> None:
    field_threshold = 0.85
    doc_threshold = settings.DOCUMENT_CONFIDENCE_THRESHOLD

    print("\n" + "=" * 88)
    print("COMPARATIVA DE MOTORES OCR")
    print("=" * 88)
    print(
        f"{'Motor':<24}{'Campos':>10}{'Docs >=70%':>12}"
        f"{'Latencia':>12}{'Memoria':>11}{'Fallos':>9}{'Gate':>8}"
    )
    print("-" * 88)

    for report in reports:
        if not report.documents:
            print(f"{report.engine:<24}{'(no disponible)':>50}")
            continue

        passes = report.field_accuracy >= field_threshold and report.document_accuracy >= 1.0

        print(
            f"{report.engine:<24}"
            f"{report.field_accuracy:>9.1%} "
            f"{report.document_accuracy:>11.1%} "
            f"{report.mean_latency_ms:>10.0f} ms"
            f"{report.peak_memory_mb:>9.1f} MB"
            f"{report.failures:>9}"
            f"{'  OK' if passes else '  NO':>8}"
        )

    print("-" * 88)
    print(
        f"Criterios de la propuesta: precision de campos clave >= {field_threshold:.0%} | "
        f"confianza por documento >= {doc_threshold:.0%}"
    )
    print("=" * 88)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluación comparativa de motores OCR")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--engines",
        nargs="+",
        default=list(CANDIDATE_ENGINES),
        help=f"motores a evaluar (default: {' '.join(CANDIDATE_ENGINES)})",
    )
    parser.add_argument(
        "--no-preprocess",
        action="store_true",
        help="desactiva el preprocesamiento OpenCV, para medir su aporte real",
    )
    parser.add_argument("--json", type=Path, help="guarda el reporte crudo en un archivo JSON")
    args = parser.parse_args()

    ground_truth = load_ground_truth(args.dataset)

    config = (
        PreprocessConfig(
            grayscale=False, upscale=False, denoise=False, threshold=False, deskew=False
        )
        if args.no_preprocess
        else PreprocessConfig.from_settings()
    )

    print(f"Dataset: {args.dataset} ({len(ground_truth)} documentos)")
    print(f"Preprocesamiento: {'desactivado' if args.no_preprocess else 'activado'}\n")

    reports = []
    for engine_name in args.engines:
        print(f"> {engine_name}")
        try:
            reports.append(run_engine(engine_name, args.dataset, ground_truth, config))
        except ValueError as exc:
            print(f"  ! {exc}")

    print_summary(reports)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(
                [
                    {
                        "engine": r.engine,
                        "field_accuracy": r.field_accuracy,
                        "document_accuracy": r.document_accuracy,
                        "mean_latency_ms": r.mean_latency_ms,
                        "peak_memory_mb": r.peak_memory_mb,
                        "documents": [vars(d) for d in r.documents],
                    }
                    for r in reports
                ],
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print(f"\nReporte guardado en {args.json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
