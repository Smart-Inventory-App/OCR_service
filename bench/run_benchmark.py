"""Evaluación comparativa de motores OCR.

La propuesta lo pide explícitamente:

    "Durante el desarrollo del proyecto se realizará una evaluación comparativa
    entre estas herramientas mediante pruebas sobre datos reales, considerando
    métricas como precisión de reconocimiento, tiempo de respuesta y consumo de
    recursos."

Corre cada motor sobre las mismas imágenes **a través del coordinador**, es
decir con la misma estrategia de intentos que usa la App, y emite una tabla
comparativa: precisión por campo, documentos que superan el umbral, latencia
total y cuánto de esa latencia es el motor.

    python -m bench.run_benchmark --dataset bench/dataset
    python -m bench.run_benchmark --dataset bench/sample_dataset --engines tesseract

`bench/sample_dataset/` es el conjunto sintético de control, versionado en el
repositorio. `bench/dataset/` es el conjunto de trabajo: está vacío en el
repositorio y es donde se colocan las imágenes propias (ver el README).

Requiere que estén levantados los servicios de los motores que se evalúen
(`ocr_service_tesseract`, `ocr_service_easyocr` y `ocr_service_paddleocr`); el
que no responda se reporta como no disponible y se omite, sin tumbar la corrida.
El consumo de memoria ya no se mide desde acá: cada motor
es un contenedor aparte y su memoria se lee con `docker stats`, que es la medida
que importa para decidir el despliegue.
"""

import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from rapidfuzz import fuzz

from app.core.config import settings
from app.engines import factory
from app.parsers.text import fold
from app.schemas.ocr import DocType, OCRResult
from app.services import ocr_service

DEFAULT_DATASET = Path("bench/dataset")
GROUND_TRUTH_FILE = "ground_truth.json"

# Los motores que se evalúan por defecto: los tres que nombra la propuesta más
# el híbrido. Se derivan de la factory y no se listan a mano, porque una lista
# propia se desincroniza en cuanto se agrega un motor —y eso ya pasó: PaddleOCR
# quedó fuera del default y el benchmark dejó de medirlo salvo que se lo pidiera
# explícitamente, mientras el README seguía reportando sus números.
CANDIDATE_ENGINES = factory.VALID_ENGINES

# Un nombre de producto se da por correcto si coincide con esta similitud: el
# motor puede equivocarse en un carácter sin que el campo deje de servir.
NAME_SIMILARITY = 90

FIELD_ACCURACY_TARGET = 0.85
"""Criterio de la propuesta: precisión ≥ 85% en la extracción de campos clave."""

# Cuáles son los campos **clave** de cada tipo de documento. La propuesta fija el
# umbral del 85% sobre ellos —"nombre de producto, fecha de vencimiento y, en
# caso de tickets, cantidad y fecha de compra"— y no sobre todo lo que se
# extrae. `marca` y `detalles` enriquecen el alta pero no la bloquean, así que
# promediarlos con los clave en un solo número esconde el dato que hay que
# mirar: sobre las fotos del proyecto la diferencia entre uno y otro es la
# diferencia entre pasar el criterio y no pasarlo.
#
# Se replican los `KEY_FIELDS` de cada parser en lugar de importarlos porque acá
# hay que nombrar además los ítems del ticket, que no son un campo del documento.
KEY_FIELDS_BY_DOC_TYPE: dict[str, tuple[str, ...]] = {
    "product_label": ("nombre_producto",),
    "expiry_date": ("fecha_vencimiento",),
    "receipt": ("fecha_compra", "items"),
}


@dataclass
class DocumentScore:
    name: str
    doc_type: str
    correct_fields: int = 0
    total_fields: int = 0
    document_confidence: float = 0.0
    elapsed_ms: float = 0.0
    attempts: int = 0
    rotation: int = 0
    error: str | None = None
    missed: list[str] = field(default_factory=list)
    per_field: dict[str, bool] = field(default_factory=dict)

    @property
    def field_accuracy(self) -> float:
        return self.correct_fields / self.total_fields if self.total_fields else 0.0

    def key_field_tally(self) -> tuple[int, int]:
        """Aciertos y total contando **sólo** los campos clave de este documento."""
        key = KEY_FIELDS_BY_DOC_TYPE.get(self.doc_type, ())
        scored = [self.per_field[name] for name in key if name in self.per_field]

        return sum(scored), len(scored)


@dataclass
class EngineReport:
    engine: str
    documents: list[DocumentScore] = field(default_factory=list)

    @property
    def field_accuracy(self) -> float:
        """Precisión sobre **todos** los campos del ground truth, clave o no."""
        correct = sum(d.correct_fields for d in self.documents)
        total = sum(d.total_fields for d in self.documents)
        return correct / total if total else 0.0

    @property
    def key_field_accuracy(self) -> float:
        """Precisión sobre los campos clave: **el criterio de la propuesta**."""
        tallies = [d.key_field_tally() for d in self.documents]
        correct = sum(hits for hits, _ in tallies)
        total = sum(count for _, count in tallies)
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
    def mean_attempts(self) -> float:
        counts = [d.attempts for d in self.documents if d.error is None]
        return statistics.mean(counts) if counts else 0.0

    @property
    def failures(self) -> int:
        return sum(1 for d in self.documents if d.error is not None)

    @property
    def passes_gate(self) -> bool:
        """El umbral del 85% se mide sobre los campos **clave**, como la propuesta."""
        return (
            self.key_field_accuracy >= FIELD_ACCURACY_TARGET
            and self.document_accuracy >= 1.0
        )

    def accuracy_by_field(self) -> dict[str, tuple[int, int]]:
        """Aciertos por nombre de campo.

        La propuesta fija el umbral sobre los **campos clave**, no sobre el
        promedio de todo lo que se extrae: mezclar `nombre_producto` con campos
        accesorios en un solo número esconde justamente el dato que hay que
        mirar.
        """
        tally: dict[str, list[int]] = {}
        for document in self.documents:
            for name, correct in document.per_field.items():
                entry = tally.setdefault(name, [0, 0])
                entry[0] += int(correct)
                entry[1] += 1

        return {name: (hits, total) for name, (hits, total) in tally.items()}


def load_ground_truth(dataset: Path) -> dict:
    path = dataset / GROUND_TRUTH_FILE
    if not path.exists():
        raise SystemExit(
            f"No se encontró {path}.\n"
            f"Colocá las imágenes en {dataset}/ junto a un {GROUND_TRUTH_FILE} "
            f"(ver el README para el formato)."
        )

    return json.loads(path.read_text(encoding="utf-8"))


def score_document(
    result: OCRResult, expected: dict
) -> tuple[int, int, list[str], dict[str, bool]]:
    """Compara contra el ground truth. Devuelve (aciertos, total, fallos, por campo)."""
    correct = 0
    total = 0
    missed: list[str] = []
    per_field: dict[str, bool] = {}

    for name, value in expected.get("fields", {}).items():
        total += 1
        extracted = result.fields.get(name)
        hit = extracted is not None and _matches(extracted.value, value)
        per_field[name] = hit

        if hit:
            correct += 1
        else:
            got = extracted.value if extracted else None
            missed.append(f"{name}: esperaba '{value}', obtuvo '{got}'")

    expected_items = expected.get("items", [])
    if expected_items:
        extracted_items = {
            item.nombre_producto.value: item.cantidad.value for item in result.items
        }
        for item in expected_items:
            total += 1
            match = next(
                (
                    quantity
                    for extracted_name, quantity in extracted_items.items()
                    if _matches(extracted_name, item["nombre_producto"])
                ),
                None,
            )
            hit = match is not None and match == item.get("cantidad")
            per_field["items"] = per_field.get("items", True) and hit

            if hit:
                correct += 1
            else:
                missed.append(f"ítem '{item['nombre_producto']}' no reconocido")

    return correct, total, missed, per_field


def _matches(extracted, expected) -> bool:
    """Compara campos tolerando diferencias de caja, acentos y espacios."""
    if extracted is None:
        return False

    if isinstance(expected, str) and isinstance(extracted, str):
        # El nombre y la marca se comparan por similitud y también por inclusión:
        # el motor suele leer el campo correcto acompañado de una palabra de más.
        left, right = fold(extracted), fold(expected)
        return fuzz.ratio(left, right) >= NAME_SIMILARITY or right in left

    return extracted == expected


def run_engine(engine_name: str, dataset: Path, ground_truth: dict) -> EngineReport:
    report = EngineReport(engine=engine_name)

    factory.reset_cache()
    engine = factory.build_engine(engine_name)

    if not engine.is_available():
        print(f"  ! motor '{engine_name}' no disponible: se omite")
        return report

    for filename, expected in sorted(ground_truth.items()):
        path = dataset / filename
        score = DocumentScore(
            name=filename, doc_type=expected.get("doc_type", "product_label")
        )

        if not path.exists():
            score.error = "imagen faltante"
            report.documents.append(score)
            continue

        try:
            started = time.perf_counter()
            result = ocr_service.process_sync(
                path.read_bytes(), DocType(score.doc_type), engine_name=engine_name
            )
            score.elapsed_ms = (time.perf_counter() - started) * 1000
            (
                score.correct_fields,
                score.total_fields,
                score.missed,
                score.per_field,
            ) = score_document(result, expected)
            score.document_confidence = result.document_confidence
            score.attempts = result.attempts
            score.rotation = result.rotation

        except Exception as exc:  # el benchmark no debe abortar por un documento
            score.error = str(exc)

        report.documents.append(score)
        _print_document(score)

    return report


def _print_document(score: DocumentScore) -> None:
    if score.error:
        print(f"  {score.name:<28} ERROR: {score.error[:80]}")
        return

    print(
        f"  {score.name:<28} campos {score.correct_fields}/{score.total_fields} "
        f"({score.field_accuracy:>5.1%})  conf {score.document_confidence:.2f}  "
        f"rot {score.rotation:>3}  {score.attempts} int  {score.elapsed_ms:>7.0f} ms"
    )
    for miss in score.missed:
        print(f"      · {miss}")


def print_summary(reports: list[EngineReport]) -> None:
    doc_threshold = settings.DOCUMENT_CONFIDENCE_THRESHOLD

    print("\n" + "=" * 92)
    print("COMPARATIVA DE MOTORES OCR")
    print("=" * 92)
    print(
        f"{'Motor':<20}{'Clave':>9}{'Todos':>9}{'Docs >=70%':>13}"
        f"{'Latencia':>13}{'Intentos':>11}{'Fallos':>9}{'Gate':>8}"
    )
    print("-" * 92)

    for report in reports:
        if not report.documents:
            print(f"{report.engine:<20}{'(no disponible)':>50}")
            continue

        print(
            f"{report.engine:<20}"
            f"{report.key_field_accuracy:>8.1%} "
            f"{report.field_accuracy:>8.1%} "
            f"{report.document_accuracy:>12.1%} "
            f"{report.mean_latency_ms:>10.0f} ms"
            f"{report.mean_attempts:>11.1f}"
            f"{report.failures:>9}"
            f"{'  OK' if report.passes_gate else '  NO':>8}"
        )

    print("-" * 92)
    print(
        f"Criterios de la propuesta: precision de campos CLAVE >= "
        f"{FIELD_ACCURACY_TARGET:.0%} | confianza por documento >= {doc_threshold:.0%}"
    )
    print(
        "  Clave = "
        + " | ".join(
            f"{doc_type}: {', '.join(fields)}"
            for doc_type, fields in KEY_FIELDS_BY_DOC_TYPE.items()
        )
    )
    print("  Todos = incluye los accesorios (marca, detalles, total, comercio)")
    print("=" * 92)

    print("\nDETALLE POR CAMPO")
    print("-" * 92)
    for report in reports:
        if not report.documents:
            continue
        detail = "  ".join(
            f"{name}: {hits}/{total} ({hits / total:.0%})"
            for name, (hits, total) in sorted(report.accuracy_by_field().items())
        )
        print(f"{report.engine:<22}{detail}")
    print("=" * 92)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluación comparativa de motores OCR")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--engines",
        nargs="+",
        default=list(CANDIDATE_ENGINES),
        help=f"motores a evaluar (default: {' '.join(CANDIDATE_ENGINES)})",
    )
    parser.add_argument("--json", type=Path, help="guarda el reporte crudo en un archivo JSON")
    args = parser.parse_args()

    dataset = args.dataset
    ground_truth = load_ground_truth(dataset)

    print(f"Dataset: {dataset} ({len(ground_truth)} documentos)")
    for name in factory.CONCRETE_ENGINES:
        print(f"{name + ':':<12}{factory.build_engine(name).base_url}")
    print()

    reports = []
    for engine_name in args.engines:
        print(f"> {engine_name}")
        try:
            reports.append(run_engine(engine_name, dataset, ground_truth))
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
                        "key_field_accuracy": r.key_field_accuracy,
                        "field_accuracy": r.field_accuracy,
                        "document_accuracy": r.document_accuracy,
                        "mean_latency_ms": r.mean_latency_ms,
                        "mean_attempts": r.mean_attempts,
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
