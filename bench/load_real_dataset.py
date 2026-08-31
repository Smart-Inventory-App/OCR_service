"""Arma `bench/dataset/` a partir del directorio de fotos del proyecto.

Las fotos reales viven fuera del repositorio (en `Repos/imagenes/`), separadas
por tipo de documento y cada carpeta con su propio `ground_truth.json`. El
benchmark, en cambio, espera **un solo** directorio con un ground truth único.
Este script hace esa traducción, que si no hay que rehacer a mano cada vez que se
agregan fotos.

    python -m bench.load_real_dataset
    python -m bench.load_real_dataset --source ../imagenes --dataset bench/dataset

Los nombres se prefijan con el tipo de documento porque las carpetas repiten los
nombres de archivo: la etiqueta y el sello del mismo producto son los dos
`milk.jpeg`, y sin prefijo uno pisaría al otro.

Las imágenes copiadas **no se versionan** (`bench/dataset/*` está en el
`.gitignore`); el ground truth generado sí, para que quede registrado contra qué
se midió.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

DEFAULT_SOURCE = Path("../imagenes")
DEFAULT_DATASET = Path("bench/dataset")
GROUND_TRUTH_FILE = "ground_truth.json"

# Carpeta de origen → prefijo con el que se copian sus imágenes.
FOLDERS = {
    "sample_res_product": "etiqueta",
    "sample_res_product_expiredate": "vencimiento",
    "sample_res_ticket": "ticket",
}


def load(source: Path, dataset: Path) -> dict:
    """Copia las imágenes al dataset y devuelve el ground truth combinado."""
    dataset.mkdir(parents=True, exist_ok=True)
    _clear(dataset)

    merged: dict[str, dict] = {}

    for folder, prefix in FOLDERS.items():
        truth = source / folder / GROUND_TRUTH_FILE
        if not truth.exists():
            print(f"  · {folder}/ no está: se omite")
            continue

        for name, spec in json.loads(truth.read_text(encoding="utf-8")).items():
            origin = source / folder / name
            if not origin.exists():
                # Un ground truth que nombra una foto que no está no es un error
                # del script: es una foto que falta y hay que avisarlo, porque si
                # no el benchmark la cuenta como fallo del motor.
                print(f"  ! falta {folder}/{name}, nombrada en el ground truth")
                continue

            target = f"{prefix}_{name}"
            shutil.copy2(origin, dataset / target)
            merged[target] = spec

    (dataset / GROUND_TRUTH_FILE).write_text(
        json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    return merged


def _clear(dataset: Path) -> None:
    """Vacía el dataset para que no queden fotos de una corrida anterior."""
    for path in dataset.iterdir():
        if path.is_file() and path.name != ".gitkeep":
            path.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Arma bench/dataset a partir del directorio de fotos del proyecto"
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    args = parser.parse_args()

    if not args.source.exists():
        raise SystemExit(f"No existe el directorio de fotos: {args.source}")

    print(f"Origen:  {args.source}")
    print(f"Destino: {args.dataset}\n")

    merged = load(args.source, args.dataset)

    if not merged:
        raise SystemExit("No se cargó ninguna imagen: revisá el origen.")

    print(f"\n{len(merged)} documentos:")
    for name, spec in sorted(merged.items()):
        print(f"  {name:<32} {spec['doc_type']}")

    print("\nYa se puede correr:  python -m bench.run_benchmark --dataset", args.dataset)

    return 0


if __name__ == "__main__":
    sys.exit(main())
