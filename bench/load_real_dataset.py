"""Arma `bench/dataset/` a partir de un directorio de fotos propio.

Las fotos reales del proyecto **no viven en ningún repositorio**: son material de
prueba local y `bench/dataset/` llega vacío al clon. Este script las trae desde
donde estén, que hay que decirle con `--source`: no hay ruta por defecto a
propósito, porque cualquiera que se elija apunta a la máquina de alguien.

    python -m bench.load_real_dataset --source <carpeta> [--dataset bench/dataset]

Espera una carpeta por tipo de material, cada una con su `ground_truth.json`. Los
archivos copiados se prefijan con el nombre de su carpeta porque las carpetas
repiten nombres —la etiqueta y el sello del mismo producto suelen llamarse igual—
y sin prefijo uno pisaría al otro.

Se aceptan los dos formatos de ground truth que existen:

    {"archivo.jpg": {"doc_type": ..., "fields": {...}}}          plano
    {"imagenes": {"archivo.jpg": {"doc_type": ..., "codigo": ...}}}  por campos

El segundo es el que produce la transcripción manual del conjunto y nombra cada
campo directamente; se traduce al primero, que es el que consume el benchmark.

Las imágenes copiadas **no se versionan** (`bench/dataset/*` está en el
`.gitignore`); el ground truth generado sí, para que quede registrado contra qué
se midió.
"""

import argparse
import json
import shutil
from pathlib import Path

DEFAULT_DATASET = Path("bench/dataset")
GROUND_TRUTH_FILE = "ground_truth.json"

# Campos que el benchmark compara, en el orden en que los nombra la propuesta.
CAMPOS = ("nombre_producto", "fecha_vencimiento", "codigo", "fecha_compra",
          "total", "comercio", "marca", "detalles")


def normalizar(entrada: dict) -> dict | None:
    """Lleva una entrada de ground truth al formato que espera el benchmark.

    Una entrada del formato por campos trae `doc_type` y los valores sueltos; el
    benchmark quiere `doc_type` y un diccionario `fields`. Si la entrada ya viene
    en ese formato se devuelve tal cual.
    """
    if "fields" in entrada or "items" in entrada:
        return entrada

    doc_type = entrada.get("doc_type")
    if not doc_type:
        return None

    fields = {
        nombre: entrada[nombre]
        for nombre in CAMPOS
        if entrada.get(nombre) is not None
    }

    if not fields and not entrada.get("items"):
        # Sólo trae el texto visible del envase: sirve para revisar a ojo, pero
        # no para puntuar un campo. Se omite en vez de inventar un valor esperado.
        return None

    normalizada = {"doc_type": doc_type, "fields": fields}
    if entrada.get("items"):
        normalizada["items"] = entrada["items"]

    return normalizada


def leer_verdad(ruta: Path) -> dict:
    datos = json.loads(ruta.read_text(encoding="utf-8"))
    imagenes = datos.get("imagenes", datos)

    return {
        nombre: entrada
        for nombre, entrada in imagenes.items()
        if not nombre.startswith("_") and isinstance(entrada, dict)
    }


def load(source: Path, dataset: Path) -> dict:
    """Copia las imágenes al dataset y devuelve el ground truth combinado."""
    dataset.mkdir(parents=True, exist_ok=True)
    _clear(dataset)

    merged: dict[str, dict] = {}
    omitidas = 0

    for carpeta in sorted(p for p in source.iterdir() if p.is_dir()):
        truth = carpeta / GROUND_TRUTH_FILE
        if not truth.exists():
            print(f"  · {carpeta.name}/ sin {GROUND_TRUTH_FILE}: se omite")
            continue

        prefijo = carpeta.name.lower().replace(" ", "_")

        for name, spec in leer_verdad(truth).items():
            origin = carpeta / name
            if not origin.exists():
                # Un ground truth que nombra una foto que no está no es un error
                # del script: es una foto que falta y hay que avisarlo, porque si
                # no el benchmark la cuenta como fallo del motor.
                print(f"  ! falta {carpeta.name}/{name}, nombrada en el ground truth")
                continue

            normalizada = normalizar(spec)
            if normalizada is None:
                omitidas += 1
                continue

            target = f"{prefijo}_{name}"
            shutil.copy2(origin, dataset / target)
            merged[target] = normalizada

    (dataset / GROUND_TRUTH_FILE).write_text(
        json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    if omitidas:
        print(f"  · {omitidas} entradas sin campo esperado: no se pueden puntuar")

    return merged


def _clear(dataset: Path) -> None:
    """Vacía el dataset para que no queden fotos de una corrida anterior."""
    for path in dataset.iterdir():
        if path.is_file() and path.name != ".gitkeep":
            path.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Arma bench/dataset a partir de un directorio de fotos propio"
    )
    parser.add_argument(
        "--source", type=Path, required=True,
        help="carpeta con las fotos, una subcarpeta por tipo, cada una con su "
             "ground_truth.json",
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    args = parser.parse_args()

    if not args.source.exists():
        raise SystemExit(f"No existe el directorio de fotos: {args.source}")

    print(f"Origen:  {args.source}")
    print(f"Destino: {args.dataset}\n")

    merged = load(args.source, args.dataset)

    if not merged:
        raise SystemExit("No se copió ninguna imagen: revisá --source.")

    print(f"\n{len(merged)} imágenes en {args.dataset} con su ground truth.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
