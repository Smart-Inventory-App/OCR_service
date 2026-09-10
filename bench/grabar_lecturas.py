"""Regraba las lecturas de `tests/fixtures/lecturas/` desde las fotos reales.

`tests/test_fotos_reales.py` corre los parsers y el scoring contra la salida que
los motores devolvieron **de verdad** sobre unas pocas fotos del conjunto de
prueba, y no contra los motores: levantarlos cuesta minutos por documento y el
CI corre a propósito sin ellos. Este script es el que produce esas lecturas.

    docker compose up -d                       # los tres motores arriba
    python -m bench.grabar_lecturas --source <carpeta de fotos>

Las fotos **no viven en ningún repositorio** —son material de prueba local— así
que hay que decir dónde están. `--source` apunta a la carpeta raíz y cada caso
declara su ruta relativa. Lo que sí se versiona es el JSON resultante: texto,
palabras con caja y confianza, y qué motor ganó el intento.

Regrabar cambia la vara contra la que miden los tests, así que sólo tiene sentido
cuando cambia un motor o la estrategia de intentos, y conviene mirar el diff: si
una lectura empeoró, el test que la usa lo va a decir, pero el diff explica por
qué.
"""

import argparse
import json
from dataclasses import asdict
from datetime import date
from pathlib import Path

from app.schemas.ocr import DocType
from app.services import ocr_service

DESTINO = Path("tests/fixtures/lecturas")

# El "hoy" con el que se grabaron. Los tests fijan el mismo, para que un
# vencimiento plausible hoy no deje de serlo dentro de dos años.
HOY = date(2026, 9, 8)

# Una foto por cosa que puede salir mal, no una muestra representativa: para eso
# está el benchmark, que corre sobre el conjunto entero.
CASOS: tuple[tuple[str, str, DocType], ...] = (
    # nombre de la lectura        ruta relativa a --source          tipo
    # la marca impresa más grande que el nombre
    ("etiqueta_cafe", "Almacén Casa 2/1F.jpeg", DocType.PRODUCT_LABEL),
    # el nombre es la monodroga y la marca sale ilegible
    ("etiqueta_farmacia", "Farmacia/10F.jpg", DocType.PRODUCT_LABEL),
    # matriz de puntos en el canto de una botella, formato MM/AA
    ("vencimiento_sello", "Almacén/10V.jpg", DocType.EXPIRY_DATE),
    # `27FEB27`: sin separadores y con una letra mal leída
    ("vencimiento_girado", "Almacén Casa 2/4V.jpeg", DocType.EXPIRY_DATE),
    # factura B real, con su pie fiscal
    ("ticket_dia", "Tickets/dia_1.jpg", DocType.RECEIPT),
    # remito sin precios: los ítems se reconocen por la cantidad
    ("remito", "Documentos Generados/remito_02.jpg", DocType.RECEIPT),
)


def grabar(origen: Path, destino: Path) -> None:
    destino.mkdir(parents=True, exist_ok=True)

    for nombre, relativa, doc_type in CASOS:
        foto = origen / relativa
        if not foto.exists():
            print(f"[falta] {relativa}")
            continue

        output, texto, resultado = _leer(foto, doc_type)
        archivo = destino / f"{nombre}.json"
        archivo.write_text(
            json.dumps(
                {
                    "imagen": relativa,
                    "doc_type": doc_type.value,
                    "motor": output.engine,
                    "text": texto,
                    "words": [asdict(palabra) for palabra in output.words],
                },
                ensure_ascii=False,
                indent=1,
            ),
            encoding="utf-8",
        )
        print(
            f"{nombre:20} {doc_type.value:14} {output.engine:10} "
            f"conf={resultado.document_confidence:.3f} "
            f"palabras={len(output.words):3} {archivo.stat().st_size / 1024:.0f} KB"
        )


def _leer(foto: Path, doc_type: DocType):
    """Procesa la foto y devuelve la salida cruda del intento **ganador**.

    `process_sync` sólo devuelve el resultado ya parseado, así que se intercepta
    `build_result` —por donde pasa cada intento— y después se identifica cuál de
    los intentos es el que quedó.
    """
    original = ocr_service.build_result
    vistos: list[tuple] = []

    def espia(output, text, tipo, today=None):
        resultado = original(output, text, tipo, today)
        vistos.append((output, text, resultado))
        return resultado

    ocr_service.build_result = espia
    try:
        resultado = ocr_service.process_sync(
            foto.read_bytes(), doc_type, today=HOY, engine_name="hybrid"
        )
    finally:
        ocr_service.build_result = original

    output, texto, _ = next(
        candidato
        for candidato in vistos
        if candidato[2].raw_text == resultado.raw_text
        and candidato[2].document_confidence == resultado.document_confidence
    )

    return output, texto, resultado


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", required=True, type=Path, help="carpeta raíz con las fotos"
    )
    parser.add_argument("--destino", type=Path, default=DESTINO)
    args = parser.parse_args()

    grabar(args.source, args.destino)


if __name__ == "__main__":
    main()
