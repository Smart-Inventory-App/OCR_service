"""Genera un dataset sintético de etiquetas y tickets, con su ground truth.

Sirve para probar el pipeline OCR real (OpenCV + Tesseract) y el benchmark sin
tener que juntar fotos primero. **No reemplaza al dataset real**: la propuesta
exige medir la precisión "mediante pruebas sobre datos reales", y una imagen
generada por computadora no tiene el ruido, la iluminación ni las tipografías de
una foto de góndola. Úsalo para verificar que todo funciona de punta a punta.

    python -m bench.make_sample_dataset
    python -m bench.run_benchmark --dataset bench/sample_dataset
"""

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

DEFAULT_OUT = Path("bench/sample_dataset")

# Se prueba en orden hasta encontrar una tipografía disponible en el sistema.
_FONT_CANDIDATES = (
    ("C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/arialbd.ttf"),
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
     "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    ("/Library/Fonts/Arial.ttf", "/Library/Fonts/Arial Bold.ttf"),
)


def _font_paths() -> tuple[str, str] | None:
    for regular, bold in _FONT_CANDIDATES:
        if Path(regular).exists():
            return regular, bold if Path(bold).exists() else regular
    return None


class _Renderer:
    def __init__(self) -> None:
        self.paths = _font_paths()
        if self.paths is None:
            raise SystemExit(
                "No se encontró una tipografía TrueType en el sistema.\n"
                "Instalá DejaVu (Linux: fonts-dejavu-core) o usá un dataset real."
            )

    def font(self, size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
        return ImageFont.truetype(self.paths[1] if bold else self.paths[0], size)

    def render(
        self,
        path: Path,
        lines: list[tuple[str, int, bool]],
        size: tuple[int, int] = (1000, 620),
        rotate: int = 0,
    ) -> None:
        image = Image.new("RGB", size, "white")
        draw = ImageDraw.Draw(image)

        y = 40
        for text, font_size, bold in lines:
            draw.text((50, y), text, fill="black", font=self.font(font_size, bold))
            y += font_size + 26

        if rotate:
            image = image.rotate(rotate, expand=True, fillcolor="white")

        image.save(path, quality=95)


# (archivo, líneas, tamaño, rotación, ground truth)
DOCUMENTS = [
    (
        "etiqueta_leche.jpg",
        [
            ("LECHE ENTERA 1L", 62, True),
            ("La Serenisima", 34, False),
            ("VTO 15/08/2027", 40, False),
            ("LOTE A2291", 28, False),
        ],
        (1000, 620),
        0,
        {
            "doc_type": "product_label",
            "fields": {"nombre_producto": "LECHE ENTERA 1L", "fecha_vencimiento": "2027-08-15"},
        },
    ),
    (
        "etiqueta_yogur.jpg",
        [
            ("Industria Argentina", 26, False),
            ("YOGUR FIRME FRUTILLA", 58, True),
            ("CONS. PREF. 01/12/2026", 36, False),
            ("Peso neto 190g", 26, False),
        ],
        (1000, 620),
        0,
        {
            "doc_type": "product_label",
            "fields": {
                "nombre_producto": "YOGUR FIRME FRUTILLA",
                "fecha_vencimiento": "2026-12-01",
            },
        },
    ),
    (
        "etiqueta_arroz.jpg",
        [
            ("ARROZ LARGO FINO 1KG", 56, True),
            ("VENCE 20/03/2028", 38, False),
            ("SIN TACC", 26, False),
        ],
        (1000, 560),
        0,
        {
            "doc_type": "product_label",
            "fields": {
                "nombre_producto": "ARROZ LARGO FINO 1KG",
                "fecha_vencimiento": "2028-03-20",
            },
        },
    ),
    (
        # Inclinada a propósito: ejercita el paso de deskew del pipeline.
        "etiqueta_aceite_inclinada.jpg",
        [
            ("ACEITE DE GIRASOL 900ML", 54, True),
            ("VTO 30/09/2027", 38, False),
        ],
        (1000, 460),
        6,
        {
            "doc_type": "product_label",
            "fields": {
                "nombre_producto": "ACEITE DE GIRASOL 900ML",
                "fecha_vencimiento": "2027-09-30",
            },
        },
    ),
    (
        "ticket_super.jpg",
        [
            ("SUPERMERCADO LA ESQUINA", 44, True),
            ("CUIT 30-12345678-9", 30, False),
            ("FECHA 20/07/2026", 34, False),
            ("2 x Leche Entera 1L      2.400,00", 34, False),
            ("1 x Pan Lactal           1.850,50", 34, False),
            ("3 Yogur Frutilla           990,00", 34, False),
            ("TOTAL                    5.240,50", 38, True),
        ],
        (1100, 720),
        0,
        {
            "doc_type": "receipt",
            "fields": {"fecha_compra": "2026-07-20", "total": 5240.50},
            "items": [
                {"nombre_producto": "Leche Entera 1L", "cantidad": 2},
                {"nombre_producto": "Pan Lactal", "cantidad": 1},
                {"nombre_producto": "Yogur Frutilla", "cantidad": 3},
            ],
        },
    ),
    (
        "ticket_almacen.jpg",
        [
            ("ALMACEN DON JOSE", 44, True),
            ("FECHA 18/07/2026", 34, False),
            ("1 Aceite Girasol 900ml   3.100,00", 34, False),
            ("2 x Arroz Largo Fino     2.500,00", 34, False),
            ("TOTAL                    5.600,00", 38, True),
        ],
        (1100, 620),
        0,
        {
            "doc_type": "receipt",
            "fields": {"fecha_compra": "2026-07-18", "total": 5600.00},
            "items": [
                {"nombre_producto": "Aceite Girasol 900ml", "cantidad": 1},
                {"nombre_producto": "Arroz Largo Fino", "cantidad": 2},
            ],
        },
    ),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    renderer = _Renderer()

    ground_truth = {}
    for filename, lines, size, rotate, expected in DOCUMENTS:
        renderer.render(args.out / filename, lines, size, rotate)
        ground_truth[filename] = expected

    (args.out / "ground_truth.json").write_text(
        json.dumps(ground_truth, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"{len(ground_truth)} documentos generados en {args.out}")
    print(f"Probalos con: python -m bench.run_benchmark --dataset {args.out}")


if __name__ == "__main__":
    main()
