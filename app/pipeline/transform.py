"""Descripción de la preparación que se le pide a un motor.

Es **declarativa a propósito**: acá no hay OpenCV. Este servicio decide *qué*
transformación conviene para cada tipo de documento y en qué orden probar las
variantes; la ejecuta el servicio del motor, que es donde vive OpenCV.

Esa división es la que permite que el coordinador siga siendo liviano (no
instala ni Tesseract ni PyTorch ni OpenCV) sin perder el control de la
estrategia, que es donde está la inteligencia del módulo.
"""

from dataclasses import asdict, dataclass, replace

from app.core.config import settings

# Modos de binarización que entiende el contrato de los motores.
NONE = "none"
OTSU = "otsu"
ADAPTIVE = "adaptive"


@dataclass(frozen=True)
class Transform:
    """Cadena de preparación pedida al motor, más el giro de la imagen."""

    rotate: int = 0
    grayscale: bool = True
    rescale: bool = True
    min_width: int = 1600
    max_width: int = 3000
    invert: bool = True
    contrast: bool = True
    denoise: bool = False
    deskew: bool = True
    binarize: str = NONE

    def turned(self, degrees: int) -> "Transform":
        return replace(self, rotate=degrees % 360)

    def binarized(self, mode: str) -> "Transform":
        return replace(self, binarize=mode)

    def as_payload(self) -> dict:
        return asdict(self)


def label() -> Transform:
    """Preparación para etiquetas y tickets, configurable por `.env`.

    Es la cadena que hace rendir a Tesseract sobre fotos reales: grises, upscale,
    inversión si el envase es oscuro y CLAHE para la iluminación despareja.
    """
    return Transform(
        grayscale=settings.PREPROCESS_GRAYSCALE,
        rescale=settings.PREPROCESS_UPSCALE,
        min_width=settings.PREPROCESS_MIN_WIDTH,
        max_width=settings.PREPROCESS_MAX_WIDTH,
        invert=settings.PREPROCESS_INVERT,
        contrast=settings.PREPROCESS_CONTRAST,
        denoise=settings.PREPROCESS_DENOISE,
        deskew=settings.PREPROCESS_DESKEW,
        binarize=NONE,
    )


def stamp() -> Transform:
    """Preparación para el sello de vencimiento impreso sobre el envase.

    Se amplía más que una etiqueta (el sello ocupa una franja diminuta del
    cuadro) y **no se endereza**: el deskew estima el ángulo sobre todo el
    encuadre y en una foto que es casi toda envase liso se equivoca, girando el
    único renglón que importa.
    """
    return Transform(
        grayscale=True,
        rescale=True,
        min_width=settings.STAMP_MIN_WIDTH,
        max_width=settings.STAMP_MAX_WIDTH,
        invert=True,
        contrast=True,
        denoise=False,
        deskew=False,
        binarize=NONE,
    )


def stamp_adaptive() -> Transform:
    """La variante que rescata la tipografía de matriz de puntos.

    Un sello de vencimiento no es texto impreso: son puntos separados sobre una
    superficie brillante. Medido sobre las fotos reales, el umbral **adaptativo**
    es el único que los levanta — y es justo el filtro que hay que mantener
    apagado para leer una etiqueta, donde vacía el interior de los trazos
    gruesos de los logos. Que la misma cadena no sirva para los dos casos es una
    de las razones por las que el escaneo se partió en dos.

    Va **sin inversión y sin CLAHE**, al revés que todas las demás variantes, y
    también está medido: el umbral adaptativo calcula su propia referencia en
    cada ventana, así que un realce global de contraste antes no le agrega
    información —le agrega grano, y el grano es indistinguible de los puntos del
    sello—. Con CLAHE encendido, `24 OCT 27` no se lee; sin él, sí.
    """
    return replace(stamp(), binarize=ADAPTIVE, invert=False, contrast=False)


def geometric(max_width: int | None = None) -> Transform:
    """Sólo giro y acotado, sin filtros.

    Es lo que se le manda a un motor que declara `supports_preprocessing=False`:
    prepara su propia entrada y variarle los filtros no mejora nada. Lo que sí
    se le sigue pidiendo es la resolución, porque cuánto detalle necesita el
    documento lo sabe el coordinador y no el motor.
    """
    return Transform(
        grayscale=False,
        rescale=True,
        min_width=0,
        max_width=max_width or settings.ENGINE_MAX_WIDTH,
        invert=False,
        contrast=False,
        denoise=False,
        deskew=False,
        binarize=NONE,
    )
