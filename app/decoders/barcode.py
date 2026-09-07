"""Decodificación de códigos de barras.

Es la única forma de reconocimiento que corre **dentro** del coordinador, y la
excepción está justificada: un código de barras no se lee reconociendo
caracteres, se decodifica una simbología. El resultado no depende de qué motor
OCR esté configurado —un EAN-13 leído con `engine=paddleocr` y con
`engine=tesseract` tiene que dar el mismo número o algo está mal—, así que
repartir el decodificador entre los tres servicios de motor lo triplicaría y
haría que la respuesta dependiera de una elección que no debería influirla.

El peso también es otro: ZBar es una librería C de unos cientos de KB sobre la
Pillow que este servicio ya tenía, no los 2 GB de OpenCV + PyTorch que motivaron
sacar los motores a sus propios repositorios.

## Qué hace falta y qué no — medido

Sobre códigos generados y compuestos sobre fondos de foto (`tests/test_barcode.py`
reproduce las mediciones):

- **No hace falta buscar la orientación.** Al revés que el sello de vencimiento,
  que necesita las cuatro rotaciones, ZBar recorre la imagen en varias
  direcciones por su cuenta: el mismo código se decodifica a 0°, 90°, 180°, 270°
  y también inclinado 7°. Agregar la búsqueda de giro sería pagar cuatro pasadas
  para obtener exactamente el mismo resultado.
- **No hace falta realzar el contraste.** ZBar binariza internamente. Un código
  al 18 % de contraste y otro al 30 % de brillo se decodifican directo, y pasarlos
  antes por escala de grises con autocontraste no cambia nada.
- **Sí hace falta ampliar.** Es la única variable que mueve la aguja. Con el
  código ocupando 200 px de ancho o más, decodifica directo; en 180 px falla
  directo y lo recupera la ampliación al doble; en 150 px o menos no lo recupera
  nada, porque la información ya no está en la foto.
- **Sí ayuda enfocar, en una banda angosta.** Con un desenfoque gaussiano de
  radio 2,0 la máscara de enfoque es lo único que lo rescata. A partir de 2,6 no
  hay filtro que lo recupere.

Los cuatro umbrales están fijados como tests (`test_barcode.py::TestEscalera`),
así que si una versión de ZBar los corre, la suite lo dice en vez de que el
número quede viejo en este comentario.

De ahí salen los cuatro intentos, en orden de mejor apuesta y más barato
primero. El primero resuelve el caso normal en ~150 ms sobre una foto de 12 MP;
los demás sólo corren si el anterior no leyó nada.
"""

import io
import logging
import time
from dataclasses import dataclass, field

from PIL import Image, ImageFilter, UnidentifiedImageError

from app.core.config import settings

logger = logging.getLogger(__name__)

NAME = "zbar"

# Simbologías que se le piden al decodificador. Se acota a propósito en vez de
# aceptar todo lo que ZBar sabe leer: cada simbología habilitada es una pasada
# más por la imagen, y las que quedan afuera (CODABAR, DATABAR, PDF417, EAN-2/5)
# no aparecen en un envase ni en una etiqueta de envío del circuito argentino.
#
# Lo que sí entra cubre los cuatro casos reales:
#   · EAN-13 / EAN-8      el envase de góndola argentino
#   · UPC-A / UPC-E       producto importado o encargo de EE.UU.
#   · I25                 el ITF-14 de la caja de un pedido mayorista
#   · CODE-128 / CODE-39  etiqueta logística de un envío
#   · QR                  cada vez más frecuente en packaging y comprobantes
SYMBOLOGIES = (
    "EAN13",
    "EAN8",
    "UPCA",
    "UPCE",
    "I25",
    "CODE128",
    "CODE39",
    "QRCODE",
)


class DecoderUnavailable(RuntimeError):
    """ZBar no está instalado o su librería de sistema falta en la imagen."""


@dataclass(frozen=True)
class Symbol:
    """Un símbolo decodificado, con su simbología y su caja."""

    data: str
    symbology: str
    left: int = 0
    top: int = 0
    width: int = 0
    height: int = 0
    quality: int = 0


@dataclass
class BarcodeReading:
    """Resultado de recorrer la escalera de intentos."""

    symbols: list[Symbol] = field(default_factory=list)
    attempts: int = 0
    variant: str = ""
    """Qué intento resolvió la lectura, para auditoría desde el backoffice."""
    elapsed_ms: int = 0

    @property
    def found(self) -> bool:
        return bool(self.symbols)


def is_available() -> bool:
    """True si se puede decodificar de verdad.

    No alcanza con que el paquete de Python importe: `pyzbar` es un envoltorio
    sobre `libzbar`, que es una librería del sistema. Si falta, el import
    revienta con `ImportError` al buscar el .so, y eso es exactamente lo que hay
    que reportar como "no disponible" en vez de dejarlo estallar en la primera
    petición.
    """
    try:
        _pyzbar()
        return True
    except DecoderUnavailable:
        return False


def version() -> str:
    try:
        import pyzbar

        return getattr(pyzbar, "__version__", "desconocida")
    except Exception:
        return "no instalado"


def decode(content: bytes) -> BarcodeReading:
    """Decodifica los códigos de la imagen, probando las variantes necesarias.

    Se corta en la primera variante que encuentra algo: las siguientes son más
    caras y no van a encontrar un código distinto, sino el mismo o ninguno.
    """
    started = time.perf_counter()
    pyzbar = _pyzbar()

    try:
        with Image.open(io.BytesIO(content)) as opened:
            image = opened.convert("L")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError("El archivo no es una imagen válida") from exc

    reading = BarcodeReading()

    for name, build in _variants():
        candidate = build(image)
        if candidate is None:
            # La variante no aplica a esta imagen (por ejemplo, ampliar una foto
            # que ya supera el tope de píxeles). No se cuenta como intento.
            continue

        reading.attempts += 1
        symbols = _scan(pyzbar, candidate)

        if symbols:
            reading.symbols = symbols
            reading.variant = name
            break

    reading.elapsed_ms = int((time.perf_counter() - started) * 1000)

    logger.info(
        "Códigos de barras: %d símbolo(s) en %d intento(s) por '%s' (%d ms)",
        len(reading.symbols),
        reading.attempts,
        reading.variant or "ninguna variante",
        reading.elapsed_ms,
    )

    return reading


def _scan(pyzbar, image: Image.Image) -> list[Symbol]:
    """Una pasada del decodificador sobre una variante de la imagen."""
    symbols = pyzbar.decode(image, symbols=_wanted_symbols(pyzbar))

    decoded: list[Symbol] = []
    for symbol in symbols:
        try:
            data = symbol.data.decode("utf-8").strip()
        except UnicodeDecodeError:
            # Un QR puede traer bytes que no son texto. No es un dato que este
            # módulo pueda ofrecer como código de producto: se descarta.
            logger.warning("Símbolo %s con contenido no textual: se descarta", symbol.type)
            continue

        if not data:
            continue

        decoded.append(
            Symbol(
                data=data,
                symbology=str(symbol.type),
                left=int(symbol.rect.left),
                top=int(symbol.rect.top),
                width=int(symbol.rect.width),
                height=int(symbol.rect.height),
                quality=int(getattr(symbol, "quality", 0) or 0),
            )
        )

    return decoded


# ── La escalera de intentos ───────────────────────────────────────────────────


def _variants():
    """Las variantes a probar, de la más barata a la más cara.

    Cada una devuelve `None` cuando no aplica, y ahí se saltea sin contarse: no
    tiene sentido registrar como intento fallido una ampliación que nunca se hizo.
    """
    yield "directo", lambda image: image
    yield "ampliada", _upscaled

    if settings.BARCODE_SHARPEN_ENABLED:
        yield "enfocada", _sharpened
        yield "ampliada+enfocada", lambda image: _sharpened(_upscaled(image))


def _upscaled(image: Image.Image | None) -> Image.Image | None:
    """Amplía la imagen, salvo que el resultado se pase del tope de píxeles.

    El tope existe por memoria: ampliar al doble una foto de 12 MP da 48 MP, y
    aunque en escala de grises eso son ~48 MB y no los ~145 MB que costaría en
    color, sigue siendo un pico que conviene acotar. Se mide en píxeles y no en
    ancho porque lo que se quiere limitar es la memoria, que depende de los dos
    lados.
    """
    if image is None:
        return None

    factor = settings.BARCODE_UPSCALE_FACTOR
    if factor <= 1:
        return None

    target = (image.width * factor, image.height * factor)

    if target[0] * target[1] > settings.BARCODE_MAX_PIXELS:
        logger.debug(
            "Ampliación omitida: %dx%d supera el tope de %d píxeles",
            *target,
            settings.BARCODE_MAX_PIXELS,
        )
        return None

    # LANCZOS conserva el borde entre barra y espacio, que es lo único que el
    # decodificador mira. Un remuestreo bilineal lo difumina justo en la banda
    # donde la ampliación tenía que servir.
    return image.resize(target, Image.LANCZOS)


def _sharpened(image: Image.Image | None) -> Image.Image | None:
    """Máscara de enfoque, para el desenfoque leve de una foto a pulso."""
    if image is None:
        return None

    return image.filter(ImageFilter.UnsharpMask(radius=2, percent=200, threshold=2))


# ── Acceso a ZBar ─────────────────────────────────────────────────────────────


def _pyzbar():
    """Importa `pyzbar` traduciendo su falla a un error propio del dominio."""
    try:
        from pyzbar import pyzbar

        return pyzbar
    except Exception as exc:  # ImportError, y el OSError de la lib de sistema
        raise DecoderUnavailable(
            "El decodificador de códigos de barras no está disponible: "
            f"falta `pyzbar` o la librería de sistema `libzbar` ({exc})"
        ) from exc


def _wanted_symbols(pyzbar) -> list:
    """Traduce `SYMBOLOGIES` a los enums de ZBar, salteando los que no existan."""
    return [
        getattr(pyzbar.ZBarSymbol, name)
        for name in SYMBOLOGIES
        if hasattr(pyzbar.ZBarSymbol, name)
    ]
