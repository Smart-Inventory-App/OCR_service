# Servicio de OCR — Smart Inventory

**Coordinador** del módulo de **Reconocimiento Óptico de Caracteres**. Lee etiquetas de producto, sellos de vencimiento y comprobantes, y devuelve los campos estructurados que el Core necesita para dar de alta inventario automáticamente, con una **confianza por campo** que indica cuándo hace falta que el usuario confirme el dato.

Este servicio **no ejecuta ningún motor OCR**. Los motores viven en sus propios repositorios y este los coordina:

```
                     ┌──────────────────────────┐
   App ──────────────►      ocr_service         │  puerto 8003
                     │  ──────────────────────  │
                     │  estrategia de intentos  │
                     │  parsing de campos       │
                     │  confianza y umbrales    │
                     │  cola, caché, storage    │
                     └───────┬──────────┬───────┘
                             │          │
              POST /engine/run          POST /engine/run
                             ▼          ▼
              ocr_service_tesseract   ocr_service_easyocr
                    puerto 8005            puerto 8006
                  (rápido, liviano)     (preciso, pesado)
```

| Caso | `doc_type` | Campos que extrae |
|------|-----------|-------------------|
| **A** | `product_label` | `nombre_producto`, `marca`, `detalles` |
| **B** | `expiry_date` | `fecha_vencimiento` |
| **C** | `receipt` | por ítem: `nombre_producto`, `cantidad` · del documento: `fecha_compra`, `total`, `comercio` |

---

## Evolución del Módulo

Esta sección documenta los tres cambios grandes que atravesó el OCR, con lo que los motivó y lo que se midió después de cada uno. Las secciones siguientes describen el estado actual.

### 1. Escaneo: Etiqueta y Vencimiento

**Antes.** Una sola foto de la etiqueta tenía que dar el nombre del producto *y* su fecha de vencimiento. **El problema:** Medido sobre fotos reales de producto, ese caso casi no existe:

- La fecha **no está en la etiqueta**. Está impresa en un canto del envase, en una cara distinta de la que trae el nombre y la marca. Una foto que abarque las dos deja las dos ilegibles.
- Cuando sí entran juntas, **la preparación de imagen que sirve para una arruina la otra**. El logo de una etiqueta se lee mejor sobre escala de grises con buen contraste; el sello de vencimiento, impreso en matriz de puntos sobre una superficie brillante, sólo se lee con **umbral adaptativo** — que es exactamente el filtro que vacía el interior de los trazos gruesos de un logo.

**Ahora.** Son dos tipos de documento con su propia foto, su propia cadena de preparación y su propio parser:

- `product_label` extrae `nombre_producto`, `marca` y `detalles`. Ya no busca fechas.
- `expiry_date` extrae únicamente `fecha_vencimiento`, y puede permitirse ser mucho más exigente: toda la confianza del documento depende de ese campo.

Los criterios de aceptación de la propuesta (`nombre de producto` y `fecha de vencimiento` como campos clave) se siguen cubriendo entre los dos escaneos; lo que cambió es que cada uno se mide y se optimiza por separado.

### 2. Modularización de los Motores OCR

**Antes.** `ocr_service` importaba `pytesseract` y `easyocr`, y su imagen incluía el binario de Tesseract, su paquete de idioma, OpenCV y PyTorch.

**El problema:** Un contenedor de **2,5 GB** que reservaba **~1,3 GB de memoria** para tener EasyOCR cargado, aunque la mayoría de las peticiones se resolvieran con Tesseract en medio segundo. Cada cambio en un parser obligaba a reconstruir y redesplegar todo eso. **Ahora:** Tres repositorios con una división de responsabilidades explícita:

| | `ocr_service` | `ocr_service_tesseract` | `ocr_service_easyocr` |
|---|---|---|---|
| Decide qué documento es | ✅ | ❌ | ❌ |
| Decide giro, filtros y segmentación | ✅ | ❌ | ❌ |
| Decide cuántas veces reintentar | ✅ | ❌ | ❌ |
| Decide cuándo pagar el motor caro | ✅ | ❌ | ❌ |
| Extrae campos y calcula confianza | ✅ | ❌ | ❌ |
| Ejecuta el motor | ❌ | ✅ | ✅ |

Los servicios de motor reciben una imagen y una transformación, y devuelven palabras con su caja y su confianza. No saben qué documento están mirando.

El costo del coordinador después de la mudanza:

| | Antes | Ahora |
|---|---|---|
| Imagen Docker | ~2,5 GB | **~770 MB** |
| Memoria en ejecución | ~1,3 GB | **~30 MB** |
| Dependencias pesadas | Tesseract + OpenCV + PyTorch | ninguna |

Y una ganancia operativa: **el motor caro se enciende sólo cuando se lo usa**. Si `ocr_service_easyocr` está apagado, el coordinador lo reporta como no disponible, el banco de pruebas deshabilita esa opción y el modo híbrido degrada a Tesseract en lugar de fallar.

> El modo **híbrido no es un motor** y por eso no tiene repositorio: es la
> decisión de cuándo vale la pena pagar el motor caro, y esa decisión es del
> coordinador.

### 3. Precisión sobre Fotos Reales de Producto

El conjunto de trabajo son fotos de celular de productos reales, con fondo doméstico, iluminación despareja y envases brillantes y curvos (`bench/sample_res_product/` y `bench/sample_res_product_expiredate/`). Cinco cambios, todos medidos:

**a) Búsqueda de orientación.** Las cuatro fotos de vencimiento del conjunto llegan **giradas un cuarto de vuelta**: el dato está impreso en un canto del envase y no hay forma cómoda de fotografiarlo derecho. Ningún motor lee texto girado 90°. El coordinador prueba las cuatro orientaciones y conserva la que lee.

> Se evaluó usar el detector de orientación de Tesseract (OSD, `--psm 0`) para
> evitar la búsqueda. **No sirve acá:** sobre estas mismas fotos falló en la
> mitad de los casos por "too few characters", y en las que respondió se
> equivocó, con confianza menor a 3. Probar las cuatro sale más barato que
> confiar en él.

**b) Umbral adaptativo para el sello.** Es lo único que levanta la matriz de puntos. Va **sin inversión y sin CLAHE**, al revés que el resto de las variantes: el umbral adaptativo calcula su propia referencia en cada ventana, así que un realce global de contraste antes no le agrega información, le agrega grano — y el grano es indistinguible de los puntos del sello. Con CLAHE encendido, `24 OCT 27` no se lee; sin él, sí.

**c) Reparación de la lectura del sello.** Un sello impreso con puntos produce confusiones sistemáticas que el postproceso corrige donde el contexto no deja lugar a dudas:

| Leído | Real | Cómo se recupera |
|---|---|---|
| `24 GCT 27` | `24 OCT 27` | mes de tres caracteres resuelto por similitud |
| `27FEBZ7` | `27FEB27` | dígitos reparados a los lados de un mes literal |
| `VIO 15/08/2027` | `VTO 15/08/2027` | `T` e `I` son dos palos verticales |

Cada reparación **baja la confianza** del campo: es una lectura recuperada, no una lectura limpia, y el usuario tiene que poder notarlo.

**d) La fecha de elaboración ya no gana.** Un sello real trae las dos fechas juntas (`ENV.: 05/01/26  VTO.: 05/01/27`, `VTO: 15/11/26  ELAB: 19/05/26`). Devolver la primera que aparece da la fecha equivocada la mitad de las veces, así que las anunciadas como elaboración se descartan y, entre las anunciadas como vencimiento, se elige la posterior.

**e) El nombre del producto no es la línea más grande.** Era la heurística anterior y está equivocada: en un envase argentino la tipografía más grande es casi siempre **la marca**.

| Envase | Línea más grande | Nombre real |
|---|---|---|
| Café La Virginia | `LaVirginia` | Café Instantáneo |
| Harina Pureza | `PUREZA` | Harina |
| Café Cabrales | `Cabrales` | Café Tostado Molido |

Ahora el **nombre** es el bloque que nombra una categoría de producto (café, leche, harina, arroz, aceite…), la **marca** es el rótulo grande que queda al sacarlo, y los **detalles** son el contenido y los calificativos (`1 litro`, `250 g`, `Larga Vida`, `Sabor Naranja`). Además:

- Los renglones se agrupan **por geometría y no por orden de lectura**: entre dos líneas de un mismo título el motor cuela renglones de basura de otra zona de la foto, y siguiendo el orden de lectura ese intruso parte el nombre. Es lo que permite recuperar `Leche` + `Entera` → `Leche Entera` y `CAFÉ` + `TOSTADO` + `MOLIDO` → `CAFÉ TOSTADO MOLIDO`.
- Un bloque leído con confianza menor a 0,5 **no se ofrece**. Sobre un envase con fotos y fondo el motor devuelve cosas como `AÚ RARA,` o `MAXanm) sibok`, y ofrecer eso como nombre del producto es peor que no ofrecer nada: sin candidato el campo queda vacío y `needs_review` dispara la validación manual asistida.
- La marca debe ser **tipográficamente prominente** (al menos el 60 % de la letra más grande). Sin ese piso, cuando la marca real no se lee, el campo se llenaba con el primer fragmento de la letra chica legal (`do Nero:`, de "Contenido Neto").

**f) El híbrido decide con el criterio de aceptación, no con un número aparte.** Antes saltaba al motor de respaldo si la confianza global no llegaba a 0,70. Sobre una etiqueta eso falla: Tesseract lee `LaVirginia` con total nitidez y lo ofrece como nombre del producto — confianza alta y campo equivocado. Ahora se salta el respaldo sólo si la lectura **no necesita revisión**, que es la misma condición que decide si el producto se puede dar de alta sin intervención.

Y cuando sí se llama al respaldo, se le ahorra la búsqueda de orientación: se busca el giro con el motor barato —medio segundo por intento— y al caro se le pide **una sola** corrida, ya en la orientación que se sabe buena.

---

## Resultados Medidos - Benchmark

Todo lo que sigue se reproduce con `python -m bench.run_benchmark` (ver [Benchmark](#benchmark--evaluación-comparativa-de-motores)).

### Fotos Reales de Etiqueta — `bench/sample_res_product/` (5 productos)

Fotos de celular con fondo doméstico e iluminación despareja: café instantáneo, café molido, harina, agua saborizada y leche.

| Motor | `nombre_producto` | `marca` | Campos | Docs ≥70% | Latencia |
|-------|-------------------|---------|--------|-----------|----------|
| `tesseract` | 2/5 (40 %) | 0/5 (0 %) | 20 % | 0 % | 2,8 s |
| `easyocr` | 3/5 (60 %) | 2/5 (40 %) | 50 % | 40 % | 4,1 s |
| **`hybrid`** | **4/5 (80 %)** | **2/5 (40 %)** | **60 %** | **40 %** | 7,0 s |

**El campo clave que fija la propuesta es `nombre_producto`, y el híbrido llega a 80 %** — cerca del 85 % exigido, contra el 40 % de Tesseract solo. La `marca` es un campo accesorio y no bloquea el alta.

Los tres casos que fallan y por qué:

| Producto | Qué pasa |
|---|---|
| Agua saborizada *Levité Cero* | logotipo manuscrito sobre botella curva y transparente: el motor devuelve `levilê Yilla dlSur`. No hay parser que recupere eso. |
| Leche *aro* | la marca está impresa **girada 90°** en el lateral del envase; el nombre (`Leche Entera`) se lee perfecto. |
| Café *Cabrales* | la marca cruza el envase en diagonal, sobre un fondo metalizado. |

En los tres el servicio **no inventa un valor**: devuelve `null` y marca `needs_review`, que dispara la validación manual asistida. Es el comportamiento que la propuesta prevé para los casos que la automatización no cubre.

### Fotos Reales del Sello de Vencimiento — `bench/sample_res_product_expiredate/` (4 productos)

Sellos impresos sobre el envase, **los cuatro girados un cuarto de vuelta** y tres de ellos en matriz de puntos.

| Motor | `fecha_vencimiento` | Docs ≥70% | Latencia | Intentos |
|-------|---------------------|-----------|----------|----------|
| `tesseract` | 3/4 (75 %) | 25 % | 6,7 s | 8,0 |
| `easyocr` | 3/4 (75 %) | 25 % | 17,5 s | 3,8 |
| **`hybrid`** | **4/4 (100 %)** | 50 % | 14,2 s | 9,8 |

**El híbrido acierta las cuatro fechas.** Es el mejor argumento a favor del enfoque que la propuesta deja abierto: los motores son complementarios, no redundantes.

| Producto | Sello | Tesseract | EasyOCR |
|---|---|---|---|
| Café La Virginia | `24 OCT 27` | ✅ (leyó `24 GCT 27`) | ❌ |
| Café Cabrales | `VTO.: 05/01/27` | ✅ | ✅ |
| Harina Pureza | `VTO: 15/11/26` | ✅ | ❌ |
| Leche aro | `27FEB27` | ❌ (leyó `27FRRE`) | ✅ (leyó `27FEBZ7`) |

> **Sobre la confianza.** Los valores son correctos, pero sólo 2 de 4 documentos
> superan el umbral de 0,70. No es un defecto del parser: es que el motor mismo
> reporta 0,4–0,7 de confianza sobre una tipografía de puntos, y el scoring **no
> la infla**. El efecto práctico es que estas lecturas llegan a la App
> precargadas y marcadas con `needs_review`, para que el usuario confirme la
> fecha con un toque en lugar de tipearla. Es exactamente el mecanismo de
> "validación manual asistida" que la propuesta prevé para los escenarios donde
> la automatización no alcanza.

> **Sobre la latencia.** Un escaneo de vencimiento cuesta más que uno de etiqueta
> porque puede probar hasta cuatro orientaciones × dos preparaciones. Corta
> apenas una lectura supera el umbral —la harina resuelve en 3 intentos—, y es un
> trabajo asincrónico que la App consulta por *polling*, así que no bloquea la
> interfaz. Se puede desactivar con `ROTATION_SEARCH_ENABLED=false` a costa de
> fallar en las fotos giradas.

### Conjunto Sintético de Control — `bench/sample_dataset/` (9 documentos)

| Motor | Campos | Docs ≥70% | Latencia | Gate |
|-------|--------|-----------|----------|------|
| `tesseract` | 100 % | 88,9 % | 742 ms | NO |
| `easyocr` | 100 % | 88,9 % | 1331 ms | NO |
| **`hybrid`** | **100 %** | **100 %** | 1016 ms | **OK** |

El conjunto sintético cubre los tres tipos de documento e incluye un sello girado 90° y un sello con las dos fechas juntas. **El híbrido pasa el gate completo.** Sirve para verificar que el pipeline funciona de punta a punta; no reemplaza al dataset real, porque una imagen generada no tiene el ruido ni la iluminación de una foto.

### Consumo de Recursos

Medido con `docker stats` sobre los contenedores en reposo y en proceso:

| Servicio | Imagen | Memoria en reposo | Latencia por documento |
|---|---|---|---|
| `ocr_service` | ~770 MB | ~30 MB | — (coordina) |
| `ocr_service_tesseract` | ~1,05 GB | ~70 MB | 0,2–0,8 s por intento |
| `ocr_service_easyocr` | ~2,58 GB | ~450 MB | 3–5 s por intento |

---

## Funcionamiento

El pipeline tiene cinco etapas. La 3 es la única que ocurre fuera de este servicio.

```
   imagen (bytes)
        │
   1 ▸ validar        services/image_loader.py    descarga por URL o recibe el binario
        │                                          comprueba que sea una imagen
        ▼
   2 ▸ planificar     services/ocr_service.py     qué giro, filtros y segmentación
        │             pipeline/transform.py        probar, y en qué orden
        ▼
   3 ▸ reconocer      engines/remote.py           ──HTTP──► servicio del motor
        │                                          devuelve palabras + caja + confianza
        ▼
   4 ▸ limpiar        pipeline/postprocess.py     corrige O→0, Z→2, repara sellos
        │
        ▼
   5 ▸ parsear        parsers/                    regex + heurísticas → campos ISO
        │             parsers/scoring.py          confianza por campo y por documento
        ▼
   OCRResult { fields, items, document_confidence, needs_review, rotation, attempts }
```

### 1. Estrategia de Intentos

Ninguna combinación de giro y filtros gana en todas las imágenes. En vez de elegir una a ciegas se prueban unas pocas **en orden de mejor apuesta** y se conserva la de mayor `document_confidence`; si una alcanza el umbral, no se gasta tiempo en las demás. Es el camino de "reconfiguración del pipeline de procesamiento" que la propuesta prevé como plan de acción.

| `doc_type` | Intentos | Por qué |
|---|---|---|
| `product_label` | 3: filtros base con segmentación de etiqueta, con segmentación automática, y binarizada con Otsu | la foto viene derecha; lo que varía es cómo tratar el envase |
| `expiry_date` | 8: cuatro giros × (umbral adaptativo, sin binarizar) | el sello llega girado y es matriz de puntos |
| `receipt` | 3: filtros base en bloque, con Otsu, y con segmentación automática | el ticket viene derecho y es texto plano |

Un motor que declara `supports_preprocessing: false` recibe **un solo intento por orientación**: prepara su propia entrada y variarle los filtros no cambiaría nada, pero cada corrida cuesta segundos.

### 2. Motores Detrás de la Interfaz

Todo el servicio depende sólo de la interfaz `OCREngine` (`engines/base.py`):

```python
class OCREngine(Protocol):
    name: str
    supports_preprocessing: bool
    def run(self, content: bytes, lang: str, layout: str,
            transform: Transform | None) -> RawOCROutput: ...
    def is_available(self) -> bool: ...
```

- **`tesseract`** — rápido y liviano. Da **confianza por palabra**, que es la
  base del cálculo de `document_confidence`.
- **`easyocr`** — basado en deep learning. Mucho mejor en envases curvos,
  brillantes y con tipografías de fantasía; mucho más lento y pesado.
- **`hybrid`** — el "enfoque híbrido" que la propuesta no descarta. **Es el que
  mejor mide.**

Cambiar de motor es una variable de entorno:

```env
OCR_ENGINE=hybrid
```

También se puede elegir **por petición**, que es lo que usa el banco de pruebas para compararlos:

```jsonc
{ "image_url": "...", "doc_type": "expiry_date", "engine": "hybrid" }
```

> **Por qué el híbrido gana.** Los motores son **complementarios, no
> redundantes**. Sobre el sello de vencimiento del café, Tesseract acierta y
> EasyOCR no; sobre el de la leche, al revés. Sobre la etiqueta de la harina,
> Tesseract no levanta nada y EasyOCR la lee entera. Combinarlos recupera lo
> mejor de cada uno — que es precisamente el argumento con que la propuesta deja
> abierta esta opción.

### 3. Parsing

- **Etiqueta** (`parsers/product_label.py`) — nombre por categoría de producto, marca por prominencia tipográfica, detalles por contenido y calificativos. Ver [Evolución § 3.e](#3-precisión-sobre-fotos-reales-de-producto).
- **Vencimiento** (`parsers/expiry.py` + `parsers/dates.py`) — formatos argentinos habituales: `15/08/2027`, `15-08-27`, `2027-08-15`, `15 AGO 2027`, `27FEB27`, `08/2027`, y las palabras que las anuncian (`VTO`, `VENCE`, `CONS. PREF.`, `Consumir preferentemente antes de`, `BEST BEFORE`). Todo se normaliza a ISO `YYYY-MM-DD`; si el formato sólo trae mes y año se asume el último día del mes.
- **Ticket** (`parsers/receipt.py`) — segmenta líneas de ítem separando primero el importe final, luego la cantidad (`2 x Leche`, `Leche x 2`, `3 Yogur`). Una línea es ítem **sólo si trae precio o cantidad explícita**: así los encabezados y los pies (`TOTAL`, `CUIT`, `FECHA`) no se confunden con productos.

### 4. Confianza y `needs_review`

Los dos umbrales son **exactamente** los criterios de aceptación de la propuesta, y son configurables:

```env
FIELD_CONFIDENCE_THRESHOLD=0.85      # precisión ≥ 85% en campos clave
DOCUMENT_CONFIDENCE_THRESHOLD=0.70   # precisión global por documento ≥ 70%
```

- **`field.confidence` = confianza del motor × confianza del parser.** Se multiplica (y no se promedia) porque ambas condiciones son necesarias: un texto leído con nitidez pero que no matchea el patrón esperado no es un campo confiable, y viceversa. El parser penaliza señales concretas: un vencimiento sin palabra clave, un mes reparado por similitud, un día inferido, una fecha en el pasado, un nombre que no nombra ninguna categoría conocida.
- **`document_confidence`** — promedio ponderado de los campos. `nombre_producto` y las fechas pesan 1,0; `marca`, `comercio` y `total` pesan 0,5; `detalles`, 0,25.
- **`needs_review`** — `true` si algún **campo clave** queda bajo 0,85, o si un ticket no arrojó ningún ítem. Los campos clave son `nombre_producto` para la etiqueta y `fecha_vencimiento` para el sello; `marca` y `detalles` enriquecen el alta pero no la bloquean, porque son datos que el usuario completa de memoria y exigirlos mandaría a revisión etiquetas perfectamente leídas.

`raw_text`, `rotation` y `attempts` viajan siempre en la respuesta, para auditoría desde el backoffice y para depurar la estrategia.

---

## Encaje en la Arquitectura

Procesamiento **asincrónico** para evitar bloqueos en la experiencia de usuario:

```
App (React Native)
   │  1) POST /ocr/scan  (la foto, multipart)
   ▼
ocr_service ──2) POST /storage/temp──► image_storage_service ──► Cloudinary (bazaar/ocr-temp)
   │           ◄── { url, expires_at } ──┘
   │  3) encola el job en Redis y responde 202 { job_id }
   ▼
worker  descarga la imagen → planifica intentos → llama al motor → parsea → puntúa
   │    guarda el resultado en Redis (TTL 24 h)
   │    4) DELETE /storage/temp  ← descarta la imagen, siempre
   ▼
App    5) GET /ocr/jobs/{job_id} → { status, result, needs_review }
   │
   ▼
app_backend  persiste el inventario en PostgreSQL
backoffice_backend  valida/monitorea los datos OCR cargados
```

**La imagen es efímera.** Sólo existe en `bazaar/ocr-temp` mientras el OCR trabaja; el worker la descarta al terminar, haya salido bien o mal. Si el worker se cae, el TTL del storage la purga.

### Puertos

| Servicio | Puerto |
|----------|--------|
| `app_backend` (Core) | `8000` |
| `auth` | `8001` |
| `backoffice_backend` | `8002` |
| **`ocr_service`** | **`8003`** |
| `image_storage_service` | `8004` |
| `ocr_service_tesseract` | `8005` |
| `ocr_service_easyocr` | `8006` |

---

## Stack

- Python 3.11 + FastAPI
- Redis — cola de trabajos y caché de resultados
- Pillow — sólo para validar que el binario recibido sea una imagen
- `python-dateutil` / `rapidfuzz` — normalización de fechas y similitud de texto
- JWT HS256 — valida los tokens del servicio Auth (mismo `JWT_SECRET`)

**Ni OpenCV, ni pytesseract, ni easyocr, ni PyTorch.** Este servicio coordina, parsea y puntúa; no reconoce.

---

## Endpoints

Todos los `/ocr/*` requieren `Authorization: Bearer <token>`, excepto `livez`, `readyz` y `engines`.

### `POST /ocr/scan` — Punto de Entrada de la App

Sube la foto como imagen temporal, encola el job y la descarta al terminar.

**Body:** `multipart/form-data`

| Campo | Tipo | Requerido | Default |
|-------|------|-----------|---------|
| `file` | archivo (`jpeg`/`png`/`webp`) | sí | — |
| `doc_type` | `product_label` \| `expiry_date` \| `receipt` | no | `product_label` |
| `lang` | string | no | `spa` |
| `engine` | `tesseract` \| `easyocr` \| `hybrid` | no | el de `OCR_ENGINE` |

**Respuesta `202`:** `{ "job_id": "9f2c1a...", "status": "queued" }`

**Errores:** `400` formato no soportado, archivo vacío o binario que no es una imagen · `401` token inválido · `502` el storage no respondió · `503` cola no disponible.

> **El flujo de la App son dos escaneos.** Primero la etiqueta
> (`doc_type=product_label`) para el nombre, la marca y los detalles; después,
> si el producto tiene vencimiento, la foto del sello
> (`doc_type=expiry_date`). Son dos llamadas independientes: no hay que mandar
> las dos imágenes juntas, y un producto sin fecha simplemente no hace la
> segunda.

### `POST /ocr/jobs` — Encolar una Imagen Ya Subida

Para cuando el Core ya tiene la URL de Cloudinary.

```jsonc
{
  "image_url": "https://res.cloudinary.com/.../bazaar/ocr-temp/abc.jpg",
  "doc_type": "expiry_date",     // "product_label" | "expiry_date" | "receipt"
  "lang": "spa",                 // opcional
  "engine": "hybrid",            // opcional
  "discard_after": true          // opcional: descartar la imagen al terminar
}
```

### `GET /ocr/jobs/{job_id}` — Estado y Resultado

La App hace *polling* contra este endpoint.

```jsonc
{
  "job_id": "9f2c1a...",
  "status": "done",                 // queued | processing | done | failed
  "doc_type": "product_label",
  "result": {
    "fields": {
      "nombre_producto": { "value": "LECHE ENTERA",  "confidence": 0.96 },
      "marca":           { "value": "La Serenisima", "confidence": 0.88 },
      "detalles":        { "value": "1 litro",       "confidence": 0.95 }
    },
    "items": [],                    // poblado sólo en doc_type=receipt
    "document_confidence": 0.93,    // criterio ≥ 0.70
    "needs_review": false,          // true → la App pide confirmación al usuario
    "engine": "tesseract",          // el motor cuya lectura se conservó
    "rotation": 0,                  // giro con el que se leyó
    "attempts": 1,                  // combinaciones que hizo falta probar
    "raw_text": "La Serenisima\nLECHE ENTERA\n1 litro",
    "processing_ms": 229
  },
  "error": null,
  "created_at": "2026-08-05T21:14:03+00:00",
  "updated_at": "2026-08-05T21:14:04+00:00"
}
```

Para `doc_type=expiry_date`, `fields` trae únicamente `fecha_vencimiento`. Para `doc_type=receipt`, `items` es una lista de `{ "nombre_producto": {...}, "cantidad": {...} }` y `fields` trae `fecha_compra`, `total` y `comercio`.

**Errores:** `404` job inexistente o expirado · `503` cola no disponible.

> El token del llamador se guarda en una clave de Redis **separada** del estado
> del job (se necesita para descartar la imagen temporal) y se borra en cuanto el
> job termina: nunca sale por la API.

### `POST /ocr/sync` — Versión Sincrónica (URL)

Mismo body que `/ocr/jobs`, pero procesa y devuelve el `result` en la misma respuesta. Para el benchmark, el backoffice y debugging. **No usar desde la App:** bloquea la UX.

### `POST /ocr/sync/upload` — Versión Sincrónica (binario)

Igual, con `multipart/form-data`. Es la vía **standalone**: ejercita el pipeline completo sin Cloudinary y sin Redis.

### `GET /ocr/engines` — Estado de los Motores

Público. Lo usa el banco de pruebas para no ofrecer un motor apagado.

```json
[
  { "name": "tesseract", "url": "http://tpp_ocr_tesseract:8005", "available": true,
    "version": "5.5.0", "languages": ["eng","osd","spa"], "supports_preprocessing": true },
  { "name": "easyocr", "url": "http://tpp_ocr_easyocr:8006", "available": true,
    "version": "1.7.2", "languages": ["eng","spa"], "supports_preprocessing": false }
]
```

### `GET /ocr/livez` · `GET /ocr/readyz`

- `livez` → `{ "status": "ok" }`. Sin dependencias.
- `readyz` → verifica Redis **y** que el motor configurado responda. Se comprueba
  el motor configurado, no todos: que EasyOCR esté apagado no debería marcar el
  servicio como caído si el que se usa es Tesseract.

---

## Instalación y Ejecución

Los tres servicios se comunican por la red externa `tpp-net`. Si no existe:

```bash
docker network create tpp-net
```

**1. Levantar al menos un motor.**

```bash
cd ../ocr_service_tesseract && cp .env.example .env && docker-compose up -d --build
```

Opcionalmente, el motor pesado (la primera build descarga PyTorch y tarda varios
minutos):

```bash
cd ../ocr_service_easyocr && cp .env.example .env && docker-compose up -d --build
```

**2. Levantar el coordinador** (API + worker + Redis):

```bash
cp .env.example .env
```

```bash
docker-compose up -d --build
```

**3. Verificar:**

```bash
curl http://localhost:8003/ocr/readyz
```

```bash
curl http://localhost:8003/ocr/engines
```

Swagger en `http://localhost:8003/docs`. Logs del worker con `docker-compose logs -f worker`.

> `JWT_SECRET` en `.env` **debe** ser el mismo que usa el servicio `auth`, o
> todos los `/ocr/*` responderán `401`.

### Sin Docker

Este servicio no tiene dependencias de sistema, así que corre directo:

```bash
python -m venv .venv && .venv\Scripts\activate && pip install -r requirements.txt
```

Apuntar los motores, Redis y el storage a `localhost` en el `.env`:

```env
TESSERACT_SERVICE_URL=http://localhost:8005
EASYOCR_SERVICE_URL=http://localhost:8006
REDIS_URL=redis://localhost:6379/0
STORAGE_SERVICE_URL=http://localhost:8004
```

```bash
uvicorn app.main:app --reload --port 8003
```

```bash
python -m app.worker
```

> **Sin Redis** la API arranca igual y `POST /ocr/sync/upload` funciona
> perfectamente: la caché es opcional y el modo sincrónico no usa la cola. Sólo
> los endpoints asincrónicos devolverán `503`.

---

## Banco de Pruebas en el Navegador

```
http://localhost:8003/ocr/playground
```

Página servida por el propio servicio. Replica el patrón de autenticación del backoffice: se inicia sesión **una sola vez** contra el servicio Auth, el token queda en `localStorage` y se adjunta automáticamente en cada consulta; si el servidor responde `401`, la sesión se cierra y vuelve al login.

Permite subir una imagen y elegir:

- el **tipo de documento** — los tres, con una ayuda que explica qué foto espera cada uno;
- el **motor** — `tesseract`, `easyocr` o `hybrid`. La tabla **Motores** muestra cuáles están arriba y con qué versión, y **deshabilita en el selector los que están apagados**, para que no se pueda pedir un motor que no va a correr;
- el **modo** — sincrónico o asincrónico, este último con *polling* real del job.

El resultado muestra los campos con barras de confianza coloreadas según los umbrales (verde ≥ 0,85, ámbar ≥ 0,70, rojo por debajo) y, además de la confianza y el tiempo, **el motor cuya lectura se conservó, el giro aplicado y cuántos intentos hicieron falta**. En modo híbrido el motor que figura puede no ser el que se pidió: es la forma de ver cuál de los dos ganó de verdad.

Es la forma práctica de reproducir la evaluación comparativa: se sube la misma foto tres veces cambiando sólo el motor y se contrastan `document_confidence`, `processing_ms` y el texto crudo. La caché distingue por motor, así que comparar no devuelve resultados cacheados.

> Sólo se sirve con `DEBUG=true`. Con `DEBUG=false` devuelve `404` y no figura en
> el OpenAPI: es una herramienta de desarrollo, no parte del contrato.

Si el servicio Auth no corre en `http://localhost:8001`, ajustá `AUTH_PUBLIC_URL` en el `.env` (es la URL tal como la ve el **navegador**, no el hostname interno de la red de Docker).

### Desde la Terminal

```bash
curl -X POST http://localhost:8003/ocr/sync/upload -H "Authorization: Bearer $TOKEN" -F "file=@etiqueta.jpg" -F "doc_type=product_label" -F "engine=hybrid"
```

```bash
curl -X POST http://localhost:8003/ocr/sync/upload -H "Authorization: Bearer $TOKEN" -F "file=@vencimiento.jpg" -F "doc_type=expiry_date"
```

Flujo asincrónico completo (sube, encola y descarta la imagen):

```bash
curl -X POST http://localhost:8003/ocr/scan -H "Authorization: Bearer $TOKEN" -F "file=@ticket.jpg" -F "doc_type=receipt"
```

```bash
curl http://localhost:8003/ocr/jobs/<job_id> -H "Authorization: Bearer $TOKEN"
```

---

## Tests

```bash
pytest
```

**202 tests, 93 % de cobertura.** No requieren los servicios de motor ni un Redis real: los motores se sustituyen por dobles con confianza controlada (y por `respx` cuando se verifica el contrato HTTP), y Redis por `fakeredis`. Lo que sí se ejercita de verdad es todo lo que este servicio aporta.

| Archivo | Qué cubre |
|---------|-----------|
| `test_strategy.py` | los planes de intentos por tipo de documento y la orquestación del híbrido |
| `test_engines.py` | el contrato HTTP con los motores, la factory y el modo híbrido |
| `test_dates.py` | todos los formatos de fecha y su normalización a ISO |
| `test_parsers.py` | etiquetas, sellos, tickets y el cálculo de confianza |
| `test_routes.py` | los endpoints, el ciclo de vida de la imagen temporal, auth, caché |
| `test_clients.py` | clientes HTTP y el loop del worker |
| `test_acceptance.py` | **el *gate* del módulo**: precisión ≥ 85% por campo y ≥ 70% por documento |

Los dos repositorios de motor tienen sus propias suites (66 y 36 tests), que cubren la cadena de OpenCV y la traducción de la salida de cada motor.

---

## Benchmark — Evaluación Comparativa de Motores

La propuesta lo pide explícitamente:

> "Durante el desarrollo del proyecto se realizará una evaluación comparativa
> entre estas herramientas mediante pruebas sobre datos reales, considerando
> métricas como precisión de reconocimiento, tiempo de respuesta y consumo de
> recursos."

```bash
python -m bench.run_benchmark --dataset bench/sample_res_product
```

```bash
python -m bench.run_benchmark --dataset bench/sample_res_product_expiredate
```

```bash
python -m bench.run_benchmark --dataset bench/sample_dataset --engines tesseract
```

Corre cada motor sobre las mismas imágenes **a través del coordinador**, es decir con la misma estrategia de intentos que usa la App. Requiere los servicios de motor levantados.

El consumo de memoria ya no se mide desde acá: cada motor es un contenedor aparte y su memoria se lee con `docker stats`, que es la medida que importa para decidir el despliegue.

### Leer la Salida

```
Motor                     Campos   Docs >=70%     Latencia   Intentos   Fallos    Gate
tesseract                 75.0%        25.0%       6513 ms        8.0        0      NO
easyocr                   75.0%        25.0%      15223 ms        3.8        0      NO
hybrid                   100.0%        50.0%      12512 ms        9.5        0      NO

DETALLE POR CAMPO
hybrid                fecha_vencimiento: 4/4 (100%)
```

- **Campos** — precisión en campos clave vs. ground truth → criterio **≥ 85 %**.
- **Docs ≥70%** — documentos que alcanzan la confianza mínima → criterio **≥ 70 %**.
- **Intentos** — cuántas combinaciones de giro y filtros hizo falta probar.
- **Detalle por campo** — la propuesta fija el umbral sobre los campos *clave*; mezclarlos con los accesorios en un solo número escondería el dato que importa.
- **Gate** — si el motor cumple ambos criterios de aceptación.

### Dataset Propio

Las imágenes reales del proyecto no se versionan. Colocá las tuyas en `bench/dataset/` junto a un `ground_truth.json`:

```json
{
  "etiqueta_leche.jpg": {
    "doc_type": "product_label",
    "fields": { "nombre_producto": "LECHE ENTERA", "marca": "La Serenisima" }
  },
  "vencimiento_leche.jpg": {
    "doc_type": "expiry_date",
    "fields": { "fecha_vencimiento": "2027-08-15" }
  },
  "ticket_super_01.jpg": {
    "doc_type": "receipt",
    "fields": { "fecha_compra": "2026-07-20", "total": 5240.50 },
    "items": [
      { "nombre_producto": "Leche Entera 1L", "cantidad": 2 },
      { "nombre_producto": "Pan Lactal", "cantidad": 1 }
    ]
  }
}
```

Los nombres se comparan con similitud difusa (`rapidfuzz`, ≥ 90) o por inclusión: un carácter mal leído, o una palabra de más, no invalidan el campo. Fechas, cantidades y totales se comparan exacto.

### Sin Fotos a Mano

```bash
python -m bench.make_sample_dataset
```

Genera el conjunto sintético de `bench/sample_dataset/` con su ground truth: cuatro etiquetas (una inclinada), tres sellos de vencimiento (uno girado 90°, uno con las dos fechas juntas) y dos tickets. Necesita una tipografía TrueType en el sistema.

### PaddleOCR

La propuesta también nombra PaddleOCR. Con la arquitectura actual, incorporarlo **no toca este repositorio**: alcanza con crear un `ocr_service_paddleocr` que exponga el mismo contrato (`POST /engine/run`, `GET /engine/info`, `/livez`, `/readyz`) y agregar su URL a la configuración. La interfaz `OCREngine` no cambia, y el modo híbrido lo puede tomar como principal o como respaldo.

---

## Configuración

Todas las variables, con sus defaults, están en [`.env.example`](.env.example). Las que más importan:

| Variable | Default | Para qué |
|----------|---------|----------|
| `JWT_SECRET` | — | **debe coincidir con el del servicio Auth** |
| `TESSERACT_SERVICE_URL` | `http://tpp_ocr_tesseract:8005` | dónde vive el motor liviano |
| `EASYOCR_SERVICE_URL` | `http://tpp_ocr_easyocr:8006` | dónde vive el motor pesado |
| `ENGINE_API_KEY` | *(vacía)* | clave compartida con los motores |
| `ENGINE_TIMEOUT_SECONDS` | `120` | EasyOCR tarda segundos por documento |
| `OCR_ENGINE` | `hybrid` | `tesseract` \| `easyocr` \| `hybrid` |
| `FIELD_CONFIDENCE_THRESHOLD` | `0.85` | umbral por campo → dispara `needs_review` |
| `DOCUMENT_CONFIDENCE_THRESHOLD` | `0.70` | umbral por documento |
| `HYBRID_FALLBACK_THRESHOLD` | `0.70` | confianza bajo la cual se llama al respaldo |
| `RETRY_ENABLED` / `RETRY_THRESHOLD` | `true` / `0.75` | estrategia de reintentos |
| `ROTATION_SEARCH_ENABLED` | `true` | búsqueda de orientación en el sello |
| `STAMP_MIN_WIDTH` / `STAMP_MAX_WIDTH` | `2400` / `3600` | cuánto se amplía el sello |
| `ENGINE_MAX_WIDTH` / `ENGINE_STAMP_MAX_WIDTH` | `800` / `1280` | resolución que se le pide a un motor que prepara su propia entrada |
| `PREPROCESS_*` | ver `.env.example` | cada paso de la cadena para etiquetas y tickets |
| `REDIS_URL` | `redis://redis:6379/0` | cola y caché |
| `STORAGE_SERVICE_URL` | `http://tpp_storage_service:8000` | subir/descartar la imagen temporal |

---

## Notas de implementación

**La cola usa listas de Redis (`RPUSH`/`BLPOP`), no RQ.** RQ depende de `fork()`, que no existe en Windows, y habría obligado a un camino distinto en desarrollo y en producción. Con listas de Redis el mismo código corre igual en las dos partes, sin dependencias extra, y sigue siendo "Redis como cola de trabajos" como plantea la propuesta. La interfaz (`enqueue`, `dequeue`, `get_job`) queda intacta si más adelante se quiere migrar a RQ, arq o Celery.

**La orquestación corre en un thread aparte.** Un escaneo de vencimiento son hasta ocho llamadas HTTP bloqueantes encadenadas: `ocr_service.process()` las ejecuta con `asyncio.to_thread` para no trabar el event loop, ni el de FastAPI ni el del worker. `process_sync()` expone la versión bloqueante, que es la que usa el benchmark.

**Caché por hash de imagen.** Antes de procesar se busca `sha256(imagen) + doc_type + lang + motor` en Redis. Reprocesar la misma foto (reintento de la App, doble tap) sale de la caché. El motor forma parte de la clave para que comparar motores no devuelva un resultado cacheado. Si Redis no está, se saltea sin fallar.

**Un motor caído no se disfraza de lectura vacía.** Si ningún intento llegó a correr porque el servicio del motor no responde, el error se propaga como `502` en vez de devolver un resultado sin campos, que el usuario podría confundir con una foto ilegible.

**Un solo camino de código.** `/ocr/sync` y el worker comparten `ocr_service.process()`; los tres parsers comparten `scoring.py` y `parsers/text.py`; los dos motores comparten el contrato de `engines/base.py`. No hay lógica de OCR duplicada.
