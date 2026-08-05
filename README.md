# Servicio de OCR — Smart Inventory

Microservicio de **Reconocimiento Óptico de Caracteres**. Lee etiquetas de producto y comprobantes (tickets, remitos, facturas) y devuelve los campos estructurados que el Core necesita para dar de alta inventario automáticamente, con una **confianza por campo** que indica cuándo hace falta que el usuario confirme el dato.

Cubre los siguientes casos de entrada:

| Caso | Documento | Campos que extrae |
|------|-----------|-------------------|
| **A** | Etiqueta de producto | `nombre_producto`, `fecha_vencimiento` |
| **B** | Ticket / remito / factura | por ítem: `nombre_producto`, `cantidad` · del documento: `fecha_compra`, `total`, `comercio` |

---

## Funcionamiento del OCR

El pipeline tiene cinco etapas. Cada una vive en su propio módulo y se puede probar por separado.

```
   imagen (bytes)
        │
   1 ▸ cargar          services/image_loader.py    descarga por URL o recibe el binario
        │                                          decodifica a matriz BGR (OpenCV)
        ▼
   2 ▸ preprocesar     pipeline/preprocess.py      grises → upscale → denoise → deskew → binarizar
        │
        ▼
   3 ▸ reconocer       engines/                    Tesseract (o EasyOCR / híbrido)
        │                                          devuelve palabras + caja + confianza
        ▼
   4 ▸ limpiar         pipeline/postprocess.py     corrige O → 0, l → 1 en tokens numéricos
        │
        ▼
   5 ▸ parsear         parsers/                    regex + heurísticas → campos ISO
        │              parsers/scoring.py          confianza por campo y por documento
        ▼
   OCRResult { fields, items, document_confidence, needs_review, raw_text }
```

### 1. Preprocesamiento (OpenCV)

Tesseract requiere preprocesamiento (OpenCV) para obtener buenos resultados con tipografías variadas, fondos complejos, baja iluminación o **imágenes inclinadas**. La cadena aplica, en orden:

| Paso | Qué hace | Razón |
|------|----------|---------|
| `to_grayscale` | BGR → 1 canal | el color no aporta al reconocimiento de texto |
| `upscale` | lleva el ancho a ≥ 1000 px | Tesseract rinde mejor con ~300 DPI equivalentes |
| `denoise` | `bilateralFilter` | quita ruido **preservando** el borde de los caracteres |
| `deskew` | estima el ángulo con `minAreaRect` y rota | corrige fotos inclinadas |
| `threshold` | `adaptiveThreshold` gaussiano | robusto ante iluminación desigual |

Cada paso se activa o desactiva por `.env` (`PREPROCESS_*`), así el benchmark puede medir cuánto aporta realmente cada uno.

### 2. Motor Detrás de la Interfaz

Todo el servicio depende sólo de la interfaz `OCREngine` (`engines/base.py`):

```python
class OCREngine(Protocol):
    name: str
    def run(self, image: np.ndarray, lang: str = "spa", layout: str = "block") -> RawOCROutput: ...
    def is_available(self) -> bool: ...
```

- **`tesseract`** — motor del MVP. Usa `image_to_data`, que da **confianza por palabra**: es la base del cálculo de `document_confidence`.
- **`easyocr`** — alternativa a evaluar y respaldo del modo híbrido. Su import es **diferido**: la dependencia no está en el deploy y el servicio arranca igual (`is_available()` devuelve `False`).
- **`hybrid`** — el "enfoque híbrido" que la propuesta no descarta: corre Tesseract y, si la confianza media queda bajo `HYBRID_FALLBACK_THRESHOLD`, reintenta con el respaldo y conserva la mejor de las dos lecturas.

`layout` es una pista agnóstica del motor: `sparse` para etiquetas (texto disperso), `block` para tickets (columna única). Tesseract la traduce a su *page segmentation mode*.

Cambiar de motor es una variable de entorno, sin tocar el resto del código:

```env
OCR_ENGINE=hybrid
```

### 3. Parsing

- **Fechas** (`parsers/dates.py`) — formatos argentinos habituales: `15/08/2027`, `15-08-27`, `15.08.2027`, `2027-08-15`, `15 AGO 2027`, `08/2027`, y las etiquetas que las anuncian (`VTO`, `VENCE`, `CONS. PREF.`, `Consumir preferentemente antes de`, `BEST BEFORE`). Todo se normaliza a ISO `YYYY-MM-DD`. Si el formato sólo trae mes y año se asume el **último día del mes**.
- **Etiqueta** (`parsers/product_label.py`) — el nombre es la línea con **mayor altura de caja** (la tipografía más grande de la etiqueta), descartando líneas de ruido (`LOTE`, `CUIT`, `SIN TACC`, fechas). El vencimiento prioriza la fecha precedida por palabra clave y, si no hay, la primera no vencida.
- **Ticket** (`parsers/receipt.py`) — segmenta líneas de ítem separando primero el importe final, luego la cantidad (`2 x Leche`, `Leche x 2`, `3 Yogur`). Una línea es ítem **sólo si trae precio o cantidad explícita**: así los encabezados y los pies (`TOTAL`, `CUIT`, `FECHA`) no se confunden con productos.

### 4. Confianza y `needs_review`

Los dos umbrales son **exactamente** los criterios de aceptación y son configurables:

```env
FIELD_CONFIDENCE_THRESHOLD=0.85      # precisión ≥ 85% en campos clave
DOCUMENT_CONFIDENCE_THRESHOLD=0.70   # precisión global por documento ≥ 70%
```

- **`field.confidence` = confianza del motor × confianza del parser.** Se multiplica (y no se promedia) porque ambas condiciones son necesarias: un texto leído con nitidez pero que no matchea el patrón esperado no es un campo confiable, y viceversa. El parser penaliza señales concretas: un vencimiento sin palabra clave, un día inferido, una fecha de vencimiento en el pasado (suele ser la de elaboración), una cantidad asumida en 1.
- **`document_confidence`** — promedio ponderado de los campos clave (los nombres y fechas pesan el doble que `total` y `comercio`).
- **`needs_review`** — `true` si algún campo clave queda bajo 0.85, o si un ticket no arrojó ningún ítem. Es el disparador de la **validación manual asistida** que pide la propuesta: la App muestra los campos precargados para que el usuario confirme o corrija.

`raw_text` viaja siempre en la respuesta, para auditoría desde el backoffice.

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
worker  descarga la imagen → preprocesa → OCR → parsea → puntúa
   │    guarda el resultado en Redis (TTL 24 h)
   │    4) DELETE /storage/temp  ← descarta la imagen, siempre
   ▼
App    5) GET /ocr/jobs/{job_id} → { status, result, needs_review }
   │
   ▼
app_backend  persiste el inventario en PostgreSQL
backoffice_backend  valida/monitorea los datos OCR cargados
```

**La imagen es efímera.** Sólo existe en `bazaar/ocr-temp` mientras el OCR le aplica sus filtros; el worker la descarta al terminar, haya salido bien o mal. Si el worker se cae, el TTL del storage la purga (`POST /storage/temp/purge`). Ver el README de `image_storage_service`.

### Puertos

| Servicio | Puerto |
|----------|--------|
| `app_backend` (Core) | `8000` |
| `auth` | `8001` |
| `backoffice_backend` | `8002` |
| **`ocr_service`** | **`8003`** |
| `image_storage_service` | `8004` |

---

## Stack

- Python 3.11 + FastAPI
- OpenCV (`opencv-python-headless`) — preprocesamiento
- Tesseract (`pytesseract`) + paquete de idioma **español**
- Redis — cola de trabajos y caché de resultados
- `python-dateutil` / `rapidfuzz` — normalización de fechas y nombres
- JWT HS256 — valida los tokens del servicio Auth (mismo `JWT_SECRET`)

---

## Endpoints

Todos los `/ocr/*` requieren `Authorization: Bearer <token>`, excepto `livez` y `readyz`.

### `POST /ocr/scan` — punto de entrada de la App

Sube la foto como imagen temporal, encola el job y la descarta al terminar.

**Body:** `multipart/form-data`

| Campo | Tipo | Requerido | Default |
|-------|------|-----------|---------|
| `file` | archivo (`jpeg`/`png`/`webp`) | sí | — |
| `doc_type` | `product_label` \| `receipt` | no | `product_label` |
| `lang` | string | no | `spa` |

**Respuesta `202`:**
```json
{ "job_id": "9f2c1a...", "status": "queued" }
```

**Errores:** `400` formato no soportado o archivo vacío · `401` token inválido · `502` el storage no respondió · `503` cola no disponible.

---

### `POST /ocr/jobs` — encolar una imagen ya subida

Para cuando el Core ya tiene la URL de Cloudinary.

**Body:** `application/json`
```jsonc
{
  "image_url": "https://res.cloudinary.com/.../bazaar/ocr-temp/abc.jpg",
  "doc_type": "product_label",   // "product_label" | "receipt"
  "lang": "spa",                 // opcional
  "discard_after": true          // opcional: descartar la imagen al terminar
}
```

**Respuesta `202`:** `{ "job_id": "9f2c1a...", "status": "queued" }`

---

### `GET /ocr/jobs/{job_id}` — estado y resultado

La App hace *polling* contra este endpoint.

```jsonc
{
  "job_id": "9f2c1a...",
  "status": "done",                 // queued | processing | done | failed
  "doc_type": "product_label",
  "result": {
    "fields": {
      "nombre_producto":   { "value": "Leche La Serenísima 1L", "confidence": 0.92 },
      "fecha_vencimiento": { "value": "2027-08-15",             "confidence": 0.88 }
    },
    "items": [],                    // poblado sólo en doc_type=receipt
    "document_confidence": 0.90,    // criterio ≥ 0.70
    "needs_review": false,          // true → la App pide confirmación al usuario
    "engine": "tesseract",
    "raw_text": "LECHE ENTERA 1L\nVTO 15/08/2027",
    "processing_ms": 412
  },
  "error": null,
  "created_at": "2026-07-29T20:14:03+00:00",
  "updated_at": "2026-07-29T20:14:04+00:00"
}
```

Para `doc_type=receipt`, `items` es una lista de `{ "nombre_producto": {...}, "cantidad": {...} }` y `fields` trae `fecha_compra`, `total` y `comercio`.

**Errores:** `404` job inexistente o expirado · `503` cola no disponible.

> El token del llamador se guarda en una clave de Redis **separada** del estado del job (se necesita
> para descartar la imagen temporal) y se borra en cuanto el job termina: nunca sale por la API.

---

### `POST /ocr/sync` — versión sincrónica (URL)

Mismo body que `/ocr/jobs`, pero procesa y devuelve el `result` en la misma respuesta. Para el benchmark, el backoffice y debugging. **No usar desde la App:** bloquea la UX.

### `POST /ocr/sync/upload` — versión sincrónica (binario)

Igual, con `multipart/form-data`. Es la vía **standalone**: ejercita el pipeline completo sin Cloudinary y sin Redis.

```json
{ "doc_type": "product_label", "result": { "...": "..." } }
```

### `GET /ocr/livez` · `GET /ocr/readyz`

- `livez` → `{ "status": "ok" }`. Sin dependencias.
- `readyz` → verifica Redis **y** que el motor OCR se pueda ejecutar.
  `503` con el detalle de qué falló:

```json
{ "status": "error", "checks": { "redis": false, "engine": true }, "engine_name": "tesseract" }
```

---

## Instalación y Ejecución

### Opción A — Docker (recomendada)

Es la única vía que trae **Tesseract y el idioma español ya instalados**.

La red `tpp-net` es externa y compartida con el resto de los servicios. Si no existe todavía:

```bash
docker network create tpp-net
```

Configurar el entorno y levantar (API + worker + Redis):

```bash
cp .env.example .env
```

```bash
docker-compose up -d --build
```

Verificar:

```bash
curl http://localhost:8003/ocr/readyz
```

Swagger en `http://localhost:8003/docs`. Logs del worker:

```bash
docker-compose logs -f worker
```

> `JWT_SECRET` en `.env` **debe** ser el mismo que usa el servicio `auth`, o todos los `/ocr/*`
> responderán `401`.

---

### Opción B — Local

**1. Instalar Tesseract** (no se instala con pip).

- **Windows:** descargar el instalador de [UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki), marcando el idioma **Spanish** en los componentes. Después, en `.env`:

  ```env
  TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
  ```

- **Linux:** `sudo apt-get install -y tesseract-ocr tesseract-ocr-spa`
- **macOS:** `brew install tesseract tesseract-lang`

Comprobar que el idioma está: `tesseract --list-langs` debe incluir `spa`. Si falta, el servicio cae a inglés y lo avisa por log (no falla).

**2. Instalar dependencias:**

```bash
python -m venv .venv
```

```bash
.venv\Scripts\activate
```

```bash
pip install -r requirements.txt
```

**3. Configurar:**

```bash
cp .env.example .env
```

Para correr sin Docker, apuntar Redis y el storage a `localhost`:

```env
REDIS_URL=redis://localhost:6379/0
STORAGE_SERVICE_URL=http://localhost:8004
```

**4. Levantar la API:**

```bash
uvicorn app.main:app --reload --port 8003
```

**5. Levantar el worker** (otra terminal — es el que procesa la cola):

```bash
python -m app.worker
```

> **Sin Redis** la API arranca igual y `POST /ocr/sync/upload` funciona perfectamente: la caché es
> opcional y el modo sincrónico no usa la cola. Sólo los endpoints asincrónicos devolverán `503`.
> Es la forma más rápida de probar el OCR sin levantar nada más.

---

## Pruebas y Ejecución

### Banco de Pruebas en el Navegador (recomendado)

```
http://localhost:8003/ocr/playground
```

Página de prueba servida por el propio servicio. Replica el patrón de autenticación del backoffice: se inicia sesión **una sola vez** contra el servicio Auth, el token queda en `localStorage` y se adjunta automáticamente en cada consulta; si el servidor responde `401`, la sesión se cierra y vuelve al login, igual que el interceptor de axios del backoffice.

Permite subir una imagen con el explorador de archivos, elegir el tipo de documento y el modo (sincrónico o asincrónico, este último con *polling* real del job), y muestra los campos con barras de confianza coloreadas según los umbrales: verde ≥ 0.85, ámbar ≥ 0.70, rojo por debajo. También expone el texto crudo y el JSON completo de la respuesta.

> Sólo se sirve con `DEBUG=true`. Con `DEBUG=false` devuelve `404` y no figura en el OpenAPI:
> es una herramienta de desarrollo, no parte del contrato del servicio.

Si el servicio Auth no corre en `http://localhost:8001`, ajustá `AUTH_PUBLIC_URL` en el `.env` (es la URL tal como la ve el **navegador**, no el hostname interno de la red de Docker).

### Desde la Terminal

Con el servicio arriba y un token válido del servicio `auth`:

```bash
curl -X POST http://localhost:8003/ocr/sync/upload -H "Authorization: Bearer $TOKEN" -F "file=@etiqueta.jpg" -F "doc_type=product_label"
```

Flujo asincrónico completo (sube, encola y descarta la imagen):

```bash
curl -X POST http://localhost:8003/ocr/scan -H "Authorization: Bearer $TOKEN" -F "file=@ticket.jpg" -F "doc_type=receipt"
```

```bash
curl http://localhost:8003/ocr/jobs/<job_id> -H "Authorization: Bearer $TOKEN"
```

### Sin Fotos a Mano

Se puede generar un dataset sintético (etiquetas y tickets, uno inclinado a propósito) con su ground truth:

```bash
python -m bench.make_sample_dataset
```

Queda en `bench/sample_dataset/`. Sirve para verificar que todo funciona de punta a punta, pero **no reemplaza al dataset real**: una imagen generada no tiene el ruido ni la iluminación de una foto.

---

## Tests

```bash
pytest
```

**159 tests, 92% de cobertura.** No requieren Tesseract ni Redis: el motor se sustituye por un doble con confianza controlada y Redis por `fakeredis`. El pipeline OpenCV, los parsers y el scoring se ejercitan de verdad.

| Archivo | Qué cubre |
|---------|-----------|
| `test_dates.py` | todos los formatos de fecha y su normalización a ISO |
| `test_preprocess.py` | cada paso OpenCV (incluido deskew sobre una imagen rotada) y el postproceso |
| `test_parsers.py` | etiquetas, tickets, y el cálculo de confianza |
| `test_engines.py` | interfaz, factory, modo híbrido, Tesseract con `pytesseract` mockeado |
| `test_routes.py` | los endpoints, el ciclo de vida de la imagen temporal, auth, caché |
| `test_clients.py` | clientes HTTP y el loop del worker |
| `test_acceptance.py` | **el *gate* del módulo**: precisión ≥ 85% por campo y ≥ 70% por documento |

`test_acceptance.py` es el que verifica los criterios de aceptación de la propuesta sobre un conjunto de validación con ground truth, y comprueba también la contraparte: que un documento borroso o ilegible **sí** quede marcado con `needs_review`.

---

## Benchmark — Evaluación comparativa de motores OCR

La propuesta exige una evaluación comparativa entre Tesseract, EasyOCR y PaddleOCR sobre datos reales, midiendo precisión, tiempo de respuesta y consumo de recursos:

```bash
python -m bench.run_benchmark
```

Compara **Tesseract**, **EasyOCR** y **PaddleOCR** sobre datos reales midiendo *precisión de reconocimiento*, *tiempo de respuesta* y *consumo de recursos*.

No forma parte del deploy del MVP: las dependencias pesadas están comentadas en `requirements.txt`.

---

## 1. Preparar el Dataset

Las imágenes reales **no se versionan** (`.gitignore` las excluye). Colocá en `bench/dataset/`:

```
bench/dataset/
├── ground_truth.json
├── etiqueta_leche.jpg
├── etiqueta_yogur.jpg
└── ticket_super_01.jpg
```

### Formato de `ground_truth.json`

La clave es el nombre del archivo; `doc_type` es `product_label` o `receipt`.

```json
{
  "etiqueta_leche.jpg": {
    "doc_type": "product_label",
    "fields": {
      "nombre_producto": "Leche La Serenísima 1L",
      "fecha_vencimiento": "2027-08-15"
    }
  },
  "ticket_super_01.jpg": {
    "doc_type": "receipt",
    "fields": {
      "fecha_compra": "2026-07-20",
      "total": 5240.50
    },
    "items": [
      { "nombre_producto": "Leche Entera 1L", "cantidad": 2 },
      { "nombre_producto": "Pan Lactal", "cantidad": 1 }
    ]
  }
}
```

Los nombres de producto se comparan con similitud difusa (`rapidfuzz`, ≥ 90): un carácter mal leído no invalida el campo. Fechas, cantidades y totales se comparan exacto.

> Para que el resultado sea representativo, el conjunto debe incluir los escenarios difíciles que
> menciona la propuesta: tipografías variadas, fondos complejos, baja iluminación e imágenes inclinadas.

---

## 2. Instalar los Motores Alternativos

```bash
pip install easyocr
```

```bash
pip install paddlepaddle paddleocr
```

> PaddleOCR todavía no tiene adaptador en `app/engines/`. Para incluirlo, agregar
> `paddleocr_engine.py` siguiendo el patrón de `easyocr_engine.py` (import diferido +
> `is_available()`) y registrarlo en `factory._BUILDERS`. La interfaz `OCREngine` no cambia.

---

## 3. Correr el Benchmark

```bash
python -m bench.run_benchmark
```

Un solo motor:

```bash
python -m bench.run_benchmark --engines tesseract
```

Medir cuánto aporta el preprocesamiento OpenCV (corre sin filtros y se compara):

```bash
python -m bench.run_benchmark --no-preprocess
```

Guardar el reporte crudo:

```bash
python -m bench.run_benchmark --json bench/results/comparativa.json
```

---

## 4. Leer la salida

```
========================================================================================
COMPARATIVA DE MOTORES OCR
========================================================================================
Motor                       Campos   Docs ≥70%    Latencia    Memoria   Fallos    Gate
----------------------------------------------------------------------------------------
tesseract                    88.9%      100.0%       412 ms      38.2 MB       0     OK
easyocr                      92.3%      100.0%      2870 ms     512.4 MB       0     OK
----------------------------------------------------------------------------------------
Criterios de la propuesta: precisión de campos clave ≥ 85% · confianza por documento ≥ 70%
========================================================================================
```

- **Campos** — precisión en campos clave vs. ground truth → criterio **≥ 85%**.
- **Docs ≥70%** — documentos que alcanzan la confianza mínima → criterio **≥ 70%** por documento.
- **Latencia** / **Memoria** — costo computacional, el otro eje de decisión de la propuesta.
- **Gate** — si el motor cumple ambos criterios de aceptación.

Si ningún motor pasa el gate, la propuesta prevé este orden de acción: ajustar hiperparámetros, reconfigurar el pipeline, mejorar el preprocesamiento, evaluar modelos alternativos y —como último recurso— validación manual asistida (que el servicio ya implementa vía `needs_review`).

---

## Configuración

Todas las variables, con sus defaults, están en [`.env.example`](.env.example). Las que más importan:

| Variable | Default | Para qué |
|----------|---------|----------|
| `JWT_SECRET` | — | **debe coincidir con el del servicio Auth** |
| `OCR_ENGINE` | `tesseract` | `tesseract` \| `easyocr` \| `hybrid` |
| `OCR_LANG` | `spa` | idioma del motor |
| `TESSERACT_CMD` | *(vacío)* | ruta al binario; vacío = usar el del `PATH` |
| `FIELD_CONFIDENCE_THRESHOLD` | `0.85` | umbral por campo → dispara `needs_review` |
| `DOCUMENT_CONFIDENCE_THRESHOLD` | `0.70` | umbral por documento |
| `REDIS_URL` | `redis://redis:6379/0` | cola y caché |
| `JOB_TTL_SECONDS` | `86400` | cuánto vive el resultado de un job |
| `STORAGE_SERVICE_URL` | `http://tpp_storage_service:8000` | para subir/descartar la imagen temporal |
| `PREPROCESS_*` | `true` | activa o desactiva cada paso de OpenCV |

---

## Notas de implementación

**La cola usa listas de Redis (`RPUSH`/`BLPOP`), no RQ.** RQ depende de `fork()`, que no existe en Windows, y habría obligado a un camino distinto en desarrollo y en producción. Con listas de Redis el mismo código corre igual en las dos partes, sin dependencias extra, y sigue siendo "Redis como cola de trabajos" como plantea la propuesta. La interfaz (`enqueue`, `dequeue`, `get_job`) queda intacta si más adelante se quiere migrar a RQ, arq o Celery.

**El OCR corre en un thread aparte.** OpenCV y Tesseract son CPU-bound y bloqueantes: `ocr_service.process()` los ejecuta con `asyncio.to_thread` para no trabar el event loop, ni el de FastAPI ni el del worker. `process_sync()` expone la versión bloqueante, que es la que usa el benchmark.

**Caché por hash de imagen.** Antes de procesar se busca `sha256(imagen) + doc_type + lang` en Redis. Reprocesar la misma foto (reintento de la App, doble tap) sale de la caché. Si Redis no está, se saltea sin fallar.

**Un solo camino de código.** `/ocr/sync` y el worker comparten `ocr_service.process()`; los dos parsers comparten `scoring.py` y `parsers/text.py`; los dos motores comparten `RawOCROutput.from_words()`. No hay lógica de OCR duplicada.
