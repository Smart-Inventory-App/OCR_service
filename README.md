# Servicio de OCR — Smart Inventory

**Coordinador** del módulo de **Reconocimiento Óptico de Caracteres**. Lee etiquetas de producto, sellos de vencimiento y comprobantes, y devuelve los campos estructurados que el Core necesita para dar de alta inventario automáticamente, con una **confianza por campo** que indica cuándo hace falta que el usuario confirme el dato.

Este servicio **no ejecuta ningún motor OCR**. Los motores viven en sus propios repositorios y este los coordina:

```
                     ┌─────────────────────────────┐
   App ──────────────►         ocr_service         │  puerto 8003
                     │  ─────────────────────────  │
                     │     estrategia de intentos  │
                     │     parsing de campos       │
                     │     confianza y umbrales    │
                     │     cola, caché, storage    │
                     └──────────────┬──────────────┘
                                    │
                             POST /engine/run
                                    │
            ┌───────────────────────┼──────────────────────┐
            │                       │                      │
            ▼                       ▼                      ▼
    ocr_service_tesseract   ocr_service_easyocr    ocr_service_paddleocr
    puerto 8005             puerto 8006            puerto 8007
    (rápido, liviano)       (preciso, pesado)      (el más preciso, pesado)
```

Los tres motores son los que la propuesta manda comparar. Ninguno de ellos sabe qué documento está mirando: reciben una imagen y una transformación, y devuelven palabras con su caja y su confianza.

| Caso | `doc_type` | Campos que extrae |
|------|-----------|-------------------|
| **A** | `product_label` | `nombre_producto`, `marca`, `detalles` |
| **B** | `expiry_date` | `fecha_vencimiento` |
| **C** | `receipt` | por ítem: `nombre_producto`, `cantidad` · del documento: `fecha_compra`, `total`, `comercio` |
| **D** | `barcode` | `codigo` (GTIN normalizado a EAN-13), `simbologia`, `pais_registro` |

---

## Evolución del Módulo

Esta sección documenta los cinco cambios grandes que atravesó el OCR, con lo que los motivó y lo que se midió después de cada uno. Las secciones siguientes describen el estado actual.

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

Los servicios de motor reciben una imagen y una transformación, y devuelven palabras con su caja y su confianza. No saben qué documento están mirando.

El costo del coordinador después de la mudanza:

| | Antes | Ahora |
|---|---|---|
| Imagen Docker | ~2,5 GB | **~770 MB** |
| Memoria en ejecución | ~1,3 GB | **~30 MB** |
| Dependencias pesadas | Tesseract + OpenCV + PyTorch | ninguna |

Y una ganancia operativa: **el motor caro se enciende sólo cuando se lo usa**. Si un motor pesado está apagado, el coordinador lo reporta como no disponible, el banco de pruebas deshabilita esa opción y el modo híbrido degrada a Tesseract en lugar de fallar.

> El modo **híbrido no es un motor** y por eso no tiene repositorio: es la decisión de cuándo vale la pena pagar el motor caro, y esa decisión es del coordinador.

### 3. Precisión sobre Fotos Reales de Producto

El conjunto de trabajo son fotos de celular de productos reales, con fondo doméstico, iluminación despareja y envases brillantes y curvos. Es un conjunto propio y **no se versiona** (ver [Dataset Propio](#dataset-propio)). Cinco cambios, todos medidos:

**a) Búsqueda de orientación.** Las cuatro fotos de vencimiento del conjunto llegan **giradas un cuarto de vuelta**: el dato está impreso en un canto del envase y no hay forma cómoda de fotografiarlo derecho. Ningún motor lee texto girado 90°. El coordinador prueba las cuatro orientaciones y conserva la que lee.

> Se evaluó usar el detector de orientación de Tesseract (OSD, `--psm 0`) para evitar la búsqueda. **No sirve acá:** sobre estas mismas fotos falló en la mitad de los casos por "too few characters", y en las que respondió se equivocó, con confianza menor a 3. Probar las cuatro sale más barato que confiar en él.

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

### 4. El Tercer Motor: PaddleOCR

**Antes.** De los tres motores que nombra la propuesta estaban implementados dos. PaddleOCR quedaba pendiente y la comparativa, incompleta.

**El problema:** con Tesseract y EasyOCR el módulo **no llegaba a los dos criterios de aceptación a la vez** sobre fotos reales. El híbrido acertaba los ocho campos clave, pero con confianzas de 0,57–0,89: la mitad de los documentos no superaba el 0,70 por documento y terminaba en revisión manual igual. Subir la confianza a mano habría sido mentirle al usuario; hacía falta un motor que **leyera mejor de verdad**.

**Ahora.** [`ocr_service_paddleocr`](../ocr_service_paddleocr) expone el mismo contrato que los otros dos, en el puerto 8007. Dos resultados medidos:

- **Es el único motor que pasa los dos criterios por sí solo** (87,5 % en campos clave, 100 % de documentos sobre 0,70), y con él de respaldo el híbrido también los pasa.
- **Lee `27FEB27` sin que haya que repararlo.** Es el sello que Tesseract leía `27FRRE` y EasyOCR `27FEBZ7`; como no hay reparación, la confianza no se penaliza y el documento supera el umbral en vez de quedar en 0,57.

Incorporarlo costó **tres líneas** en este repositorio —una URL, una rama en la *factory*, una entrada en `CONCRETE_ENGINES`— y ni un cambio en los parsers, el scoring, las rutas o la estrategia. Es la validación del punto 2: la modularización se pagó sola la primera vez que hubo que agregar un motor.

**Lo que sí hubo que arreglar** fue un supuesto del parser de etiquetas. PaddleOCR detecta **regiones de texto**, no palabras, así que devuelve `CAFÉINSTANTÁNEO` pegado donde Tesseract devuelve dos palabras. El reconocimiento de la categoría de producto —de la que cuelga todo el nombre— buscaba la palabra entera y no la encontraba, y el campo caía en la marca (`LaVirginia`). Ahora la categoría se reconoce también pegada, pero **sólo si lo que sigue es otro término del vocabulario**: `cafe` + `instantaneo` sí, `yogur` + `isimo` no — porque `Yogurisimo` es una marca, y aceptarla como nombre sería volver a la confusión que arregló el punto 3.e. El mismo arreglo cubre el caso inverso: una `O` de logotipo delante de `CAFÉ`, que la reparación de versalitas partidas pegaba en `ocafe`.

El efecto sobre las fotos reales: PaddleOCR pasó de 58,3 % a **83,3 %** de campos y de 62,5 % a **100 %** de documentos sobre el umbral.

### 5. Comprobantes Fiscales Reales

**Antes.** El caso `receipt` estaba validado contra tickets **sintéticos**: dos imágenes generadas con una tipografía limpia, encabezado, tres ítems y un total. Sobre eso los cuatro motores daban 100 %.

**El problema:** la primera factura B real —de DIA, con su encabezado fiscal, su columna de IVA y su pie de programa de fidelidad— rompió el parser en tres lugares distintos, y ninguno tenía que ver con el motor. PaddleOCR leyó el comprobante casi perfecto; lo que falló fue interpretarlo.

| Qué salía | Por qué |
|---|---|
| `total = 25,00` en vez de `16015.25` | El comprobante **mezcla convenciones**: lista los ítems con coma decimal (`8909,00`) y el total con punto (`16015.25`). El patrón de importes empezaba con `\d{1,3}`, así que mordía `160`, y el resto de las cifras se leían como importes sueltos: el último, `25`, ganaba por ser el último de la línea. |
| `INDIRECTOS`, `AHORRO OFERTAS`, `TOT AHORRO` y `Suma de sus pagos` cargados como **productos** | Debajo del total, un comprobante argentino imprime media docena de renglones con importe —régimen de transparencia fiscal, tributos, medios de pago, ahorro del programa de fidelidad— y todos tienen exactamente la forma de un ítem. |
| `PAN ARTESANO BIMBO (21,00)` como nombre | `(21,00)` es la **columna de alícuota de IVA** de una factura A/B, que va pegada a la descripción. |

**Ahora.** Los importes se tokenizan completos y la convención se deduce de cada número (manda el separador que aparece último; con puntos solos, es decimal sólo si quedan una o dos cifras detrás, porque `1.234` son mil doscientos treinta y cuatro). El pie fiscal se filtra. La alícuota se recorta del nombre.

Sobre la factura real, el resultado pasó de 4/6 campos a **6/6**: `fecha_compra`, `total`, `comercio` y los tres ítems con su cantidad, con `document_confidence` 0,86 y `needs_review: false` — es decir, un comprobante que carga inventario sin intervención.

**La lección, que vale para el módulo entero:** el conjunto sintético sirve para verificar que el pipeline no se rompió, pero **no descubre nada**. Los tres bugs de arriba vivieron todo este tiempo detrás de un 100 % verde, y aparecieron con la primera foto real. Lo mismo pasó con las etiquetas: la primera foto de un condimento mostró que el vocabulario de categorías no tenía **ninguna** especia, y que cuando la marca y el nombre están impresos en cuerpos parecidos —`Alicante` sobre `Perejil`— el agrupamiento por geometría los une y el nombre sale con la marca adelante.

---

## Resultados Medidos - Benchmark

Todo lo que sigue se reproduce con `python -m bench.run_benchmark` (ver [Benchmark](#benchmark--evaluación-comparativa-de-motores)).

### El Criterio Se Mide sobre los Campos **Clave**

La propuesta fija el 85 % sobre "campos clave (nombre de producto, fecha de vencimiento y, en caso de tickets, cantidad, fecha de compra)". `marca`, `detalles`, `total` y `comercio` enriquecen el alta pero **no la bloquean**, y promediarlos con los clave en un solo número esconde justamente el dato que hay que mirar. Por eso el benchmark reporta las dos columnas:

- **Clave** — `nombre_producto`, `fecha_vencimiento`, y en el ticket `fecha_compra` y los ítems. Es el criterio de aceptación y es lo que decide el *gate*.
- **Todos** — incluye los accesorios. Sirve para ver cuánto se está extrayendo de más, no para aprobar el módulo.

### Pruebas con Fotos Reales

Fotos de celular con fondo doméstico, iluminación despareja y envases brillantes y curvos: café instantáneo *La Virginia*, café molido *Cabrales*, harina *Pureza*, leche *aro*, perejil *Alicante*, agua *cellier* y yogur *ilolay*, la mayoría con su etiqueta y su sello de vencimiento. Los sellos llegan **girados un cuarto de vuelta** y varios están impresos en matriz de puntos. La factura es una **factura B real de DIA**, con su encabezado fiscal, su columna de IVA y su pie de programa de fidelidad.

| Motor | Clave | Todos | Docs ≥70% | Latencia | Gate |
|-------|-------|-------|-----------|----------|------|
| `tesseract` | 53,3 % (8/15) | 37,5 % | 7,1 % | 4,2 s | NO |
| `easyocr` | 69,2 % (9/13) | 59,1 % | 28,6 % | 11,1 s | NO |
| **`paddleocr`** | **86,7 % (13/15)** | **87,5 %** | **92,9 %** | 19,6 s | casi |
| **`hybrid`** (Tesseract + PaddleOCR) | **86,7 % (13/15)** | **87,5 %** | **92,9 %** | 19,0 s | casi |

Por campo:

| Motor | `nombre_producto` | `fecha_vencimiento` | `marca` | Factura (`fecha_compra`·`total`·`comercio`·ítems) |
|-------|-------------------|---------------------|---------|---------------------------------------------------|
| `tesseract` | 3/7 | 4/6 | 0/7 | 0 · 1 · 0 · 0 |
| `easyocr` | 5/7 | 3/4 | 4/7 | 0 · 0 · 0 · 0 |
| **`paddleocr`** | **6/7** | **5/6** | **6/7** | **1 · 1 · 1 · 1** |

**PaddleOCR supera el 85 % en campos clave** sobre casi el doble de documentos que la medición anterior, y con los tres tipos de documento representados. Los dos motores livianos quedan lejos: sobre fotos reales Tesseract no llega ni a la mitad.

> **Los dos fallos de EasyOCR** son del contenedor, no del motor: se quedó sin memoria durante la corrida y dejó de responder a dos peticiones (Docker Desktop reparte ~2 GB entre *todos* los contenedores y con los dos motores pesados arriba no alcanza). Sus porcentajes están calculados sobre los 12 documentos que sí completó, así que **no son comparables** con los de los otros tres. Es también el motivo por el que el coordinador propaga un motor caído como `502` en vez de devolver una lectura vacía.

Los tres campos clave que ningún motor resuelve:

| Documento | Qué pasa |
|---|---|
| Etiqueta café *Cabrales* | PaddleOCR lee `CAFÉ`, `TOSTADO` y `MOLIDO`, pero en tres regiones que no se funden en un bloque, así que el nombre sale incompleto. |
| Sello yogur *ilolay* | `L15/09/26` en matriz de puntos sobre una tapa blanca curva y brillante, fotografiada en ángulo. **Ningún motor lo detecta en ninguna de las cuatro orientaciones ni a 3200 px de ancho**: no es un problema de parser ni de resolución. |
| Sello café *La Virginia* | resuelto, pero sólo tras probar la cuarta orientación (180°), lo que dispara la latencia a 46 s. |

En los tres el servicio **no inventa un valor**: devuelve `null` y marca `needs_review`, que dispara la validación manual asistida que la propuesta prevé. El único documento que queda bajo el umbral por documento es el sello del yogur, y por eso el *gate* completo —que exige el 100 % de los documentos— todavía no cierra.

### Conjunto Sintético de Control — `bench/sample_dataset/` (9 documentos)

Es el único conjunto que cubre los **tres** tipos de documento: incluye dos tickets con sus ítems, cantidades, fecha de compra y total.

| Motor | Clave | Todos | Docs ≥70% | Latencia | Gate |
|-------|-------|-------|-----------|----------|------|
| `tesseract` | 100 % | 100 % | 88,9 % | 674 ms | NO |
| `easyocr` | 100 % | 100 % | 88,9 % | 1237 ms | NO |
| `paddleocr` | 100 % | 95,0 % | **100 %** | 4324 ms | **OK** |
| **`hybrid`** | **100 %** | **100 %** | **100 %** | **1474 ms** | **OK** |

Sobre documentos limpios el híbrido resuelve casi todo con Tesseract y sólo paga el motor caro donde hace falta: **pasa el gate completo en 1,5 s por documento**, doce veces más rápido que sobre las fotos reales. Los ítems del ticket (`fecha_compra` 2/2, `items` 2/2, `total` 2/2) salen perfectos en los cuatro motores.

El conjunto sintético sirve para verificar que el pipeline funciona de punta a punta; **no reemplaza al dataset real**, porque una imagen generada no tiene el ruido ni la iluminación de una foto — se ve en que Tesseract pasa de 100 % acá a 62,5 % sobre fotos.

### Consumo de Recursos

Medido con `docker stats` sobre los contenedores en reposo y en proceso:

| Servicio | Imagen | Memoria en reposo | Latencia por documento |
|---|---|---|---|
| `ocr_service` | ~770 MB | ~30 MB | — (coordina) |
| `ocr_service_tesseract` | ~1,05 GB | ~70 MB | 0,2–0,8 s por intento |
| `ocr_service_easyocr` | ~2,58 GB | ~350 MB | 3–5 s por intento |
| `ocr_service_paddleocr` | ~2,68 GB | ~350 MB | 4–15 s por intento |

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
| `barcode` | 4: directo, ampliado, enfocado y ampliado+enfocado | no usa motor OCR |

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

- **`tesseract`** — rápido y liviano. Da **confianza por palabra**, que es la base del cálculo de `document_confidence`.
- **`easyocr`** — basado en deep learning. Mucho mejor en envases curvos, brillantes y con tipografías de fantasía; mucho más lento y pesado.
- **`paddleocr`** — el más preciso de los tres sobre fotos reales, y el único que pasa los dos criterios de aceptación por sí solo. Detecta **regiones de texto** en vez de palabras, así que a veces devuelve dos palabras pegadas.
- **`hybrid`** — el "enfoque híbrido" que la propuesta no descarta: Tesseract como principal y, cuando su lectura no alcanza, PaddleOCR de respaldo.

Cambiar de motor es una variable de entorno:

```env
OCR_ENGINE=hybrid
```

También se puede elegir **por petición**, que es lo que usa el banco de pruebas para compararlos:

```jsonc
{ "image_url": "...", "doc_type": "expiry_date", "engine": "hybrid" }
```

> **Por qué conviene combinarlos.** Los motores son **complementarios, no redundantes**. Sobre el sello de la harina, Tesseract acierta y EasyOCR no; sobre el de la leche, al revés. Sobre la etiqueta de la harina, Tesseract no levanta nada y los otros dos la leen entera. Combinar recupera lo mejor de cada uno — que es precisamente el argumento con que la propuesta deja abierta esta opción.
>
> Cuál va de respaldo **no se eligió, se midió**: con EasyOCR el híbrido acierta más campos clave pero con confianzas bajo el umbral, y con PaddleOCR acierta uno menos y pasa el gate. Es una variable (`HYBRID_FALLBACK_ENGINE`) y el benchmark es el que la decide; si mañana el dataset crece y la respuesta cambia, se cambia el `.env` y no el código.

### 3. Parsing

- **Etiqueta** (`parsers/product_label.py`) — nombre por categoría de producto, marca por prominencia tipográfica, detalles por contenido y calificativos. Ver [Evolución § 3.e](#3-precisión-sobre-fotos-reales-de-producto).
- **Vencimiento** (`parsers/expiry.py` + `parsers/dates.py`) — formatos argentinos habituales: `15/08/2027`, `15-08-27`, `15 08 2027`, `2027-08-15`, `15 AGO 2027`, `27FEB27`, `08/2027`, `AGO 2027`, y las palabras que las anuncian (`VTO`, `VENCE`, `VENCIMIENTO`, `CONS. PREF.`, `Consumir preferentemente antes de`, `BEST BEFORE`, y la `V` sola del envase de farmacia). Todo se normaliza a ISO `YYYY-MM-DD`; si el formato sólo trae mes y año se asume el último día del mes.

Dos reglas gobiernan lo que **no** se acepta, y las dos salieron de medir sobre las fotos del conjunto:

- **Mes/año con año de dos dígitos (`03/28`) sólo cuenta si lo anuncia una palabra de vencimiento.** Suelto es indistinguible de un día y un mes. Con el anuncio adelante no hay ambigüedad: nadie imprime un día y un mes después de «vence». Es el formato más frecuente en el envase de farmacia, y `L 1279 V 03/28` —donde `L` es el lote y `V` el vencimiento— sale bien en un solo intento.
- **Los dos separadores de un día/mes/año tienen que ser el mismo.** `15/08/2027` y `15 08 2027` valen; `1 09/27` no, porque ese `1` es un resto del lote y la lectura correcta es «septiembre de 2027». Antes se devolvía el 1 de septiembre, que es exactamente el tipo de dato falso que el usuario podría aceptar sin mirar.
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
| `ocr_service_paddleocr` | `8007` |

---

## Stack

- Python 3.11 + FastAPI
- Redis — cola de trabajos y caché de resultados
- Pillow — valida el binario recibido y arma las variantes del decodificador
- `pyzbar` / `libzbar0` — decodificación de códigos de barras
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
| `doc_type` | `product_label` \| `expiry_date` \| `receipt` \| `barcode` | no | `product_label` |
| `lang` | string | no | `spa` |
| `engine` | `tesseract` \| `easyocr` \| `paddleocr` \| `hybrid` | no | el de `OCR_ENGINE` |

**Respuesta `202`:** `{ "job_id": "9f2c1a...", "status": "queued" }`

**Errores:** `400` formato no soportado, archivo vacío o binario que no es una imagen · `401` token inválido · `502` el storage no respondió · `503` cola no disponible.

> **El flujo de la App son dos escaneos.** Primero la etiqueta (`doc_type=product_label`) para el nombre, la marca y los detalles; después,
> si el producto tiene vencimiento, la foto del sello (`doc_type=expiry_date`). Son dos llamadas independientes: no hay que mandar las dos imágenes juntas, y un producto sin fecha simplemente no hace la segunda.

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

> El token del llamador se guarda en una clave de Redis **separada** del estado del job (se necesita para descartar la imagen temporal) y se borra en cuanto el job termina: nunca sale por la API.

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
    "version": "1.7.2", "languages": ["eng","spa"], "supports_preprocessing": false },
  { "name": "paddleocr", "url": "http://tpp_ocr_paddleocr:8007", "available": true,
    "version": "3.7.0", "languages": ["eng","spa"], "supports_preprocessing": false }
]
```

### `GET /ocr/livez` · `GET /ocr/readyz`

- `livez` → `{ "status": "ok" }`. Sin dependencias.
- `readyz` → verifica Redis **y** que el motor configurado responda. Se comprueba el motor configurado, no todos: que EasyOCR esté apagado no debería marcar el servicio como caído si el que se usa es Tesseract.

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

Los motores pesados son opcionales, pero el híbrido por defecto usa PaddleOCR de respaldo. La primera build de cada uno descarga el framework y los pesos, y tarda varios minutos:

```bash
cd ../ocr_service_paddleocr && cp .env.example .env && docker-compose up -d --build
```

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

> `JWT_SECRET` en `.env` **debe** ser el mismo que usa el servicio `auth`, o todos los `/ocr/*` responderán `401`.

### Sin Docker

Este servicio no tiene dependencias de sistema, así que corre directo:

```bash
python -m venv .venv && .venv\Scripts\activate && pip install -r requirements.txt
```

Apuntar los motores, Redis y el storage a `localhost` en el `.env`:

```env
TESSERACT_SERVICE_URL=http://localhost:8005
EASYOCR_SERVICE_URL=http://localhost:8006
PADDLEOCR_SERVICE_URL=http://localhost:8007
REDIS_URL=redis://localhost:6379/0
STORAGE_SERVICE_URL=http://localhost:8004
```

```bash
uvicorn app.main:app --reload --port 8003
```

```bash
python -m app.worker
```

> **Sin Redis** la API arranca igual y `POST /ocr/sync/upload` funciona perfectamente: la caché es opcional y el modo sincrónico no usa la cola. Sólo los endpoints asincrónicos devolverán `503`.

---

## Banco de Pruebas en el Navegador

```
http://localhost:8003/ocr/playground
```

Página servida por el propio servicio. Replica el patrón de autenticación del backoffice: se inicia sesión **una sola vez** contra el servicio Auth, el token queda en `localStorage` y se adjunta automáticamente en cada consulta; si el servidor responde `401`, la sesión se cierra y vuelve al login.

Permite subir una imagen y elegir:

- el **tipo de documento** — los tres, con una ayuda que explica qué foto espera cada uno;
- el **motor** — `tesseract`, `easyocr`, `paddleocr` o `hybrid`. La tabla **Motores** muestra cuáles están arriba y con qué versión, y **deshabilita en el selector los que están apagados**, para que no se pueda pedir un motor que no va a correr;
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

**406 tests, 95 % de cobertura, en unos 20 segundos.** No requieren los servicios de motor ni un Redis real: los motores se sustituyen por dobles con confianza controlada (y por `respx` cuando se verifica el contrato HTTP), y Redis por `fakeredis`. Lo que sí se ejercita de verdad es todo lo que este servicio aporta.

| Archivo | Qué cubre |
|---------|-----------|
| `test_strategy.py` | los planes de intentos por tipo de documento y la orquestación del híbrido |
| `test_engines.py` | el contrato HTTP con los motores, la factory y el modo híbrido |
| `test_dates.py` | todos los formatos de fecha y su normalización a ISO |
| `test_parsers.py` | etiquetas, sellos, tickets y el cálculo de confianza |
| `test_routes.py` | los endpoints, el ciclo de vida de la imagen temporal, auth, caché |
| `test_clients.py` | clientes HTTP y el loop del worker |
| `test_acceptance.py` | **el *gate* del módulo**: precisión ≥ 85% por campo y ≥ 70% por documento |
| `test_barcode.py` | la escalera de intentos del decodificador, la normalización a GTIN y el dominio argentino |
| `test_fotos_reales.py` | **fotos del conjunto de prueba**, sin levantar un motor (ver abajo) |

#### Fotos Reales Sin Motores

`test_fotos_reales.py` contesta la pregunta que ninguna imagen dibujada en memoria puede contestar —si el módulo resuelve una foto de verdad— y lo hace sin depender de los motores, que es lo que le permite correr en el CI. La pregunta se parte igual que el módulo:

- **El código de barras se decodifica acá**, con ZBar, que es una librería del sistema y no un modelo. La foto entra entera y se verifica el resultado final. Son las dos únicas imágenes versionadas: un EAN-13 de farmacia y el código impreso de 4080 px que **sólo se lee reduciendo** —la foto que motivó el cambio con más impacto medido del módulo—.
- **El texto lo reconocen los motores, que viven en otros servicios.** Lo que se versiona es **la salida que devolvieron de verdad** (`tests/fixtures/lecturas/*.json`: texto y palabras con caja y confianza, 40 KB) y contra eso corren los parsers y el scoring. Cubre marca-contra-nombre, monodroga, `MM/AA`, `27FEB27`, una factura B real y un remito sin precios.

Son 40 KB y milisegundos, en lugar de 1,7 MB de fotos y siete minutos de inferencia. Para regrabar las lecturas cuando cambie un motor o la estrategia de intentos: `python -m bench.grabar_lecturas --source <carpeta de fotos>` con los servicios arriba. Las fotos **no viven en ningún repositorio**: son material de prueba local.

Los tres repositorios de motor tienen sus propias suites, que cubren la cadena de OpenCV y la traducción de la salida de cada motor:

| Repositorio | Tests | Cobertura |
|---|---|---|
| `ocr_service_tesseract` | 71 | 96 % |
| `ocr_service_easyocr` | 42 | 92 % |
| `ocr_service_paddleocr` | 46 | 92 % |

`ocr_service_easyocr` y `ocr_service_paddleocr` incluyen `test_bandas_reales.py`, que ejercita la detección de bandas de texto sobre **una foto de envase real** —la única imagen versionada en cada uno, de 103 KB— sin cargar el modelo. Son los dos motores a los que el coordinador les manda recortes, y esa función no se puede probar con un rectángulo dibujado.

**565 tests en total** en los cuatro repositorios del módulo, todos en el CI y sin ningún motor instalado.

### Integración con los motores (fuera del CI)

Todo lo de arriba corre sin motores, y eso es lo que lo hace utilizable como red de regresión. Pero hay afirmaciones que un doble **no puede desmentir**: contesta lo que uno le dijo que conteste, así que una suposición equivocada sobre el otro lado sobrevive intacta. Para eso está `integration_tests/`, que habla con los tres servicios de verdad.

```bash
docker compose up -d
pytest -c pytest-integration.ini
```

Vive **fuera de `tests/`** a propósito, y ahí está toda la separación: el CI corre `pytest tests/`, así que nunca las recoge — no hay marcador que recordar ni variable que setear para que el build siga siendo verde sin Docker. Tienen su propio archivo de configuración porque también necesitan lo contrario que el CI: el motor híbrido en vez del doble, las URLs apuntando a `localhost` en vez de a la red de Docker, y sin cobertura (escribir `coverage.xml` acá pisaría el del CI). Si los motores no están, la sesión entera se saltea con el motivo puesto en lugar de fallar test por test.

| Archivo | Qué verifica | Por qué no alcanza un doble |
|---|---|---|
| `test_contrato_motores.py` | los tres responden y se identifican; devuelven palabras con caja y confianza sobre una foto real; la `Transform` viaja por HTTP y **se aplica**; cada uno **declara** si prepara su propia entrada | `attempt_plan` arma un plan distinto según esa capacidad: si el motor cambiara de opinión, los planes se armarían mal y ningún doble lo notaría |
| `test_extremo_a_extremo.py` | que el código de barras dé el mismo número con `engine=tesseract` y con `engine=paddleocr`; que las lecturas grabadas **sigan representando al módulo** | lo primero es la razón por la que ZBar corre en el coordinador y no repartido entre los motores; lo segundo es el riesgo del fixture barato — si un motor mejora o cambia la estrategia de intentos, `tests/fixtures/lecturas/` queda viejo y nada lo dice |

La segunda mitad necesita las fotos del conjunto de prueba, que **no viven en el repositorio**. Sin ellas se saltea sola; con ellas se le dice dónde están:

```bash
OCR_FOTOS="/ruta/al/conjunto" pytest -c pytest-integration.ini
```

Son 17 tests y tardan unos 4 minutos —medido—, no los 20 segundos de la suite del CI: la mitad son inferencias reales sobre fotos de hasta 4000 px. Cuándo correrlas: al tocar el contrato con los motores, al cambiar la estrategia de intentos, al actualizar la imagen de un motor, y antes de integrar el módulo con el resto del sistema. Si falla `test_la_lectura_grabada_sigue_representando_al_modulo`, el mensaje dice qué hacer: mirar el diff de `python -m bench.grabar_lecturas` antes de tocar nada, porque puede estar fallando por una **mejora** del motor y no por una regresión.

> Ninguna suite del CI lee el dataset del proyecto: vive fuera del repositorio y los tests se sostienen con imágenes generadas en memoria, dos fotos versionadas de código de barras, lecturas grabadas, motores dobles y `fakeredis`. Las fotos son para el **benchmark** —que mide precisión, no evita regresiones— y para `integration_tests/`, que se pide a mano.

---

## Benchmark — Evaluación Comparativa de Motores

La propuesta lo pide explícitamente:

> "Durante el desarrollo del proyecto se realizará una evaluación comparativa entre estas herramientas mediante pruebas sobre datos reales, considerando métricas como precisión de reconocimiento, tiempo de respuesta y consumo de recursos."

```bash
python -m bench.run_benchmark --dataset bench/dataset
```

```bash
python -m bench.run_benchmark --dataset bench/sample_dataset --engines tesseract
```

Corre cada motor sobre las mismas imágenes **a través del coordinador**, es decir con la misma estrategia de intentos que usa la App. Requiere los servicios de motor levantados.

El consumo de memoria ya no se mide desde acá: cada motor es un contenedor aparte y su memoria se lee con `docker stats`, que es la medida que importa para decidir el despliegue.

### Leer la Salida

```
Motor                   Clave    Todos   Docs >=70%     Latencia   Intentos   Fallos    Gate
tesseract              62.5%    41.7%        12.5%       4327 ms        5.5        0      NO
easyocr                75.0%    66.7%        37.5%       9218 ms        2.4        0      NO
paddleocr              87.5%    83.3%       100.0%      18285 ms        1.8        0      OK
hybrid                100.0%    83.3%        50.0%       9858 ms        6.9        0      NO

DETALLE POR CAMPO
paddleocr             fecha_vencimiento: 4/4 (100%)  marca: 3/4 (75%)  nombre_producto: 3/4 (75%)
```

- **Clave** — precisión en los campos que la propuesta llama clave (`nombre_producto`, `fecha_vencimiento`, y en el ticket `fecha_compra` y los ítems) → criterio **≥ 85 %**. **Es la columna que decide el gate.**
- **Todos** — la misma precisión incluyendo los accesorios (`marca`, `detalles`, `total`, `comercio`). Informativa: mezclar los accesorios con los clave en un solo número escondería el dato que importa.
- **Docs ≥70%** — documentos que alcanzan la confianza mínima → criterio **≥ 70 %**.
- **Intentos** — cuántas combinaciones de giro y filtros hizo falta probar.
- **Gate** — si el motor cumple **ambos** criterios de aceptación. Hay que cumplir los dos: acertar el campo con una confianza que no llega al umbral igual manda el documento a revisión manual.

### Dataset Propio

Las imágenes reales del proyecto **no se versionan**: `bench/dataset/` está en el `.gitignore` y llega vacío al repositorio.

Si están en una carpeta propia —una subcarpeta por tipo de material, cada una con su `ground_truth.json`— el dataset se arma solo. **Hay que indicar dónde**: no existe una ruta por defecto, porque cualquiera apuntaría a la máquina de alguien.

```bash
python -m bench.load_real_dataset --source <carpeta con las fotos>
```

Copia las imágenes prefijadas con el nombre de su carpeta (la etiqueta y el sello del mismo producto suelen llamarse igual, y sin prefijo uno pisaría al otro), une los ground truth en uno y **avisa de las fotos que el ground truth nombra pero no están**, que si no el benchmark las cuenta como fallo del motor. Hay que volver a correrlo cada vez que se agregan fotos.

Acepta las dos formas de ground truth: la plana que consume el benchmark (`{"archivo.jpg": {"doc_type": ..., "fields": {...}}}`) y la que nombra los campos directamente bajo una clave `imagenes`, que es la que sale de transcribir el conjunto a mano. Las entradas sin ningún campo esperado —sólo el texto visible del envase— se omiten: sirven para revisar a ojo, no para puntuar.

También se puede armar a mano: colocá las imágenes en `bench/dataset/` junto a un `ground_truth.json`:

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

### Los Tres Motores de la Propuesta

Con [`ocr_service_paddleocr`](../ocr_service_paddleocr) quedan implementados los tres motores que la propuesta manda comparar, y la evaluación comparativa los cubre a todos.

Vale la pena registrar **cuánto costó incorporar el tercero**, porque es la prueba de que la modularización del punto 2 valió la pena: el repositorio nuevo expone el mismo contrato (`POST /engine/run`, `GET /engine/info`, `/livez`, `/readyz`) y de este lado el cambio fueron **tres líneas** — una URL en la configuración, una rama en la *factory* y una entrada en `CONCRETE_ENGINES`. Ni los parsers, ni el scoring, ni la estrategia de intentos, ni las rutas se tocaron. La interfaz `OCREngine` no cambió, y el modo híbrido lo toma como respaldo o como principal sin saber nada de él.

Agregar un cuarto motor cuesta exactamente lo mismo.

---

## Configuración

Todas las variables, con sus defaults, están en [`.env.example`](.env.example). Las que más importan:

| Variable | Default | Para qué |
|----------|---------|----------|
| `JWT_SECRET` | — | **debe coincidir con el del servicio Auth** |
| `TESSERACT_SERVICE_URL` | `http://tpp_ocr_tesseract:8005` | dónde vive el motor liviano |
| `EASYOCR_SERVICE_URL` | `http://tpp_ocr_easyocr:8006` | dónde vive un motor pesado |
| `PADDLEOCR_SERVICE_URL` | `http://tpp_ocr_paddleocr:8007` | dónde vive el otro |
| `ENGINE_API_KEY` | *(vacía)* | clave compartida con los motores |
| `ENGINE_TIMEOUT_SECONDS` | `120` | EasyOCR tarda segundos por documento |
| `OCR_ENGINE` | `hybrid` | `tesseract` \| `easyocr` \| `paddleocr` \| `hybrid` |
| `FIELD_CONFIDENCE_THRESHOLD` | `0.85` | umbral por campo → dispara `needs_review` |
| `DOCUMENT_CONFIDENCE_THRESHOLD` | `0.70` | umbral por documento |
| `HYBRID_FALLBACK_THRESHOLD` | `0.70` | confianza bajo la cual se llama al respaldo |
| `HYBRID_FALLBACK_ENGINE` | `paddleocr` | cuál es el respaldo — lo decidió el benchmark |
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
