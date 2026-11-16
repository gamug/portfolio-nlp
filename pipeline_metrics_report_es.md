# Precisión y recall (uno contra el resto) de los pasos del pipeline

Fecha: 2026-11-16 (las cifras de las secciones 5.1 y 3.3 son del 2026-11-16 y del 2026-11-15; las demás, de las fechas que cada sección indica). El número del encabezado es el número de paso propio del pipeline; las secciones siguen el orden en que corren los pasos. "El dueño" es el término del proyecto para quien marca y decide. Pasos cubiertos, en el orden del pipeline: scope (con el paso de sector), cut, NER y la elección de la empresa, summarization, sentiment, category.

Todas las cifras de abajo salen de los documentos del repositorio y de archivos de resultados versionados, o se recalcularon a partir de archivos versionados (las tablas de category
y de sector se recalcularon para este informe a partir de `docs/category_results_2026_11_13/*.csv` y `gold/sector_validation.csv`; las cifras del candidato de
category son los archivos del PR #196, `docs/category_results_2026_11_15/`, y sus corridas de MLflow, sección 6). **Las cifras medidas sobre la corrida real del corpus son
la muestra de datos reales de category (6.4), las marcas de empresa principal sobre piezas (3.3) y la verificación de sentiment sobre 190 piezas (5.1)**; la muestra de category se leyó del almacén RESULTS (solo lectura: `news_category`, `news_summary`, `sub_news`); todas las demás salen de archivos versionados o
de resultados.

**Uno contra el resto (OVR)** significa: para cada clase, precisión = de lo que el paso llamó de esa clase, la proporción que es correcta; recall = de los miembros verdaderos de
esa clase, la proporción que el paso encontró. Donde un paso no tiene clases de ese tipo, o el dueño del proyecto nunca las midió, la tabla lo dice en lugar de
inventar una cifra.

**Qué métricas se conservan.** Para todo modelo entrenado en el proyecto (los clasificadores de scope, el clasificador de pares del cut, el modelo de relevancia, el modelo de pares y el stack de sentiment, el candidato de category) este informe conserva solo dos tipos de métricas: **de entrenamiento-validación** (validación cruzada, puntajes fuera de pliegue y de una división de validación) y **held-out** (puntajes sobre textos con los que el modelo no se entrenó ni se ajustó).
Lo que se quitó en esta versión es toda otra medida: la proporción de ítems de una corrida que un paso etiquetó (una proporción de salida de una corrida de prueba) y los conteos de una corrida del corpus, que dicen qué produce un modelo, no si acierta. Se conservan las descripciones de los datasets, las narrativas de selección del modelo y las limitaciones. Los modelos que no se entrenan aquí (summarization, el modelo zero-shot del paso de category, las reglas) conservan las cifras que tenían.
Las proporciones y los conteos quitados siguen en los documentos de los pasos (`docs/companies_step.md`, `docs/main_company_training.md`, `docs/summary_step.md`), que no son informes de todo el pipeline y los conservan. Las descripciones de datasets conservan las cifras de calidad de etiquetas de sus conjuntos (por ejemplo la verificación de etiquetas de entrenamiento del cut): describen los datos, no un modelo.

## Resumen

| Paso | Qué se mide | ¿Tiene precisión y recall OVR por clase? | Cifra principal | ¿Conjunto reservado (held-out)? |
|---|---|---|---|---|
| Scope | una de 4 clases por texto (`single`, `multi`, `sector`, `out_of_scope`) | **Sí**, cuatro clases (sección 1.1) | acuerdo en el held-out 83.6% [78.1, 88.9]; recall de `out_of_scope` 53%; precisión en cuarentena 90.8% | Sí (200 textos, vistos muchas veces), más 225 leídos por el dueño |
| Sector | el sector GICS de 2 dígitos asignado a un texto `sector` | **Solo precisión** (por sector, de 2 a 5 textos cada uno); el recall no se midió | 87.5% de los sectores asignados correctos (35 de 40) | Sí (un sorteo hecho después de construir el paso), un solo revisor |
| Cut | las piezas ("sub-news") que el cut hace de un texto con varias empresas | **Sin clases.** Tres preguntas de sí/no por pieza (parecidas a una precisión) | 84.6% de las piezas fieles; 75.5% limpias | Sí (28 textos, 143 piezas) |
| Empresa: NER | los tramos de organización que encuentra NER | No por clase sobre los textos actuales; una corrida anterior juzgada (v1, artículos completos) | recall 0.84, precisión 0.66 (corrida anterior) | Juzgado por un LLM, no por un gold |
| Empresa: relevancia (qué empresa es la principal) | si un par (texto, empresa) es "principal" | Sí, para las dos clases, fuera de pliegue (out-of-fold) | precisión de la empresa principal 90.0%, recall 94.7% con umbral 0.5 (validación cruzada sobre resúmenes); **sobre 119 filas marcadas de 59 piezas, los textos que lee el paso: precisión 90.9%, recall 30.0%** | **Sí**: las marcas del dueño sobre piezas que el modelo nunca vio (sección 3.3, parciales: 119 filas de 59 piezas); el 94.7% es validación cruzada sobre 809 resúmenes marcados, no held-out |
| Summarization | el resumen de un texto de más de unos 2,000 caracteres | **Sin clases.** Puntajes de un juez LLM (1 a 5) | **DeepSeek, el summarizer del paso desde T-206:** fidelidad 4.99, cobertura 4.54, alucinación 0.7% (300 artículos); el paso anterior con distilbart: 4.60, 3.24, 15.0% | El juez también es DeepSeek (calificó sus propios resúmenes); no un gold; la corrida sobre el corpus no se mide |
| Sentiment | positivo, negativo, neutral para un par (texto, empresa) | **Sí**, tres clases | puntaje 0.814; P/R negativo 0.747 / 0.748 (validación cruzada); sobre 190 piezas reales: exactitud 73.7%, precisión de negativo 69.4% | **No** para el 0.814 (validación cruzada anidada, 809 filas); **sí** para las 190 piezas (sección 5.1) |
| Category | 14 etiquetas (6 grupos) para un par (texto, empresa); el candidato lee 12 | **Sí**, por etiqueta (precisión, recall y ROC) | modelo del paso: 43.0% en el held-out. **Candidato (sin conectar): 54.1% en el held-out, 71.3% en 300 sub-news reales (etiquetas del asistente), 87.8% fuera de pliegue (etiquetas revisadas)** | Sí (135 filas single, evaluadas una sola vez); la barra de 80% **no se cumple** |

**Lee primero la última columna.** Solo el scope, el cut, category, la prueba limpia de sector y las dos verificaciones sobre piezas de la corrida real (las marcas de empresa principal del dueño, sección 3.3, y la verificación de sentiment sobre 190 piezas, sección 5.1) son medidas sobre textos con los que el modelo o las reglas no eligieron
sus ajustes; las cifras de validación cruzada del modelo de relevancia y del stack de sentiment son fuera de pliegue y no son held-out. Ninguno de los pasos ha sido evaluado por un segundo
lector. La barra que el dueño fijó para las etiquetas es 80% (D24 para category; 70% para precisión y recall del negativo en sentiment, D32).

## 1. Scope (paso 1, `docs/scope_training.md`)

Dos clasificadores NLI afinados sobre `MoritzLaurer/deberta-v3-base-zeroshot-v2.0` (etapa A: empresa / sector / no relacionado; etapa B: single / multi) asignan a cada texto una de cuatro clases:
`single`, `multi`, `sector`, `out_of_scope`. El modelo lee el título y los primeros 800 caracteres. `out_of_scope` detiene el texto; no se calcula nada más para él.

### Descripción del dataset

Todos los conjuntos son textos de noticias financieras (artículos del crawl; la muestra held-out va de 2007 a 2026, 193 de 200 de un sitio de noticias, 6 de un portal financiero, 1 de un sitio de investigación). Cada texto es un cuerpo único; el modelo lee su título y los primeros 800 caracteres.
Las etiquetas son uno de seis alcances (`single`, `multi`, `sector`, `market_economy`, `investing_general`, `unrelated`); el clasificador fusiona los últimos tres en `out_of_scope`, lo que da cuatro clases.

| Conjunto | Textos | Conteo por clase | Columnas | Cómo se sorteó |
|---|---|---|---|---|
| Train 1 (`category_gold_v2.csv`) | 284 | single 99, multi 78, sector 43, unrelated 33, market_economy 17, investing_general 14 | `article_id`, `session`, `ticker`, `scope`, `company_labels`, `mentions`, `sub_industry`, `notes` | 240 cuerpos fijados para una compuerta de prompt (estratificados sobre el scope v1) y 44 cuerpos de sector |
| Train 2 (`scope_enrich.csv`) | 51 | sobre todo sector (26), single 14, market_economy 7, el resto 1 o 2 cada uno | como arriba más `rule` | sorteado con reglas para tener más `sector` y `out_of_scope`; más fácil que el corpus |
| Train 3 (`scope_hook.csv`) | 30 | single 18, market_economy 7, investing_general 3, sector 2 | como arriba | sorteado por regla: una empresa del universo en el título y al menos tres términos de mercado o macro en los primeros 800 caracteres |
| **Held-out** (`category_gold_v2_test.csv`) | 200 | single 116, multi 44, market_economy 14, unrelated 10, investing_general 9, sector 7 | como arriba más `stratum` y `frame` | sorteado de una muestra de 1,000 textos del corpus; `stratum` y `frame` dan el peso de cada texto respecto al corpus (los tamaños de `frame` suman 1,000) |
| Validación en cuarentena (`scope_quarantined_validation.csv`) | 225 | 100 predichos `out_of_scope`, 50 `single`, 50 `multi`, 25 `sector` | `draw`, `review_order`, `article_id`, `ticker`, `predicted_scope`, `validator_first`, `correct_first`, `validator`, `correct_scope`, `changed`, `notes` | sorteado por clase predicha entre los textos que el paso apartó como `quarantined`; ninguno en un archivo gold; un sí o no por texto, con el alcance que debería tener cuando es no |

Los textos del held-out traen `article_id`, `ticker`, `company`, `pub_date`, `source_domain`, `title` y `text`; el texto mide 2,557 caracteres en promedio (mediana 2,260, máximo 6,020; los textos de entrenamiento promedian 3,123, mediana 2,899).
El held-out se revisó una vez más contra reglas escritas (14 cambios de alcance) después de ver sus primeros resultados, así que existen dos claves y las cifras deben nombrar la clave.

**El paso de sector (1.2) tiene sus propios conjuntos.** Datos etiquetados: 70 textos de un solo sector del gold etiquetado, agrupados al sector GICS de 2 dígitos, y 100 filas reales de una primera revisión de sí/no (un sector mostrado, `correct` sí o no), cada conjunto dividido por posición (filas pares afinan, filas impares prueban).
La prueba limpia (`sector_validation.csv`, `review` = `fresh`) son 57 filas sorteadas con la configuración final, por cuota: 40 con sector asignado, 12 enviadas a `out_of_scope`, 5 controles (se muestra el sector menos parecido, se espera `no`). Columnas: `review`, `review_order`, `article_id`, `shown_sector`,
`control`, `correct`, `right_sector`, `notes`, `status`, `candidate`, `margin`, `vote_sector`, `sure`, `step_revision`. Sus 167 filas (110 de la primera revisión, 57 limpias) son una medida del paso, nunca datos de entrenamiento.

### Selección del modelo

La tarea: decir `single`, `multi`, `sector` o `out_of_scope` a partir del título y la apertura, con la barra en 80% de acuerdo (bajada desde 90% el 2026-11-14). Acuerdo ponderado en el held-out (la clave anterior):

| Enfoque | Acuerdo | Qué enseñó |
|---|---|---|
| Siempre `single` | 58.2% | el piso |
| Reglas actuales (`size_triage`) | 68.2% | buenas en `single` (91%), casi ciegas a `sector` y `out_of_scope` |
| TF-IDF y regresión logística, título | 66.4% | una bolsa de palabras no puede contar empresas |
| Título más rasgos del matcher | 68.1% | sin mejora sobre las reglas |
| NER más rasgos de relevancia | 71.1% | una mejora real, débil en `sector` y `out_of_scope`; 4.6 h para el corpus |
| Dos etapas sobre embeddings de oraciones (bge-base) | 74.5% | mejor equilibrado; el tamaño se eligió tras ver tres sobre los mismos 200 |
| Dos etapas NLI, zero-shot y afinadas con 284 | 72.4% y 73.4%; solo la etapa B 90.9% entre textos de empresa | no hace falta un paso de embeddings |
| **Dos etapas NLI, ambas afinadas, de extremo a extremo** | 79.9%, luego 83.5% con 51 etiquetas más de clases raras | la mejor; recall de `sector` 82% |
| Igual, más 30 etiquetas "hook" tal como se etiquetaron primero | 80.4% | se hallaron etiquetas contradictorias; etiquetas revisadas contra reglas escritas |
| **Seleccionado: el modelo fijado tras la revisión de etiquetas** | **83.6% [78.1, 88.9]** | ganancia pareada sobre las reglas +16.5 [9.6, 23.4] |

El modelo seleccionado son dos copias afinadas del `MoritzLaurer/deberta-v3-base-zeroshot-v2.0` fijado (etapa A empresa / sector / no relacionado; etapa B single / multi). Se eligió por su ganancia pareada sobre las reglas, porque reutiliza un modelo ya fijado (sin modelo nuevo ni paso de embeddings) y porque
cuesta dos pasadas cortas por texto. Se aceptó a propósito un compromiso conocido: el recall de `out_of_scope` es cerca de la mitad mientras el de `single` es 92%, porque un `single` leído como `out_of_scope` se detiene en el paso 1 y nunca llega a las vistas de empresa, y lo contrario solo cuesta cómputo.
El paso de sector eligió el embedder pequeño `BAAI/bge-small-en-v1.5` con descripciones de sector v2, cuatro clases de "no es un sector" y una compuerta (el margen o los votos de las empresas coinciden; un empate no es coincidencia); los embedders más grandes ganaron de 11 a 15 puntos de precisión en una mitad y perdieron de 12 a 18 en la otra, y no se adoptaron.

### 1.1 Clasificador de scope

**200 textos held-out** (nunca entrenados). Uno contra el resto, recalculado a partir de las predicciones guardadas (`data/scope_end_to_end_with81b.csv`, la clave del primer ajuste: 26 `out_of_scope`).
Ponderado significa ponderado al corpus por estrato.

| Clase (n gold) | Precisión | Recall | Precisión ponderada | Recall ponderado |
|---|---|---|---|---|
| `single` (118) | 85.7% | 91.5% | 86.9% | 91.5% |
| `multi` (47) | 97.4% | 78.7% | 97.2% | 79.5% |
| `sector` (9) | 46.2% | 66.7% | 47.0% | 82.4% |
| `out_of_scope` (26) | 65.2% | 57.7% | 55.2% | 49.3% |
| Acuerdo global | 83.0% | | 83.5% | |

Los pesos fijados, evaluados por el documento con la clave nueva (33 `out_of_scope`), dan **83.6% [78.1, 88.9]** de acuerdo ponderado (las reglas llegan a 67.2%; ganancia pareada +16.5 [9.6, 23.4]) y recall de `out_of_scope` 53% [32, 74];
la barra que fijó el dueño es 80%. Esas cifras por clase no se recalcularon aquí (la clave difiere), así que la tabla de arriba es la corrida del primer ajuste, no los pesos fijados.

Matriz de confusión (filas gold, columnas predicho):

| Gold \ Predicho | single | multi | sector | out_of_scope |
|---|---|---|---|---|
| single | 108 | 0 | 4 | 6 |
| multi | 8 | 37 | 1 | 1 |
| sector | 2 | 0 | 6 | 1 |
| out_of_scope | 8 | 1 | 2 | 15 |

**Muestra en cuarentena** (225 textos leídos por el dueño; el sorteo fue por clase predicha, así que esto es solo precisión):

| Clase predicha | Leídos | Correctos (clave final) |
|---|---|---|
| `out_of_scope` | 100 | 93.0% [86.3, 96.6] |
| `single` | 50 | 90.0% [78.6, 95.7] |
| `multi` | 50 | 82.0% [69.2, 90.2] |
| `sector` | 25 | 88.0% [70.0, 95.8] |
| Todas, ponderado | 225 | **90.8% [86.9, 94.7]** |

**Qué se pierde.** En el held-out, noticias reales de empresa o de sector leídas como `out_of_scope`: 8 de 174 (4.6% ponderado), y 7 de las 100 decisiones `out_of_scope` de la muestra en cuarentena eran erróneas (4 de las 7 eran historias de sector).
Siete de las 8 pérdidas del held-out nombran la empresa en el título o en los primeros 800 caracteres, que es todo lo que lee el modelo, y los errores no se concentran en textos que nombran varias empresas (10 de los 23 textos de noticias con alcance erróneo, frente a 47% de todos los textos de noticias).

**Límites.** El held-out se vio muchas veces y su clave se corrigió después de ver los errores; sus intervalos de `sector` y `out_of_scope` son amplios (9 y 26 textos). El ruido por semilla casi no se midió (tres ajustes, todos con semilla 1).
El recall de `out_of_scope` sobre los textos en cuarentena y la auditoría del gold bajo las reglas 17, 18 y 21 se cerraron por decisión, no se hicieron (T-179, T-178). Las dos cifras de arriba (83.6% y 90.8%) son poblaciones y claves distintas.

### 1.2 Sector (paso 4, `docs/sector_step.md`)

El paso asigna a cada texto `sector` un sector GICS (11 clases) a partir de embeddings de oraciones (bge-small, fijado), los votos de sector de las empresas que el texto
nombra y cuatro clases de "no es un sector". La compuerta conserva una respuesta cuando el margen del embedding es grande o los votos de las empresas coinciden; en caso contrario el texto
no recibe sector (44% de los textos de sector no reciben ninguno, por diseño).

**La prueba limpia** (2026-11-13): 57 filas sorteadas por cuota con la configuración final, de textos que no están ni en el gold ni en la primera revisión; el dueño respondió sí o no
(`gold/sector_validation.csv`, conjunto 12). 40 sectores asignados, 5 controles, 12 enviados fuera.

| Medida | n | Correctos | Intervalo 95% |
|---|---|---|---|
| El sector asignado es correcto (regla de votos del sorteo) | 40 | 35 (87.5%) | 74 a 95% |
| Lo mismo con la regla tal como es ahora (un empate no es coincidencia) | 39 | 35 (89.7%) | 76 a 96% |
| Controles (se muestra el sector menos parecido) | 5 | 0 sí | |
| Enviados a fuera de alcance y el dueño está de acuerdo | 12 | 7 (58%) | 32 a 81% |

**Precisión por sector** (asignados en la prueba limpia, recalculada desde el archivo versionado; 40 filas): los conteos por sector son minúsculos y el documento dice que las
cifras por sector **no son utilizables**. Se listan para que se vean, no para citarlas.

| Código GICS | Sector | Asignados | Correctos | Precisión |
|---|---|---|---|---|
| 10 | Energía | 3 | 3 | 1.00 |
| 15 | Materiales | 3 | 2 | 0.67 |
| 20 | Industriales | 4 | 4 | 1.00 |
| 25 | Consumo discrecional | 5 | 4 | 0.80 |
| 30 | Consumo básico | 2 | 1 | 0.50 |
| 35 | Salud | 4 | 3 | 0.75 |
| 40 | Financieros | 5 | 4 | 0.80 |
| 45 | Tecnología de la información | 5 | 5 | 1.00 |
| 50 | Servicios de comunicación | 4 | 4 | 1.00 |
| 55 | Servicios públicos | 2 | 2 | 1.00 |
| 60 | Bienes raíces | 3 | 3 | 1.00 |
| | **Todos** | **40** | **35** | **0.875** |

Los 5 errores: dos no eran textos de sector (inversión, mercado), tres eran de otro sector (una nota de internet europeo leída como consumo discrecional,
una historia de tiendas de un dólar leída como consumo básico, una historia de alimentos leída como salud).

**Recall: no se midió por sector.** Lo que se sabe:
- 44% de los textos `sector` no reciben sector. De los 12 enviados fuera en la prueba, el dueño habría dado un sector a 5, y la mejor conjetura del paso fue ese sector en 4.
  Estimado con las tasas del corpus (sobre 12 textos, muy amplio): por cada 100 textos `sector`, 56 se asignan, unos 50 de ellos correctos; la compuerta renuncia a cerca de una quinta parte de los
  sectores correctos disponibles (unos 15 de 65).
- Materiales, Consumo básico y Servicios públicos son los sectores que el paso da menos en la muestra aleatoria; Materiales no tiene un texto limpio etiquetado en el gold.
- Mitades de ajuste y de prueba (las decisiones se tomaron sobre ellas, así que son optimistas): la compuerta conserva 65% de los textos con 83.6% de precisión.

**Límites.** Un revisor, un sorteo, de 39 a 40 filas; los intervalos miden 20 puntos de ancho. Cerca de la mitad de las respuestas erróneas de la primera revisión no eran textos de sector en absoluto,
así que el paso de scope alimenta a este. La barra de D24 (80% contra el gold held-out) **no se mide** para la subindustria (T-167).

## 2. Cut (paso 1.2, `docs/scope_training.md` sección 8.3)

El extractor `units` (corte de referencia `units5-d3e2cbfa830e`) divide un texto `multi` en piezas, cada una pensada para contener la historia de una empresa. El cut **no tiene clases**,
así que no hay tabla de uno contra el resto. El dueño marcó cada pieza con tres preguntas de sí/no, que se leen como una precisión por propiedad:

| Pregunta | Sí cuando |
|---|---|
| `faithful` | la pieza dice lo que dice el artículo, sin nada añadido ni cambiado (la barra del extractor es 80%, D32) |
| `one_story` | cuenta la noticia de una empresa o de una historia |
| `complete` | se lee como un todo (no es una cláusula cortada a la mitad ni un encabezado solo) |

### Descripción del dataset

Detrás del cut hay tres tipos de datos; el cut es un modelo que decide a qué empresa pertenece cada párrafo de un cuerpo.

| Conjunto | Tamaño | Columnas y etiquetas | Características |
|---|---|---|---|
| Etiquetas de entrenamiento de atribución (`company_segments_train.csv`) | 33,624 párrafos de 2,834 cuerpos; 21,682 `lead`, 11,942 `continuation` | `article_id`, `body_hash`, `year`, `split`, `segment_index`, `ticker`, `kind` (solo ids y posiciones, sin texto) | las etiquetas son automáticas, de cuerpos escritos con una sección por empresa; no están etiquetadas a mano |
| Verificación de etiquetas (`segment_check_gold.csv`) | 200 párrafos (100 `lead`, 100 `continuation`) | `key`, `article_id`, `segment_index`, `ticker`, `kind`, `year`, `correct`, `notes` | las etiquetas `lead` acertaron 88.0%, las `continuation` 38.0%, por eso el modelo entrena solo con `lead` |
| Conjuntos de fidelidad (`faithfulness_gold.csv`, conjunto de confirmación) | primer conjunto 30 cuerpos narrativos, 437 textos (336 fieles, 101 no); conjunto de confirmación 30 cuerpos distintos, 395 textos (345 fieles, 50 no) | `key`, `article_id`, `ticker`, `text_hash`, `extractors`, `revisions`, `pub_date`, `faithful`, `notes` | cuerpos narrativos con varias empresas; textos de cinco extractores, con el extractor oculto y las filas mezcladas; el grupo de confirmación se agotó, así que no se puede sortear un tercer conjunto |
| **Muestra del cut** (`cut_sample_marks.csv` y textos, las cifras de la sección 2) | 28 cuerpos, 143 piezas (5.1 por cuerpo, máximo 13), todas de un mismo sitio de noticias, publicadas de 2007 a 2026 | `review_order`, `article_id`, `segment_index`, `ticker`, `pub_date`, `source_domain`, `title`, `text` (la pieza), `body` (el cuerpo completo, primeros 6,000 caracteres), `company`, `faithful`, `one_story`, `complete`, `notes`; `kind` almacenado | sorteada al azar del corte `units5-d3e2cbfa830e`; las piezas son cortas (media 434 caracteres, mediana 266, máximo 3,380); `ticker` y `company` son la etiqueta del crawl para el artículo, no una marca de la pieza; `kind`: `kept` 110, `wrong_scope` 15, `ambiguous` 14, `wrong_company` 4 |

### Selección del modelo

Se compararon cinco extractores sobre los conjuntos de fidelidad; todos cortan el cuerpo en piezas y deciden a qué empresa pertenece cada pieza. La barra es 80% de textos fieles.

| Extractor | Cómo corta y decide |
|---|---|
| `baseline` | oraciones; una oración pertenece a las empresas que nombra, o a las de la oración anterior; sin modelo |
| `model` | oraciones; una oración que no nombra a nadie va a la empresa a la que el modelo entrenado da el puntaje más alto (al menos 0.5) |
| **`units`** | líneas, viñetas y cláusulas, sin el pie de página; decide el modelo entrenado |
| `units2` | como `units`, con una guarda sobre formas cortas de nombres y sobre líneas que abren con nombres y un guion |
| `units3` | como `units2`, y una pieza de esa línea también pertenece a las empresas que ella misma nombra |

El modelo entrenado es un clasificador de pares ("¿esta oración trata de esta empresa?") afinado sobre el codificador del modelo de sentiment, con el nombre de la empresa enmascarado y la oración anterior como contexto, solo con etiquetas `lead`.
`units` se seleccionó el 2026-11-08: en el conjunto de confirmación pasa la barra (86.1% [80.9, 90.7] sobre toda empresa nombrada, 95.3% [92.0, 98.2] sobre empresas reales), con textos más cortos que el baseline (570 caracteres frente a 952) con la misma proporción de fieles (86.5% y 86.6%) y
menos texto de otras empresas. No se muestra mejor que el baseline en cuerpos nuevos; se mantuvo porque no es peor y es más magro. El margen es estrecho: pasar de una a tres marcas a `no` baja el extremo inferior de 80%. Luego el cut se releyó como `units4`, y **`units5` (`units5-d3e2cbfa830e`) es el cut final**;
su checkpoint está fijado a la copia privada del Hub `gamug/FinBERT-news-attribution@436ad700`.

**Muestra**: 28 textos sorteados al azar del cut, 143 piezas, marcadas por el dueño el 2026-11-12. Proporción de sí con intervalo de Wilson de 95% sobre piezas:

| Nivel | Piezas (textos) | faithful | one_story | complete | limpias (3 sí) | Textos todos limpios |
|---|---|---|---|---|---|---|
| **Todas (la cifra a citar)** | 143 (28) | 84.6% [77.8, 89.6] | 78.3% [70.9, 84.3] | 83.2% [76.2, 88.5] | 75.5% [67.9, 81.8] | 16 de 28 |
| Sin las piezas de textos con alcance erróneo | 128 (27) | 89.1% [82.5, 93.4] | 82.0% [74.5, 87.7] | 87.5% [80.7, 92.2] | 78.9% [71.0, 85.1] | 16 de 27 |
| También sin los textos ambiguos | 114 (26) | 93.9% [87.9, 97.0] | por construcción | 91.2% [84.6, 95.2] | por construcción | por construcción |

El segundo y el tercer nivel se eligieron después de que el dueño viera los fallos; son el cut con los errores de sus entradas apartados, no cifras para citar solas.
Los intervalos que remuestrean textos completos son más amplios (para `faithful` sobre todas las piezas: [74.8, 92.9]).

**Frente a la barra de 80% para `faithful`**: la estimación puntual pasa (84.6%), el extremo inferior del intervalo (77.8%) no.

**La empresa principal por pieza** es una medida del paso 3 (el modelo de relevancia), no del cut: sus cifras de entrenamiento-validación y held-out están en las secciones 3.2 y 3.3.

**Qué no se mide.**
- El **recall** del cut (si el cut encontró todas las historias del texto): no se midió.
- **Cómo le va al cut en todos los artículos del crawl.** Solo corre sobre textos que el paso de scope llamó `multi`; no existe cifra held-out de su salida sobre todo el corpus (los conteos de la corrida del corpus no se conservan aquí).

## 3. NER y la elección de la empresa (paso 3, `docs/companies_step.md`)

El paso 3 tiene dos partes que se miden por separado: NER (encuentra tramos de organización) y el modelo de relevancia (decide cuáles empresas de un texto son las principales).

### Descripción del dataset

No hay un gold de NER sobre los textos que lee el paso 3. Lo que existe:

| Conjunto | Tamaño | Columnas y etiquetas | Características |
|---|---|---|---|
| Datos de entrenamiento de NER (FiNER-ORD, usados por quien publicó el modelo) | no se conservan en este repositorio | tramos de entidad de tipo `PER`, `LOC`, `ORG` | datos públicos de NER de noticias financieras; el repo registra el modelo, no los datos |
| Evaluación juzgada (v1, `docs/evaluation.md`) | 1,000 artículos completos muestreados, 953 procesados | tramos juzgados `wrong` o `missed` por un juez LLM | no es un gold hecho a mano; artículos completos, no los resúmenes ni piezas que lee el paso 3 |
| Entrenamiento y evaluación de relevancia (los 809 resúmenes marcados) | 809 pares (texto, empresa) de 855 resúmenes etiquetados (dos sorteos, 450 y 405) | `key`, `article_id`, `ticker`, `company`, `body_hash`, `rule_scope`, `revision`, `predicted`, `pool_class_size`, `block`, `batch`, `slice`, `text_hash`, `pub_date`, más la marca | cada fila es un texto (un resumen de un artículo, unas pocas oraciones) y una empresa; relevante = la marca tiene dirección (positiva o negativa), no relevante = neutral (202 de las 809 son neutrales); los sorteos equilibran las clases que predijo el modelo de sentiment anterior, por eso las cifras usan pesos |
| Marcas de empresa principal sobre sub-news (`data/main_company/to_tag.csv`, local) | 290 filas sobre 153 sub-news del corte `units5` (la primera versión tenía 353 sobre 180) | `item_id`, `texto`, `texto_es`, `ticker`, `empresa`, `es_principal` (si es la empresa principal, `yes`/`no`), `falta_la_principal` (si falta la empresa principal), `notas` | seis bandas de relevancia; 119 filas de 59 ítems están marcadas (100 `yes`, 19 `no`) |

### Selección del modelo

**NER.** `gamug/sec-bert-finer-ord-ner` (SEC-BERT-BASE afinado sobre FiNER-ORD, fijado en `ba7b9e43`) es el modelo del pipeline; solo encuentra tramos de organización, que un matcher convierte en tickers del universo. Los documentos leídos para este informe **no registran comparación con otro modelo de NER**, solo la corrida juzgada de la sección 3.1. La opción `--candidates ner+text`
(NER más el matcher aplicado sobre el texto, sin coincidencias en minúsculas y sin coincidencias dentro de líneas de divulgación) se eligió sobre `ner` solo por el dueño el 2026-11-13, por la proporción de ítems que etiqueta (una proporción de salida de una corrida de prueba, no una métrica de corrección: sus cifras no se conservan aquí; véase `docs/companies_step.md`, "Dry run").

**Relevancia (cuál empresa es la principal).** La base es la lectura zero-shot del modelo de category (`MoritzLaurer/deberta-v3-base-zeroshot-v2.0`, "este texto trata principalmente de <empresa>"), la mejor de nueve variantes zero-shot por AUC (0.743; el ranking del propio modelo de sentiment fue 0.795). Afinado sobre las 809 marcas, su AUC fuera de pliegue es 0.90, y el
modelo publicado (`gamug/deberta-v3-company-relevance@8812c8bc`, privado) es el que corre el paso. El umbral 0.5 se eligió sobre esos puntajes fuera de pliegue (recall de principal 94.7%, precisión 90.0%); los otros valores probados (0.3, 0.7, 0.9) cambian poco en resúmenes, pero en sub-news la proporción de ítems con empresa principal se mueve mucho a lo largo del rango (una proporción de salida de una corrida de prueba, que no se conserva aquí), lo que los 809 resúmenes no pueden resolver.

### 3.1 NER (`gamug/sec-bert-finer-ord-ner`)

La única medida es la evaluación v1 de 2026-09: un juez LLM sobre 1,000 **artículos completos** muestreados (953 procesados), no sobre los resúmenes ni las piezas del paso 3.

| Medida | Valor |
|---|---|
| Micro-F1 de tramos | 0.7418 |
| Recall / precisión | 0.84 / 0.66 (el modelo sobrepredice) |
| Proporción de tramos predichos que el juez llama erróneos | 33.8% |
| F1 por tipo | ORG 0.75, LOC 0.82, PER 0.66 |

Solo los tramos `ORG` importan para el paso 3. **No hay un gold OVR por clase sobre los textos que lee el paso 3**, y el juez no es el dueño. Los tramos se convierten en tickers con un matcher, que tiene
vacíos conocidos (nombres cortos como `Disney` o `JPMorgan`, T-181). Los tramos crudos no se almacenan.

### 3.2 Relevancia: si una empresa es "principal" en un texto (`gamug/deberta-v3-company-relevance`)

Dos clases: principal (relevante) y no principal. Puntajes fuera de pliegue de los **809 resúmenes marcados**, sin ponderar, umbral 0.5 (el valor del paso):

| | Valor |
|---|---|
| AUC fuera de pliegue | 0.90 |
| Clase **principal**: recall ("filas relevantes conservadas") | 94.7% |
| Clase **principal**: precisión | 90.0% |
| Clase **no principal**: recall (1 menos "filas no relevantes llamadas principales", 31.7%) | 68.3% |
| Clase no principal: precisión | no figura en el registro |

En los otros umbrales (0.30, 0.50, 0.70, 0.90): recall de principal 96.2 / 94.7 / 92.3 / 87.3%, precisión 89.7 / 90.0 / 91.1 / 91.7%, proporción de filas no principales llamadas principales 33.2 / 31.7 / 27.2 / 23.8%.
Entre una cuarta parte y un tercio de las filas que el dueño marcó "no relevante" se llaman principales, sea cual sea el umbral.

**Límites.**
- Las cifras de validación cruzada de arriba **no son held-out** (el umbral se eligió sobre ellas), y el modelo se entrenó sobre resúmenes. **Su validación held-out es sobre piezas: las marcas parciales del dueño de la sección 3.3** (119 filas de 59 piezas, textos que el modelo nunca vio: precisión de principal 90.9%, recall 30.0% con 0.5) (T-165, T-170). Las proporciones de ítems con empresa principal de una corrida de prueba que da el documento del paso (`docs/companies_step.md`, "Dry run") no se conservan aquí (dicen qué produce el modelo, no si acierta); se mueven mucho con el umbral, y nada dice cuál valor es el correcto.
- No se midió la división entre "no se nombra ninguna empresa del universo" y "se nombra en una forma que el matcher no conoce".
- Los candidatos de texto que añade el matcher (`ner+text`) los revisó el desarrollador, no el dueño, sobre unos pocos pares; el acuerdo de la empresa principal con el ticker del crawl es un piso de corrección, no una medida. Ninguna de las dos cifras se conserva aquí.

### 3.3 Empresa principal en piezas: las marcas parciales del dueño (T-170)

El conjunto de la verificación de empresa principal (`data/main_company/to_tag.csv`, local y fuera de git; estratos en `to_tag.strata.json`) tiene 290 filas sobre 153 sub-news del corte `units5`
(`units5-d3e2cbfa830e`, empresas `companies-f349536cf701`, semilla 20261113), seis bandas de relevancia de 30 ítems (la primera versión tenía 353 filas sobre 180 ítems; el archivo de estratos aún lista 180). El dueño marcó `es_principal` (`yes` o `no`: si esta empresa es la principal de la pieza)
en **119 filas de 59 ítems** (100 `yes`, 19 `no`; 171 filas siguen sin marcar). La decisión del modelo es su puntaje de relevancia con el umbral del paso (0.5), tomado de `to_tag.strata.json`. Calculado el 2026-11-15 sobre el archivo tal como estaba entonces, sin ponderar
(las bandas se muestrearon por igual, no en proporción a la población, así que no son tasas del corpus).

| Clase | n marcadas | Precisión | Recall |
|---|---|---|---|
| Principal (`yes`) a 0.5 | 100 | 90.9% [76.4, 96.9] (30/33) | 30.0% [21.9, 39.6] (30/100) |
| No principal (`no`) a 0.5 | 19 | 18.6% (16/86) | 84.2% (16/19) |
| Principal a 0.3 | 100 | 90.0% [78.6, 95.7] (45/50) | 45.0% [35.6, 54.8] (45/100) |
| Principal a 0.7 | 100 | 85.0% [64.0, 94.8] (17/20) | 17.0% [10.9, 25.5] (17/100) |

El puntaje separa los `yes` de los `no` del dueño con un AUC de **0.663** (sin ponderar), contra 0.896 fuera de pliegue sobre resúmenes: más de 5 puntos por debajo, el disparador de la regla de T-170 (las sub-news tienen su propio umbral, elegido sobre las sub-news marcadas). Las marcas son parciales y la regla del dueño es más amplia que la del modelo (abajo), así que es una razón para decidir, no la decisión.
Las marcas, sus textos y el scorer están versionados: `gold/main_company_marks.csv`, `gold/texts/main_company_marks_texts.csv`, `scripts/score_main_company_marks_2026_11_16.py`; la cuenta completa es `docs/main_company_training.md`.

Leído por ítem (59 piezas): 53 tienen una empresa que el dueño marcó como principal; el umbral del modelo encuentra al menos una de ellas en 30 (56.6%). El modelo declara una empresa principal en 32 piezas, y el dueño coincide en al menos una en 30 (93.8%).

Proporción de `yes` por banda de relevancia del ítem (la banda 0 tiene los puntajes más bajos): banda 0, 19 de 21 filas; banda 1, 19 de 25; banda 2, 20 de 22; banda 3, 20 de 20; banda 4, 10 de 11; banda 5, 12 de 20.

**Qué dicen las cifras.**
- **Cuando el modelo declara principal a una empresa, casi siempre acierta** (precisión 0.91 a 0.5, 0.85 a 0.7). **Se le escapa la mayoría de lo que el dueño llama principal** (recall 0.30): 19 de las 21 filas de la banda más baja son `yes`.
- **Tres piezas aportan 21 de las 119 filas**, y todas sus filas son `yes` (ítems 335155, 343149 y 260642: 10, 7 y 4 empresas). Las dos primeras son listas de tickers, y el modelo da a sus empresas puntajes de 0.01 a 0.16. Sin esas tres piezas quedan 98 filas (79 `yes`, 19 `no`):
  precisión de principal 30 de 33 = 0.91, recall 30 de 79 = 0.38, precisión de no principal 16 de 65 = 0.25, recall 16 de 19 = 0.84. El recall sigue bajo, así que las listas no son toda la causa.
- La regla del dueño puede ser más amplia que la del modelo: las piezas con varias empresas reciben varios `yes`, mientras que el modelo se entrenó para elegir aquella de la que trata principalmente el texto (los 809 resúmenes). Las marcas no dicen cuál lectura se quiere; eso es una decisión, no un error del marcado.
- La segunda columna, `falta_la_principal` (si falta la empresa principal en la lista), está llena en las 119 filas marcadas (114 `no`, 5 `yes`) y coincide con `es_principal` salvo en una fila (`yes` y `yes`). No se puntúa aquí.
- 59 piezas y un solo lector; 41% de las filas están marcadas y la muestra es estratificada, así que los intervalos son anchos y las cifras no son tasas del corpus.

Lo que respalda: cuando el paso nombra una empresa principal en una pieza, suele ser una que el dueño acepta; no respalda que el paso encuentre las empresas principales del dueño (recall 0.30 a 0.45). Una tasa sobre el corpus no está respaldada.

## 4. Summarization (paso 2, `docs/summary_step.md`, `docs/summarization_training.md`)

Desde T-206 (dueño, 2026-11-15) el paso resume con **DeepSeek**, a través de una API, un texto de más de unos 2,000 caracteres (una llamada por texto, prompt `config/summary_prompt.md`, temperatura 0); `--summariser local` conserva `sshleifer/distilbart-cnn-12-6` (chunk-and-reduce, el brazo B de abajo) y la cadena v1 sigue con distilbart. Un texto de hasta unos 2,000 caracteres se almacena tal cual. Summarization **no tiene clases**, así que no hay precisión ni recall: la única medida es un juez LLM (DeepSeek, temperatura 0) que puntúa fidelidad, cobertura y concisión de 1 a 5
y lista alucinaciones. **La corrida sobre el corpus no se mide**: las cifras son de 300 artículos, y el juez de la corrida no debe ser DeepSeek (`docs/summary_step.md`).

### Descripción del dataset

No hay datos etiquetados: la evaluación usa la salida almacenada del pipeline y un juez. La muestra del experimento son **300 artículos, 100 por nivel de `num_chunks`** (1, 2 y 3 o más fragmentos), sorteados con semilla 1 de los artículos de una corrida de evaluación anterior (288 en inglés, 12 en español; el summarizer solo lee inglés).
La salida del juez por resumen es `faithfulness`, `coverage` y `conciseness` (1 a 5), una lista de `hallucinations` y una justificación; los resúmenes idénticos se juzgan una vez (en caché por artículo y hash del resumen).
El corpus sobre el que corre el paso: 93,034 cuerpos single (59,432 de más de 2,000 caracteres) y 161,574 sub-news del corte de referencia (5,048 de más de 2,000 caracteres); el resto se almacena tal cual.

### Selección del modelo

**Modelo.** DeepSeek (`deepseek-chat`, que la API responde como `deepseek-flash`), elegido por el dueño el 2026-11-15 con el brazo D del experimento de abajo (pareado contra el paso anterior: fidelidad +0.39, cobertura +1.29, alucinación -14.3 puntos); cada fila de la corrida guarda el nombre de modelo que dio la API. El modelo anterior, `sshleifer/distilbart-cnn-12-6` (fijado en `a4f8f3ea`, heredado de la etapa v1 y no entrenado aquí), queda tras `--summariser local`; antes de DeepSeek nunca se comparó con otro summarizer, solo con formas de usarlo.
**Límite de tamaño.** Un texto de menos de 2,000 caracteres se almacena tal cual. El límite se eligió a partir del conteo de textos que irían al modelo: 1,500 caracteres 75,368 cuerpos single (81.0%) y 9,249 sub-news (5.7%), 17.6 h; **2,000 caracteres 59,432 (63.9%) y 5,048 (3.1%), 13.9 h**; 3,000 caracteres 31,992 (34.4%) y 1,731 (1.1%), 7.5 h; 4,000 caracteres 17,621 (18.9%) y 708 (0.4%), 4.1 h. Las horas son las cifras locales de distilbart (GPU). Con DeepSeek el costo del límite son llamadas y dinero: `docs/summary_step.md` y `docs/summarization_training.md` (T-206) cuentan 79,283 ítems enviados bajo la misma regla de tamaño (unos 310.8 millones de caracteres) y esperan unos $16 a $32 a las tarifas del 2026-11-15, por confirmar.
**Método de los brazos con distilbart (2026-11-13).** Se compararon seis brazos (tabla de esta sección); se mantuvo el chunk-and-reduce actual (brazo B) porque las alternativas no lo superaron en todas las medidas: unir los resúmenes de fragmentos sin compuerta (C1) es peor (+10.7 puntos de alucinación), la compuerta lo recupera (C2), el primer fragmento solo (C0) y las colas literales (C3) son más fieles pero menos cubrientes o más largas. Ese experimento no cambió nada en producción; el brazo DeepSeek (D) vino después.

**El experimento de 2026-11-13** (300 artículos, 100 por nivel de `num_chunks`, 288 en inglés y 12 en español; la misma muestra que la corrida 44). Más alto es mejor, salvo alucinación. El brazo **B es el paso anterior** (distilbart); **D es DeepSeek, el paso desde T-206** (corrido el 2026-11-15 sobre los mismos 300 artículos y el mismo juez).

| Brazo | Qué es | Fidelidad | Cobertura | Concisión | Alucinación | Tokens |
|---|---|---|---|---|---|---|
| A | resumen v1 almacenado, rejuzgado | 4.54 | 3.23 | 4.07 | 19.7% | 71 |
| B | el paso anterior (distilbart, resume de nuevo; `--summariser local`) | 4.60 | 3.24 | 4.08 | 15.0% | 69 |
| C0 | solo el fragmento 1 | 4.70 | 3.31 | 4.33 | 13.0% | 69 |
| C1 | resúmenes de fragmentos unidos, sin compuerta NLI | 4.39 | 3.47 | 3.23 | 25.7% | 122 |
| C2 | C1 con la compuerta NLI | 4.64 | 3.31 | 3.44 | 14.7% | 100 |
| C3 | fragmento 1 filtrado más las primeras oraciones literales de los fragmentos siguientes | 4.81 | 3.13 | 3.32 | 6.7% | 137 |
| **D** | **DeepSeek, el paso desde T-206** | **4.99** | **4.54** | **4.98** | **0.7%** | **86 (tokens de salida por llamada)** |

Por nivel para el paso anterior (B), fidelidad / cobertura / alucinación: 1 fragmento 4.67 / 3.81 / 14%; 2 fragmentos 4.64 / 3.11 / 11%; 3+ fragmentos 4.49 / 2.81 / 20%. La fidelidad baja y la alucinación sube con el número de fragmentos.
Pareado contra B: C0 +0.10 de fidelidad [+0.01, +0.18]; C3 +0.21 [+0.13, +0.30] de fidelidad y -8.3 puntos de alucinación [-12.0, -4.7], con -0.11 de cobertura; C1 es peor (-0.21 de fidelidad, +10.7 puntos).

**DeepSeek (brazo D)** contra B, pareado por artículo (intervalos bootstrap de 95%): fidelidad 4.99 contra 4.60 (+0.39 [+0.29, +0.49]), cobertura 4.54 contra 3.24 (+1.29 [+1.21, +1.39]), concisión 4.98 contra 4.08 (+0.90 [+0.80, +1.00]), alucinación 0.7% contra 15.0% (-14.3 puntos [-18.7, -10.3]). Por nivel (1 / 2 / 3+ fragmentos), D: fidelidad 4.97 / 4.99 / 5.00, cobertura 4.87 / 4.50 / 4.24, alucinación 1% / 1% / 0%. Con temperatura 0 dos corridas dan palabras distintas (30 artículos dos veces: 0 idénticos, solapamiento medio de palabras 0.66) pero los puntajes se mantuvieron (fidelidad 5.00 y 5.00, cobertura 4.70 y 4.63). Una recalificación a ciegas de 24 pares por el asistente (Claude, no una persona; el brazo oculto) coincidió: fidelidad 4.92 contra 4.42, cobertura 4.75 contra 3.08, alucinación 4% contra 21%. Las cifras están en `docs/summarization_results_2026_11_15/`.

**Línea base anterior** (etapa v1, 1,000 artículos muestreados, `docs/evaluation.md`): fidelidad media 4.87, alucinación en 5.4% de los resúmenes, cobertura 3.02: precisos pero poco completos. Es otra corrida de juez y otra muestra, así que no es comparable con la tabla de arriba.

**Límites.** Una muestra, un juez, una corrida por brazo. La cobertura del juez es un puntaje global: **nada verifica si las empresas de las que trata el texto, o su evento principal, están en el resumen** (sin verificación por empresa). El umbral NLI (0.5) no se afinó.
**DeepSeek fue summarizer y juez** en la evidencia del brazo D, así que sus puntajes son su propia lectura de su propio texto (la recalificación a ciegas de 24 pares es pequeña, y los brazos se adivinan por el estilo); la corrida sobre el corpus debe juzgarla otro modelo. El texto del artículo sale de la máquina hacia la API (aceptado por el dueño). Un resumen es salida de modelo, no un hecho, y el texto de DeepSeek no es repetible con temperatura 0. Los brazos con distilbart son trabajo exploratorio; el paso con DeepSeek está construido (#200) y su corrida sobre el corpus aún no se mide.

## 5. Sentiment (paso 6, `docs/sentiment_step.md`, `docs/sentiment_training.md`)

El modelo de pares (texto y empresa, `sentiment_pair`) da tres probabilidades de clase; se apilan con el puntaje de relevancia mediante una regresión logística
(`config/sentiment_stack.json`). Las cifras son de la **validación cruzada anidada** sobre las **809 filas marcadas** (ponderada, D37), **sin held-out**, sobre toda la
población marcada (incluye pares de otras empresas), mientras que el paso puntúa solo pares principales. El gold de 180 textos no se ha evaluado con el stack (T-148c4, T-193). **La única verificación held-out es la sección 5.1:** el paso tal como corrió sobre el corpus, evaluado sobre **190 piezas etiquetadas por el dueño**: exactitud 73.7% [67.0, 79.4] (bajo la barra de 80% de D32), precisión 83.0% / 69.4% / 56.7% y recall en la muestra 76.0% / 80.6% / 53.1% para positivo / negativo / neutral; neutral casi nunca se predice (recall ponderado 4.4%).

### Descripción del dataset

La etiqueta es **lo que un texto dice sobre una empresa**, no el sentimiento de sus oraciones. Cada fila es un texto (normalmente un resumen de un artículo, unas pocas oraciones) y una empresa; la marca es el impacto en el precio que un inversionista leería sobre esa empresa. Si el texto no da ningún hecho sobre la empresa, la marca es neutral, así que neutral mezcla "hechos sin dirección" y "ningún hecho sobre la empresa".

| Conjunto | Tamaño | Conteo por clase | Columnas | Características |
|---|---|---|---|---|
| Pool de oraciones (plata) | 5,800 oraciones fusionadas, 3,865 tras equilibrar | 1,321 negativas, 1,321 neutrales, 1,223 positivas | no está en los archivos gold | oraciones de los artículos del proyecto, etiquetadas por un LLM una a una, sin atarlas a una empresa; entrenó el modelo de partida |
| Gold de 180 textos (`sentiment_gold.csv`) | 180: resúmenes y textos de empresa (120) y textos de extractor (60) | 82 positivos, 65 negativos, 33 neutrales | `key`, `article_id`, `ticker`, `kind`, `part`, `text_hash`, `revision`, `stratum`, `extractor`, `predicted`, `population`, `pub_date`, `sentiment`, `notes` | el examen de la regla de fijación; nunca se afinó con él; aún no se evalúa con el stack |
| Conjunto de entrenamiento 1 (`sentiment_train.csv`) | 450 resúmenes (424 leídos como par) | 232 positivos, 113 negativos, 105 neutrales | `key`, `article_id`, `ticker`, `company`, `body_hash`, `rule_scope`, `revision`, `predicted`, `pool_class_size`, `block`, `batch`, `slice`, `text_hash`, `pub_date`, más la marca | tres lotes de 150, 150 por clase que predijo el modelo anterior |
| Conjunto de entrenamiento 2 (`sentiment_train_2.csv`) | 405 resúmenes (385 leídos como par) | 178 positivos, 116 negativos, 111 neutrales | como arriba | mismo diseño, 135 por clase predicha |
| Replay | 341 oraciones sorteadas de 925 elegibles (las que nombran la empresa de su artículo) | | | se leen como par con esa empresa, mezcladas en todo conjunto de entrenamiento |
| **Verificación held-out** (`gold/sentiment_val190.csv`, sección 5.1) | 190 piezas de la corrida sobre el corpus, dos sorteos (90 + 100); nunca de entrenamiento | 96 positivas, 62 negativas, 32 neutrales | `key`, `article_id`, `item_id`, `ticker`, `batch`, `text_source`, `text_hash`, `predicted`, `predicted_score`, `relevance_score`, `pub_date`, las tres revisiones, `sentiment`, `notes`; los textos están en `gold/texts/sentiment_val190_texts.csv` | piezas (sub-news) del paso de resumen guardado, no resúmenes de artículos; 38 son resúmenes de distilbart, el resto textos cortos originales; la llamada del propio modelo se guarda aparte de la etiqueta del dueño |

Los sorteos equilibran las clases que predijo el modelo anterior, no las clases verdaderas, así que toda cifra se pondera por el tamaño del pool de la clase predicha sobre los textos sorteados de ella. Juntos, los dos conjuntos de entrenamiento dan 809 filas de pares. Ningún texto del gold ni de un conjunto held-out está en un conjunto de entrenamiento ni en el replay.
Los textos de sector, mercado y economía (46 filas) se etiquetan por su alcance y se dejan fuera del modelo de pares.

### Selección del modelo

El modelo de pares lee `[empresa] [SEP] [fragmento de texto]` y promedia las probabilidades de los fragmentos; una regresión logística (el stack) combina sus log-probabilidades con el puntaje de relevancia. Validado de forma cruzada, anidada y ponderada (sin held-out), sobre las 809 filas:

| Candidato | Puntaje (D37) | Qué enseñó |
|---|---|---|
| Modelo v1 fijado, solo texto | 0.648 (60.4% de acuerdo en el gold de 180 textos) | el punto de partida; no oye a la empresa |
| Modelo de pares, la receta de la grilla (27 corridas: épocas 2, 3, 4; tasa de aprendizaje 1e-5, 2e-5, 3e-5; 3 semillas) | 0.678 en 83 textos de validación; **0.737** por validación cruzada | ganancia +0.089 sobre el modelo fijado; el recall neutral sigue en 0.442 |
| Replay más grande (809 oraciones) | 0.723 | no mejora; se queda con 341 |
| Banderas de mención, prior neutral, pistas, un umbral | 0.737 a 0.740 | nada: tres cuartas partes de los textos neutrales sí nombran la empresa |
| Compuerta dura "empresa no nombrada, entonces neutral" | 0.664 a 0.678 | sube el recall neutral, baja la precisión |
| Más relevancia, compuerta | 0.789 | recall neutral 0.754, precisión 0.680 |
| **Más relevancia, stack (seleccionado)** | **0.814** | precisión neutral 0.787, recall 0.686; negativo 0.747 / 0.748 |
| `ProsusAI/finbert` simple como punto de partida, con stack | 0.811 | lo mismo dentro del ruido; se queda el endpoint propio del proyecto |

Seleccionado el 2026-11-10 (la verificación held-out de la sección 5.1 vino después y no participó en la selección): el modelo de pares iniciado desde el endpoint del proyecto `gamug/FinBERT-financial-news`, entrenado con las 809 filas más las 341 oraciones de replay, apilado con el modelo de relevancia afinado. La fijación no está decidida: espera al gold de 180 textos según la regla de D35.

| Clase | Modelo de pares solo P / R | Compuerta P / R | **Stack (el paso) P / R** |
|---|---|---|---|
| Positivo | 0.761 / 0.834 | 0.835 / 0.804 | **0.807 / 0.855** |
| Negativo | 0.680 / 0.745 | 0.756 / 0.734 | **0.747 / 0.748** |
| Neutral | 0.634 / 0.442 | 0.680 / 0.754 | **0.787 / 0.686** |
| P / R macro | 0.692 / 0.674 | 0.757 / 0.764 | **0.780 / 0.763** |
| ROC OVR macro | 0.845 | 0.845 (por construcción) | **0.898** |
| Puntaje (D37) | 0.737 | 0.789 | **0.814** |

Frente al modelo v1 fijado sobre el mismo corte (puntaje 0.648), el stack gana +0.077 [+0.057, +0.095] sobre el modelo de pares solo. La precisión y el recall del negativo están por encima de la barra de D32 (70%) en
la validación cruzada.

**Límites.**
- Las cifras de validación cruzada no son held-out (cerca de una docena de variantes se eligieron sobre las mismas filas), así que son optimistas; los modelos finales son más confiados que los modelos de pliegue sobre los que se ajustó el stack (no se midió). **La verificación held-out de la sección 5.1 es la medida que hay que citar, y es menor** (exactitud 73.7% sobre 190 piezas de la corrida real; recall de neutral en la muestra 53.1% frente a 0.686 en la validación cruzada; ponderado 4.4%): no miden lo mismo (resúmenes marcados, con pares de otras empresas, frente a pares principales de piezas), así que la brecha no es una medida del optimismo.
- Un control con relevancia barajada suma a lo sumo +0.005, así que la ganancia del stack es información real, no ruido.
- **Neutral es la clase más débil por sí sola** (recall 0.44 sin el modelo de relevancia); el fallo no es "la empresa está ausente": el modelo oye una empresa y una polaridad pero no distingue si el texto trata de la empresa.
- El paso lee resúmenes de a lo sumo unos 2,000 caracteres, en fragmentos de a lo sumo el espacio que la empresa deja en 512 tokens, y promedia las probabilidades de los fragmentos.
- Un texto de sector no recibe sentiment (D42).

### 5.1 Verificación held-out sobre 190 piezas de la corrida real (2026-11-16)

La primera verificación held-out del paso de sentiment sobre textos que ningún modelo ni ajuste vio: el paso tal como corrió sobre el corpus, el modelo de pares y el stack de la revisión
`sentiment-5999fe131d8f` (`gamug/FinBERT-company-sentiment@9a7b1eb3`), evaluado sobre **190 piezas etiquetadas por el dueño** (un lector) en dos sorteos.
Las marcas, sus textos y el scorer están versionados: `gold/sentiment_val190.csv`, `gold/texts/sentiment_val190_texts.csv`, `scripts/score_sentiment_val190_2026_11_16.py`
(`scripts/build_val_gold_2026_11_16.py` construye los archivos de gold desde los archivos de trabajo locales).

| Elemento | Primer sorteo (90) | Segundo sorteo (100) |
|---|---|---|
| Fuente | sub-news del paso de resumen guardado (`news_summary`, revisión `summary-7a9bd2062bc4`) con una empresa principal del paso de empresas (`companies-f349536cf701`, relevancia 0.7 o más) y un puntaje de sentiment; el texto es el que lee el paso de sentiment | igual, con un universo más estricto |
| Sorteo | 30 por clase que predijo el modelo (10 resúmenes guardados y 20 textos cortos originales cada una), un texto por artículo | al azar simple, uno por artículo |
| Universo | 57,826 candidatas (positivo 38,559, negativo 18,847, neutral 420) | cerca de 21,000 candidatas |
| Filtros | 250 caracteres o más y dos oraciones; mayormente letras; sin listas de tickers ni líneas de divulgación; la empresa nombrada en el texto; inglés; ningún artículo de `gold/sentiment_gold.csv` ni de los conjuntos de entrenamiento | los mismos, más relevancia 0.9 o más, de 300 a 1,500 caracteres, la empresa nombrada dos veces y dentro de los primeros 100 caracteres, menos de dos llamadas de analistas, ningún artículo del primer sorteo (ni de una muestra anterior de 200 textos que se descartó y no se versionó) |
| Etiquetas | positivo 39, negativo 27, neutral 24 | positivo 57, negativo 35, neutral 8 |
| Columnas | `review_order`, `key`, `article_id`, `ticker`, `pub_date`, `title`, `text`, `spanish_title`, `spanish_text`, `company`, `sentiment`, `notes` (la predicción está en un `key.csv` aparte, con una columna `batch`) | las mismas |

Los textos miden de 250 a 1,626 caracteres (mediana cerca de 400). El título y el texto en español son una ayuda de lectura (`news_nlp.translate`) y no se copian a `gold/`. Una fila tiene la nota "company is not the main" y se conserva.
Los scripts son `scripts/export_sentiment_val_2026_11_15.py` y `scripts/export_sentiment_val100_2026_11_15.py`; los archivos de trabajo siguen locales (`data/sentiment_val90/`, fuera de git).

**Resultados por sorteo** (precisión exacta dentro de cada clase predicha, intervalo de Wilson 95%; el recall en la muestra es la proporción de la clase etiquetada hallada; el recall ponderado da a cada clase predicha el tamaño de su universo sobre las piezas sorteadas de ella):

Primer sorteo, 90 (estratificado por la predicción): exactitud 66.7% [56.4, 75.5] (60 de 90), ponderada 74.4%.

| Clase | Precisión | Recall (muestra) | Recall (ponderado) |
|---|---|---|---|
| Positivo | 80.0% [62.7, 90.5] (24/30) | 61.5% (24/39) | 88.8% |
| Negativo | 63.3% [45.5, 78.1] (19/30) | 70.4% (19/27) | 69.7% |
| Neutral | 56.7% [39.2, 72.6] (17/30) | 70.8% (17/24) | 4.0% |

Segundo sorteo, 100 (al azar, sin necesidad de ponderar): exactitud 80.0% [71.1, 86.7] (80 de 100).

| Clase | Precisión | Recall |
|---|---|---|
| Positivo | 84.5% [73.1, 91.6] (49/58) | 86.0% [74.7, 92.7] (49/57) |
| Negativo | 73.8% [58.9, 84.7] (31/42) | 88.6% [74.0, 95.5] (31/35) |
| Neutral | ninguna llamada (0/0) | 0% [0.0, 32.4] (0/8) |

**Las 190 juntas** (una aproximación: los dos sorteos tienen universos distintos, y las cifras ponderadas los tratan como muestras del universo de 57,826 del primero):

| Clase | Precisión | Recall (muestra) | Recall (ponderado) | F1 (muestra) | F1 (ponderado) |
|---|---|---|---|---|---|
| Positivo | 83.0% [73.8, 89.4] (73/88) | 76.0% [66.6, 83.5] (73/96) | 89.4% | 79.3% | 86.1% |
| Negativo | 69.4% [58.0, 78.9] (50/72) | 80.6% [69.1, 88.6] (50/62) | 78.6% | 74.6% | 73.7% |
| Neutral | 56.7% [39.2, 72.6] (17/30) | 53.1% [36.4, 69.1] (17/32) | 4.4% | 54.8% | 8.2% |

Exactitud: 73.7% [67.0, 79.4] (140 de 190); ponderada al universo 78.4%.
Confusión (filas el modelo, columnas la etiqueta del dueño; positivo / negativo / neutral): positivo 73 / 8 / 7; negativo 14 / 50 / 8; neutral 9 / 4 / 17.

**Frente a las barras de D32** (80% de acuerdo; precisión y recall de negativo de al menos 70%): la exactitud junta, 73.7% [67.0, 79.4], está **bajo 80%** (78.4% ponderada al universo, también bajo); el segundo sorteo solo, 80.0% [71.1, 86.7], está en la barra por estimación puntual, con un intervalo que llega bien por debajo. La precisión de negativo, 69.4% [58.0, 78.9], está **bajo 70%** por estimación puntual; el recall de negativo, 80.6% [69.1, 88.6], la supera.

**Relación con la decisión del pin.** Esta no es la compuerta de D35. La regla de D35 evalúa el gold de 180 textos sobre pares principales (T-148c4, T-193); esa evaluación no se ha hecho y `config/pipeline_models.json` sigue nombrando el modelo v1. Estas 190 piezas no se sortearon con el diseño de D35; miden el paso tal como corrió sobre el corpus. Son evidencia para la decisión del dueño; si reemplazan la evaluación del gold de 180 le toca decirlo al dueño.

**Qué dice.**
- **Positivo** es la clase más fuerte (precisión 83.0%, recall ponderado 89.4%).
- **La precisión de negativo** está bajo la barra en la cifra junta y sobre ella en el segundo sorteo, más limpio (73.8%); el recall de negativo la supera en ambos.
- **Neutral casi nunca se predice**: el modelo llama neutrales a 420 de los 57,826 textos del universo (0.7%), mientras que el dueño etiquetó neutral el 16.8% de las 190 (32); por eso el recall ponderado es 4.4%. Su precisión (56.7%) es moderada. La validación cruzada del modelo de pares dio un recall de neutral de 0.686; sobre piezas reales es mucho menor (no miden lo mismo: resúmenes marcados frente a pares principales de piezas). Registrado como T-212 (propuesto; el dueño decide la corrección).
- El segundo sorteo es más fácil (exactitud 80.0% frente a 66.7%): su filtro hace que la empresa sea el tema del texto, y el primer sorteo incluye también textos donde se menciona de pasada, que el modelo lee como una dirección.

**Límites.** El conjunto junto tiene 190 etiquetas (positivo 96, negativo 62, neutral 32) y 190 llamadas (88, 72, 30). **Positivo y negativo descansan en más de 60 etiquetas y más de 70 llamadas cada uno; neutral es la clase delgada**: 32 etiquetas y 30 llamadas, así que su precisión [39.2, 72.6] y su recall [36.4, 69.1] son anchos.
Los recalls ponderados toman sus proporciones condicionales de las 190 piezas y los tamaños de clase del universo de 57,826 del primer sorteo (los dos sorteos tienen universos distintos: una aproximación). El recall ponderado de neutral, 4.4%, es el de las etiquetas neutral halladas entre las piezas llamadas positivas (7 de 88) y las llamadas negativas (8 de 72), escaladas por 38,559 y 18,847, más las llamadas neutrales (17 de 30, escaladas por 420): es una tasa construida con pocas etiquetas y es una pista, no una medición.
El segundo sorteo es al azar y no necesita ponderación, pero tiene solo 8 etiquetas neutral. El universo son las candidatas tras los filtros, no todo el corpus; las piezas son de la revisión de resumen con distilbart `summary-7a9bd2062bc4` (38 de las 190 son resúmenes; los resúmenes de DeepSeek de la corrida sobre el corpus no están cubiertos). Un lector, sin segunda lectura; la línea entre neutral y una dirección no se fijó de antemano y depende de la regla usada.

## 6. Category (paso 5, `docs/category_step.md`, `docs/category_train.md`)

Aquí se miden dos modelos. **El paso corre el modelo zero-shot sin modificar** (6.1). Un **candidato afinado** (la receta de entailment de `config/category_model.json`, 12 etiquetas, una etiqueta por fila, reajustado con 1,216 filas) **no está
conectado** al paso; su evaluación es de 6.2 a 6.5. Ninguno alcanza la barra de 80% (D24). Cada corrida está en MLflow, experimento `news_nlp_train/category` (`uv run mlflow ui --backend-store-uri ./mlruns`), y los
pesos del candidato son el modelo registrado `category-entailment-single-12` (versión 1, marcado "candidate, not wired").

### Descripción del dataset

Cada fila es un par (texto, empresa) con una etiqueta de la taxonomía de `docs/category-taxonomy.md`: **14 etiquetas en 6 grupos** para el modelo del paso; el candidato lee 12 (`competitive_actions` y `peer_comparison` leídas como `other`).

| Conjunto | Tamaño | Características | Columnas |
|---|---|---|---|
| Filas de entrenamiento (hoja anterior) | 225 filas con etiqueta de empresa | cuerpos single y cuerpos multi estructurados divididos por empresa, del conjunto de entrenamiento de scope; cuerpos multi narrativos excluidos; las hipótesis del baseline se escribieron contra ellas | `article_id`, `company`, `label`, `pred` (archivos de resultados) |
| **Filas de entrenamiento de una sola etiqueta (candidato)** | 1,216: una hoja de 1,000 filas reducida a una etiqueta por fila (999) más 217 de las 225 filas anteriores | 12 etiquetas; `earnings_performance` 547, `market_analyst_sentiment` 214, `product_innovation` 90, `mergers_acquisitions` 81, `legal_regulatory` 45, `leadership_governance` 40, `capital_shareholder_returns` 40, `reputational_harm` 39, `labor_human_capital` 36, `partnerships_business_dev` 32, `other` 32, `investor_activity` 20; 82% son segmentos cortos de empresa de resúmenes de mercado (mediana 272 caracteres); las etiquetas se revisaron después de ver los primeros resultados | por fila: texto, empresa, etiqueta, `fold` (5 pliegues agrupados por artículo) |
| **Held-out** | 200 textos, de los cuales 135 filas single son evaluables (116 cuerpos single); 44 textos `multi` narrativos no se evalúan | artículos completos (mediana 2,085 caracteres, ninguno bajo 600), convención de etiquetado anterior (las calificaciones de analistas podían ser `investor_activity`); evaluado una sola vez | `article_id`, `company`, `kind`, `label`, `pred`, más los 14 puntajes de etiqueta en la corrida mixta |
| Muestra de sub-news reales | 300 filas | un sorteo con semilla en dos lotes de 150 de los pares de sub-news de la primera corrida del corpus cuyo texto resumido mide menos de 1,000 caracteres (142,588 de 159,401 pares); un texto por cuerpo; ninguno de un cuerpo etiquetado; 107 de las 300 están marcadas como poco claras | texto, empresa, etiqueta, banderas |

### Selección del modelo

El paso corre el zero-shot sin modificar `MoritzLaurer/deberta-v3-base-zeroshot-v2.0` en su fijación: premisa `<empresa>: <primeros 2,000 caracteres>`, una hipótesis por etiqueta ("The news about <empresa> is about <definición>"), la mejor probabilidad de entailment da la etiqueta y su grupo da el grupo (6 grupos, opción B). **Ni él ni ningún candidato alcanza la barra de 80% (D24).**

| Candidato | Resultado | Veredicto |
|---|---|---|
| **Zero-shot sin modificar (el paso)** | held-out 43.0% (etiqueta), 55.6% (grupo); 56.4% en las filas de entrenamiento | provisional, corre en producción |
| Afinado plano, validación cruzada de 5 pliegues sobre 225 filas | 64.0% fuera de pliegue | mejor que zero-shot fuera de pliegue |
| Afinado mixto (etiquetas más grupos), 225 filas | 62.2% fuera de pliegue; held-out 52.6% (plano), 48.9% (dos pasos), grupo 62.2% | no publicado, no usado |
| **Candidato de entailment**, reajustado con 1,216 filas, 12 etiquetas | 87.8% fuera de pliegue; **held-out 54.1%** [45.9, 62.3]; 71.3% en 300 sub-news reales | registrado como `category-entailment-single-12` v1, "candidate, not wired" |
| DeBERTa-v3-base simple reentrenado con las mismas filas y pliegues | 85.7% fuera de pliegue | 2 puntos por debajo de la receta de entailment |

El held-out es la única medida que es held-out y falla; el candidato cae lejos de su 87.8% porque sus filas de entrenamiento son segmentos cortos de resúmenes de mercado y las filas del held-out son artículos completos. No es un reemplazo del modelo del paso antes de que se cumpla la barra en el held-out.

### 6.1 El modelo que corre el paso (zero-shot sin modificar)

**Provisional.** El modelo es el `MoritzLaurer/deberta-v3-base-zeroshot-v2.0` sin modificar (zero-shot), con la premisa `<empresa>: <texto, 2,000 caracteres>` y una hipótesis por etiqueta.
**No** cumple la barra de 80% (D24); las etiquetas del dueño son la referencia.

| | Filas de entrenamiento (225) | Filas single del held-out (135) |
|---|---|---|
| Acuerdo de etiqueta | 56.4% | **43.0%** |
| Acuerdo de grupo (6 grupos) | 62.7% | 55.6% |

La cifra del held-out es la honesta: las definiciones se escribieron contra las filas de entrenamiento. El held-out tiene 135 filas evaluables de 200 textos; los 44 textos `multi` narrativos no se evalúan.
Precisión y recall por etiqueta, recalculados a partir de `baseline_heldout.csv` y `baseline_train.csv`:

| Etiqueta | n held-out | P held-out | R held-out | n entrenamiento | P entrenamiento | R entrenamiento |
|---|---|---|---|---|---|---|
| earnings_performance | 27 | 0.71 | 0.74 | 64 | 0.71 | 0.95 |
| product_innovation | 22 | 0.62 | 0.45 | 26 | 0.62 | 0.58 |
| market_analyst_sentiment | 17 | 0.27 | 0.53 | 21 | 0.35 | 0.43 |
| investor_activity | 16 | 0.50 | 0.19 | 8 | 0.43 | 0.38 |
| legal_regulatory | 11 | 0.40 | 0.55 | 6 | 0.40 | 0.33 |
| leadership_governance | 11 | 1.00 | 0.18 | 9 | 0.60 | 0.33 |
| capital_shareholder_returns | 7 | n/a (nunca predicha) | 0.00 | 6 | 0.00 | 0.00 |
| reputational_harm | 7 | 1.00 | 0.14 | 23 | 0.86 | 0.26 |
| mergers_acquisitions | 5 | 0.31 | 0.80 | 15 | 0.44 | 0.73 |
| partnerships_business_dev | 5 | 0.25 | 0.20 | 17 | 0.67 | 0.35 |
| labor_human_capital | 3 | 0.50 | 0.67 | 19 | 0.69 | 0.58 |
| other | 3 | 0.00 | 0.00 | 3 | 0.00 | 0.00 |
| competitive_actions | 1 | 0.00 | 0.00 | 4 | 0.00 | 0.00 |
| peer_comparison | 0 | n/a | n/a | 4 | 0.00 | 0.00 |

Lee las etiquetas con menos de unas 10 filas held-out como anécdotas: una fila acertada o errónea mueve la cifra 10 puntos o más. Solo `earnings_performance` es confiable (recall cerca de 95% en entrenamiento, 74% en held-out).
`market_analyst_sentiment` está sobrepredicha (33 predichas para 17 verdaderas), `mergers_acquisitions` también (13 para 5).

**Variante afinada (no publicada, no usada por el paso).** El afinado de receta mixta alcanzó 62.2% fuera de pliegue en entrenamiento y 48.9% en el held-out (etiqueta, dos pasos), y 62.2% en el grupo del held-out;
su ROC OVR en el held-out es **0.798** macro sobre las 13 etiquetas con filas held-out (0.783 sobre los grupos). Tampoco alcanza la barra. La precisión y el recall por clase de esa variante
están en `docs/category_results_2026_11_13/` (`heldout_mixed_scores.csv`) y no se tabularon para este informe.

**El grupo es el campo más confiable** (el propio consejo del documento): 55.6% en held-out, frente a 43.0% de la etiqueta.

### 6.2 Candidato: validación cruzada (fuera de pliegue, 1,216 filas de una etiqueta, 5 pliegues agrupados por artículo)

Filas: la hoja de 1,000 filas del dueño reducida a una etiqueta por fila (999) más las 225 filas anteriores releídas (217 conservadas); 12 etiquetas (`competitive_actions` y `peer_comparison` leídas como `other`). **Las etiquetas se revisaron después de ver los
primeros resultados, y por el asistente**, así que esto mide qué tan bien aprende la receta esas etiquetas; no es una exactitud limpia. Las filas son segmentos cortos de empresa de resúmenes de mercado (mediana 272 caracteres), no artículos completos.

Exactitud **87.8%**, F1 macro 0.748, puntaje de selección D37 **0.821** (media macro de precisión, recall y ROC OVR). Las mismas filas evaluadas con el modelo sin modificar: 75.5% / 0.525 / D37 0.680 (1,214 de las filas).
DeBERTa-v3-base simple reentrenado con las mismas filas y pliegues: 85.7% / 0.705 / D37 0.792, es decir 2 puntos por debajo de la receta de entailment (sin probar significancia).

| Etiqueta | Filas | Precisión | Recall | ROC AUC |
|---|---|---|---|---|
| earnings_performance | 547 | 0.97 | 0.97 | 1.00 |
| mergers_acquisitions | 81 | 0.80 | 0.89 | 0.99 |
| leadership_governance | 40 | 0.88 | 0.90 | 1.00 |
| legal_regulatory | 45 | 0.72 | 0.73 | 0.98 |
| product_innovation | 90 | 0.87 | 0.86 | 0.99 |
| capital_shareholder_returns | 40 | 0.78 | 0.70 | 0.96 |
| labor_human_capital | 36 | 0.84 | 0.86 | 1.00 |
| market_analyst_sentiment | 214 | 0.89 | 0.88 | 0.98 |
| partnerships_business_dev | 32 | 0.77 | 0.75 | 0.97 |
| reputational_harm | 39 | 0.71 | 0.74 | 0.98 |
| investor_activity | 20 | 0.50 | 0.50 | 0.93 |
| other | 32 | 0.29 | 0.19 | 0.80 |

Los intervalos de 95% (por artículo) son de alrededor de ±0.12 a ±0.15 para las clases con 20 a 45 filas (`docs/category_results_2026_11_15/ovr_single.md`).

### 6.3 Candidato: el held-out de 200, evaluado una sola vez (135 filas single)

El candidato se reajustó con las 1,216 filas y se evaluó **una sola vez** sobre las filas single del held-out (los 44 textos narrativos no se evalúan). Las etiquetas del held-out son las del dueño de la convención anterior (las calificaciones de analistas pueden ser
`investor_activity`); no se revisaron. **Acuerdo de etiqueta 54.1% [45.9%, 62.3%]**, acuerdo de grupo 65.2% [57.4%, 72.7%], F1 macro 0.448, puntaje D37 0.615. El modelo sin modificar sobre las mismas filas: 43.7% (43.0% con 14 etiquetas).
**La barra de 80% no se cumple.**

| Etiqueta | Filas | Predichas | Precisión | Recall | ROC AUC |
|---|---|---|---|---|---|
| earnings_performance | 27 | 34 | 0.74 | 0.93 | 0.96 |
| product_innovation | 22 | 25 | 0.60 | 0.68 | 0.89 |
| market_analyst_sentiment | 17 | 19 | 0.42 | 0.47 | 0.79 |
| investor_activity | 16 | 3 | 0.67 | 0.12 | 0.77 |
| leadership_governance | 11 | 5 | 1.00 | 0.45 | 0.64 |
| legal_regulatory | 11 | 14 | 0.43 | 0.55 | 0.92 |
| capital_shareholder_returns | 7 | 4 | 0.50 | 0.29 | 0.73 |
| reputational_harm | 7 | 4 | 0.50 | 0.29 | 0.93 |
| mergers_acquisitions | 5 | 11 | 0.36 | 0.80 | 0.90 |
| partnerships_business_dev | 5 | 1 | 1.00 | 0.20 | 0.73 |
| other | 4 | 9 | 0.11 | 0.25 | 0.71 |
| labor_human_capital | 3 | 6 | 0.33 | 0.67 | 0.98 |

Ocho etiquetas tienen menos de 12 filas: léelas como anécdotas (los intervalos llegan a 0 y 1). Solo `earnings_performance` es ajustada. La mayor confusión es `investor_activity` predicha como `market_analyst_sentiment` (8 filas),
la convención que cambió. **Por qué cae tan lejos de 87.8%:** 82% de las filas de entrenamiento son segmentos cortos de resúmenes de mercado (mediana 272 caracteres) y las filas del held-out son artículos completos (mediana 2,085, ninguno bajo 600 caracteres).

### 6.4 Candidato: una muestra de sub-news reales (300 filas del almacén RESULTS real)

Lo que lee el paso: pares de sub-news de la primera corrida del corpus (`category-65f33b55b26b`, corte `units5-d3e2cbfa830e`) cuyo **texto resumido mide menos de 1,000 caracteres** (`news_summary.summary_text`; 142,588 de 159,401 pares), un
sorteo con semilla de 300 en dos lotes de 150, un texto por cuerpo, ninguno de un artículo o cuerpo de algún conjunto etiquetado. **Las etiquetas son del asistente** (a ciegas de ambos modelos, reglas de `docs/category-taxonomy.md`), no del dueño;
107 de las 300 se marcaron poco claras. El asistente también etiquetó las filas de entrenamiento, así que el acuerdo probablemente es más alto que el que daría un etiquetador independiente.

| | Acuerdo de etiqueta | F1 macro | Etiquetador seguro (193) | Etiquetador inseguro (107) |
|---|---|---|---|---|
| **Candidato** | **71.3%** [66.0%, 76.3%] | 0.637 | 83.9% | 48.6% |
| Sin modificar (el paso) | 65.7% [60.3%, 71.0%] | 0.483 | 79.3% | 41.1% |

Pareado: el candidato acierta donde el modelo sin modificar se equivoca en 38 filas, lo inverso en 21 (prueba de signos p = 0.036). Lotes: 69.3% y 73.3% (candidato), 62.7% y 68.7% (sin modificar).

**Por clase, las 300 filas (candidato / modelo seleccionado)**, precisión y recall con intervalos de 95% (remuestreados por artículo):

| clase | filas | precisión [95%] | recall [95%] | ROC AUC |
|---|---|---|---|---|
| market_analyst_sentiment | 99 | 0.81 [0.73, 0.88] | 0.80 [0.72, 0.87] | 0.95 |
| earnings_performance | 56 | 0.82 [0.72, 0.91] | 0.89 [0.80, 0.96] | 0.98 |
| mergers_acquisitions | 30 | 0.80 [0.65, 0.94] | 0.80 [0.64, 0.94] | 0.98 |
| partnerships_business_dev | 26 | 0.65 [0.48, 0.81] | 0.77 [0.58, 0.93] | 0.95 |
| product_innovation | 18 | 0.46 [0.29, 0.65] | 0.72 [0.50, 0.92] | 0.93 |
| legal_regulatory | 13 | 0.77 [0.50, 1.00] | 0.77 [0.54, 1.00] | 0.99 |
| investor_activity | 7 | 0.75 [0.40, 1.00] | 0.86 [0.50, 1.00] | 0.92 |
| leadership_governance | 5 | 0.29 [0.00, 0.67] | 0.40 [0.00, 1.00] | 0.98 |
| other | 39 | 0.30 [0.00, 0.64] | 0.08 [0.00, 0.18] | 0.68 |

No están en esa tabla (menos de 5 filas): `labor_human_capital` (4 filas) precisión 1.00 [1.00, 1.00], recall 1.00 [1.00, 1.00], ROC 1.00; `reputational_harm` (2) precisión 0.33 [0.00, 0.83], recall 1.00 [1.00, 1.00], ROC 0.99;
`capital_shareholder_returns` (1) precisión 0.25 [0.00, 1.00], recall 1.00 [1.00, 1.00], ROC 1.00.

**Las mismas 300 filas, el modelo que corre el paso (zero-shot sin modificar, 14 etiquetas leídas como 12):**

| clase | filas | precisión [95%] | recall [95%] | ROC AUC |
|---|---|---|---|---|
| market_analyst_sentiment | 99 | 0.67 [0.59, 0.74] | 0.90 [0.84, 0.95] | 0.87 |
| earnings_performance | 56 | 0.79 [0.68, 0.89] | 0.88 [0.78, 0.95] | 0.97 |
| mergers_acquisitions | 30 | 0.59 [0.44, 0.75] | 0.73 [0.58, 0.88] | 0.96 |
| partnerships_business_dev | 26 | 0.78 [0.56, 0.95] | 0.54 [0.32, 0.75] | 0.91 |
| product_innovation | 18 | 0.45 [0.17, 0.78] | 0.28 [0.08, 0.50] | 0.76 |
| legal_regulatory | 13 | 1.00 [1.00, 1.00] | 0.38 [0.12, 0.67] | 0.97 |
| investor_activity | 7 | 0.75 [0.00, 1.00] | 0.43 [0.00, 0.83] | 0.84 |
| leadership_governance | 5 | 0.33 [0.00, 1.00] | 0.20 [0.00, 0.67] | 0.95 |
| other | 39 | 0.28 [0.08, 0.50] | 0.13 [0.03, 0.24] | 0.46 |

(También: `labor_human_capital` 4 filas 0.75 / 0.75 / 0.99; `reputational_harm` 2 filas 0.20 / 0.50 / 0.92; `capital_shareholder_returns` 1 fila, nunca predicha, recall 0.00, ROC 0.94.)

Las etiquetas con menos de 10 filas (`investor_activity`, `leadership_governance` y las tres bajo la tabla) son anécdotas. **`other` es el fallo en ambos modelos**: el candidato acierta 3 de 39 y el modelo sin modificar 5 (ROC 0.68 y 0.46); el modelo sin modificar sobreasigna
`market_analyst_sentiment` (133 predichas para 99 verdaderas), lo que el candidato corrige en buena parte.

### 6.5 El candidato frente al modelo que corre el paso

| Medida | Sin modificar (el paso) | Candidato | Barra (D24) |
|---|---|---|---|
| Held-out, 135 filas single, etiquetas anteriores del dueño | 43.7% | **54.1%** | 80%: no se cumple |
| Sub-news reales, 300 filas, etiquetas del asistente | 65.7% | **71.3%** | no definida para este conjunto |
| Fuera de pliegue, 1,216 filas, etiquetas revisadas | 75.5% | 87.8% | no es una medida held-out |

Las tres medidas discrepan porque son textos distintos y etiquetadores distintos; el held-out es la única etiquetada por el dueño y held-out, y es la que falla. El candidato es mejor que el modelo del paso en las tres, y
la ganancia es estadísticamente clara en las 300 filas reales. No debería reemplazar al modelo fijado antes de que se cumpla la barra en el held-out (T-201 a T-203).

## Lo que estas cifras no dicen

- **Los pasos no se comparan con una misma vara.** Cada uno se mide sobre su propia muestra, con su propio revisor y su propia definición de "correcto"; la tabla de resumen muestra cuáles son held-out.
- **Las cadenas de errores no están en estas cifras.** Un scope erróneo llega al cut, al paso de empresa, al paso de sector y al paso de category como entrada mala; por ejemplo, cerca de la mitad de las respuestas erróneas de sector de la primera revisión no eran textos de sector,
  y 15 de las 143 piezas muestreadas venían de textos que el paso de scope debió detener.
- **Tres medidas son sobre la corrida real del corpus**: la muestra de datos reales de category (6.4, etiquetas del asistente), las marcas de empresa principal sobre piezas (3.3, parciales) y la verificación de sentiment sobre 190 piezas (5.1, etiquetas del dueño). El embudo (textos por paso, y cuántos llegan a un snapshot) sale del almacén RESULTS real y no está en este informe.
- **Las tres medidas del candidato de category no son una misma vara** (6.5): fuera de pliegue sobre etiquetas que revisó el asistente, held-out sobre las etiquetas anteriores del dueño (artículos completos), y una muestra de datos reales etiquetada por el asistente (segmentos cortos). Solo el held-out es independiente, y falla la barra (54.1% frente a 80%).
- **No se mide en absoluto:** el recall de sector por sector, el recall del cut, la empresa principal por pieza más allá de las 119 filas marcadas de la sección 3.3, NER sobre los textos propios del paso 3, la relevancia sobre piezas más allá de las marcas parciales de la sección 3.3, y el puntaje held-out del stack de sentiment más allá de las 190 piezas de la sección 5.1.

## De dónde sale cada cifra

| Paso | Fuente |
|---|---|
| Scope | `docs/scope_training.md` (secciones 5, 6, 9); `data/scope_end_to_end_with81b.csv` y `gold/category_gold_v2_test.csv` (tabla por clase recalculada) |
| Sector | `docs/sector_step.md` ("The clean test", "What this does not show"); `gold/sector_validation.csv` (precisión por sector recalculada) |
| Cut | `docs/scope_training.md` sección 8.3; `gold/cut_sample_marks.csv`; `scripts/cut_sample_metrics_2026_11_12.py` |
| Empresa: NER | `docs/evaluation.md` (línea base de 2026-09, `ner`) |
| Empresa: relevancia | `docs/companies_step.md` (tabla de umbrales); `docs/sentiment_training.md` ítem 6 (AUC 0.90); `docs/main_company_training.md` y `gold/main_company_marks.csv` (las marcas sobre piezas, versionadas) |
| Summarization | `docs/summarization_training.md` (brazos, niveles y el brazo D de DeepSeek); `docs/summarization_results_2026_11_15/` (cifras versionadas del brazo D); `docs/evaluation.md` (línea base v1); `docs/summary_step.md` (límites) |
| Sentiment | `docs/sentiment_step.md` ("What the figures are"); `docs/sentiment_training.md` (tabla del stack, validación cruzada anidada; la verificación held-out sobre 190 piezas, sección 5.1: `gold/sentiment_val190.csv`, `gold/texts/sentiment_val190_texts.csv`, `scripts/score_sentiment_val190_2026_11_16.py`) |
| Category (el modelo del paso, 6.1) | `docs/category_step.md`; `docs/category_results_2026_11_13/` (tablas por etiqueta recalculadas desde `baseline_heldout.csv` y `baseline_train.csv`) |
| Category (candidato, 6.2 a 6.5) | PR #196: `docs/category_train.md`; `docs/category_results_2026_11_15/` (`ovr_single.md` fuera de pliegue; `heldout_selected.md` held-out, evaluado una sola vez por `scripts/category_heldout_2026_11_15.py`; `real_sample.md` y `scripts/category_real_sample_2026_11_15.py` para las 300 filas reales, etiquetas en `gold/category_real_sample.csv`); experimento MLflow `news_nlp_train/category` (`scripts/category_mlflow_log_2026_11_15.py`) y el modelo registrado `category-entailment-single-12`; puntajes D37 de las corridas de MLflow |
