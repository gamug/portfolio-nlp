# Precisión y recall (uno contra el resto) de los pasos del pipeline

Fecha: 2026-11-20 (los held-out del NER y de la relevancia son del 2026-11-15, 2026-11-16 y 2026-11-20; el held-out de sentiment se recalculó el 2026-11-20 con el re-etiquetado del dueño, versión 2; los demás, de las fechas que cada sección indica). El número del encabezado es el número de paso propio del pipeline; las secciones siguen el orden en que corren los pasos. "El dueño" es el término del proyecto para quien marca y decide. Pasos cubiertos, en el orden del pipeline: scope (con el paso de sector), cut, NER y la elección de la empresa, summarization, sentiment, category. Al final, una sección junta las métricas de todos los modelos.

Todas las cifras de abajo salen de los documentos del repositorio y de archivos de resultados versionados, o se recalcularon a partir de archivos versionados (las tablas de category
y de sector se recalcularon para este informe a partir de `docs/category_results_2026_11_13/*.csv` y `gold/sector_validation.csv`; las cifras de category son los archivos del PR #196, `docs/category_results_2026_11_15/`, y sus corridas de MLflow, sección 6; las del NER, `docs/ner_results_2026_11_16/`). **Las cifras medidas sobre textos de la corrida real del corpus son
el held-out del NER (100 textos del paso 2), las marcas de empresa principal sobre piezas (sección 3.2), la verificación de sentiment sobre 190 piezas (sección 5) y la muestra de datos reales de category (sección 6)**; la muestra de category se leyó del almacén RESULTS (solo lectura: `news_category`, `news_summary`, `sub_news`); todas las demás salen de archivos versionados o
de resultados.

**Uno contra el resto (OVR)** significa: para cada clase, precisión = de lo que el paso llamó de esa clase, la proporción que es correcta; recall = de los miembros verdaderos de
esa clase, la proporción que el paso encontró. Donde un paso no tiene clases de ese tipo, o el dueño del proyecto nunca las midió, la tabla lo dice en lugar de
inventar una cifra.

**Qué métricas se conservan.** Para todo modelo entrenado en el proyecto (los clasificadores de scope, el clasificador de pares del cut, el modelo de relevancia, el modelo de pares y el stack de sentiment, el fine-tune de category) este informe conserva solo dos tipos de métricas: **de entrenamiento-validación** (validación cruzada, puntajes fuera de pliegue y de una división de validación) y **held-out** (puntajes sobre textos con los que el modelo no se entrenó ni se ajustó).
Lo que se quitó en esta versión es toda otra medida: la proporción de ítems de una corrida que un paso etiquetó (una proporción de salida de una corrida de prueba) y los conteos de una corrida del corpus, que dicen qué produce un modelo, no si acierta. Se conservan las descripciones de los datasets, las narrativas de selección del modelo y las limitaciones. Los modelos que no se entrenan aquí (summarization, el modelo zero-shot anterior del paso de category, las reglas) conservan las cifras que tenían.
Las proporciones y los conteos quitados siguen en los documentos de los pasos (`docs/companies_step.md`, `docs/main_company_training.md`, `docs/summary_step.md`), que no son informes de todo el pipeline y los conservan. Las descripciones de datasets conservan las cifras de calidad de etiquetas de sus conjuntos (por ejemplo la verificación de etiquetas de entrenamiento del cut): describen los datos, no un modelo.

## Resumen

| Paso | Qué se mide | ¿Tiene precisión y recall OVR por clase? | Cifra principal | ¿Conjunto reservado (held-out)? |
|---|---|---|---|---|
| Scope | una de 4 clases por texto (`single`, `multi`, `sector`, `out_of_scope`) | **Sí**, cuatro clases (sección 1.1) | acuerdo en el held-out 83.6% [78.1, 88.9]; recall de `out_of_scope` 53%; precisión en cuarentena 90.8% | Sí (200 textos, vistos muchas veces), más 225 leídos por el dueño |
| Sector | el sector GICS de 2 dígitos asignado a un texto `sector` | **Solo precisión** (por sector, de 5 a 37 textos cada uno); el recall no se midió | 85.9% de los sectores asignados correctos (176 de 205, 357 respuestas); 96.7% (176 de 182) sin los 23 `no` que dicen que el texto no es de sector | Sí (dos sorteos hechos después de construir el paso, 2026-11-13 y 2026-11-18), un solo revisor |
| Cut | las piezas ("sub-news") que el cut hace de un texto con varias empresas | **Sin clases.** Tres preguntas de sí/no por pieza (parecidas a una precisión) | 84.6% de las piezas fieles; 75.5% limpias | Sí (28 textos, 143 piezas) |
| Empresa: NER | los tramos de organización que encuentra NER (y las empresas del universo que el matcher saca de ellos) | **Sin clases**: precisión, recall y F1 sobre 100 textos etiquetados por el dueño (321 organizaciones) | empresas del universo: precisión 90.1%, recall 95.4% con las etiquetas del dueño tal como están (92.6% / 95.5% con la superposición del desarrollador; recall 98.7% con `ner+text`); todas las organizaciones: 72.6% / 90.4% con la superposición | **Sí** (100 textos, un revisor; 14 etiquetas que no están en el texto, superposición: 9 reemplazadas, 1 conservada, 4 descartadas); además una corrida v1 juzgada por un LLM |
| Empresa: relevancia (qué empresa es la principal) | si un par (texto, empresa) es "principal" | Sí, para las dos clases | validación cruzada (809 resúmenes): precisión de la empresa principal 90.0%, recall 94.7% con umbral 0.5; **held-out (185 filas marcadas de 104 artículos, los textos que lee el paso), por artículo, la única principal que elige el modelo (titular, decidido el 2026-11-21): recall 100.0% (88/88), precisión 84.6% (88/104); con umbral 0.5: recall 69.3%, precisión 91.0%; filas: precisión 92.2%, recall 49.0%**. Con los marcados de este conjunto la recall de elección única no puede bajar de 100%, así que la precisión es la prueba del modelo (`docs/main_company_training.md`, "Decision, 2026-11-21") | **Sí** (marcas parciales sobre piezas, un revisor); la validación cruzada no lo es |
| Summarization | el resumen de un texto de más de unos 2,000 caracteres | **Sin clases.** Puntajes de un juez LLM (1 a 5) | **DeepSeek, el summarizer del paso desde T-206:** fidelidad 4.99, cobertura 4.54, alucinación 0.7% (300 artículos); el paso anterior con distilbart: 4.60, 3.24, 15.0% | El juez también es DeepSeek (calificó sus propios resúmenes); una recalificación a ciegas de 24 pares por el asistente (Claude) coincidió en términos generales (fidelidad 4.92, cobertura 4.75, alucinación 4%); no es un gold y es pequeña; la corrida sobre el corpus no se mide |
| Sentiment | positivo, negativo, neutral para un par (texto, empresa) | **Sí**, tres clases | validación cruzada anidada (809 filas): puntaje 0.814, negativo P/R 0.747 / 0.748; **held-out (190 piezas reales, versión 2 del 2026-11-20): exactitud 78.9%, negativo 70.8% / 83.6%, neutral 80.0% / 68.6%** (las cifras de neutral se movieron con el re-etiquetado) | **Sí** (190 piezas, un lector; el re-etiquetado se hizo con la etiqueta del modelo visible); la validación cruzada no lo es |
| Category | 14 etiquetas (6 grupos) para un par (texto, empresa); el modelo fijado en el paso lee 12; **planeado: 8 (7 medidas más `other`, no medida)** | **Sí**, por etiqueta (precisión y recall) | **oficial (decisión del dueño, 2026-11-17): 88.0% [83.8, 91.2] de coincidencia con un juez LLM a ciegas sobre 300 sub-news reales** (la etiqueta del modelo es la primera o la segunda del juez; el modelo aprendió las etiquetas del dueño, que el asistente revisó); con las cuatro etiquetas débiles movidas a `other`: 92.0% (94.4% sobre las 7 etiquetas medidas). Además: fuera de pliegue 91.5%, held-out 57.1%, 82.1% en 300 sub-news reales (etiquetas conjuntas), sobre las 7 etiquetas medidas (modelo de 12 etiquetas reasignado, no reentrenado) | La barra de 80% (D24) se **cumple** con la cifra oficial (88.0%) |

**Lee primero la última columna.** Solo el scope, el cut, el NER, la relevancia sobre piezas, el sentiment sobre 190 piezas, category y la prueba limpia de sector son medidas sobre textos con los que el modelo o las reglas no eligieron
sus ajustes; las cifras de validación cruzada de relevancia y de sentiment son fuera de pliegue y no son held-out. Ninguno de los pasos ha sido evaluado por un segundo
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
Las dos cifras de arriba (83.6% y 90.8%) son poblaciones y claves distintas.

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

**El conjunto más grande (2026-11-19, `docs/sector_training.md`):** el dueño respondió 357 filas, las 57 de arriba más 300 sorteadas el 2026-11-18 (165 asignadas, 15 controles, 120 enviadas fuera). Dos medidas,
sobre todo el archivo y sin los `no` cuya nota dice que el texto no es de sector (`general_economy`, `multi`, `single`, `out_of_scope`, inversión: errores del paso de scope):

| Medida | Todo el archivo (357) | Sin los `no` de "no es de sector" |
|---|---|---|
| El sector asignado es correcto | 176/205 = 85.9% [80.4, 90.0] | 176/182 = 96.7% [93.0, 98.5] |
| Dejados pasar por el margen | 134/150 = 89.3% | 134/138 = 97.1% |
| Dejados pasar solo por el voto | 42/55 = 76.4% | 42/44 = 95.5% |
| Controles con respuesta `yes` | 0/20 | 0/20 |
| Enviados fuera y el dueño está de acuerdo | 101/132 = 76.5% [68.6, 82.9] (75.0% con dos respuestas contradictorias leídas como `no`) | no aplica (allí un `no` dice que el descarte fue un error) |

Las 23 respuestas excluidas son el 11% de las filas asignadas. Cinco de las primeras 57 respuestas cambiaron desde el gold versionado, dos contradicen su propia nota y 28 `no` de enviados fuera no nombran sector,
así que el costo de la compuerta sigue siendo una cota superior (como máximo una quinta parte de los sectores correctos); todo en `docs/sector_training.md`. Las marcas están versionadas en `gold/sector_fresh_marks_2026_11_19.csv`.

**Uno contra el resto por sector (2026-11-19, 337 filas sin controles; `docs/sector_training.md`):** precisión, y recall como cota superior (los 28 `no` de enviados fuera que no nombran sector solo pueden bajarlo; el sorteo no es una muestra del corpus).

| Código GICS | Sector | Asignados | Correctos | Precisión [95%] | Precisión sin los `no` de "no es de sector" | Realmente de este sector (conocidos) | Encontrados | Recall, cota superior |
|---|---|---|---|---|---|---|---|---|
| 10 | Energy | 19 | 18 | 94.7% [75.4, 99.1] | 18/18 | 18 | 18 | 100.0% |
| 15 | Materials | 5 | 3 | 60.0% [23.1, 88.2] | 3/4 | 3 | 3 | 100.0% |
| 20 | Industrials | 25 | 21 | 84.0% [65.3, 93.6] | 21/21 | 21 | 21 | 100.0% |
| 25 | Consumer Discretionary | 25 | 21 | 84.0% [65.3, 93.6] | 21/22 | 25 | 21 | 84.0% |
| 30 | Consumer Staples | 7 | 5 | 71.4% [35.9, 91.8] | 5/7 | 7 | 5 | 71.4% |
| 35 | Health Care | 15 | 14 | 93.3% [70.2, 98.8] | 14/15 | 14 | 14 | 100.0% |
| 40 | Financials | 37 | 30 | 81.1% [65.8, 90.5] | 30/30 | 30 | 30 | 100.0% |
| 45 | Information Technology | 36 | 30 | 83.3% [68.1, 92.1] | 30/30 | 31 | 30 | 96.8% |
| 50 | Communication Services | 18 | 17 | 94.4% [74.2, 99.0] | 17/17 | 19 | 17 | 89.5% |
| 55 | Utilities | 5 | 5 | 100.0% [56.6, 100.0] | 5/5 | 5 | 5 | 100.0% |
| 60 | Real Estate | 13 | 12 | 92.3% [66.7, 98.6] | 12/13 | 12 | 12 | 100.0% |
| — | **Sin sector** (enviados fuera) | 132 | 101 | 76.5% [68.6, 82.9] | — | 124 | 101 | 81.5% |

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

Se compararon cinco extractores sobre los conjuntos de fidelidad; todos cortan el cuerpo en piezas y deciden a qué empresa pertenece cada pieza. Después se comparó `units` con dos lecturas posteriores, `units4` y `units5`. La barra es 80% de textos fieles.

| Extractor | Cómo corta y decide |
|---|---|
| `baseline` | oraciones; una oración pertenece a las empresas que nombra, o a las de la oración anterior; sin modelo |
| `model` | oraciones; una oración que no nombra a nadie va a la empresa a la que el modelo entrenado da el puntaje más alto (al menos 0.5) |
| **`units`** | líneas, viñetas y cláusulas, sin el pie de página; decide el modelo entrenado (selección del 2026-11-08) |
| `units2` | como `units`, con una guarda sobre formas cortas de nombres y sobre líneas que abren con nombres y un guion |
| `units3` | como `units2`, y una pieza de esa línea también pertenece a las empresas que ella misma nombra |
| `units4` | como `units`, con el resolvedor simple (sin guarda); el encabezado de una oración "Nombres – texto" se lee por oración (también a mitad de párrafo) y un ticker entre paréntesis forma parte del encabezado; la oración va a las empresas del universo del encabezado y a las que su texto nombra, y a nadie si el encabezado no nombra ninguna; una oración con encabezado no se corta en cláusulas (T-176; el corte del 2026-11-11 al 2026-11-12) |
| **`units5`** | como `units4`, y además lee los encabezados que `units4` no leía: un nombre corto en mayúsculas ("SAP —", "UBS –"), una etiqueta de grupo ("Memory stocks —"), un punto dentro de un nombre ("Amazon.com (AMZN) –") y un texto en minúsculas tras el guion solo cuando el encabezado trae su ticker entre paréntesis; una oración sin encabezado tras un encabezado de nadie va a las empresas del universo que nombra (T-177; **el corte final desde el 2026-11-12**) |

El modelo entrenado es un clasificador de pares ("¿esta oración trata de esta empresa?") afinado sobre el codificador del modelo de sentiment, con el nombre de la empresa enmascarado y la oración anterior como contexto, solo con etiquetas `lead`.
`units` se seleccionó el 2026-11-08: en el conjunto de confirmación pasa la barra (86.1% [80.9, 90.7] sobre toda empresa nombrada, 95.3% [92.0, 98.2] sobre empresas reales), con textos más cortos que el baseline (570 caracteres frente a 952) con la misma proporción de fieles (86.5% y 86.6%) y
menos texto de otras empresas. No se muestra mejor que el baseline en cuerpos nuevos; se mantuvo porque no es peor y es más magro. El margen es estrecho: pasar de una a tres marcas a `no` baja el extremo inferior de 80%.

**`units4` y `units5` (lecturas de `units`, mismo modelo entrenado).** La primera corrida del cut sobre 232 cuerpos `multi` mostró dos fallos de `units`: 162 textos de empresa (en 79 cuerpos) con dos o más encabezados "Nombre —" y 6 textos que eran solo un encabezado. Sobre esos 232 cuerpos (`scripts/compare_cut_readings_2026_11_11.py`, GPU):

| | `units` | `units4` | `units5` |
|---|---|---|---|
| textos de empresa | 1,671 | 1,589 | 1,638 |
| textos con varios encabezados | 162 (79 cuerpos) | 23 (18 cuerpos) | 10 (los diez solapes legítimos, leídos con `several_headings_owner`) |
| textos que son solo un encabezado | 6 | 0 | no se registra |
| fieles sobre los textos con marca del dueño (marcas anteriores de fidelidad) | no se compara aquí | 304 de 313, 97.1% [94.5, 99.1] | 351 de 359, 97.8% [95.5, 99.4] |

La ganancia de `units5` sobre `units4` viene de textos añadidos que son fieles, no de textos ya existentes que mejoren (`docs/extractor_training.md`, actualización del 2026-11-11; `docs/scope_training.md` sección 8.1). **`units5` (`units5-d3e2cbfa830e`) es el corte final** (dueño, 2026-11-12, tras la muestra de sí/no de la sección de abajo); su checkpoint está fijado a la copia privada del Hub `gamug/FinBERT-news-attribution@436ad700`.

**Muestra**: 28 textos sorteados al azar del cut, 143 piezas, marcadas por el dueño el 2026-11-12. Proporción de sí con intervalo de Wilson de 95% sobre piezas:

| Nivel | Piezas (textos) | faithful | one_story | complete | limpias (3 sí) | Textos todos limpios |
|---|---|---|---|---|---|---|
| **Todas (la cifra a citar)** | 143 (28) | 84.6% [77.8, 89.6] | 78.3% [70.9, 84.3] | 83.2% [76.2, 88.5] | 75.5% [67.9, 81.8] | 16 de 28 |
| Sin las piezas de textos con alcance erróneo | 128 (27) | 89.1% [82.5, 93.4] | 82.0% [74.5, 87.7] | 87.5% [80.7, 92.2] | 78.9% [71.0, 85.1] | 16 de 27 |
| También sin los textos ambiguos | 114 (26) | 93.9% [87.9, 97.0] | por construcción | 91.2% [84.6, 95.2] | por construcción | por construcción |

El segundo y el tercer nivel se eligieron después de que el dueño viera los fallos; son el cut con los errores de sus entradas apartados, no cifras para citar solas.
Los intervalos que remuestrean textos completos son más amplios (para `faithful` sobre todas las piezas: [74.8, 92.9]).

**Frente a la barra de 80% para `faithful`**: la estimación puntual pasa (84.6%), el extremo inferior del intervalo (77.8%) no.

**La empresa principal por pieza** es una medida del paso 3 (el modelo de relevancia), no del cut: sus cifras de entrenamiento-validación y held-out están en la sección 3.2.

**Qué no se mide.**
- El **recall** del cut (si el cut encontró todas las historias del texto): no se midió.

## 3. NER y la elección de la empresa (paso 3, `docs/companies_step.md`)

El paso 3 tiene dos partes que se miden por separado: NER (encuentra tramos de organización) y el modelo de relevancia (decide cuáles empresas de un texto son las principales). Cada una tiene su sección y una sola tabla de métricas, con la validación y el held-out juntos.

### Descripción del dataset

El NER tiene desde el 2026-11-16 un conjunto held-out del dueño sobre textos del paso 2; la relevancia tiene sus 809 resúmenes marcados y, además, marcas sobre piezas. Lo que existe:

| Conjunto | Tamaño | Columnas y etiquetas | Características |
|---|---|---|---|
| Datos de entrenamiento de NER (FiNER-ORD, usados por quien publicó el modelo) | no se conservan en este repositorio | tramos de entidad de tipo `PER`, `LOC`, `ORG` | datos públicos de NER de noticias financieras; el repo registra el modelo, no los datos |
| Evaluación juzgada (v1, `docs/evaluation.md`) | 1,000 artículos completos muestreados, 953 procesados | tramos juzgados `wrong` o `missed` por un juez LLM | no es un gold hecho a mano; artículos completos, no los resúmenes ni piezas que lee el paso 3 |
| **Held-out de NER** (`data/ner_val100/to_tag.csv`, local y fuera de git; `docs/ner_training.md`) | 100 textos, 321 organizaciones | `review_order`, `article_id`, `pub_date`, `title`, `text`, `spanish_title`, `spanish_text`, `key`, `organizations`, `notes` | el dueño escribe las organizaciones de cada texto separadas por comas, tal como aparecen en el texto; los textos son resúmenes u originales del paso 2 (`summary-7a9bd2062bc4`); **14 etiquetas de 11 filas no están escritas en el texto** (nombres en español, erratas, nombres pegados, un texto defectuoso, cuatro de otro artículo): el desarrollador propone una superposición (9 reemplazadas, 1 conservada, 4 descartadas) y el archivo no se tocó (T-213) |
| Entrenamiento y evaluación de relevancia (los 809 resúmenes marcados) | 809 pares (texto, empresa) de 855 resúmenes etiquetados (dos sorteos, 450 y 405) | `key`, `article_id`, `ticker`, `company`, `body_hash`, `rule_scope`, `revision`, `predicted`, `pool_class_size`, `block`, `batch`, `slice`, `text_hash`, `pub_date`, más la marca | cada fila es un texto (un resumen de un artículo, unas pocas oraciones) y una empresa; relevante = la marca tiene dirección (positiva o negativa), no relevante = neutral (202 de las 809 son neutrales); los sorteos equilibran las clases que predijo el modelo de sentiment anterior, por eso las cifras usan pesos |
| **Held-out de relevancia**: marcas de empresa principal sobre sub-news (`data/main_company/to_tag.csv`, local; `gold/main_company_marks_2026_11_20.csv`, la versión anterior del 2026-11-16 en `gold/main_company_marks.csv`) | 185 filas sobre 104 sub-news del corte `units5` (la primera versión tenía 353 sobre 180); 185 filas de 104 ítems están marcadas (145 `yes`, 40 `no`; la versión del 2026-11-16: 119 filas de 59 ítems) | `item_id`, `texto`, `texto_es`, `ticker`, `empresa`, `es_principal` (si es la empresa principal, `yes`/`no`), `falta_la_principal`, `notas` | seis bandas de relevancia sorteadas con 30 ítems cada una, quedan de 8 a 29 ítems por banda (empresas `companies-f349536cf701`, semilla 20261113); el modelo nunca vio sub-news; un lector |

### Selección del modelo

**NER.** `gamug/sec-bert-finer-ord-ner` (SEC-BERT-BASE afinado sobre FiNER-ORD, fijado en `ba7b9e43`) es el modelo del pipeline; solo encuentra tramos de organización, que un matcher convierte en tickers del universo. Los documentos leídos para este informe **no registran comparación con otro modelo de NER**: se mide sobre la corrida juzgada v1 y sobre el held-out de 100 textos de la sección 3.1; no se cambió ni se fijó nada con esa medida. La opción `--candidates ner+text`
(NER más el matcher aplicado sobre el texto, sin coincidencias en minúsculas y sin coincidencias dentro de líneas de divulgación) se eligió sobre `ner` solo por el dueño el 2026-11-13, por la proporción de ítems que etiqueta (una proporción de salida de una corrida de prueba, no una métrica de corrección: sus cifras no se conservan aquí; véase `docs/companies_step.md`, "Dry run"); el held-out de 100 textos la mide ahora.

**Relevancia (cuál empresa es la principal).** La base es la lectura zero-shot del modelo de category (`MoritzLaurer/deberta-v3-base-zeroshot-v2.0`, "este texto trata principalmente de <empresa>"), la mejor de nueve variantes zero-shot por AUC (0.743; el ranking del propio modelo de sentiment fue 0.795). Afinado sobre las 809 marcas, su AUC fuera de pliegue es 0.90, y el
modelo publicado (`gamug/deberta-v3-company-relevance@8812c8bc`, privado) es el que corre el paso. El umbral 0.5 se eligió sobre esos puntajes fuera de pliegue (recall de principal 94.7%, precisión 90.0%); los otros valores probados (0.3, 0.7, 0.9) cambian poco en resúmenes, pero en sub-news la proporción de ítems con empresa principal se mueve mucho a lo largo del rango (una proporción de salida de una corrida de prueba, que no se conserva aquí), lo que los 809 resúmenes no pueden resolver; las marcas sobre piezas (abajo) lo miden.

### 3.1 NER (`gamug/sec-bert-finer-ord-ner`)

NER no tiene clases por tipo que importen aquí: solo cuentan los tramos `ORG`, y lo que el paso 3 conserva son las empresas del universo que el matcher saca de ellos. Una sola tabla: la evaluación juzgada v1 (la validación) y el held-out del dueño (100 textos, 321 organizaciones; precisión, recall y F1 por texto como conjuntos de empresas canónicas; las 14 etiquetas que no están en el texto se tratan en las primeras filas con la **superposición** del desarrollador, una propuesta que el dueño debe confirmar: 9 reemplazadas, 1 conservada, 4 descartadas; las etiquetas tal como están siguen a la tabla).

| Medida | Evaluación juzgada v1 (1,000 artículos completos, juez LLM) | Held-out del dueño (100 textos) |
|---|---|---|
| Todas las organizaciones: precisión | 0.66 (el modelo sobrepredice; 33.8% de los tramos predichos los llama erróneos el juez) | 72.6% |
| Todas las organizaciones: recall | 0.84 | 90.4% |
| Todas las organizaciones: F1 | micro-F1 de tramos 0.7418 (ORG 0.75, LOC 0.82, PER 0.66) | 0.805 |
| **Empresas del universo** (lo que el paso 3 conserva): precisión | no se midió | **92.6%** (150/162) |
| **Empresas del universo**: recall | no se midió | **95.5%** (150/157) |
| **Empresas del universo**: F1 | no se midió | **0.940** |
| Empresas del universo con `ner+text` (el matcher también sobre el texto): precisión / recall / F1 | no se midió | 90.1% / **98.7%** / 0.942 |

Con las 14 etiquetas tal como están (sin superposición): empresas del universo 90.1% / 95.4% / 0.927 y todas las organizaciones 70.3% / 87.0% / 0.777; quitarlas empeora la precisión, porque una etiqueta mal escrita de una empresa real deja el acierto del modelo como "de más".

**Qué dice y qué no.**
- **De las 7 empresas del universo que faltan, 5 no las etiqueta el NER** (Meta Platforms, Hormel, MSFT, Oracle, Las Vegas Sands) y 2 las etiqueta pero el matcher no las asigna (`ON Semiconductor`, `Amazon`); `ner+text` recupera las 5. Las 12 empresas de más son nombres de empresa reales que el dueño no escribió (`Royal Caribbean`, `PayPal`, `Intel`, `Trane`...); no se leyó cada texto para ver si trata de ellas.
- Los tramos de más sobre todas las organizaciones son medios, instituciones y productos (CNBC, Reserva Federal, Congress, iPhone) sin ticker, que el paso 3 descarta; no son un defecto del paso.
- Las etiquetas son **todas las organizaciones nombradas**, no las principales (eso lo mide la sección 3.2). La lista de alias con que se comparan los nombres (Google y Alphabet, Facebook y Meta...) se eligió después de ver los errores, así que está ajustada a este conjunto (sin ella, sobre todas las organizaciones, 68.0% / 86.4%).
- 100 textos, un revisor, sin intervalo; resúmenes del paso 2 antiguo (distilbart), no los de DeepSeek de la corrida del corpus. La superposición sobre las 14 etiquetas (9 reemplazadas, 1 conservada, 4 descartadas) es del desarrollador; los textos defectuosos (palabras pegadas como `NHTSAsaid`) son del crawl, no del modelo.
- El juez v1 no es un gold ni el dueño, y puntúa artículos completos. Los tramos se convierten en tickers con un matcher con vacíos conocidos (nombres cortos como `Disney` o `JPMorgan`, T-181); los tramos crudos no se almacenan. Detalle: `docs/ner_training.md`; cifras: `docs/ner_results_2026_11_16/`; script: `scripts/ner_eval_2026_11_16.py`.

### 3.2 Relevancia: si una empresa es "principal" en un texto (`gamug/deberta-v3-company-relevance`)

Dos clases: principal y no principal. Una sola tabla: la validación cruzada (puntajes fuera de pliegue de los 809 resúmenes marcados, sin ponderar) y el held-out (las 185 filas que el dueño marcó sobre 104 artículos del corte `units5`, versión del 2026-11-20, sin ponderar salvo donde la fila diga reponderado: las bandas se muestrearon por igual, no son tasas del corpus). La recall y la precisión del held-out son **por artículo** (ver la limitación abajo). Intervalos de 95% donde se calcularon.

| Medida | Validación cruzada (809 resúmenes) | Held-out (185 filas, 104 artículos) |
|---|---|---|
| AUC, filas | 0.90 | **0.829** [0.730, 0.912] |
| AUC, artículos (mejor puntaje de cada artículo) | no figura en el registro | 0.766 [0.662, 0.873] |
| **Recall de elección única: el titular (decidido el 2026-11-21)** | no figura en el registro | **100.0% [95.8, 100.0] (88/88)**; remuestreo de 100.0% a 100.0%; precisión **84.6% [76.5, 90.3] (88/104)** |
| Recall por artículo a 0.5 (la regla del paso; secundaria) | no figura en el registro | 69.3% [59.0, 78.0] (61/88); reponderado por tamaño de banda 88.0% |
| Precisión por artículo a 0.5 | no figura en el registro | 91.0% [81.8, 95.8] (61/67) |
| Recall por artículo a 0.1 / 0.3 / 0.7 | no figura en el registro | 97.7% (86/88) / 83.0% (73/88) / 53.4% (47/88) |
| Principal a 0.5 (el paso): precisión, filas | 90.0% | 92.2% [84.0, 96.4] (71/77) |
| Principal a 0.5: recall, filas | 94.7% | 49.0% [41.0, 57.0] (71/145) |
| No principal a 0.5: precisión, filas | no figura en el registro | 31.5% (34/108) |
| No principal a 0.5: recall, filas | 68.3% (1 menos 31.7% de filas no relevantes llamadas principales) | 85.0% (34/40) |
| Principal a 0.3: precisión / recall, filas | 89.7% / 96.2% | 91.2% [84.1, 95.3] / 64.1% [56.1, 71.5] |
| Principal a 0.7: precisión / recall, filas | 91.1% / 92.3% | 93.2% [83.8, 97.3] / 37.9% [30.4, 46.0] |
| Principal a 0.9: precisión / recall, filas | 91.7% / 87.3% | no se calculó |

**La limitación de una sola principal.** El modelo de relevancia se entrenó para elegir la única empresa de la que trata principalmente un texto. El dueño marca cada empresa principal de un artículo, así que un artículo con varias principales del dueño se puntúa por filas solo en parte: un artículo con seis principales del dueño donde el modelo nombra dos tiene una recall por filas de un tercio, aunque encontró una empresa principal. Es una limitación del diseño de la tarea, no un error del modelo. Por eso la recall y la precisión titulares cuentan **por artículo**: un artículo se encuentra cuando la empresa principal del modelo es una que el dueño marcó principal. Las cifras por filas quedan como cifras secundarias. La clase no principal se mide por filas, porque el `no` del dueño es una respuesta explícita.

Escenarios: las mismas marcas, sin los artículos de cada exclusión (ninguna empresa en absoluto; empresa principal faltante en la lista o fuera del universo S&P 500). Sin ponderar, a 0.5.

| Escenario (artículos, filas) | Recall por artículo | Precisión por artículo | AUC por artículo | Recall por filas | Recall de no principal por filas |
|---|---|---|---|---|---|
| Todos (104, 185) | 69.3% (61/88) | 91.0% (61/67) | 0.766 | 49.0% (71/145) | 85.0% (34/40) |
| Sin los artículos sin empresa (102, 183) | 69.3% (61/88) | 92.4% (61/66) | 0.777 | 49.0% (71/145) | 86.8% (33/38) |
| Sin los artículos con principal faltante (76, 138) | 73.0% (54/74) | 98.2% (54/55) | n/a | 51.2% (64/125) | 92.3% (12/13) |
| Sin ambos (74, 136) | 73.0% (54/74) | 100.0% (54/54) | n/a | 51.2% (64/125) | 100.0% (11/11) |

**Qué dice y qué no dice.**
- **Su elección única es una principal del dueño en cada artículo que tiene una (88 de 88), y nombra una empresa que el dueño no marca como principal en 16 de los 104 artículos** (precisión 84.6%). **Con la regla de umbral, a 0.5 encuentra siete de cada diez artículos** con una principal del dueño (69.3%), y 97.7% a 0.1; cuando declara principal a una empresa, la regla de umbral acierta el 91.0% de las veces (de 89% a 92% de 0.1 a 0.7). El AUC por filas es **0.829** [0.730, 0.912] contra 0.90 fuera de pliegue, más de 5 puntos por debajo: el disparador de la regla de T-170 (las sub-news tienen su propio umbral, elegido sobre las sub-news marcadas); la decisión es del dueño y sigue abierta (`docs/main_company_training.md`).
- **La limitación de una sola principal no explica la brecha**: los artículos con una principal del dueño se encuentran el 70.0% de las veces, los que tienen varias el 68.4%. La recall por filas a 0.5 es menor en los artículos con varias (36 de 95 filas, 37.9%, frente a 35 de 50, 70.0%), y eso es la limitación del diseño, que se ve en las cifras por filas.
- **Las empresas faltantes tampoco explican la recall de umbral**: 28 artículos tienen la empresa principal faltante en la lista o fuera del S&P 500. Sin ellos, la recall por artículo a 0.5 sube de 69.3% a 73.0%.
- **Esos escenarios conservan pocas filas `no`**: sin los 28 artículos quedan 13 de las 40 filas `no`, así que la recall de no principal por filas y el AUC por artículo de los dos últimos escenarios descansan en muy pocas filas; el AUC por artículo no está definido cuando no queda ningún artículo sin principal.
- Las varias principales del dueño son reales, y el modelo se entrenó para elegir una: es una decisión del diseño, no un error del marcado.
- Un solo lector, 40 filas `no`: los intervalos son anchos y las cifras no son tasas del corpus (las bandas se sortearon por igual, y 76 de los 180 artículos sorteados al principio ya no están en el archivo).
- No se midió la división entre "no se nombra ninguna empresa del universo" y "se nombra en una forma que el matcher no conoce". Los candidatos de texto de `ner+text` los revisó el desarrollador sobre unos pocos pares; ninguna de las dos cifras se conserva aquí.
- Archivos: `gold/main_company_marks_2026_11_20.csv` (la versión anterior: `gold/main_company_marks.csv`), `gold/texts/main_company_marks_texts_2026_11_20.csv`, `scripts/score_main_company_marks_2026_11_16.py`; la cuenta completa es `docs/main_company_training.md`.

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

**Límites.** Una muestra, un juez, una corrida por brazo. La cobertura del juez es un puntaje global: **nada verifica si las empresas de las que trata el texto, o su evento principal, están en el resumen** (sin verificación por empresa).
**DeepSeek fue summarizer y juez** en la evidencia del brazo D, así que sus puntajes son su propia lectura de su propio texto (la recalificación a ciegas de 24 pares es pequeña, y los brazos se adivinan por el estilo). El texto de DeepSeek no es repetible con temperatura 0 (una repetición de 30 artículos conservó los puntajes). Las cifras son de 300 artículos; la corrida sobre el corpus aún no se mide.

## 5. Sentiment (paso 6, `docs/sentiment_step.md`, `docs/sentiment_training.md`)

El modelo de pares (texto y empresa, `sentiment_pair`) da tres probabilidades de clase; se apilan con el puntaje de relevancia mediante una regresión logística
(`config/sentiment_stack.json`). Se mide de dos maneras, en una sola tabla de abajo: la **validación cruzada anidada** sobre las **809 filas marcadas** (ponderada, D37; no es held-out; toda la población marcada, con pares de otras empresas, mientras que el paso puntúa solo pares principales) y un **held-out de 190 piezas etiquetadas por el dueño** (un lector) de la corrida real del corpus, revisión `sentiment-5999fe131d8f` (`gamug/FinBERT-company-sentiment@9a7b1eb3`). El gold de 180 textos no se ha evaluado con el stack (T-148c4).

### Descripción del dataset

La etiqueta es **lo que un texto dice sobre una empresa**, no el sentimiento de sus oraciones. Cada fila es un texto (normalmente un resumen de un artículo, unas pocas oraciones) y una empresa; la marca es el impacto en el precio que un inversionista leería sobre esa empresa. Si el texto no da ningún hecho sobre la empresa, la marca es neutral, así que neutral mezcla "hechos sin dirección" y "ningún hecho sobre la empresa".

| Conjunto | Tamaño | Conteo por clase | Columnas | Características |
|---|---|---|---|---|
| Pool de oraciones (plata) | 5,800 oraciones fusionadas, 3,865 tras equilibrar | 1,321 negativas, 1,321 neutrales, 1,223 positivas | no está en los archivos gold | oraciones de los artículos del proyecto, etiquetadas por un LLM una a una, sin atarlas a una empresa; entrenó el modelo de partida |
| Gold de 180 textos (`sentiment_gold.csv`) | 180: resúmenes y textos de empresa (120) y textos de extractor (60) | 82 positivos, 65 negativos, 33 neutrales | `key`, `article_id`, `ticker`, `kind`, `part`, `text_hash`, `revision`, `stratum`, `extractor`, `predicted`, `population`, `pub_date`, `sentiment`, `notes` | el examen de la regla de fijación; nunca se afinó con él; aún no se evalúa con el stack |
| Conjunto de entrenamiento 1 (`sentiment_train.csv`) | 450 resúmenes (424 leídos como par) | 232 positivos, 113 negativos, 105 neutrales | `key`, `article_id`, `ticker`, `company`, `body_hash`, `rule_scope`, `revision`, `predicted`, `pool_class_size`, `block`, `batch`, `slice`, `text_hash`, `pub_date`, más la marca | tres lotes de 150, 150 por clase que predijo el modelo anterior |
| Conjunto de entrenamiento 2 (`sentiment_train_2.csv`) | 405 resúmenes (385 leídos como par) | 178 positivos, 116 negativos, 111 neutrales | como arriba | mismo diseño, 135 por clase predicha |
| Replay | 341 oraciones sorteadas de 925 elegibles (las que nombran la empresa de su artículo) | | | se leen como par con esa empresa, mezcladas en todo conjunto de entrenamiento |
| **Held-out** (`gold/sentiment_val190_2026_11_20_v2.csv`, versión 2; la versión del 2026-11-16 `gold/sentiment_val190.csv` se conserva) | 190 piezas de la corrida sobre el corpus, dos sorteos (90 + 100); nunca de entrenamiento | 94 positivas, 61 negativas, 35 neutrales (versión 2; 96 / 62 / 32 antes del re-etiquetado del 2026-11-20) | `key`, `article_id`, `item_id`, `ticker`, `batch`, `text_source`, `text_hash`, `predicted`, `predicted_score`, `relevance_score`, `pub_date`, las tres revisiones, `sentiment`, `notes`; los textos están en `gold/texts/sentiment_val190_texts.csv` | piezas (sub-news) del paso de resumen guardado (`summary-7a9bd2062bc4`), con una empresa principal del paso de empresas y un puntaje de sentiment; no resúmenes de artículos: 38 son resúmenes de distilbart, el resto textos cortos originales; sorteo 1 (90): 30 por clase predicha, sin textos de los conjuntos de entrenamiento ni del gold; sorteo 2 (100): al azar, filtro más estricto (relevancia 0.9 o más, empresa nombrada dos veces y al inicio); la llamada del propio modelo se guarda aparte de la etiqueta del dueño |

Los sorteos equilibran las clases que predijo el modelo anterior, no las clases verdaderas, así que las cifras del held-out describen la muestra. No se ponderan al pool, y la muestra no estima el corpus. Juntos, los dos conjuntos de entrenamiento dan 809 filas de pares. Ningún texto del gold ni de un conjunto held-out está en un conjunto de entrenamiento ni en el replay.
Los textos de sector, mercado y economía (46 filas) se etiquetan por su alcance y se dejan fuera del modelo de pares.

### Selección del modelo

El modelo de pares lee `[empresa] [SEP] [fragmento de texto]` y promedia las probabilidades de los fragmentos; una regresión logística (el stack) combina sus log-probabilidades con el puntaje de relevancia. Validado de forma cruzada, anidada y ponderada (métricas de entrenamiento-validación), sobre las 809 filas:

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

Seleccionado el 2026-11-10 (el held-out de 190 piezas vino después y no participó en la selección): el modelo de pares iniciado desde el endpoint del proyecto `gamug/FinBERT-financial-news`, entrenado con las 809 filas más las 341 oraciones de replay, apilado con el modelo de relevancia afinado. La fijación no está decidida: espera al gold de 180 textos según la regla de D35.

**Métricas del stack (el paso), validación y held-out en una sola tabla.** Precisión (P) y recall (R) uno contra el resto; el held-out lleva intervalos de Wilson de 95%.

| Medida | Validación cruzada anidada (809 filas, ponderada) | Held-out (190 piezas de la corrida real) |
|---|---|---|
| Positivo: P / R | 0.807 / 0.855 | P 85.2% [76.3, 91.2] (75/88); R 79.8% [70.6, 86.7] (75/94) |
| Negativo: P / R | 0.747 / 0.748 | P 70.8% [59.5, 80.1] (51/72); R 83.6% [72.4, 90.8] (51/61) |
| Neutral: P / R | 0.787 / 0.686 | P 80.0% [62.7, 90.5] (24/30); R 68.6% [52.0, 81.4] (24/35); antes del re-etiquetado: P 56.7%, R 53.1% |
| P / R macro | 0.780 / 0.763 | no se calculó |
| ROC OVR macro | 0.898 | no se calculó |
| Puntaje (D37) / exactitud | puntaje **0.814** | exactitud **78.9%** [72.6, 84.1] (150 de 190) |

Los candidatos intermedios (modelo de pares solo 0.737, con compuerta 0.789) y el modelo v1 fijado (0.648) están en la tabla de selección de arriba; el stack gana +0.077 [+0.057, +0.095] sobre el modelo de pares solo.

**Frente a las barras de D32** (80% de acuerdo; precisión y recall de negativo de al menos 70%): en la validación cruzada el negativo está por encima (0.747 / 0.748). En el held-out la exactitud, 78.9%, está **bajo 80%** por estimación puntual; la precisión de negativo, 70.8%, y el recall, 83.6%, superan 70%.

**Límites.**
- **La validación cruzada no es held-out**, así que es optimista; los modelos finales son más confiados que los modelos de pliegue sobre los que se ajustó el stack (no se midió). El held-out es menor, pero **no miden lo mismo** (resúmenes marcados, con pares de otras empresas, frente a pares principales de piezas de la corrida real), así que la brecha no es una medida del optimismo.
- **Neutral: la proporción en la población no se mide.** El modelo llama neutrales a 420 de los 57,826 textos candidatos (0.7%). Las 190 piezas no pueden decir cuántos textos neutrales tiene el conjunto ni cuántos encuentra el modelo, así que no se reporta un recall poblacional. El re-etiquetado del 2026-11-20 cambió 11 etiquetas, 7 de ellas a llamadas neutrales que el modelo ya había hecho; por eso la precisión de neutral de 80% es en parte el re-etiquetado y no una medida ciega. Registrado como T-212 (propuesto; el dueño decide la corrección). Neutral es también la clase más débil por sí sola (recall 0.44 sin el modelo de relevancia): el modelo oye una empresa y una polaridad pero no distingue si el texto trata de la empresa.
- **Los dos sorteos del held-out difieren**: el primero (90, estratificado por la predicción) da exactitud 75.6% [65.8, 83.3] y el segundo (100, al azar con un filtro que hace de la empresa el tema) 82.0% [73.3, 88.3]; neutral tiene 35 etiquetas y 30 llamadas, con intervalos anchos. Las 190 son de la revisión de resumen con distilbart; los resúmenes de DeepSeek de la corrida del corpus no están cubiertos. Un lector, sin segunda lectura ciega; la línea entre neutral y una dirección no se fijó de antemano.
- **No es la compuerta de D35** (el gold de 180 textos sobre pares principales, T-148c4): `config/pipeline_models.json` sigue nombrando el modelo v1; el held-out es evidencia para la decisión del dueño. Marcas, textos y scorer versionados: `gold/sentiment_val190_2026_11_20_v2.csv`, `gold/texts/sentiment_val190_texts.csv`, `scripts/score_sentiment_val190_2026_11_16.py`.
- Un control con relevancia barajada suma a lo sumo +0.005, así que la ganancia del stack es información real, no ruido.
- El paso lee resúmenes de a lo sumo unos 2,000 caracteres, en fragmentos de a lo sumo el espacio que la empresa deja en 512 tokens, y promedia las probabilidades de los fragmentos. Un texto de sector no recibe sentiment (D42).

## 6. Category (paso 5, `docs/category_step.md`, `docs/category_training.md`)

> **Cambio planeado (no construido; el fijado, el paso y todas las cifras de abajo siguen siendo del modelo de 12 etiquetas).** El owner planea sacar `product_innovation`, `leadership_governance`, `capital_shareholder_returns` y `reputational_harm` del modelo para mejorar sus métricas, y reunirlas en `other`. El modelo quedaría con **8 etiquetas: 7 medidas** (`earnings_performance`, `market_analyst_sentiment`, `mergers_acquisitions`, `legal_regulatory`, `labor_human_capital`, `partnerships_business_dev`, `investor_activity`) **y `other`, que queda como una etiqueta no medida**: significa "el modelo no tiene nada que decir de este texto", así que no se reporta precisión ni recall para ella. Las cifras de abajo no se recalcularon para esta definición y deben volver a medirse cuando se construya (archivos de etiquetas nuevos, un ajuste nuevo, un fijado nuevo).

**Desde el 2026-11-15 (T-203, D43) el paso corre el fine-tune de entailment** (la receta de `config/category_model.json`, 12 etiquetas, una etiqueta por fila, reajustado con 1,216 filas; el fijado es `config/category_step.json`, `category-entailment-single-12`), **fijado por decisión del dueño** (la cifra oficial, 88.0%, está sobre la barra de 80% de D24). El modelo anterior del paso, el zero-shot sin modificar, se conserva como comparación. Cada corrida está en MLflow, experimento `news_nlp_train/category` (`uv run mlflow ui --backend-store-uri ./mlruns`); los pesos son el modelo registrado `category-entailment-single-12` (versión 1) y la copia privada del Hub `gamug/deberta-v3-news-category-12`.

### Descripción del dataset

Cada fila es un par (texto, empresa) con una etiqueta de la taxonomía de `docs/category-taxonomy.md`: **14 etiquetas en 6 grupos** para el modelo anterior del paso (zero-shot); el fine-tune que corre el paso lee 12 (`competitive_actions` y `peer_comparison` leídas como `other`).

| Conjunto | Tamaño | Características | Columnas |
|---|---|---|---|
| Filas de entrenamiento (hoja anterior) | 225 filas con etiqueta de empresa | cuerpos single y cuerpos multi estructurados divididos por empresa, del conjunto de entrenamiento de scope; cuerpos multi narrativos excluidos; las hipótesis del baseline se escribieron contra ellas | `article_id`, `company`, `label`, `pred` (archivos de resultados) |
| **Filas de entrenamiento de una sola etiqueta (fine-tune del paso)** | 1,216: una hoja de 1,000 filas reducida a una etiqueta por fila (999) más 217 de las 225 filas anteriores | 12 etiquetas; `earnings_performance` 547, `market_analyst_sentiment` 214, `product_innovation` 90, `mergers_acquisitions` 81, `legal_regulatory` 45, `leadership_governance` 40, `capital_shareholder_returns` 40, `reputational_harm` 39, `labor_human_capital` 36, `partnerships_business_dev` 32, `other` 32, `investor_activity` 20; 82% son segmentos cortos de empresa de resúmenes de mercado (mediana 272 caracteres); las etiquetas se revisaron después de ver los primeros resultados | por fila: texto, empresa, etiqueta, `fold` (5 pliegues agrupados por artículo) |
| **Held-out** | 200 textos, de los cuales 135 filas single son evaluables (116 cuerpos single); 44 textos `multi` narrativos no se evalúan | artículos completos (mediana 2,085 caracteres, ninguno bajo 600), convención de etiquetado anterior (las calificaciones de analistas podían ser `investor_activity`); evaluado una sola vez | `article_id`, `company`, `kind`, `label`, `pred`, más los 14 puntajes de etiqueta en la corrida mixta |
| Muestra de sub-news reales | 300 filas | un sorteo con semilla en dos lotes de 150 de los pares de sub-news de la primera corrida del corpus cuyo texto resumido mide menos de 1,000 caracteres (142,588 de 159,401 pares); un texto por cuerpo; ninguno de un cuerpo etiquetado; 107 de las 300 están marcadas como poco claras | texto, empresa, etiqueta, banderas |

### Selección del modelo

El paso corre el fine-tune de entailment fijado; el modelo anterior, el zero-shot sin modificar `MoritzLaurer/deberta-v3-base-zeroshot-v2.0` en su fijación, lee la premisa `<empresa>: <primeros 2,000 caracteres>` con una hipótesis por etiqueta ("The news about <empresa> is about <definición>"), y la mejor probabilidad de entailment da la etiqueta y su grupo da el grupo (6 grupos, opción B). **Ni uno ni otro alcanza la barra de 80% (D24) en el held-out.**

| Candidato | Resultado | Veredicto |
|---|---|---|
| Zero-shot sin modificar (el modelo anterior del paso) | held-out 43.0% (etiqueta, 14 etiquetas), 55.6% (grupo); 56.4% en las filas de entrenamiento | reemplazado por el fine-tune |
| Afinado plano, validación cruzada de 5 pliegues sobre 225 filas | 64.0% fuera de pliegue | mejor que zero-shot fuera de pliegue |
| Afinado mixto (etiquetas más grupos), 225 filas | 62.2% fuera de pliegue; held-out 52.6% (plano), 48.9% (dos pasos), grupo 62.2% | no publicado, no usado |
| **Fine-tune de entailment**, reajustado con 1,216 filas, 12 etiquetas | 87.8% fuera de pliegue; **held-out 54.1%** [45.9, 62.3]; 71.3% en 300 sub-news reales | **el modelo del paso (T-203)**, registrado como `category-entailment-single-12` v1 |
| DeBERTa-v3-base simple reentrenado con las mismas filas y pliegues | 85.7% fuera de pliegue | 2 puntos por debajo de la receta de entailment |

El held-out es la única medida que es held-out y falla; el fine-tune cae lejos de su 87.8% porque sus filas de entrenamiento son segmentos cortos de resúmenes de mercado y las filas del held-out son artículos completos.

**Métricas del fine-tune (el modelo del paso), las tres evaluaciones en una sola tabla.** Precisión y recall uno contra el resto por etiqueta, con el número de filas entre paréntesis (las filas del modelo de 12 etiquetas: 1,216, 135 y 300; las filas de exactitud y F1 macro usan la distribución planeada, 975, 84 y 235 filas, ver la nota bajo la tabla). **Fuera de pliegue:** 1,216 filas de una etiqueta, 5 pliegues agrupados por artículo; las etiquetas se revisaron después de ver los primeros resultados y por el asistente, así que mide qué tan bien aprende la receta esas etiquetas (filas: segmentos cortos de resúmenes de mercado, mediana 272 caracteres). **Held-out:** 135 filas single de 200 textos, etiquetas del dueño de la convención anterior (artículos completos, mediana 2,085 caracteres), evaluado **una sola vez**. **Sub-news reales:** 300 filas de la primera corrida del corpus (texto resumido de menos de 1,000 caracteres, un texto por cuerpo), etiquetadas en conjunto por el dueño y el asistente, a ciegas de ambos modelos (107 de las 300 marcadas poco claras).

| Etiqueta | Fuera de pliegue P / R (filas) | Held-out P / R (filas) | Sub-news reales P / R (filas) |
|---|---|---|---|
| earnings_performance | 0.97 / 0.97 (547) | 0.74 / 0.93 (27) | 0.82 / 0.89 (56) |
| market_analyst_sentiment | 0.89 / 0.88 (214) | 0.42 / 0.47 (17) | 0.81 / 0.80 (99) |
| mergers_acquisitions | 0.80 / 0.89 (81) | 0.36 / 0.80 (5) | 0.80 / 0.80 (30) |
| legal_regulatory | 0.72 / 0.73 (45) | 0.43 / 0.55 (11) | 0.77 / 0.77 (13) |
| labor_human_capital | 0.84 / 0.86 (36) | 0.33 / 0.67 (3) | 1.00 / 1.00 (4) |
| partnerships_business_dev | 0.77 / 0.75 (32) | 1.00 / 0.20 (5) | 0.65 / 0.77 (26) |
| investor_activity | 0.50 / 0.50 (20) | 0.67 / 0.12 (16) | 0.75 / 0.86 (7) |
| other (comodín, no medida) | — | — | — |
| **Acuerdo de etiqueta (exactitud), distribución planeada** | **91.5%** [89.6, 93.1] (892 de 975) | **57.1%** [46.5, 67.2] (48 de 84) | **82.1%** [76.7, 86.5] (193 de 235) |
| F1 macro (7 etiquetas medidas) | 0.791 | 0.462 | 0.818 |
| Puntaje de selección (D37) | no se recalculó | no se recalculó | no se calculó |
| Zero-shot sin modificar (el modelo anterior): exactitud, distribución planeada | 85.6% [83.3, 87.7] (833 de 973) | 53.6% [43.0, 63.8] (45 de 84) | 78.7% [73.1, 83.5] (185 de 235) |

**Quitadas de la tabla para la definición planeada:** `product_innovation`, `leadership_governance`, `capital_shareholder_returns` y `reputational_harm` (sus filas contarían como `other`), y `other` se muestra como no medida. **Cómo se obtuvieron las filas de abajo:** las predicciones del modelo de 12 etiquetas y las etiquetas se reasignan (las cuatro etiquetas se leen como `other`); **no es un modelo de 8 etiquetas reentrenado**, que debe medirse cuando se construya. **La exactitud y el F1 macro se calculan sobre las filas cuya etiqueta es una de las 7 medidas** (975, 84 y 235 filas; una de ellas predicha como `other` cuenta como error; las filas etiquetadas `other` no se puntúan); el F1 macro es sobre todas las filas, uno contra el resto en las 7 etiquetas, como la tabla por etiqueta. La precisión y el recall por etiqueta de arriba no cambian con la reasignación (las predicciones y filas de las 7 etiquetas son las mismas). Los intervalos son de Wilson (los de 12 etiquetas eran bootstrap por artículo); el puntaje de selección (D37) no se recalculó. El modelo sin modificar se lee igual, sobre las filas que comparten ambos modelos (973 de sus 1,214 filas fuera de pliegue). **Reproducir:** `scripts/category_planned_8_2026_11_16.py` (solo archivos versionados, sin GPU) escribe `docs/category_results_2026_11_15/planned_8_labels.json`; una prueba fija las cifras citadas aquí. Las cifras de 12 etiquetas (87.8%, 54.1%, 71.3%; F1 macro 0.748, 0.448, 0.637) están en `docs/category_training.md` y `docs/category_results_2026_11_15/`.

Frente a la barra de 80% (D24): **se cumple** con la medida oficial de abajo (88.0% [83.8, 91.2], el juez sobre 300 sub-news reales); held-out 54.1% (57.1% con la distribución planeada), acuerdo de grupo en el held-out 65.2% [57.4, 72.7].

**La cifra oficial y las cuatro etiquetas débiles movidas a `other` (dueño, 2026-11-17).** El dueño toma como medida oficial del modelo el **juez LLM a ciegas sobre las 300 sub-news reales** (`docs/category_results_2026_11_15/judge_step7_n300.md`; `deepseek-chat`, nunca vio la etiqueta de un modelo, aplica las reglas de etiquetado del proyecto): el modelo coincide con el juez cuando su etiqueta es la **primera o la segunda** del juez: **88.0% [83.8, 91.2] (264 de 300)** para el modelo fijado, contra 81.0% [76.2, 85.0] (243 de 300) del zero-shot sin modificar. La coincidencia estricta (solo la primera etiqueta) es 77.0% contra 71.0%. Leer las mismas filas con la definición planeada (`scripts/category_planned_8_2026_11_16.py`, nada reentrenado: las cuatro etiquetas débiles cuentan como `other` en ambos lados) sube todas las medidas; "8 etiquetas" puntúa `other` como una etiqueta, "7 medidas" deja fuera las filas cuya referencia es `other` (la definición planeada no la mide):

| Medida (modelo fijado) | 12 etiquetas (como corre el paso) | 8 etiquetas (`other` puntuada) | 7 etiquetas medidas (`other` no puntuada) |
|---|---|---|---|
| **Juez sobre 300 sub-news reales, oficial (primera o segunda etiqueta)** | **88.0%** | **92.0% [88.4, 94.6]** | **94.4% [90.9, 96.7]** (252 filas) |
| Juez, estricto (solo la primera etiqueta) | 77.0% | 80.0% [75.1, 84.1] | 81.7% [76.5, 86.0] |
| El mismo juez, zero-shot sin modificar (primera o segunda) | 81.0% | 85.7% | 90.9% |
| Fuera de pliegue (1,216 filas) | 87.8% | 89.4% [87.5, 91.0] | 91.5% |
| Held-out del dueño (135 filas, una vez) | 54.1% | 60.0% [51.6, 67.9] | 57.1% (84 filas) |
| 300 sub-news reales, etiquetas conjuntas | 71.3% | 75.7% [70.5, 80.2] | 82.1% (235 filas) |

Cómo se mide: el juez es `deepseek-chat`, a ciegas de la etiqueta del modelo, con las reglas de etiquetado del proyecto, y sus etiquetas no se repiten exactamente de una corrida a otra (2 de 80 cambiaron); las filas de 8 y 7 etiquetas leen el modelo de 12 etiquetas a través de la fusión, y el modelo de 8 etiquetas se mide cuando se construya (T-214).

**Nota sobre el held-out (lectura del owner):** esta muestra estaba muy sucia y **no es representativa de los textos que se van a categorizar**, que son los resúmenes del paso de resumen del pipeline. Las cifras del held-out (54.1% tal como está, 57.1% con la distribución planeada) deben leerse como una prueba sobre un conjunto difícil y ruidoso, no como la exactitud esperada en producción. Con la distribución planeada solo **84 de sus 135 filas se puntúan** (intervalo 46.5 a 67.2), así que además es pequeño.

**Qué dice y qué no.** (Las viñetas son sobre el modelo de 12 etiquetas, con sus cifras 87.8%, 54.1% y 71.3%, salvo que digan otra cosa.)
- **Las tres medidas discrepan porque son textos y etiquetadores distintos**; el held-out es la única que es held-out, y es la que falla (su muestra estaba sucia, nota arriba). Cae lejos de 87.8% por la diferencia de textos (segmentos cortos frente a artículos completos) y porque la convención de etiquetado cambió (la mayor confusión es `investor_activity` predicha como `market_analyst_sentiment`, 8 filas).
- **El fine-tune es mejor que el modelo anterior en las tres**, y la ganancia es clara en las 300 filas reales (acierta donde el anterior se equivoca en 38 filas y al revés en 21; prueba de signos p = 0.036). El asistente participó en el etiquetado de las filas de entrenamiento y de las 300 filas, así que el acuerdo en las 300 filas probablemente es más alto que el de un etiquetador independiente (83.9% donde el etiquetador estaba seguro, 48.6% donde no).
- **Solo `earnings_performance` es ajustada.** Las etiquetas con menos de unas 10 filas en un conjunto son anécdotas (una fila mueve la cifra 10 puntos o más; los intervalos de 95% de la validación cruzada son de ±0.12 a ±0.15 para las clases de 20 a 45 filas). **`other` es el fallo:** el fine-tune acierta 3 de 39 en las sub-news reales (ROC 0.68; el modelo anterior, 5 de 39, ROC 0.46); con la definición planeada `other` no se mide y este fallo sale de las cifras.
- Las tablas por etiqueta del modelo anterior (14 etiquetas), del afinado mixto y los intervalos están en `docs/category_results_2026_11_13/` y `docs/category_results_2026_11_15/` (`ovr_single.md`, `heldout_selected.md`, `real_sample.md`); las corridas, en MLflow.

## Lo que estas cifras no dicen

- **Los pasos no se comparan con una misma vara.** Cada uno se mide sobre su propia muestra, con su propio revisor y su propia definición de "correcto"; la tabla de resumen muestra cuáles son held-out.
- **Las cadenas de errores no están en estas cifras.** Un scope erróneo llega al cut, al paso de empresa, al paso de sector y al paso de category como entrada mala; por ejemplo, cerca de la mitad de las respuestas erróneas de sector de la primera revisión no eran textos de sector,
  y 15 de las 143 piezas muestreadas venían de textos que el paso de scope debió detener.
- **Cuatro medidas son sobre textos de la corrida real del corpus**: el held-out del NER (100 textos del paso 2, 3.1), las marcas de empresa principal sobre piezas (3.2, parciales), la verificación de sentiment sobre 190 piezas (5) y la muestra de datos reales de category (6, etiquetas conjuntas). El embudo (textos por paso, y cuántos llegan a un snapshot) sale del almacén RESULTS real y no está en este informe.
- **Las tres medidas de category no son una misma vara** (6): fuera de pliegue sobre etiquetas que revisó el asistente, held-out sobre las etiquetas anteriores del dueño (artículos completos), y una muestra de datos reales etiquetada en conjunto por el dueño y el asistente (segmentos cortos). La medida oficial es una cuarta, el juez LLM a ciegas sobre las sub-news reales (88.0%, 6), contando su primera o segunda etiqueta.
- **No se mide en absoluto:** el recall de sector por sector, el recall del cut, la empresa principal por pieza más allá de las 185 filas marcadas de la sección 3.2, el NER sobre los resúmenes de DeepSeek, la relevancia sobre piezas más allá de esas marcas, y el puntaje held-out del stack de sentiment más allá de las 190 piezas de la sección 5.

## De dónde sale cada cifra

| Paso | Fuente |
|---|---|
| Scope | `docs/scope_training.md` (secciones 5, 6, 9); `data/scope_end_to_end_with81b.csv` y `gold/category_gold_v2_test.csv` (tabla por clase recalculada) |
| Sector | `docs/sector_step.md` ("The clean test", "What this does not show"); `docs/sector_training.md` (357 respuestas); `gold/sector_validation.csv` (los primeros 57); `scripts/score_sector_fresh_2026_11_19.py` |
| Cut | `docs/scope_training.md` secciones 8.1 y 8.3; `docs/extractor_training.md` (actualización del 2026-11-11, `units4` y `units5`); `gold/cut_sample_marks.csv`; `scripts/cut_sample_metrics_2026_11_12.py` |
| Empresa: NER | `docs/ner_training.md` (el held-out de 100 textos); `docs/ner_results_2026_11_16/` (`ner_metrics.json`, `ner_errors.csv`); `scripts/ner_eval_2026_11_16.py`; `docs/evaluation.md` (línea base v1 de 2026-09, `ner`) |
| Empresa: relevancia | `docs/companies_step.md` (tabla de umbrales); `docs/sentiment_training.md` ítem 6 (AUC 0.90); `docs/main_company_training.md` y `gold/main_company_marks_2026_11_20.csv` (las marcas sobre piezas, el held-out, versionadas) |
| Summarization | `docs/summarization_training.md` (brazos, niveles y el brazo D de DeepSeek); `docs/summarization_results_2026_11_15/` (cifras versionadas del brazo D); `docs/evaluation.md` (línea base v1); `docs/summary_step.md` (límites) |
| Sentiment | `docs/sentiment_step.md` ("What the figures are"); `docs/sentiment_training.md` (tabla del stack, validación cruzada anidada); el held-out de 190 piezas: `gold/sentiment_val190_2026_11_20_v2.csv`, `gold/texts/sentiment_val190_texts.csv`, `scripts/score_sentiment_val190_2026_11_16.py` |
| Category | `docs/category_step.md`; `docs/category_training.md`; `docs/category_results_2026_11_13/` (zero-shot anterior, tablas recalculadas desde `baseline_heldout.csv` y `baseline_train.csv`); `docs/category_results_2026_11_15/` (`ovr_single.md` fuera de pliegue; `heldout_selected.md` held-out, evaluado una sola vez por `scripts/category_heldout_2026_11_15.py`; `judge_step7_n300.md` (el juez a ciegas, cifra oficial), `planned_8_labels.json` (la fusión), `real_sample.md` y `scripts/category_real_sample_2026_11_15.py` para las 300 filas reales, etiquetas en `gold/category_real_sample.csv`); experimento MLflow `news_nlp_train/category` (`scripts/category_mlflow_log_2026_11_15.py`) y el modelo registrado `category-entailment-single-12`; puntajes D37 de las corridas de MLflow |

## Todos los modelos juntos

Una tabla con las métricas generales de cada modelo, para mostrar los resultados de un vistazo. Todas las cifras están medidas arriba, en la sección de cada paso, y se copian aquí tal cual; "validación" es validación cruzada o fuera de pliegue, y "held-out" son textos con los que el modelo no se entrenó ni se ajustó. Un guion es que esa medida no existe.

| Paso | Modelo | Validación (entrenamiento) | Held-out | Barra y estado |
|---|---|---|---|---|
| Scope (paso 1) | dos clasificadores NLI afinados sobre `deberta-v3-base-zeroshot-v2.0` | — | acuerdo ponderado 83.6% [78.1, 88.9] (200 textos); precisión 90.8% en cuarentena (225); recall de `out_of_scope` 53% | 80%: se cumple por estimación puntual (extremo inferior 78.1) |
| Sector (paso 4) | `bge-small-en-v1.5` con compuerta | — | 85.9% de los sectores asignados correctos (176 de 205); 96.7% sin los `no` de "no es de sector"; recall no medido | sin barra medida (D24 para subindustria no medida, T-167) |
| Cut (paso 1.2) | `units5` con el clasificador de pares (`gamug/FinBERT-news-attribution`) | — | 84.6% [77.8, 89.6] de piezas fieles; 75.5% limpias (143 piezas) | 80% fieles: la estimación pasa, el extremo inferior no |
| NER (paso 3) | `gamug/sec-bert-finer-ord-ner` | juez LLM v1 (1,000 artículos): F1 de tramos 0.7418, ORG 0.75 | empresas del universo: P 92.6%, R 95.5%, F1 0.940 (R 98.7% con `ner+text`); todas las organizaciones: P 72.6%, R 90.4% (100 textos; la superposición sobre 14 etiquetas: 9 reemplazadas, 1 conservada, 4 descartadas); tal como están, universo P 90.1%, R 95.4% | sin barra |
| Relevancia (paso 3) | `gamug/deberta-v3-company-relevance` | AUC 0.90; principal P 90.0%, R 94.7% (809 resúmenes) | AUC 0.829 (filas); recall de elección única 100.0% (88/88), precisión 84.6% (88/104); a 0.5, recall por artículo 69.3%, precisión 91.0% (185 filas de 104 artículos) | sin barra; el umbral de sub-news lo decide el dueño (T-170) |
| Summarization (paso 2) | DeepSeek (`deepseek-chat`) | — | sin held-out; juez LLM sobre 300 artículos: fidelidad 4.99, cobertura 4.54, alucinación 0.7% (el juez también es DeepSeek) | sin barra; la corrida del corpus no se mide |
| Sentiment (paso 6) | modelo de pares + stack (`gamug/FinBERT-company-sentiment`) | puntaje 0.814; negativo P/R 0.747 / 0.748 (809 filas) | exactitud 78.9% [72.6, 84.1]; negativo P 70.8%, R 83.6% (190 piezas) | 80% de acuerdo y 70% en el negativo (D32): **no se cumple** (exactitud agregada 78.9%) |
| Category (paso 5) | fine-tune de entailment (`category-entailment-single-12`) | exactitud 91.5%, F1 macro 0.791 (975 filas de las 7 etiquetas medidas) | **juez sobre 300 sub-news reales (oficial): 88.0%**; exactitud 57.1% [46.5, 67.2] (84 filas, held-out); 82.1% en 235 sub-news reales (etiquetas conjuntas) | 80% (D24): **se cumple** con la cifra oficial del juez (88.0%, 300 sub-news reales); fijado por decisión del dueño; las cifras son del modelo de 12 etiquetas reasignado a la distribución planeada, no reentrenado (sección 6; 12 etiquetas: 87.8%, 54.1%, 71.3%) |
