# Enfoque cerrado: Knowledge Hub interno de Tinámica

## Caso de uso

Recomendador de recursos de conocimiento interno para consultores. El problema que resuelve: en una consultora, el conocimiento generado en cada proyecto (plantillas, código reutilizable, informes, casos de éxito) queda disperso y es difícil de encontrar para quien empieza un proyecto nuevo. Esto provoca que se reinvierta tiempo en resolver problemas ya resueltos en otro proyecto o vertical. El objetivo del recomendador es, dado el perfil de un consultor y el contexto del proyecto en el que está, sugerirle los recursos internos más relevantes para él, acelerando el arranque de proyectos y fomentando que el conocimiento se reutilice entre verticales (banca, retail, industria, etc.) en lugar de quedarse aislado dentro de cada equipo.

## Usuarios

Los usuarios del sistema son los consultores de Tinámica. Cada consultor se describe con un perfil compuesto por:
- **Rol**: p. ej. data scientist, ingeniero de datos, consultor funcional, arquitecto...
- **Seniority**: nivel de experiencia (junior, senior, manager...).
- **Tecnologías**: stack tecnológico que domina o utiliza habitualmente (Python, Spark, cloud, etc.).
- **Sector del cliente actual**: la vertical de negocio en la que está trabajando en su proyecto en curso (banca, seguros, retail, industria...).

Este perfil es el que permite tanto encontrar consultores con gustos/necesidades parecidas (para la parte colaborativa del sistema) como razonar por contenido cuando no hay histórico suficiente.

## Ítems

Los ítems son los recursos de conocimiento disponibles en el hub, agrupados en dos familias:
- **Técnicos**: plantillas de proyecto, snippets o componentes de código reutilizables, documentación técnica.
- **Casos de éxito**: informes de proyecto y casos de éxito, organizados por sector y cliente (banca, retail, industria, etc.), que recogen cómo se resolvió un problema similar en el pasado.

Cada recurso lleva asociada una metadata que lo describe independientemente de si tiene interacciones o no: **tecnología** a la que pertenece, **sector** al que aplica y **tipo de recurso** (plantilla, snippet, documentación, informe, caso de éxito). Esta metadata es la que se usará para las recomendaciones basadas en contenido.

## Señal de interacción

La señal de interacción entre consultor y recurso es **explícita**: un rating de **1 a 5** que el consultor asigna al recurso después de haberlo usado, indicando cuánta utilidad le ha aportado. Esta señal explícita es la que alimentará todos los modelos de la progresión técnica, y es equivalente a las puntuaciones de películas o productos que se usan habitualmente en los sistemas de recomendación clásicos, pero aplicada a recursos de conocimiento interno.

## Progresión técnica en la demo

La demo se plantea como una progresión de modelos de complejidad creciente, de forma que cada paso muestre las limitaciones del anterior y justifique por qué se incorpora la siguiente técnica:

1. **TF-IDF (recomendación basada en contenido / cold-start)**: recomienda recursos comparando su metadata (tecnología, sector, tipo) mediante similitud de texto, sin necesidad de que existan ratings previos. Resuelve el problema del **cold-start**: un consultor recién incorporado o un recurso recién publicado, al no tener todavía historial de interacciones, no pueden ser recomendados por los modelos colaborativos, pero sí por este basado en contenido.

2. **Filtrado colaborativo (Collaborative Filtering, CF) basado en memoria — modelo baseline**: recomienda a partir de los patrones de interacción entre consultores y recursos, sin mirar el contenido de estos. Se implementa en dos variantes: *user-based* (busca consultores con gustos parecidos y recomienda lo que a ellos les ha resultado útil) e *item-based* (busca recursos parecidos entre sí en función de quién los ha valorado de forma similar). Sirve como modelo de referencia frente al que comparar los siguientes.

3. **Factorización de matrices (Matrix Factorization, MF) clásica**: en vez de calcular similitudes directas como el CF baseline, aprende para cada consultor y cada recurso un vector de "factores latentes" (características implícitas no etiquetadas manualmente) de forma que el producto de ambos vectores prediga el rating. Es el enfoque de filtrado colaborativo *basado en modelo* que popularizaron sistemas como el de Netflix, y mejora al baseline al capturar patrones más sutiles que la similitud directa.

4. **MF con Keras (deep learning)**: sobre la misma idea de factores latentes, se sustituye el producto simple por una red neuronal (embeddings + capa densa) que combina los vectores de consultor y recurso, permitiendo capturar relaciones no lineales entre ellos que el producto escalar de MF no puede representar.

5. **AutoRec (deep learning avanzado)**: un autoencoder que aprende a reconstruir el patrón completo de ratings de un consultor -en vez de aprender un vector latente por consultor y otro por recurso como en los pasos 3 y 4-, comprimiéndolo en una capa oculta más pequeña y reconstruyéndolo a la salida. Sirve como pieza de mayor lucimiento técnico de la demo, mostrando cómo el deep learning se aplica también sobre el problema de recomendación en su forma más directa: reconstrucción de la matriz de ratings.

## Estructura del proyecto

```
recommenders/
├── pyproject.toml   # dependencias del proyecto, gestionadas con uv
├── scripts/         # un script por paso de la progresión (01_tfidf.py ... 05_autorec.py)
├── utils/           # código compartido por todos los scripts (datos, evaluación, métricas, logging)
├── dataset/         # CSVs sintéticos (consultores, recursos, ratings) y su generador
└── logs/            # salida de cada ejecución, una por script
```

## Cómo ejecutar

El proyecto usa [uv](https://docs.astral.sh/uv/) para gestionar la versión de Python y las dependencias (`pyproject.toml` + `uv.lock`), así que no hace falta crear ni activar manualmente un virtualenv.

```bash
uv sync                        # crea el entorno (.venv) con las dependencias fijadas en uv.lock
uv run scripts/01_tfidf.py     # ...02_cf.py, 03_mf.py, 04_mf_keras.py, 05_autorec.py
```

Cada script deja su log (métricas incluidas) en `logs/<nombre_del_script>.log`.
