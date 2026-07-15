![Portada del proyecto](data/imagen_readmeML.png)

# Análisis Exploratorio y Segmentación de Clientes ·

> Parte del proyecto correspondiente al **análisis exploratorio de datos (EDA)** y a la **segmentación de clientes** (aprendizaje no supervisado). Rama: `feature/eda`.

## Contexto

Campaña de telemarketing de una entidad bancaria para la contratación de un **depósito a plazo**. El objetivo del proyecto es predecir qué clientes contratan el producto, para priorizar a quién contactar y optimizar el esfuerzo comercial.

- **Conjunto de datos:** Bank Marketing — UCI Machine Learning Repository
- **Registros:** 45.211 clientes · **Variables:** 16 predictoras + objetivo (`y`: yes/no)
- **Enfoque:** híbrido → segmentación de clientes (no supervisado) + clasificación (supervisado)

## Contenido de esta parte

| Archivo | Descripción |
|---|---|
| `EDA_dirigido_PersonaA.ipynb` | EDA dirigido: calidad del dato, variable objetivo, análisis univariante y relación con el objetivo. |
| `Modulo_Segmentacion_Clientes.ipynb` | Segmentación con K-Means: elección de algoritmo y de *k*, perfilado de segmentos y guardado de artefactos. |
| `Memoria_Tecnica_EDA.docx` | Memoria técnica del EDA (documento formal). |
| `models/scaler_segmentacion.pkl` | Escalador ajustado sobre *train* para la segmentación. |
| `models/kmeans_segmentacion.pkl` | Modelo de clustering (K-Means, k=4) entrenado. |

## Cómo ejecutar

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```

1. Coloca el fichero `bank-full.csv` junto a los notebooks (o ajusta la variable `DATA_PATH`).
2. **Importante:** el separador del CSV es `;` → `pd.read_csv(ruta, sep=";")`.
3. Ejecuta los notebooks de arriba abajo. Ambos se ejecutan sin errores y generan sus figuras en línea.

## Metodología

El **EDA descriptivo** se realiza sobre el conjunto de datos completo, para comprender su estructura. La prevención de la **fuga de datos** (*data leakage*) se aplica en las fases que *ajustan parámetros*: el preprocesado y la segmentación se ajustan **exclusivamente sobre el conjunto de entrenamiento** (partición estratificada 80/20, `random_state=42`).

## Principales hallazgos del EDA

- **Variable objetivo desbalanceada:** solo el **11,7 %** de los clientes contactados contrata (88,3 % / 11,7 %). → Se recomienda medir con **F1, recall y ROC-AUC** (no *accuracy*) y estratificar la partición.
- **Fuga de datos en `duration`:** es la variable más correlacionada con el objetivo (r ≈ 0,39; 537 s de media en los "sí" frente a 221 s en los "no"), pero su valor **solo se conoce tras la llamada**. → **Se excluye** del conjunto de predictores.
- **Asimetría fuerte** en `balance`, `campaign` y `previous`. → Transformación logarítmica/robusta.
- **Valores ausentes camuflados:** no hay nulos explícitos, pero la categoría `unknown` los sustituye (`poutcome` ~82 %, `contact` ~29 %, `education` ~4 %, `job` ~0,6 %). → Tratar como categoría; el `unknown` de `contact` aporta señal (baja conversión).
- **Valor centinela:** `pdays = -1` (~82 %) significa "sin contacto previo". → Recodificar como indicador binario.
- **Señales de negocio:** un resultado previo `success` dispara la conversión al **64,7 %**; hay fuerte **estacionalidad** (marzo, sep, oct y dic altos; mayo el más bajo); la **edad** influye en forma de U (jóvenes y mayores contratan más); y la ausencia de deudas se asocia a mayor conversión.

### Hallazgo → acción → responsable

| Hallazgo | Acción recomendada | Responsable |
|---|---|---|
| Objetivo desbalanceado (88/12) | Métricas F1/recall; `stratify` | Modelado |
| `duration` = fuga de datos | Excluir del modelo | Equipo |
| Asimetría en variables numéricas | Transformación logarítmica/robusta | Preprocesado |
| `pdays = -1` (centinela) | Recodificar como marca binaria | Preprocesado |
| `unknown` = ausencia camuflada | Tratar como categoría | Preprocesado |
| Cardinalidad baja (≤ 12) | One-Hot Encoding | Preprocesado |
| `poutcome`, mes, edad, situación crediticia | Conservar (alto valor predictivo) | Modelado |

## Segmentación de clientes

Se agrupa la cartera con **K-Means** para identificar perfiles accionables y aportar una variable adicional al clasificador.

- **Variables:** perfil demográfico y financiero disponible **antes** del contacto (`age`, `log_balance`, `campaign`, `previous`, hipoteca, préstamo, contacto previo). **Sin `y` y sin `duration`.**
- **Elección de algoritmo:** se comparó K-Means con DBSCAN y clustering jerárquico. Se elige **K-Means** por escalabilidad e interpretabilidad (DBSCAN fragmenta con ruido; el jerárquico no escala a ~36.000 filas).
- **Número de segmentos:** `k = 4`. Aunque el silhouette es máximo en k=3, se prioriza la **interpretabilidad y accionabilidad** de los perfiles (validado también con Davies-Bouldin, Calinski-Harabasz y método del codo).

### Resultados (tasa de suscripción por segmento)

| Segmento | % cartera | Conversión |
|---|---|---|
| Contactados previamente | 18 % | **23,3 %** |
| Sin cargas financieras | 32 % | 13,5 % |
| Con hipoteca | 37 % | 6,7 % |
| Con préstamo · saldo reducido | 14 % | 6,3 % |

La conversión varía **del 23 % al 6 %** según el perfil (casi ×4): existe separación real, lo que permite **priorizar** los perfiles de mayor propensión.

### Integración con el pipeline

- La **etiqueta de segmento** se incorpora como **una variable predictora más** en el preprocesado (tratada como categórica → One-Hot).
- La segmentación está **cerrada**: se entrena una sola vez sobre *train* y se **serializan** el `scaler` y el `KMeans`. En inferencia, cada cliente se asigna a un segmento (`kmeans.predict`) **antes** de puntuarse; no se reentrena.

---

*Análisis realizado sobre el conjunto de entrenamiento (partición estratificada 80/20). El EDA describe sobre el conjunto completo; las transformaciones que ajustan parámetros se aplican solo sobre train.*

## Feature Engineering

### Eliminación de variables con fuga de información
- `duration` se elimina completamente.  
  Su valor depende del resultado de la llamada → **no puede usarse para predecir antes de llamar**.

### Recodificación de variables
- `pdays = -1` indica *ausencia de contacto previo*.  
- Se transforma en una variable binaria:  
  **0 = sin contacto previo**, **1 = contacto previo**.

### Recodificación de categorías
- `poutcome = unknown` se recodifica como `no_previous_contact`.  
- Todos los registros con `unknown` en `poutcome` coinciden con `pdays = -1`, por lo que ambas variables representan la misma condición: **ausencia de contacto previo**.

### Imputación de valores faltantes reales
- `job = unknown` aparece en solo **0.6%** de los registros.  
- Se imputa con la **moda del conjunto de entrenamiento** para evitar distorsión.

### Codificación de variables binarias
- `default`, `housing`, `loan` → se convierten a **0/1**.

### Conservación de categorías válidas
- `education` y `contact` conservan la categoría `unknown`, dado que su presencia en el dataset es del **4.1%** y **28.8%** respectivamente. Imputarlas podría introducir ruido, modificar la estructura real de los datos y reducir la capacidad del modelo para capturar patrones asociados a estos valores.

---

## Preparación del modelado

### Separación Train/Test
Se realiza un **split estratificado** para preservar la proporción de clases.

### Identificación de tipos de variables
- Variables numéricas  
- Variables categóricas  
- Variables binarias ya transformadas  

---

## Pipelines para modelos

Se construyen **3 pipelines independientes**, optimizados para cada familia de modelos:

### Modelos que requieren transformación de sesgo + escalado + OneHot
- Logistic Regression  
- KNN  
- SVM  

Incluyen:
- Transformación de sesgo (`log1p` o `cbrt`)  
- `StandardScaler` para numéricas  
- `OneHotEncoder` para categóricas  

### Modelos que no requieren escalado pero sí OneHot
- DecisionTree  
- RandomForest  
- XGBoost  
- LightGBM  

Incluyen:
- Numéricas en *passthrough*  
- `OneHotEncoder` para categóricas  

### CatBoost
- No requiere escalado  
- No requiere OneHot  
- Maneja categóricas de forma nativa  
- Gestiona internamente el desbalanceo



---







-------- ESTRUCTURA PROPUESTA------

![Portada del proyecto](data/imagen_readmeML.png)

## Descripción del problema
Campaña de telemarketing de una entidad bancaria para la contratación de un **depósito a plazo**.  
El objetivo del proyecto es predecir qué clientes contratan el producto, para priorizar a quién contactar y optimizar el esfuerzo comercial.

---

## Dataset utilizado
- **Conjunto de datos:** Bank Marketing — UCI Machine Learning Repository  
- **Registros:** 45.211 clientes  
- **Variables:** 16 predictoras + objetivo (`y`: yes/no)  
- **Enfoque:** híbrido → segmentación de clientes (no supervisado) + clasificación (supervisado)
- **Fuente oficial:** https://archive.ics.uci.edu/dataset/222/bank+marketing

---

# Análisis Exploratorio y Segmentación de Clientes

## Contenido de esta parte

| Archivo | Descripción |
|---|---|
| `EDA_dirigido_PersonaA.ipynb` | EDA dirigido: calidad del dato, variable objetivo, análisis univariante y relación con el objetivo. |
| `Modulo_Segmentacion_Clientes.ipynb` | Segmentación con K-Means: elección de algoritmo y de *k*, perfilado de segmentos y guardado de artefactos. |
| `Memoria_Tecnica_EDA.docx` | Memoria técnica del EDA (documento formal). |
| `models/scaler_segmentacion.pkl` | Escalador ajustado sobre *train* para la segmentación. |
| `models/kmeans_segmentacion.pkl` | Modelo de clustering (K-Means, k=4) entrenado. |

## Cómo ejecutar

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```

1. Coloca el fichero `bank-full.csv` junto a los notebooks (o ajusta la variable `DATA_PATH`).  
2. **Importante:** el separador del CSV es `;` → `pd.read_csv(ruta, sep=";")`.  
3. Ejecuta los notebooks de arriba abajo. Ambos se ejecutan sin errores y generan sus figuras en línea.

## Metodología

El **EDA descriptivo** se realiza sobre el conjunto de datos completo, para comprender su estructura.  
La prevención de la **fuga de datos** (*data leakage*) se aplica en las fases que *ajustan parámetros*:  
el preprocesado y la segmentación se ajustan **exclusivamente sobre el conjunto de entrenamiento** (partición estratificada 80/20, `random_state=42`).

## Principales hallazgos del EDA

- **Variable objetivo desbalanceada:** solo el **11,7 %** de los clientes contactados contrata (88,3 % / 11,7 %).  
- **Fuga de datos en `duration`:** es la variable más correlacionada con el objetivo, pero solo se conoce tras la llamada → **se excluye**.  
- **Asimetría fuerte** en `balance`, `campaign` y `previous`.  
- **Valores ausentes camuflados:** `unknown` sustituye nulos en varias variables.  
- **Valor centinela:** `pdays = -1` significa “sin contacto previo”.  
- **Señales de negocio:** estacionalidad, impacto de resultados previos, forma de U en la edad, etc.

### Hallazgo → acción → responsable

| Hallazgo | Acción recomendada | Responsable |
|---|---|---|
| Objetivo desbalanceado (88/12) | Métricas F1/recall; `stratify` | Modelado |
| `duration` = fuga de datos | Excluir del modelo | Equipo |
| Asimetría en variables numéricas | Transformación logarítmica/robusta | Preprocesado |
| `pdays = -1` (centinela) | Recodificar como marca binaria | Preprocesado |
| `unknown` = ausencia camuflada | Tratar como categoría | Preprocesado |
| Cardinalidad baja (≤ 12) | One-Hot Encoding | Preprocesado |
| `poutcome`, mes, edad, situación crediticia | Conservar (alto valor predictivo) | Modelado |

## Segmentación de clientes

Se agrupa la cartera con **K-Means** para identificar perfiles accionables y aportar una variable adicional al clasificador.

- **Variables:** perfil demográfico y financiero disponible **antes** del contacto (`age`, `log_balance`, `campaign`, `previous`, hipoteca, préstamo, contacto previo).  
- **Sin `y` y sin `duration`.**  
- **Algoritmo:** K-Means (comparado con DBSCAN y jerárquico).  
- **Número de segmentos:** `k = 4`.

### Resultados (tasa de suscripción por segmento)

| Segmento | % cartera | Conversión |
|---|---|---|
| Contactados previamente | 18 % | **23,3 %** |
| Sin cargas financieras | 32 % | 13,5 % |
| Con hipoteca | 37 % | 6,7 % |
| Con préstamo · saldo reducido | 14 % | 6,3 % |

### Integración con el pipeline

- La **etiqueta de segmento** se incorpora como **una variable predictora más** en el preprocesado (tratada como categórica → One-Hot).
- La segmentación está **cerrada**: se entrena una sola vez sobre *train* y se **serializan** el `scaler` y el `KMeans`. En inferencia, cada cliente se asigna a un segmento (`kmeans.predict`) **antes** de puntuarse; no se reentrena.

---
*Análisis realizado sobre el conjunto de entrenamiento (partición estratificada 80/20). El EDA describe sobre el conjunto completo; las transformaciones que ajustan parámetros se aplican solo sobre train.*
---

# 3. Feature Engineering

### Eliminación de variables con fuga de información
- `duration` se elimina completamente.  
  Su valor depende del resultado de la llamada → **no puede usarse para predecir antes de llamar**.

### Recodificación de variables
- `pdays = -1` indica *ausencia de contacto previo*.  
- Se transforma en una variable binaria:  
  **0 = sin contacto previo**, **1 = contacto previo**.

### Renombrado de variables
- `poutcome = unknown` se recodifica como `no_previous_contact`.  
- Se alinea con la interpretación de `pdays = -1`.

### Imputación de valores faltantes reales
- `job = unknown` aparece en solo **0.6%** de los registros.  
- Se imputa con la **moda del conjunto de entrenamiento** para evitar distorsión.

### Codificación de variables binarias
- `default`, `housing`, `loan` → se convierten a **0/1**.

### Conservación de categorías válidas
- `education` y `contact` mantienen `unknown` como categoría.

---

# 4. Preparación del modelado

### Separación Train/Test
Se realiza un **split estratificado** para preservar la proporción de clases.

### Identificación de tipos de variables
- Variables numéricas  
- Variables categóricas  
- Variables binarias ya transformadas  

---

# 5. Pipelines para modelos

Se construyen **3 pipelines independientes**, optimizados para cada familia de modelos:

### Modelos que requieren transformación de sesgo + escalado + OneHot
- Logistic Regression  
- KNN  
- SVM  

Incluyen:
- Transformación de sesgo (`log1p` o `cbrt`)  
- `StandardScaler` para numéricas  
- `OneHotEncoder` para categóricas  

### Modelos que no requieren escalado pero sí OneHot
- DecisionTree  
- RandomForest  
- XGBoost  
- LightGBM  

Incluyen:
- Numéricas en *passthrough*  
- `OneHotEncoder` para categóricas  

### CatBoost
- No requiere escalado  
- No requiere OneHot  
- Maneja categóricas de forma nativa  
- Gestiona internamente el desbalanceo  

---

# 6. Modelado

- Métricas

---

# 7. Resultados del modelado


---

# 8. Conclusiones



## Autores
- Claudia 
    — GitHub
    — LinkedIn

- Marta Harana Herrera 
    — GitHub: https://github.com/MHHsim  
    — LinkedIn: https://www.linkedin.com/in/marta-harana-herrera-004a84117/

- Maria Rodriguez 
    — GitHub 
    — LinkedIn  


