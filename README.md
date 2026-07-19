![Portada del proyecto](src/img/imagen_readmeML.png)

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

## Estructura del repositorio

```
├── src/
│   ├── data_sample/    # Archivos de datos de muestra (máx. 100MB)
│   ├── img/            # Imágenes utilizadas en el proyecto
│   ├── models/         # Modelos guardados en formato pickle o joblib
│   ├── notebooks/      # Notebooks de desarrollo y pruebas
│   ├── utils/          # Módulos y funciones auxiliares
├── main.ipynb          # Notebook final del pipeline de ML
├── Presentacion.pdf    # Documento soporte de la exposición
├── README.md           # Fichero README resumen del proyecto
├── requirements.txt    # Fichero con las dependencias usadas en el proyecto y reproducible
```

---

## Tecnologías utilizadas

- **Python 3.10+**
- **pandas** y **numpy** — análisis y manipulación de datos
- **scikit-learn** — modelos base, pipelines, métricas y validación cruzada

- **Modelos de Gradient Boosting utilizados:**
  - XGBoost  
  - LightGBM  
  - CatBoost  

- **Modelos clásicos utilizados en la experimentación:**
  - Logistic Regression  
  - K-Nearest Neighbors (KNN)  
  - Support Vector Machine (SVM)  
  - Decision Tree  
  - Random Forest  

- **GridSearchCV** y **RandomizedSearchCV** — optimización de hiperparámetros  
- **Matplotlib** y **Seaborn** — visualización de resultados  
- **Jupyter Notebook** — desarrollo y análisis exploratorio  
- **Git + GitHub Desktop** — control de versiones  
- **Visual Studio Code (VS Code)** — entorno de desarrollo

---

##  Instrucciones de reproducción

Para ejecutar el proyecto y reproducir los resultados del modelo, sigue los pasos indicados a continuación:

```bash
# 1. Clonar el repositorio
git clone https://github.com/claudiafranzoni/Proyecto_Modelo_machine_learning
cd Proyecto_Modelo_machine_learning

# 2. Crear entorno virtual
python -m venv venv

# 3. Activar entorno virtual
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate

# 4. Instalar dependencias
pip install -r requirements.txt

# 5. Ejecutar notebook
jupyter notebook notebooks/main.ipynb
```

## Cargar modelo final

```python
import joblib
modelo = joblib.load("models/catboost_optimizado.pkl")
```

## Análisis Exploratorio y Segmentación de Clientes

### Contenido de esta parte

| Archivo | Descripción |
|---|---|
| `EDA_dirigido_PersonaA.ipynb` | EDA dirigido: calidad del dato, variable objetivo, análisis univariante y relación con el objetivo. |
| `Modulo_Segmentacion_Clientes.ipynb` | Segmentación con K-Means: elección de algoritmo y de *k*, perfilado de segmentos y guardado de artefactos. |
| `models/scaler_segmentacion.pkl` | Escalador ajustado sobre *train* para la segmentación. |
| `models/kmeans_segmentacion.pkl` | Modelo de clustering (K-Means, k=4) entrenado. |

### Cómo ejecutar

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```

1. Coloca el fichero `bank-full.csv` junto a los notebooks (o ajusta la variable `DATA_PATH`).  
2. **Importante:** el separador del CSV es `;` → `pd.read_csv(ruta, sep=";")`.  
3. Ejecuta los notebooks de arriba abajo. Ambos se ejecutan sin errores y generan sus figuras en línea.

### Metodología

El **EDA descriptivo** se realiza sobre el conjunto de datos completo, para comprender su estructura.  
La prevención de la **fuga de datos** (*data leakage*) se aplica en las fases que *ajustan parámetros*:  
el preprocesado y la segmentación se ajustan **exclusivamente sobre el conjunto de entrenamiento** (partición estratificada 80/20, `random_state=42`).

### Principales hallazgos del EDA

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

### Segmentación de clientes

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

## Feature Engineering

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

## Modelado

El objetivo de esta fase ha sido desarrollar un modelo de clasificación binaria supervisada capaz de estimar la probabilidad de que un cliente contrate un depósito a plazo (`y` ∈ {no, yes}). El propósito fundamental de negocio es **priorizar los contactos del *call center* antes de realizar la llamada**, optimizando los costes y aumentando el ratio de conversión.

Para garantizar la máxima rigurosidad y robustez del modelo, la estrategia se fundamentó en tres pilares:

###  1. Prevención del *Data Leakage* (Fuga de Datos)
* **Exclusiones de negocio:** Se eliminó la variable predictiva `duration` (duración de la llamada). Aunque presenta una correlación alta con la contratación ($r \approx 0.39$), este dato solo se conoce *después* de finalizada la llamada. Incluirlo falsearía las métricas e invalidaría el objetivo de priorización previa.
* **Separación de particiones:** La división de datos en conjuntos de entrenamiento (80%) y prueba (20%) —este último compuesto por **9,043 registros**— se realizó de forma previa a cualquier transformación. Todas las imputaciones, escalados y codificaciones (*One-Hot Encoding*) se ajustaron y aplicaron utilizando un entorno de **Pipeline**, evitando el filtrado de información del conjunto test.

###  2. Gestión de Clases Desbalanceadas
El conjunto de datos presenta un desbalance severo donde solo el **11.7%** de los clientes contrata el depósito. 
* Se aplicó una partición **estratificada** (`stratify=y`) para mantener las proporciones exactas en todas las matrices.
* Se configuró el tratamiento del desbalance de manera nativa en la raíz de los clasificadores: integrando parámetros como `class_weight='balanced'` en los algoritmos tradicionales y calculando el ratio de compensación exacto mediante `scale_pos_weight` en el ecosistema XGBoost.

###  3. Estrategia de Evaluación y "Torneo de Baselines"
Se descartó la exactitud (*accuracy*) al ser una métrica engañosa en problemas desbalanceados. El sistema de evaluación modular midió el rendimiento en función de tres indicadores clave de negocio:
1. **Recall (Sensibilidad):** Capacidad de no dejar escapar clientes dispuestos a contratar (minimizar el coste de oportunidad).
2. **Precision (Precisión):** Capacidad de acertar en la llamada para optimizar el tiempo del comercial (minimizar llamadas inútiles).
3. **F1-Score y ROC-AUC:** Equilibrio general entre precisión y cobertura del modelo para ordenar y segmentar a la clientela.  

---

## Resultados del Modelado 

### Fase 1: Comparativa de Modelos Base (*Baselines*)
Se evaluaron 7 algoritmos de diversa naturaleza computacional (lineales, basados en distancias y ensamblados de última generación) bajo las mismas condiciones. A continuación, se muestran los resultados ordenados por la métrica principal (*F1-Score*):

| Modelo | ROC-AUC | F1-Score (Clase 1) | Recall (Clase 1) | Precisión (Clase 1) | Decisión / Observaciones |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **CatBoost** | **0.8005** | **0.4642** | 0.6248 | 0.3693 |  **Seleccionado:** Mayor equilibrio y tratamiento nativo de categóricas. |
| **SVM** | 0.7965 | 0.4626 | 0.6229 | 0.3680 |  Descartado por su alto coste computacional y lentitud en producción. |
| **LightGBM** | 0.8012 | 0.4587 | **0.6437** | 0.3564 |  Alternativa si la estrategia comercial fuera 100% agresiva (máx. Recall). |
| **XGBoost** | 0.7749 | 0.4284 | 0.5784 | 0.3402 | Rendimiento robusto, pero superado por CatBoost. |
| **Random Forest** | 0.7878 | 0.4178 | 0.3809 | 0.4627 | Penaliza en exceso el Recall en beneficio de la precisión. |
| **Regresión Logística** | 0.7740 | 0.3827 | 0.6408 | 0.2728 | Buen Recall gracias al enriquecimiento de clústeres, baja precisión. |
| **KNN** | 0.6932 | 0.3046 | 0.2108 | **0.5493** | No apto para este nivel de desbalance al basarse en distancias simples. |

### Fase 2: Optimización de Hiperparámetros (*GridSearchCV*)
Tras seleccionar **CatBoost** por superar el umbral de 0.80 en ROC-AUC y liderar el *F1-Score*, se sometió a un ajuste fino con validación cruzada (`cv=3`) sobre los hiperparámetros de profundidad (`depth`), regularización L2 (`l2_leaf_reg`) y tasa de aprendizaje (`learning_rate`). 

La evolución del modelo sobre la muestra de prueba final es la siguiente:

| Métrica Clave | CatBoost Base | CatBoost Optimizado | Impacto Operativo |
| :--- | :---: | :---: | :--- |
| **ROC-AUC Score** | 0.8005 | **0.8038** | Mayor capacidad de discriminación en la ordenación del *score* del cliente. |
| **F1-Score (Clase 1)** | 0.4642 | **0.4744** | Incremento general del +2.2% en la nota de equilibrio del clasificador. |
| **Precisión (Clase 1)**| 0.3693 | **0.3856** | **Mejora del +4.4%.** Mayor asertividad comercial por cada llamada emitida. |
| **Recall (Clase 1)** | 0.6248 | **0.6163** | Se consolida la captura de un **61.63% del mercado real de contrataciones**. |

---

## Conclusiones y Retorno de Negocio 

El despliegue analítico realizado sobre el modelo **CatBoost Optimizado** arroja aprendizajes de negocio decisivos para el área de marketing y operaciones de la entidad bancaria:

###  1. Triplicación de la Eficiencia del Call Center
Históricamente, al realizar una campaña de telemarketing a ciegas o masiva, el banco lograba una tasa de éxito natural del **11.70%** (1 de cada 10 llamadas convertía). 
* Gracias a la priorización del modelo optimizado, la **Precisión se eleva al 38.56%** (casi 4 de cada 10 contactos recomendados terminan en firma).
* Esto se traduce directamente en un incremento operativo donde el equipo comercial **multiplica por tres su rentabilidad por hora trabajada**.

###  2. Ahorro Directo en Costes de Fricción (Matriz de Confusión)
Al evaluar el rendimiento sobre los 9,043 clientes del conjunto de prueba independiente, el clasificador demostró una altísima capacidad de filtrado financiero:
* **6,946 llamadas innecesarias evitada** (*Verdaderos Negativos*): El modelo identificó con éxito a casi 7,000 personas que iban a rechazar la oferta, ahorrando semanas de trabajo comercial en vano.
* **Captura de Oportunidad** (*Verdaderos Positivos*): El modelo permitió cerrar **652 ventas directas** realizando únicamente 1,691 llamadas totales.

###  3. Apertura de la "Caja Negra" y Alineación con el EDA
El análisis de importancia relativa de las variables (*Feature Importance*) validó la coherencia matemática y comercial del algoritmo:
* **El perfil y momento financiero (`balance`, `age`):** Se confirman como los factores demográficos y económicos más determinantes para predecir la propensión, respaldando los hallazgos del análisis exploratorio inicial.
* **La estacionalidad e histórico de contactos (`month`, `poutcome`):** El modelo no solo dicta *a quién* llamar, sino que confirma empíricamente que el mes de lanzamiento de la campaña es crucial para el cierre de operaciones.

###  4. Viabilidad y Puesta en Producción
El modelo final cumple con los más altos estándares de ingeniería de datos: sin sesgos de fuga temporal, preparado para absorber clases desbalanceadas e interpretado sin pérdida de generalización. El clasificador ha sido serializado en formato `.pkl` (`catboost_optimizado.pkl`), quedando listo para su integración en los flujos computacionales de la arquitectura de negocio actual del banco.

---

## Evaluación del Umbral  

### Cómo adaptar el comportamiento del modelo CatBoost a los objetivos del negocio bancario

En modelos de clasificación binaria, el umbral de decisión determina a partir de qué probabilidad consideramos que un cliente **sí contrataría** el producto. Aunque el valor por defecto suele ser **0.50**, en campañas bancarias este umbral tiene un impacto directo en el número de llamadas, los clientes captados y los recursos necesarios. Por eso, evaluar distintos umbrales permite ajustar el modelo a la estrategia comercial del banco.

### Efecto de modificar el umbral

- **Umbral bajo (≈ 0.30)**  
  - El modelo se vuelve más agresivo.  
  - Se identifican más clientes que sí contratarían (menos falsos negativos).  
  - Aumentan las llamadas y el coste operativo.  
  - Ideal para **maximizar captación**.

- **Umbral estándar o alto (≈ 0.50 – 0.70)**  
  - El modelo se vuelve más conservador.  
  - Se reducen las llamadas innecesarias (menos falsos positivos).  
  - Se pierden más clientes que sí contratarían.  
  - Adecuado para **optimizar eficiencia** y reducir carga del equipo.

- **Umbral adaptado al presupuesto y capacidad del equipo**  
  - Permite equilibrar **tiempo, coste y clientes captados**.  
  - Útil cuando el número de agentes es limitado o la campaña debe ajustarse a un presupuesto concreto.

### Conclusión general

A medida que subimos el umbral, el modelo exige mayor probabilidad para clasificar a un cliente como “sí”. Esto reduce llamadas innecesarias, pero también aumenta los clientes perdidos.  

Por tanto:
- Umbral alto = menos esfuerzo comercial, más eficiencia, pero más oportunidades perdidas.  
- Umbral bajo = más esfuerzo comercial, más captación, pero mayor coste operativo.

El ajuste del umbral convierte el modelo CatBoost en una herramienta flexible que puede adaptarse a cualquier estrategia del banco: vender más, ahorrar recursos o encontrar un punto intermedio.



## Autores
- Claudia Franzoni Hernández
    — GitHub: https://github.com/claudiafranzoni
    — LinkedIn: https://www.linkedin.com/in/claudia-franzoni-800529196/

- Marta Harana Herrera 
    — GitHub: https://github.com/MHHsim  
    — LinkedIn: https://www.linkedin.com/in/marta-harana-herrera-004a84117/

- Maria Rodríguez Esteras 
    — GitHub: https://github.com/Mariasares
    — LinkedIn: https://www.linkedin.com/in/mar%C3%ADa-rodes-8259403a1/  


