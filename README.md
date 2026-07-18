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
  - **XGBoost**
  - **LightGBM**
  - **CatBoost**
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

## Instrucciones de reprodución

Para ejecutar el proyecto y reproducir los resultados del modelo, sigue los pasos indicados a continuación:

git clone https://github.com/claudiafranzoni/Proyecto_Modelo_machine_learning
cd Proyecto_Modelo_machine_learning

python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt

jupyter notebook notebooks/main.ipynb

# Cargar modelo final
import joblib
modelo = joblib.load("models/catboost_optimizado.pkl")

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


