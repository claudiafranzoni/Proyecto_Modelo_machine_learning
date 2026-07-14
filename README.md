![Portada del proyecto](data/imagen_readmeML.png)

# Análisis Exploratorio y Segmentación de Clientes · Persona A

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
