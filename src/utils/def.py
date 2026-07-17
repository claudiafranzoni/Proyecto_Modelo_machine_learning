def describe_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Genera un resumen estadístico descriptivo de un DataFrame.

    Argumentos:
        df (pd.DataFrame): DataFrame a analizar.

    Retorna:
        pd.DataFrame: DataFrame con una fila por columna del input y las
        siguientes columnas: 'tipo', 'porcentaje_nulos', 'valores_unicos',
        'porcentaje_cardinalidad'.
        Retorna None si el input no es un DataFrame válido.
    """
    
    # Comprobación de que si es un DataFrame
    if not isinstance(df, pd.DataFrame):
        print("Error: el objeto proporcionado no es un DataFrame.")
        return None

    # Crear el DataFrame resultado
    resultado = pd.DataFrame(index=df.columns)

    # Tipo de dato
    resultado["tipo"] = df.dtypes.astype(str)

    # Porcentaje de nulos
    resultado["porcentaje_nulos"] = (df.isna().mean() * 100).round(2)

    # Valores únicos
    resultado["valores_unicos"] = df.nunique()

    # Porcentaje de cardinalidad
    resultado["porcentaje_cardinalidad"] = ((df.nunique() / len(df)) * 100).round(2)


    return resultado



def evaluar_modelo(modelo, X_test, y_real, nombre_modelo="Modelo"):
    """
    Función para evaluar las predicciones de un modelo de clasificación.
    Muestra las métricas principales, dibuja la Matriz de Confusión y la Curva ROC,
    y devuelve un diccionario con los resultados para la tabla final.
    """
    print(f"Evaluación del modelo: '{nombre_modelo}'")
    
    # El modelo calcula las predicciones y probabilidades de manera automática
    y_pred = modelo.predict(X_test)
    y_probabilidades = modelo.predict_proba(X_test)[:, 1]
    
    # 1) Reporte de clasificación (Métricas de texto)
    print("Reporte de Clasificación:")
    print(classification_report(y_real, y_pred))
    
    # 2) Área bajo la curva ROC (ROC-AUC) y métricas clave para el tema de estudio
    auc = roc_auc_score(y_real, y_probabilidades)
    f1 = f1_score(y_real, y_pred)
    recall = recall_score(y_real, y_pred)
    precision = precision_score(y_real, y_pred)
    
    print(f"ROC-AUC Score      : {auc:.4f}")
    print(f"F1-Score (Clase 1) : {f1:.4f}")
    print(f"Recall (Clase 1)   : {recall:.4f}")
    print(f"Precisión (Clase 1): {precision:.4f}\n")
    
    # 3) Preparar la base para dibujar 2 gráficas juntas
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    
    # GRÁFICA 1: Matriz de Confusión
    cm = confusion_matrix(y_real, y_pred)
    # Con seaborn (sns) la pinto de colores (mapa de calor)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax[0])
    ax[0].set_title('Matriz de Confusión')
    ax[0].set_ylabel('Realidad (Lo que de verdad pasó)')
    ax[0].set_xlabel('Predicción (Lo que dijo el modelo)')
    ax[0].xaxis.set_ticklabels(['No Contrata (0)', 'Sí Contrata (1)'])
    ax[0].yaxis.set_ticklabels(['No Contrata (0)', 'Sí Contrata (1)'])
    
    # GRÁFICA 2: Curva ROC
    fpr, tpr, _ = roc_curve(y_real, y_probabilidades)
    ax[1].plot(fpr, tpr, color='orange', label=f'Curva ROC (AUC = {auc:.2f})')
    ax[1].plot([0, 1], [0, 1], color='navy', linestyle='--') # La línea del azar
    ax[1].set_title('Curva ROC')
    ax[1].set_xlabel('Tasa de Falsos Positivos')
    ax[1].set_ylabel('Tasa de Verdaderos Positivos')
    ax[1].legend(loc="lower right")
    
    # Mostrar las gráficas
    plt.tight_layout()
    plt.show()
    
    # Devuelve las métricas para poder hacer la tabla comparativa al final ylas meto en un diccionario
    return {
        "Modelo": nombre_modelo,
        "ROC-AUC": auc,
        "F1-Score (Clase 1)": f1,
        "Recall (Clase 1)": recall,
        "Precisión (Clase 1)": precision
    }

print("Función 'evaluar_modelo' cargada y lista para usar.")