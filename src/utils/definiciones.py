import pandas as pd
import numpy as np

from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    f1_score,
    recall_score,
    precision_score,
    confusion_matrix,
    roc_curve
)

import matplotlib.pyplot as plt
import seaborn as sns

from typing import Optional


# ----------------------------- describe_df -----------------------------


def describe_df(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Genera un resumen estadístico descriptivo de un DataFrame.

    Argumentos:
        df (pd.DataFrame): DataFrame a analizar.

    Retorna:
        Optional[pd.DataFrame]: DataFrame con una fila por columna del input y las
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


# ----------------------------- tipifica_variables -----------------------------


def tipifica_variables(
    df: pd.DataFrame,
    umbral_categorica: int,
    umbral_continua: float
) -> Optional[pd.DataFrame]:
    """
    Clasifica las variables de un DataFrame según su cardinalidad y porcentaje de cardinalidad.

    Argumentos:
        df (pd.DataFrame): DataFrame a analizar.
        umbral_categorica (int): Umbral para categorizar las variables categóricas.
        umbral_continua (float): Umbral de porcentaje de cardinalidad para variables continuas.

    Retorna:
        Optional[pd.DataFrame]: DataFrame con dos columnas: 'nombre_variable', 'tipo_sugerido'.
        Retorna None si los parámetros no son válidos.
    """
    
    # Comprobación de DataFrame
    if not isinstance(df, pd.DataFrame):
        print("Error: el objeto proporcionado no es un DataFrame.")
        return None
    
    # Comprobación umbral_categorica, debe ser entero positivo
    if not isinstance(umbral_categorica, int) or umbral_categorica <= 0:
        print("Error: umbral_categorica debe ser un entero positivo.")
        return None

    # comprobación umbral_continua, debe ser float entre 0 y 100
    if not isinstance(umbral_continua, float) or not (0 <= umbral_continua <= 100):
        print("Error: umbral_continua debe ser un float entre 0 y 100.")
        return None
    
    # Cardinalidad y porcentaje_cardinalidad
    cardinalidad = df.nunique()
    porcentaje_cardinalidad = df.nunique() / len(df) * 100

    # Clasificación variables
    tipos = []

    for col in df.columns:
        card = cardinalidad[col]
        pct = porcentaje_cardinalidad[col]

        if card == 2:
            tipo = "Binaria"

        elif card < umbral_categorica:
            tipo = "Categórica"

        elif card >= umbral_categorica and pct >= umbral_continua:
            tipo = "Numérica Continua"

        else:
            tipo = "Numérica Discreta"

        tipos.append(tipo)
    
    # DF resultado
    resultado = pd.DataFrame({
        "nombre_variable": df.columns,
        "tipo_sugerido": tipos
    })

    return resultado

