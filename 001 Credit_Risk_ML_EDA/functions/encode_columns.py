import pandas as pd
from sklearn.preprocessing import LabelEncoder

def encode_columns(df, ohe_columns=None, label_columns=None):
    """
    Função para aplicar One-Hot Encoding (OHE) e Label Encoding em um DataFrame.
    
    Parâmetros:
    df (pd.DataFrame): DataFrame no qual será aplicado o encoding.
    ohe_columns (list, opcional): Lista de colunas a serem codificadas com One-Hot Encoding.
    label_columns (list, opcional): Lista de colunas a serem codificadas com Label Encoding.

    Retorna:
    pd.DataFrame: DataFrame com as colunas listadas codificadas.
    """
    
    # Criar uma cópia do DataFrame para evitar modificar o original
    df_encoded = df.copy()

    # Aplicar One-Hot Encoding, se as colunas forem fornecidas
    if ohe_columns is not None:
        df_encoded = pd.get_dummies(df_encoded, columns=ohe_columns, drop_first=False)

    # Aplicar Label Encoding, se as colunas forem fornecidas
    if label_columns is not None:
        label_encoder = LabelEncoder()
        for col in label_columns:
            df_encoded[col] = label_encoder.fit_transform(df_encoded[col])

    return df_encoded


 
