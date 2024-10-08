import pandas as pd

def convert_to_category(df, exclude_columns=None):
    """
    Converte as colunas categóricas (do tipo 'object') para o tipo 'category' do pandas,
    exceto aquelas especificadas no parâmetro exclude_columns.

    Parâmetros:
    df (pd.DataFrame): DataFrame a ser formatado.
    exclude_columns (list): Lista de nomes de colunas que NÃO devem ser convertidas. Padrão é None.

    Retorna:
    pd.DataFrame: DataFrame com as colunas categóricas convertidas para 'category', exceto as excluídas.
    """
    # Se exclude_columns for None, definimos como uma lista vazia
    if exclude_columns is None:
        exclude_columns = []
    
    # Percorrer as colunas do tipo 'object', exceto aquelas na lista de exclusão
    for col in df.select_dtypes(include=['object']).columns:
        if col not in exclude_columns:
            df[col] = df[col].astype('category')
    
    return df