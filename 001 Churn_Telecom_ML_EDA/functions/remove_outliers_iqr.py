import pandas as pd

def remove_outliers_iqr(df, exclude_columns=None):
    """
    Função para remover outliers das colunas numéricas de um DataFrame com base no Intervalo Interquartil (IQR).
    
    Parâmetros:
    df (pd.DataFrame): O DataFrame contendo os dados.
    exclude_columns (list, opcional): Lista de colunas a serem excluídas da verificação de outliers.
    
    Retorna:
    pd.DataFrame: O DataFrame sem os outliers nas colunas numéricas.
    """
    
    # Selecionar as colunas numéricas do DataFrame
    numerical_columns = df.select_dtypes(include=['number']).columns

    # Remover as colunas da lista de exclusão, se fornecida
    if exclude_columns:
        numerical_columns = [col for col in numerical_columns if col not in exclude_columns]
    
    # Verificar se ainda há colunas numéricas para trabalhar
    if len(numerical_columns) == 0:
        print("No numerical columns available for outlier removal after applying the exclusion.")
        return df

    # Loop para remover outliers de cada coluna numérica
    for column in numerical_columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Filtrar o DataFrame para excluir as linhas com outliers
        df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

    return df