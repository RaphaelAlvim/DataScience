import pandas as pd

def analyze_dataframe(df, exclude_columns=None):
    """
    Função que realiza uma análise do DataFrame, retornando value_counts para colunas categóricas,
    describe para colunas numéricas, range para colunas de data, e contagem de valores únicos.
    
    Parâmetros:
    df (pd.DataFrame): DataFrame a ser analisado.
    exclude_columns (list): Lista de colunas a serem excluídas da análise.
    
    Retorna:
    dict: Dicionário com os resultados da análise.
    """
    if exclude_columns is None:
        exclude_columns = []
    
    # Remover colunas especificadas
    df_filtered = df.drop(columns=exclude_columns, errors='ignore')
    
    analysis = {}
    
    # 1. Value counts para colunas categóricas
    print("\n--- Categorical Columns Value Counts (with Percentages) ---")
    categorical_columns = df_filtered.select_dtypes(include=['object', 'category']).columns
    for col in categorical_columns:
        value_counts = df_filtered[col].value_counts()
        value_percent = df_filtered[col].value_counts(normalize=True) * 100
        value_percent = value_percent.apply(lambda x: f"{x:.2f}%")  # Formatar porcentagem com duas casas decimais
        total_unique = df_filtered[col].nunique()
        value_df = pd.DataFrame({'Count': value_counts, 'Percentage': value_percent})
        print(f"\n{col} (Top 10 of {total_unique} total values):")
        print(value_df.head(10))  # Mostrar apenas os 10 valores principais
    
    # 2. Describe para colunas numéricas
    print("\n--- Numerical Columns Description ---")
    numeric_columns = df_filtered.select_dtypes(include=['number']).columns
    if not numeric_columns.empty:
        analysis['numeric_description'] = df_filtered[numeric_columns].describe()
        print(analysis['numeric_description'])
    
    # 3. Range para colunas de datas
    print("\n--- Date Columns Range ---")
    datetime_columns = df_filtered.select_dtypes(include=['datetime', 'object']).apply(pd.to_datetime, errors='coerce').dropna(axis=1).columns
    for col in datetime_columns:
        analysis[f"date_range_{col}"] = (df_filtered[col].min(), df_filtered[col].max())
        print(f"{col}: {analysis[f'date_range_{col}']}")







