import pandas as pd
import re

def camel_to_snake(name):
    """
    Converte uma string de camelCase para snake_case.
    Lida corretamente com duas letras maiúsculas seguidas, como em 'customerID', que se torna 'customer_id'.
    """
    # Primeiro, insere um underscore antes de qualquer sequência de letras maiúsculas
    name = re.sub(r'([a-z])([A-Z])', r'\1_\2', name)
    
    # Em seguida, trata casos como 'ID' para garantir que seja 'id' e converte para minúsculas
    return re.sub(r'([A-Z]+)', lambda x: x.group(0).lower(), name)

def clean_column_data(value):
    """
    Limpa os valores das colunas categóricas.
    - Remove parênteses e hífens.
    - Substitui espaços por sublinhados e converte para minúsculas.
    - Remove sublinhados extras no final.
    """
    if isinstance(value, str):
        # Remove parênteses, hífens, substitui espaços por sublinhados e remove sublinhado extra no final
        value = re.sub(r'[()\-\s]+', '_', value.strip()).lower().rstrip('_')
    return value

def format_column_names_and_data(*dfs):
    """
    Formata os nomes das colunas e os dados das colunas categóricas de vários DataFrames.
    - Nomes das colunas são convertidos de camelCase e nomes com espaços para snake_case e letras minúsculas.
    - Valores das colunas categóricas (object e category) são convertidos para letras minúsculas,
      têm espaços removidos, e substituem caracteres especiais.

    Parâmetros:
    *dfs (list of pd.DataFrame): Um ou mais DataFrames a serem formatados.

    Retorna:
    list of pd.DataFrame: Lista de DataFrames com nomes de colunas e valores de colunas categóricas formatados.
    """
    
    formatted_dfs = []
    
    # Iterar sobre cada DataFrame
    for df in dfs:
        # Formatar os nomes das colunas (de camelCase e nomes com espaços para snake_case e remover espaços)
        df.columns = [camel_to_snake(re.sub(r'\s+', '_', col.strip())) for col in df.columns]
        
        # Verificar se as colunas são do tipo object ou category e formatar os valores
        for col in df.select_dtypes(include=['object', 'category']).columns:
            df[col] = df[col].apply(clean_column_data)
        
        formatted_dfs.append(df)
    
    return formatted_dfs

# Exemplo de uso:
# df_formatted_1, df_formatted_2 = format_column_names_and_data(df1, df2)