import pandas as pd

def check_duplicates_and_missing_values(df, target_column=None):
    """
    Função que retorna informações sobre balanceamento da coluna target, duplicatas e valores ausentes em um DataFrame.

    Parâmetros:
    df (pd.DataFrame): DataFrame a ser analisado.
    target_column (str): Coluna para calcular o número de duplicatas por valor da coluna 'target' considerando apenas as duplicatas subsequentes.

    Retorna:
    None (imprime os resultados na tela)
    """
    total_rows = df.shape[0]

    # 0. Verificação do balanceamento da coluna target (se fornecida)
    if target_column and target_column in df.columns:
        target_balance_counts = df[target_column].value_counts()
        target_balance_percent = df[target_column].value_counts(normalize=True) * 100
        target_balance_df = pd.DataFrame({
            'Count': target_balance_counts,
            'Percentage': target_balance_percent.apply(lambda x: f"{x:.2f}%")
        })
        print(f"Class balance in the target column '{target_column}':")
        print(target_balance_df)
        print("\n")

    # 1. Número de linhas duplicadas (apenas subsequentes) e o percentual em relação ao total
    num_duplicates = df.duplicated().sum()  # Mantém a lógica original
    if num_duplicates > 0:
        percent_duplicates = (num_duplicates / total_rows) * 100
        print(f"Total number of duplicate rows: {num_duplicates}")
        print(f"Percentage of duplicate rows: {percent_duplicates:.2f}%\n")
    else:
        print("No duplicate rows found.\n")
    
    # 2. Número de duplicatas por valor da coluna 'target', se fornecida (mantendo a lógica das duplicatas subsequentes)
    if target_column:
        if target_column in df.columns:
            duplicated_rows = df[df.duplicated()]
            if not duplicated_rows.empty:
                duplicates_by_target = duplicated_rows[target_column].value_counts()
                print(f"Number of duplicate rows by target column '{target_column}':")
                print(duplicates_by_target)
            else:
                print(f"No duplicate rows found by target column '{target_column}'.\n")
        else:
            print(f"The column '{target_column}' does not exist in the DataFrame.\n")
    else:
        print("No target column was provided.\n")
    
    # 3. Número de linhas com valores ausentes e o percentual por coluna
    missing_values = df.isnull().sum()
    if missing_values.any():
        percent_missing_values = (missing_values / total_rows) * 100
        missing_data_info = pd.DataFrame({
            'Missing Values': missing_values,
            'Percentage': percent_missing_values.apply(lambda x: f"{x:.2f}%")
        })

        print("\nMissing values per column and percentage:")
        print(missing_data_info[missing_data_info['Missing Values'] > 0])
    else:
        print("No missing values found in the dataset.\n")
    
    # 3.1. Percentuais por valor da coluna 'target' (se fornecida)
    if target_column and target_column in df.columns:
        print(f"\nMissing values percentages per '{target_column}':")
        for value in df[target_column].unique():
            if pd.notna(value):
                subset_df = df[df[target_column] == value]
                missing_per_target = subset_df.isnull().sum()
                if missing_per_target.any():
                    total_missing_in_column = df.isnull().sum()  # Total de missing values no DataFrame original
                    percent_per_target = (missing_per_target / total_missing_in_column) * 100
                    print(f"\nFor {target_column} = {value}:")
                    print(pd.DataFrame({
                        'Missing Values': missing_per_target,
                        'Percentage': percent_per_target.apply(lambda x: f"{x:.2f}%")
                    })[missing_per_target > 0])
                else:
                    print(f"No missing values for '{target_column}' = {value}.\n")
    
    # 4. Colunas com 100% de valores ausentes
    full_missing_columns = missing_values[missing_values == total_rows].index.tolist()
    if full_missing_columns:
        print(f"\nColumns with 100% missing values: {full_missing_columns}")
    else:
        print("No columns with 100% missing values.\n")
    
    # 5. Quantidade total de linhas com valores ausentes no DataFrame e o percentual geral
    total_missing_rows = df.isnull().any(axis=1).sum()
    if total_missing_rows > 0:
        percent_total_missing_rows = (total_missing_rows / total_rows) * 100
        print(f"\nTotal rows with missing values: {total_missing_rows}")
        print(f"Percentage of rows with missing values: {percent_total_missing_rows:.2f}%\n")
    else:
        print("No rows with missing values found.\n")
    
    # 6. Linhas com mais de 50% de valores ausentes
    rows_with_many_missing = (df.isnull().sum(axis=1) / df.shape[1] > 0.5).sum()
    if rows_with_many_missing > 0:
        percent_rows_with_many_missing = (rows_with_many_missing / total_rows) * 100
        print(f"Rows with more than 50% missing values: {rows_with_many_missing} ({percent_rows_with_many_missing:.2f}%)")
    else:
        print("No rows with more than 50% missing values.\n")
    
    # 7. Colunas com valores constantes (sem variabilidade)
    constant_columns = [col for col in df.columns if df[col].nunique() == 1]
    if constant_columns:
        print(f"\nColumns with constant values (only one unique value): {constant_columns}")
    else:
        print("No columns with constant values.\n")
