import seaborn as sns
import matplotlib.pyplot as plt
import math

def barplot_categorical_columns(df, exclude_columns=None):
    """
    Função para criar gráficos de barra das colunas categóricas de um DataFrame.
    
    Parâmetros:
    df (pd.DataFrame): O DataFrame contendo os dados.
    exclude_columns (list): Lista opcional de colunas a serem excluídas dos gráficos.
    """
    
    # Selecionar as colunas categóricas do DataFrame
    categorical_columns = df.select_dtypes(include=['object']).columns

    # Remover colunas da lista de exclusão, se fornecida
    if exclude_columns:
        categorical_columns = [col for col in categorical_columns if col not in exclude_columns]

    # Verificar se ainda há colunas categóricas para plotar
    if len(categorical_columns) == 0:
        print("No categorical columns available to plot after applying the exclusion.")
        return

    # Definir o número de linhas e colunas dos subplots com base no número de colunas categóricas
    n_cols = 3  # Definir um número fixo de colunas
    n_rows = math.ceil(len(categorical_columns) / n_cols)  # Calcular dinamicamente o número de linhas necessárias

    # Criar a figura e os eixos com o número correto de linhas e colunas
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 6 * n_rows))

    # Transformar o array de 'axes' em uma lista para facilitar o loop
    axes = axes.flatten()

    # Loop para criar cada gráfico de barra
    for i, column in enumerate(categorical_columns):
        sns.countplot(x=column, data=df, ax=axes[i])
        axes[i].set_ylabel("Count")  # Definir rótulo 'Count' no eixo y
        axes[i].set_xlabel(column)  # Definir nome da coluna no eixo x

    # Remover eixos vazios, se existirem (caso o número de colunas seja menor que o número de subplots)
    for i in range(len(categorical_columns), len(axes)):
        fig.delaxes(axes[i])

    # Ajustar o layout para garantir que os gráficos não se sobreponham
    plt.tight_layout()

    # Mostrar o gráfico
    plt.show()