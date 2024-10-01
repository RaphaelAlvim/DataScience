import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import math

def boxplot_numeric_columns(df, exclude_columns=None):
    """
    Função para criar gráficos (boxplots) das colunas numéricas de um DataFrame.
    
    Parâmetros:
    df (pd.DataFrame): O DataFrame contendo os dados.
    exclude_columns (list): Lista opcional de colunas a serem excluídas dos gráficos.
    """
    
    # Função para formatar os valores do eixo x
    def number_format(x, pos):
        return '{:,.0f}'.format(x)  # Formata os números com vírgulas para separar os milhares

    # Selecionar as colunas numéricas do DataFrame
    numerical_columns = df.select_dtypes(include=['number']).columns

    # Remover colunas da lista de exclusão, se fornecida
    if exclude_columns:
        numerical_columns = [col for col in numerical_columns if col not in exclude_columns]

    # Verificar se ainda há colunas numéricas para plotar
    if len(numerical_columns) == 0:
        print("No numerical columns available to plot after applying the exclusion.")
        return

    # Definir o número de linhas e colunas dos subplots com base no número de colunas numéricas
    n_cols = 3  # Definir um número fixo de colunas
    n_rows = math.ceil(len(numerical_columns) / n_cols)  # Calcular dinamicamente o número de linhas necessárias

    # Criar a figura e os eixos com o número correto de linhas e colunas
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 6 * n_rows))

    # Transformar o array de 'axes' em uma lista para facilitar o loop
    axes = axes.flatten()

    # Loop para criar cada boxplot
    for i, column in enumerate(numerical_columns):
        sns.boxplot(x=column, data=df, ax=axes[i])
        axes[i].set_ylabel("")  # Remover o rótulo 'Frequency'
        axes[i].set_xlabel(column)
        axes[i].xaxis.set_major_formatter(FuncFormatter(number_format))  # Aplicar formatação ao eixo x

    # Remover eixos vazios, se existirem (caso o número de colunas seja menor que o número de subplots)
    for i in range(len(numerical_columns), len(axes)):
        fig.delaxes(axes[i])

    # Ajustar o layout para garantir que os gráficos não se sobreponham
    plt.tight_layout()

    # Mostrar o gráfico
    plt.show()