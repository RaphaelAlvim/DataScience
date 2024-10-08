# Função para gerar o relatório de erros

def error_report(X_test, y_test, y_pred_proba, threshold=0.5):
    """
    Esta função gera um relatório detalhado sobre os erros (falsos positivos e falsos negativos) e os acertos
    (verdadeiros positivos e verdadeiros negativos) de um modelo de classificação.
    
    Parâmetros:
    X_test: DataFrame contendo as features do conjunto de teste.
    y_test: Rótulos verdadeiros do conjunto de teste.
    y_pred_proba: Probabilidades preditas pelo modelo.
    threshold: Valor de corte para definir se uma probabilidade é classificada como 1 ou 0 (padrão 0.5).
    """
    
    # Transformar probabilidades em previsões binárias (0 ou 1)
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Identificar falsos positivos, falsos negativos, verdadeiros positivos e verdadeiros negativos
    false_positives = X_test[(y_test == 0) & (y_pred == 1)]
    false_negatives = X_test[(y_test == 1) & (y_pred == 0)]
    true_positives = X_test[(y_test == 1) & (y_pred == 1)]
    true_negatives = X_test[(y_test == 0) & (y_pred == 0)]

    # Função interna para gerar um resumo de estatísticas para cada tipo de erro/acerto
    def generate_summary(df, error_type):
        print(f"\n------ {error_type} ------\n")
        print(f"Total {error_type}: {len(df)}\n")
        
        # Exibir média das features para identificar padrões
        print("Feature Means:")
        print(df.mean())  # Média das features numéricas
        
        # Exibir moda das features categóricas
        print("\nFeature Modes:")
        print(df.mode().iloc[0])  # Moda das features categóricas
        
        # Exibir as features com maior variabilidade (desvio padrão)
        print("\nTop Features with High Variance (Standard Deviation):")
        print(df.std().sort_values(ascending=False).head())  # Variabilidade nas features
    
    # Gerar relatório para falsos positivos
    generate_summary(false_positives, "False Positives")
    
    # Gerar relatório para falsos negativos
    generate_summary(false_negatives, "False Negatives")
    
    # Gerar relatório para verdadeiros positivos
    generate_summary(true_positives, "True Positives")
    
    # Gerar relatório para verdadeiros negativos
    generate_summary(true_negatives, "True Negatives")
    
    # Conclusão do relatório
    print("\nReport Completed:")
    print(f"False Positives: {len(false_positives)} | False Negatives: {len(false_negatives)}")
    print(f"True Positives: {len(true_positives)} | True Negatives: {len(true_negatives)}")

# Exemplo de uso
# error_report(X_test, y_test, y_pred_proba)
