import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score

# Função principal para avaliar múltiplos modelos de classificação
def evaluate_multiple_classification_models(X_train, y_train, X_val, y_val, random_state=54321):

    """
    Função para treinar e avaliar múltiplos modelos de classificação e exibir métricas de desempenho.
    
    Parâmetros:
    X_train, y_train: Conjunto de treinamento.
    X_val, y_val: Conjunto de validação.
    random_state: Valor para garantir reprodutibilidade nos modelos que aceitam esse parâmetro.
    """
    
    models = {
        'Dummy': DummyClassifier(strategy='most_frequent', random_state=random_state),
        'Logistic Regression': LogisticRegression(random_state=random_state, max_iter=1000),
        'Random Forest': RandomForestClassifier(random_state=random_state),
        'K-Nearest Neighbors': KNeighborsClassifier(),
        'Support Vector Machine': SVC(probability=True, random_state=random_state),
        'LightGBM': LGBMClassifier(random_state=random_state, verbose=-1),
        'XGBoost': XGBClassifier(random_state=random_state, verbosity=0),
        'CatBoost': CatBoostClassifier(random_state=random_state, verbose=0)
    }

    # Loop pelos modelos
    for model_name, model in models.items():
        print(f"\n### {model_name} ###")
        try:
            # Treinar o modelo com os dados de treino
            model.fit(X_train, y_train)

            # Avaliar o modelo e exibir as métricas
            evaluate_classification_model(model, X_train, y_train, X_val, y_val)

        except Exception as e:
            print(f"An error occurred with {model_name}: {e}")
            continue

# Função para avaliar um modelo de classificação
def evaluate_classification_model(model, train_features, train_target, test_features, test_target):
    
    eval_stats = {}
    
    for type, features, target in (('train', train_features, train_target), ('val', test_features, test_target)):
        
        eval_stats[type] = {}
    
        # Fazer predições e estimativas de probabilidade
        pred_target = model.predict(features)
        pred_proba = model.predict_proba(features)[:, 1]
        
        # F1 Score
        f1_score_value = f1_score(target, pred_target)
        eval_stats[type]['F1 Score'] = f1_score_value
        
        # Curva ROC e ROC AUC
        roc_auc = roc_auc_score(target, pred_proba)    
        eval_stats[type]['ROC AUC'] = roc_auc

        # Curva Precision-Recall e APS
        aps = average_precision_score(target, pred_proba)
        eval_stats[type]['APS'] = aps

        # Accuracy
        accuracy = accuracy_score(target, pred_target)
        eval_stats[type]['Accuracy'] = accuracy

    # Criar um DataFrame para exibir as estatísticas de avaliação
    df_eval_stats = pd.DataFrame(eval_stats)
    df_eval_stats = df_eval_stats.round(2)
    df_eval_stats = df_eval_stats.reindex(index=('Accuracy', 'F1 Score', 'APS', 'ROC AUC'))
    
    print(df_eval_stats)
    return

# Exemplo de uso (com os dados de treino e validação já divididos):
# evaluate_multiple_classification_models(X_train, y_train, X_val, y_val, random_state=54321)
