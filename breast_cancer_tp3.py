import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

def load_and_preprocess_data():
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name='target')
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y

def evaluate_knn_baseline(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    print("Desempenho do modelo baseline (KNN):")
    print(classification_report(y_test, y_pred))
    return knn

def evaluate_logistic_regression(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    log_reg = LogisticRegression(max_iter=1000, random_state=42)
    log_reg.fit(X_train, y_train)
    y_pred = log_reg.predict(X_test)
    print("Desempenho do modelo Regressão Logística:")
    print(classification_report(y_test, y_pred))
    return log_reg

def cross_validate_and_tune(model, X, y, param_grid):
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='f1', n_jobs=-1)
    grid_search.fit(X, y)
    print("Melhores hiperparâmetros:", grid_search.best_params_)
    print("Melhor F1-score nos folds:", grid_search.best_score_)
    return grid_search.best_estimator_

def main():
    X, y = load_and_preprocess_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("--- Otimizando KNN ---")
    knn_param_grid = {'n_neighbors': range(1, 21)}
    best_knn = cross_validate_and_tune(KNeighborsClassifier(), X_train, y_train, knn_param_grid)

    print("\n--- Otimizando Regressão Logística ---")
    log_reg_param_grid = {'C': [0.01, 0.1, 1, 10, 100]}
    best_log_reg = cross_validate_and_tune(LogisticRegression(max_iter=1000, random_state=42), X_train, y_train, log_reg_param_grid)

    print("\n--- Comparação final no conjunto de TESTE ---")
    print("Desempenho final do modelo KNN Otimizado:")
    y_pred_knn = best_knn.predict(X_test)
    print(classification_report(y_test, y_pred_knn))

    print("Desempenho final do modelo Regressão Logística Otimizada:")
    y_pred_log_reg = best_log_reg.predict(X_test)
    print(classification_report(y_test, y_pred_log_reg))

if __name__ == "__main__":
    main()
