import pandas as pd  # type: ignore
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import skew, kurtosis
from sklearn.model_selection import train_test_split  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore
from sklearn.datasets import load_breast_cancer, make_classification  # type: ignore
from sklearn.neighbors import KNeighborsClassifier  # type: ignore
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, r2_score  # type: ignore
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression  # type: ignore

def load_data():
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name='target')
    return X, y

def exploratory_analysis(X):
    print(X.info())
    print(X.describe())

def split_and_scale(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test

def train_knn(X_train, y_train, n_neighbors=5):
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    return knn

def evaluate_knn(knn, X_test, y_test):
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Acurácia do modelo KNN: {accuracy:.2f}')
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Benigno', 'Maligno'])
    disp.plot(cmap=plt.cm.Purples)
    plt.savefig("Acurácia_do_modelo_KNN.png")
    return accuracy

def generate_synthetic_data(n_samples, n_features, n_informative, n_redundant, random_state=42):
    X_synthetic, y_synthetic = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        random_state=random_state
    )
    noise = np.random.normal(0, 0.1, X_synthetic.shape)
    X_synthetic_noisy = X_synthetic + noise
    return X_synthetic_noisy, y_synthetic

def evaluate_with_synthetic(knn, X_train, y_train, X_synthetic, y_synthetic, X_test, y_test):
    X_combined = np.vstack((X_train, X_synthetic))
    y_combined = np.hstack((y_train, y_synthetic))
    knn.fit(X_combined, y_combined)
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Acurácia do modelo KNN com dados sintéticos: {accuracy:.2f}')
    return accuracy

def plot_k_accuracies(X_train, y_train, X_test, y_test, k_range=range(1, 21)):
    accuracies = []
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        accuracies.append(accuracy_score(y_test, y_pred))
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, accuracies, marker='o')
    plt.title('Acurácia do Modelo KNN para Diferentes Valores de K')
    plt.xlabel('Número de Vizinhos (K)')
    plt.ylabel('Acurácia')
    plt.grid()
    plt.savefig('grafico_knn.png') 

def feature_importance_analysis(X, y):
    importances = RandomForestClassifier(random_state=42).fit(X, y).feature_importances_
    feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
    feature_importance = feature_importance.sort_values(by='Importance', ascending=False)
    print("Importância dos Atributos:")
    print(feature_importance)
    return feature_importance

def plot_real_vs_predicted(y_test_reg, y_pred_reg):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test_reg, y_pred_reg, alpha=0.7, color='blue')
    plt.plot([y_test_reg.min(), y_test_reg.max()], [y_test_reg.min(), y_test_reg.max()], color='red', linestyle='--')
    plt.title('Valores Reais vs Valores Estimados')
    plt.xlabel('Valores Reais')
    plt.ylabel('Valores Estimados')
    plt.grid()
    plt.savefig('real_vs_predicted.png')

def residual_statistics(residuos):
    print("Estatísticas dos Resíduos:")
    print(f"Média: {np.mean(residuos):.2f}")
    print(f"Variância: {np.var(residuos):.2f}")
    print(f"Assimetria: {skew(residuos):.2f}")
    print(f"Curtose: {kurtosis(residuos):.2f}")

def linear_regression_analysis(X, target_col='mean area'):
    X_reg = X.drop(columns=[target_col])
    y_reg = X[target_col]
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
    reg = LinearRegression()
    reg.fit(X_train_reg, y_train_reg)
    y_pred_reg = reg.predict(X_test_reg)
    r2 = r2_score(y_test_reg, y_pred_reg)
    print(f'Coeficiente de Determinação (R²): {r2:.2f}')
    residuos = y_test_reg - y_pred_reg
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test_reg, residuos, alpha=0.7)
    plt.axhline(0, color='green', linestyle='--')
    plt.title('Resíduos vs Valores Reais')
    plt.xlabel('Valores Reais')
    plt.ylabel('Resíduos')
    plt.grid()
    plt.savefig('Coeficiente_de_determinacao.png')
    plt.figure(figsize=(10, 6))
    plt.hist(residuos, bins=20, alpha=0.7, color='purple')
    plt.title('Distribuição dos Resíduos')
    plt.xlabel('Resíduos')
    plt.ylabel('Frequência')
    plt.grid()
    plt.savefig('histograma_residuos.png')
    plot_real_vs_predicted(y_test_reg, y_pred_reg)
    residual_statistics(residuos)

def main():
    X, y = load_data()
    exploratory_analysis(X)
    feature_importance_analysis(X, y)
    X_train, X_test, y_train, y_test = split_and_scale(X, y)
    knn = train_knn(X_train, y_train)
    evaluate_knn(knn, X_test, y_test)
    X_synthetic, y_synthetic = generate_synthetic_data(
        n_samples=200,
        n_features=X.shape[1],
        n_informative=10,
        n_redundant=5
    )
    evaluate_with_synthetic(knn, X_train, y_train, X_synthetic, y_synthetic, X_test, y_test)
    plot_k_accuracies(X_train, y_train, X_test, y_test)
    linear_regression_analysis(X, target_col='mean area')

if __name__ == "__main__":
    main()