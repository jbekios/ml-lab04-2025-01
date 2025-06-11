import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import GridSearchCV, LeaveOneOut
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ----------------------------------------
# Cargar el dataset manualmente (5 ejemplos)
# ----------------------------------------
def cargar_datos(filename):

    # Cargar los datos
    df_15 = pd.read_spss(filename)

    # Quitar la columna ID
    df_15 = df_15.drop(columns='ID')

    # Mostrar el dataframe
    columns_gds = ['GDS', 'GDS_R1', 'GDS_R2', 'GDS_R3', 'GDS_R4', 'GDS_R5']
    X = df_15.drop(columns=columns_gds)
    y = df_15['GDS']

    return X, y.astype(int)

# ----------------------------------------
# Evaluar modelo con GridSearch y Leave-One-Out
# ----------------------------------------
def evaluar_modelo_loocv(X, y, base_estimator, param_grid, nombre_modelo, bagging_params):
    # Crear pipeline con estandarización + Bagging
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("bagging", BaggingClassifier(
            estimator=base_estimator,
            n_estimators=bagging_params.get("n_estimators", 10),          # Número de modelos base
            max_samples=bagging_params.get("max_samples", 1.0),           # Proporción de muestras
            max_features=bagging_params.get("max_features", 1.0),         # Proporción de features
            bootstrap=bagging_params.get("bootstrap", True),              # Muestreo con reemplazo en muestras
            bootstrap_features=bagging_params.get("bootstrap_features", False),  # Muestreo con reemplazo en features
            random_state=bagging_params.get("random_state", 42),          # Semilla para reproducibilidad
            n_jobs=bagging_params.get("n_jobs", -1)                       # Paralelismo: usar todos los núcleos
        ))
    ])

    # GridSearch con Leave-One-Out
    grid = GridSearchCV(pipeline, param_grid=param_grid, cv=LeaveOneOut())
    grid.fit(X, y)

    y_pred = grid.predict(X)
    acc = accuracy_score(y, y_pred)

    print(f"Modelo {nombre_modelo} - Accuracy (LOOCV): {acc:.4f}")
    print("Mejores hiperparámetros:", grid.best_params_)
    return acc, grid.best_params_

# ----------------------------------------
# Función principal
# ----------------------------------------
def main(n_estimators=10, usar_bootstrap_features=False):
    
    X, y = cargar_datos('datasets/15 atributos R0-R5.sav')

    # Rango de hiperparámetros
    valores_C = list(range(1, 6))                     # [1, 2, 3, 4, 5]
    valores_gamma = [10**(-i) for i in range(1, 4)]   # [0.1, 0.01, 0.001]

    # Configuración del Bagging (modificable)
    bagging_config = {
        "n_estimators": n_estimators,
        "max_samples": 1.0,
        "max_features": 1.0,
        "bootstrap": True,
        "bootstrap_features": usar_bootstrap_features,
        "random_state": 42,
        "n_jobs": -1
    }

    # SVM Lineal
    svm_linear = SVC(kernel='linear')
    param_grid_linear = {
        "bagging__estimator__C": valores_C
    }

    print("🔍 Evaluando SVM Lineal")
    evaluar_modelo_loocv(X, y, svm_linear, param_grid_linear, "SVM Lineal", bagging_config)

    # SVM RBF
    svm_rbf = SVC(kernel='rbf')
    param_grid_rbf = {
        "bagging__estimator__C": valores_C,
        "bagging__estimator__gamma": valores_gamma
    }

    print("\n🔍 Evaluando SVM RBF")
    evaluar_modelo_loocv(X, y, svm_rbf, param_grid_rbf, "SVM RBF", bagging_config)

# ----------------------------------------
# Ejecutar con parámetros
# ----------------------------------------
if __name__ == "__main__":
    main(n_estimators=15, usar_bootstrap_features=True)
