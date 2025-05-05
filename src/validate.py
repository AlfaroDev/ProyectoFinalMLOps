import joblib
import pandas as pd
from sklearn.metrics import mean_squared_error
import numpy as np
import sys
import os

# Parámetro de umbral
THRESHOLD = 10.0  # Ajusta este umbral según el MSE esperado

# --- Cargar el MISMO dataset que en train.py ---
print("--- Debug: Cargando dataset de TRM ---")
trm_df_train_scaled = np.loadtxt('trm_df_train_scaled.csv', delimiter=',').reshape(-1, 1)
trm_df_test_scaled = np.loadtxt('trm_df_test_scaled.csv', delimiter=',').reshape(-1, 1)

time_step = 10
X_test = []
for i in range(time_step,len(trm_df_test_scaled)):
    X_test.append(trm_df_test_scaled[i-time_step:i,0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
print(f"--- Debug: Dimensiones de X_test: {X_test.shape} ---") 

# --- Cargar modelo previamente entrenado ---
model_filename = "mlruns/5e59a801d8c641bbbeaf664295e5e18d/artifacts/model/model.pkl"
model_path = os.path.abspath(os.path.join(os.getcwd(), model_filename))
print(f"--- Debug: Intentando cargar modelo desde: {model_path} ---")

try:
    model = joblib.load(model_path)
except FileNotFoundError:
    print(f"--- ERROR: No se encontró el archivo del modelo en '{model_path}'. Asegúrate de que el paso 'make train' lo haya guardado correctamente en la raíz del proyecto. ---")
    # Listar archivos en el directorio actual para depuración
    print(f"--- Debug: Archivos en {os.getcwd()}: ---")
    try:
        print(os.listdir(os.getcwd()))
    except Exception as list_err:
        print(f"(No se pudo listar el directorio: {list_err})")
    print("---")
    sys.exit(1)  # Salir con error

# --- Predicción y Validación ---
print("--- Debug: Realizando predicciones ---")
try:
    min_max_scaler = joblib.load('scaler.pkl')
    y_pred = model.predict(X_test)
    y_pred = min_max_scaler.inverse_transform(y_pred)
    real = min_max_scaler.inverse_transform(trm_df_test_scaled[time_step:len(trm_df_test_scaled),0].reshape(-1,1))
except ValueError as pred_err:
    print(f"--- ERROR durante la predicción: {pred_err} ---")
    # Imprimir información de características si el error persiste
    print(f"Modelo esperaba {model.n_features_in_} features.")
    print(f"X_test tiene {X_test.shape[1]} features.")
    sys.exit(1)

mse = mean_squared_error(real, y_pred)
print(f"🔍 MSE del modelo: {mse:.4f} (umbral: {THRESHOLD})")

# Validación
if mse <= THRESHOLD:
    print("✅ El modelo cumple los criterios de calidad.")
    sys.exit(0)  # éxito
else:
    print("❌ El modelo no cumple el umbral. Deteniendo pipeline.")
    sys.exit(1)  # error