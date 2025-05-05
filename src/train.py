import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import mlflow
import mlflow.sklearn
from sklearn.metrics import mean_squared_error
import sys
import traceback
import numpy as np
import joblib
import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Input
keras.utils.set_random_seed(64)
# Evaluación
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

print(f"--- Debug: Initial CWD: {os.getcwd()} ---")

# --- Define Paths ---
# Usar rutas absolutas dentro del workspace del runner
workspace_dir = os.getcwd() # Debería ser /home/runner/work/mlflow-deploy/mlflow-deploy
mlruns_dir = os.path.join(workspace_dir, "mlruns").replace("/", "\\")
tracking_uri = "file:\\" + os.path.abspath(mlruns_dir).replace("/", "\\")
# Definir explícitamente la ubicación base deseada para los artefactos
artifact_location = "file:\\" + os.path.abspath(mlruns_dir).replace("/", "\\")

print(f"--- Debug: Workspace Dir: {workspace_dir} ---")
print(f"--- Debug: MLRuns Dir: {mlruns_dir} ---")
print(f"--- Debug: Tracking URI: {tracking_uri} ---")
print(f"--- Debug: Desired Artifact Location Base: {artifact_location} ---")

# --- Asegurar que el directorio MLRuns exista ---
os.makedirs(mlruns_dir, exist_ok=True)

# --- Configurar MLflow ---
mlflow.set_tracking_uri(tracking_uri)

# --- Crear o Establecer Experimento Explícitamente con Artifact Location ---
experiment_name = "CI-CD-Proyecto final"
experiment_id = None # Inicializar variable
try:
    # Intentar crear el experimento, proporcionando la ubicación del artefacto
    experiment_id = mlflow.create_experiment(
        name=experiment_name,
        artifact_location=artifact_location # ¡Forzar la ubicación aquí!
    )
    print(f"--- Debug: Creado Experimento '{experiment_name}' con ID: {experiment_id} ---")
except mlflow.exceptions.MlflowException as e:
    if "RESOURCE_ALREADY_EXISTS" in str(e):
        print(f"--- Debug: Experimento '{experiment_name}' ya existe. Obteniendo ID. ---")
        # Obtener el experimento existente para conseguir su ID
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment:
            experiment_id = experiment.experiment_id
            print(f"--- Debug: ID del Experimento Existente: {experiment_id} ---")
            print(f"--- Debug: Ubicación de Artefacto del Experimento Existente: {experiment.artifact_location} ---")
            # Opcional: Verificar si la ubicación del artefacto es la correcta
            if experiment.artifact_location != artifact_location:
                 print(f"--- WARNING: La ubicación del artefacto del experimento existente ('{experiment.artifact_location}') NO coincide con la deseada ('{artifact_location}')! ---")
        else:
            # Esto no debería ocurrir si RESOURCE_ALREADY_EXISTS fue el error
            print(f"--- ERROR: No se pudo obtener el experimento existente '{experiment_name}' por nombre. ---")
            sys.exit(1)
    else:
        print(f"--- ERROR creando/obteniendo experimento: {e} ---")
        raise e # Relanzar otros errores

# Asegurarse de que tenemos un experiment_id válido
if experiment_id is None:
    print(f"--- ERROR FATAL: No se pudo obtener un ID de experimento válido para '{experiment_name}'. ---")
    sys.exit(1)

# Cargar datos desde archivos csv y entrenar Modelo ---
time_step = 10
trm_df_train_scaled = np.loadtxt('trm_df_train_scaled.csv', delimiter=',').reshape(-1, 1)
trm_df_val_scaled = np.loadtxt('trm_df_val_scaled.csv', delimiter=',').reshape(-1, 1)
trm_df_test_scaled = np.loadtxt('trm_df_test_scaled.csv', delimiter=',').reshape(-1, 1)

# Entrenamiento
X_train = []
Y_train = []
m = len(trm_df_train_scaled)

for i in range(time_step,m):
    X_train.append(trm_df_train_scaled[i-time_step:i,0])
    Y_train.append(trm_df_train_scaled[i,0])
X_train, Y_train = np.array(X_train), np.array(Y_train)

# Validación
X_val = []
Y_val = []
m = len(trm_df_val_scaled)

for i in range(time_step,m):
    X_val.append(trm_df_val_scaled[i-time_step:i,0])
    Y_val.append(trm_df_val_scaled[i,0])
X_val, Y_val = np.array(X_val), np.array(Y_val)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_val = np.reshape(X_val, (X_val.shape[0], X_val.shape[1], 1))

dim_salida = 1
na = 70

modelo = Sequential()
modelo.add(Input(shape=(X_train.shape[1],1)))
modelo.add(LSTM(units=na))
modelo.add(Dense(units=dim_salida))

modelo.compile(optimizer='rmsprop', loss='mse')
modelo.fit(X_train,Y_train,epochs=150,batch_size=9,validation_data=(X_val,Y_val),verbose=1)

# Prueba
X_test = []
for i in range(time_step,len(trm_df_test_scaled)):
    X_test.append(trm_df_test_scaled[i-time_step:i,0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
# Se importa scaler
min_max_scaler = joblib.load('scaler.pkl')
prediccion = modelo.predict(X_test)
prediccion = min_max_scaler.inverse_transform(prediccion)
real = min_max_scaler.inverse_transform(trm_df_test_scaled[time_step:len(trm_df_test_scaled),0].reshape(-1,1))
# métricas
mse_lstm = mean_squared_error(real, prediccion)
mae_lstm= mean_absolute_error(real, prediccion)
r2_lstm = r2_score(real, prediccion)

# --- Iniciar Run de MLflow ---
print(f"--- Debug: Iniciando run de MLflow en Experimento ID: {experiment_id} ---") # Añadir ID aquí
run = None
try:
    # Iniciar el run PASANDO EXPLÍCITAMENTE el experiment_id
    with mlflow.start_run(experiment_id=experiment_id) as run: # <--- CAMBIO CLAVE
        run_id = run.info.run_id
        actual_artifact_uri = run.info.artifact_uri
        print(f"--- Debug: Run ID: {run_id} ---")
        print(f"--- Debug: URI Real del Artefacto del Run: {actual_artifact_uri} ---")

        # Comprobar si coincide con el patrón esperado basado en artifact_location del experimento
        # (La artifact_uri del run incluirá el run_id)
        expected_artifact_uri_base = os.path.join(artifact_location, run_id, "artifacts")
        if actual_artifact_uri != expected_artifact_uri_base:
             print(f"--- WARNING: La URI del Artefacto del Run '{actual_artifact_uri}' no coincide exactamente con la esperada '{expected_artifact_uri_base}' (esto puede ser normal si la estructura difiere ligeramente). Lo importante es que NO sea la ruta local incorrecta. ---")
        if "/home/manuelcastiblan/" in actual_artifact_uri:
             print(f"--- ¡¡¡ERROR CRÍTICO!!!: La URI del Artefacto del Run '{actual_artifact_uri}' TODAVÍA contiene la ruta local incorrecta! ---")


        mlflow.log_metric("mse", mse_lstm)
        print(f"--- Debug: Intentando log_model con artifact_path='model' ---")

        mlflow.sklearn.log_model(
            sk_model=modelo,
            artifact_path="model"
        )
        print(f"✅ Modelo registrado correctamente. MSE: {mse_lstm:.4f}")

except Exception as e:
    print(f"\n--- ERROR durante la ejecución de MLflow ---")
    traceback.print_exc()
    print(f"--- Fin de la Traza de Error ---")
    print(f"CWD actual en el error: {os.getcwd()}")
    print(f"Tracking URI usada: {mlflow.get_tracking_uri()}")
    print(f"Experiment ID intentado: {experiment_id}") # Añadir ID aquí
    if run:
         print(f"URI del Artefacto del Run en el error: {run.info.artifact_uri}")
    else:
         print("El objeto Run no se creó con éxito.")
    sys.exit(1)
