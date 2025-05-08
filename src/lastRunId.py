from mlflow.tracking import MlflowClient
import os
# Trabajar con config.yml
import yaml

def load_config(path='config.yml'):
    with open(path, 'r') as file:
        config = yaml.safe_load(file)
    return config

config = load_config()

experiment_name = config['settings']['experiment_name']
model_path = config['paths']['model_path']

# --- Cargar modelo previamente entrenado ---
client = MlflowClient()
experiment_name = experiment_name
experiment = client.get_experiment_by_name(experiment_name)

# Listar los runs ordenados por fecha de inicio
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["start_time DESC"],
    max_results=1
)

# Obtener el run_id del último run
latest_run = runs[0]
run_id = latest_run.info.run_id

model_path = model_path.replace("runId", run_id)

model_filename = "model.pkl"
model_path = os.path.abspath(os.path.join(os.getcwd(), model_path + model_filename))
print(model_path)