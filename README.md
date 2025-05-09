## 📈 Predicción de la TRM (COP a USD) 1993 - 1996

Este proyecto busca predecir la Tasa Representativa del Mercado (TRM) de pesos colombianos (COP) respecto al dólar estadounidense (USD) mediante un red neuronal LSTM. Incluye descarga de datos de repositorio público del gobierno colombiano, preprocesamiento, entrenamiento, evaluación del modelo y despliegue automatizado con CI/CD usando GitHub Actions y MLflow.

---

### 🧠 Características del proyecto

* Manejo de variables globales con config.yml
* Descarga datos históricos de TRM desde datos.gov.co
* Preprocesa los datos.
* Entrena una red LSTM para predecir la TRM.
* Evaluar el desempeño de los modelos con MSE, MAE y R2.
* Registra ejecuciones del entrenamiento usando MLflow.
* Ejecuta un pipeline CI/CD con GitHub actions.

---

### ⚙️ Uso manual y seguimiento en MLFlow local

* En el directorio mlflow-deploy instalar el environment venv con python3 -n venv venv
* Activar el environment venv con . venv/Scripts/activate
* Instalar requisitos con python -m pip install -r mlflow-deploy/requirements.txt
* En train.py descomentar la linea mlflow.set_tracking_uri("http://127.0.0.1:8080")
* Ejecutar cada paso con:
    - python src/preprocessing.py
    - python src/train.py
    - python src/validate.py
* En http://127.0.0.1:8080 se puede ver el historial de experimentos y runs


### 🚀 Entrenamiento del modelo
* En Makefile están los pasos automatizados para descargar los datos, preprocesarlos, entrenar la red, evaluarla, obtener la ruta del modelo más reciente y subirlo como artefacto.
* Se puede ver el resultado de cada pipeline en actions de este repositorio de Github actions. 

### ✍️ Autor

* Nombre: *\[Diego Alfaro]*
* Email: *[dalfaro72747@universidadean.edu.co]*
* Proyecto para: *Trabajo Final MLOPS Maestría Ciencia de Datos Universidad EAN*