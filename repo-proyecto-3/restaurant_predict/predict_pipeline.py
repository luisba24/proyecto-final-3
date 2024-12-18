"""
Este módulo realiza la predicción para un modelo de aprendizaje automático
de un resturante en base las quejas recibidas.
"""
import pickle
from datetime import datetime
import os
import mlflow
import pandas as pd

def create_prediction():
    """
    Función para predecir y guardar los resultados.
    """
    # Cargar configuraciones de ingeniería de características
    with open('../artifacts/feature_eng_configs.pkl', 'rb') as f:
        feature_eng_configs = pickle.load(f)

    # Cargar el modelo guardado
    with open('../models/random_forest_v1.pkl', 'rb') as f:
        modelo = pickle.load(f)


    # Cargar el dataset de prueba
    data_test = pd.read_csv('../data/raw/test.csv')

    data_test.drop(["CustomerID", "PreferredCuisine",
                    "Gender", "TimeOfVisit", "AverageSpend", "Age"], axis=1, inplace=True)

    # codificación de variable MealType
    data_test['MealType'] = pd.get_dummies(data_test['MealType'], drop_first=True).astype(int)
    # codificación de Variables VisitFrequency y DiningOccasion
    data_test['VisitFrequency'] = data_test[
        'VisitFrequency'].map(feature_eng_configs['visit_freq_counts'])
    data_test['DiningOccasion'] = data_test[
        'DiningOccasion'].map(feature_eng_configs['dining_occ_counts'])
    # Cargar el scaler estándar si fue usado previamente
    with open('../artifacts/std_scaler.pkl', 'rb') as f:
        std_scaler = pickle.load(f)
    # Estandarizar las variables del dataset de prueba
    x_data_test_std = std_scaler.transform(data_test)
    # Realizar predicciones
    model_predicts = modelo.predict(x_data_test_std)

    # Crear un DataFrame con las predicciones
    predictions_df = pd.DataFrame(model_predicts, columns=["Prediction"])

    # Obtener la fecha y hora actual en el formato especificado
    timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

    # Ruta de la subcarpeta donde se almacenarán las predicciones
    output_folder = '../data/predictions'
    os.makedirs(output_folder, exist_ok=True)  # Crear la carpeta si no existe

    # Nombre del archivo basado en la fecha y hora
    output_file = os.path.join(output_folder, f'predictions-{timestamp}.csv')

    # Guardar las predicciones en un archivo CSV
    predictions_df.to_csv(output_file, index=False)
    print(f"Predicciones guardadas en: {output_file}")

    # Iniciar un experimento en MLflow
    mlflow.set_experiment("Restaurant Predictions")

    # Registrar el artefacto en MLflow
    with mlflow.start_run():
        # Guardar el archivo de predicciones en MLflow
        mlflow.log_artifact(output_file, artifact_path="predictions")
        print(f"Archivo de predicciones registrado en MLflow: {output_file}")
