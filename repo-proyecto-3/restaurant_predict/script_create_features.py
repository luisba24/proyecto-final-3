"""
Este módulo realiza la creación de características para un modelo de aprendizaje automático
de un resturante en base las quejas recibidas.
"""

import pandas as pd


def create_model_features():
    """
    Función para cargar datos, realizar ingeniería de características y guardar los resultados.
    """
    ### 1. Cargamos Datos
    dataset = pd.read_csv("../data/raw/train.csv")

    dataset.head()


    ### 3. Eliminamos variables no útiles


    dataset.drop(["CustomerID", "PreferredCuisine", "Gender", "TimeOfVisit", "AverageSpend", "Age"],
                 axis=1, inplace=True)
    dataset.head()


    ### 4. Ingeniería de Características


    dataset.isnull().mean()


    columnas_categoricas = ['VisitFrequency', 'DiningOccasion', 'MealType']

    for col in columnas_categoricas:
        categorias = dataset[col].unique()
        print(f"Columna: {col}")
        print(f"Categorías únicas: {categorias}")
        print(f"Número de categorías: {len(categorias)}\n")



    # Codificación de la variable Meal Type usando OHE


    # Usar pd.get_dummies() directamente y convertir a entero
    dataset['MealType'] = pd.get_dummies(dataset['MealType'], drop_first=True).astype(int)


    # Codificacion de las variables VisitFrequency y DiningOccasion usando Frecuency Encoder


    # Aplicar value_counts() y map()
    visit_freq_counts = dataset['VisitFrequency'].value_counts()
    dataset['VisitFrequency'] = dataset['VisitFrequency'].map(visit_freq_counts)

    dining_occ_counts = dataset['DiningOccasion'].value_counts()
    dataset['DiningOccasion'] = dataset['DiningOccasion'].map(dining_occ_counts)


    dataset.head()


    ### 5. Guardar el dataset procesado


    dataset.to_csv('../data/processed/features_for_model.csv', index=False)
    #dataset.to_csv('features_for_model.csv', index=False)


    # Guardamos valores de configuración del train


    feature_eng_configs = {
        'visit_freq_counts': visit_freq_counts,
        'dining_occ_counts': dining_occ_counts
    }

    import pickle
    with open('../artifacts/feature_eng_configs.pkl', 'wb') as f:
    #with open('feature_eng_configs.pkl', 'wb') as f:
        pickle.dump(feature_eng_configs, f)
