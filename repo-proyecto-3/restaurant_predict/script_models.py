"""
Este módulo implementa la creación de modelos de clasificación con diversas configuraciones.
Incluye la división del dataset, escalado de características, entrenamiento y evaluación de
los modelos Random Forest, Regresión Logística, SVM, KNN y Árboles de Decisión.
"""
import pandas as pd


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# modelos
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

def create_model_features():
    """
    Crea, entrena y evalúa modelos de clasificación con diversas configuraciones de hiperparámetros.
    - Carga un dataset procesado.
    - Divide los datos en conjuntos de entrenamiento y prueba.
    - Escala las características utilizando StandardScaler.
    - Entrena y evalúa modelos con configuraciones diferentes.
    - Imprime las precisiones de los modelos y guarda el mejor modelo como artefacto.
    """
    # Cargar el dataset procesado
    dataset = pd.read_csv('../data/processed/features_for_model.csv')
    #dataset = pd.read_csv('features_for_model.csv')
    # Selección de target y features
    x = dataset.drop(['HighSatisfaction'], axis=1)
    y = dataset['HighSatisfaction']
      # Split de Train y Test
    # Dividir en train y test
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=True,
                                                        random_state=2025)
    # Configuramos y calculamos el standard scaler
    # Escalar los datos
    std_scaler = StandardScaler()
    std_scaler.fit(x_train) # calcular los valores para el scaler.
    # Guardamos el scaler configurado (con datos de train) como artefacto del modelo.
    # Guardar el scaler
    import pickle
    with open('../artifacts/std_scaler.pkl', 'wb') as f:
        pickle.dump(std_scaler, f)
    #with open('std_scaler.pkl', 'wb') as f:
        #pickle.dump(std_scaler, f)
    # Creamos modelo de predicción
    x_train_std = std_scaler.transform(x_train)
    x_test_std = std_scaler.transform(x_test)
    # Definir modelos e hiperparámetros
    # Modelo 1: Random Forest
    modelo_rf = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=2025)
    modelo_rf.fit(x_train_std, y_train)
    y_preds_rf = modelo_rf.predict(x_test_std)
    accuracy_rf = accuracy_score(y_test, y_preds_rf)

    # Versión 2
    modelo_rf_2 = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    modelo_rf_2.fit(x_train_std, y_train)
    y_preds_rf_2 = modelo_rf_2.predict(x_test_std)
    accuracy_rf_2 = accuracy_score(y_test, y_preds_rf_2)

    # Versión 3
    modelo_rf_3 = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=0)
    modelo_rf_3.fit(x_train_std, y_train)
    y_preds_rf_3 = modelo_rf_3.predict(x_test_std)
    accuracy_rf_3 = accuracy_score(y_test, y_preds_rf_3)

    # Versión 4
    modelo_rf_4 = RandomForestClassifier(n_estimators=150, max_depth=15, random_state=1)
    modelo_rf_4.fit(x_train_std, y_train)
    y_preds_rf_4 = modelo_rf_4.predict(x_test_std)
    accuracy_rf_4 = accuracy_score(y_test, y_preds_rf_4)

    # Modelo 2: Regresión Logística
    modelo_rl = LogisticRegression(C=1.0, solver='liblinear', random_state=2025)
    modelo_rl.fit(x_train_std, y_train)
    y_preds_rl = modelo_rl.predict(x_test_std)
    accuracy_rl = accuracy_score(y_test, y_preds_rl)

    # Versión 2
    modelo_rl_2 = LogisticRegression(C=0.1, solver='lbfgs', random_state=42, max_iter=200)
    modelo_rl_2.fit(x_train_std, y_train)
    y_preds_rl_2 = modelo_rl_2.predict(x_test_std)
    accuracy_rl_2 = accuracy_score(y_test, y_preds_rl_2)


    # Versión 3
    modelo_rl_3 = LogisticRegression(C=10, solver='newton-cg', random_state=0)
    modelo_rl_3.fit(x_train_std, y_train)
    y_preds_rl_3 = modelo_rl_3.predict(x_test_std)
    accuracy_rl_3 = accuracy_score(y_test, y_preds_rl_3)


    # Versión 4
    modelo_rl_4 = LogisticRegression(C=0.5, solver='saga', random_state=2, max_iter=300)
    modelo_rl_4.fit(x_train_std, y_train)
    y_preds_rl_4 = modelo_rl_4.predict(x_test_std)
    accuracy_rl_4 = accuracy_score(y_test, y_preds_rl_4)

    # Modelo 3: SVC
    modelo_svc = SVC(C=1.0, kernel='rbf', random_state=2025)
    modelo_svc.fit(x_train_std, y_train)
    y_preds_svc = modelo_svc.predict(x_test_std)
    accuracy_svc = accuracy_score(y_test, y_preds_svc)


    # Versión 2
    modelo_svc_2 = SVC(C=0.5, kernel='linear', random_state=42)
    modelo_svc_2.fit(x_train_std, y_train)
    y_preds_svc_2 = modelo_svc_2.predict(x_test_std)
    accuracy_svc_2 = accuracy_score(y_test, y_preds_svc_2)


    # Versión 3
    modelo_svc_3 = SVC(C=2.0, kernel='poly', degree=3, random_state=0)
    modelo_svc_3.fit(x_train_std, y_train)
    y_preds_svc_3 = modelo_svc_3.predict(x_test_std)
    accuracy_svc_3 = accuracy_score(y_test, y_preds_svc_3)

    # Versión 4
    modelo_svc_4 = SVC(C=1.5, kernel='sigmoid', random_state=1)
    modelo_svc_4.fit(x_train_std, y_train)
    y_preds_svc_4 = modelo_svc_4.predict(x_test_std)
    accuracy_svc_4 = accuracy_score(y_test, y_preds_svc_4)

    # Modelo 4: K-Nearest Neighbors
    modelo_knn = KNeighborsClassifier(n_neighbors=5, weights='uniform')
    modelo_knn.fit(x_train_std, y_train)
    y_preds_knn = modelo_knn.predict(x_test_std)
    accuracy_knn = accuracy_score(y_test, y_preds_knn)


    # Versión 2
    modelo_knn_2 = KNeighborsClassifier(n_neighbors=10, weights='distance')
    modelo_knn_2.fit(x_train_std, y_train)
    y_preds_knn_2 = modelo_knn_2.predict(x_test_std)
    accuracy_knn_2 = accuracy_score(y_test, y_preds_knn_2)


    # Versión 3
    modelo_knn_3 = KNeighborsClassifier(n_neighbors=3, weights='uniform')
    modelo_knn_3.fit(x_train_std, y_train)
    y_preds_knn_3 = modelo_knn_3.predict(x_test_std)
    accuracy_knn_3 = accuracy_score(y_test, y_preds_knn_3)


    # Versión 4
    modelo_knn_4 = KNeighborsClassifier(n_neighbors=7, weights='distance')
    modelo_knn_4.fit(x_train_std, y_train)
    y_preds_knn_4 = modelo_knn_4.predict(x_test_std)
    accuracy_knn_4 = accuracy_score(y_test, y_preds_knn_4)

    # Modelo 5: Árbol de Decisión
    modelo_dt = DecisionTreeClassifier(max_depth=None, min_samples_split=2, random_state=2025)
    modelo_dt.fit(x_train_std, y_train)
    y_preds_dt = modelo_dt.predict(x_test_std)
    accuracy_dt = accuracy_score(y_test, y_preds_dt)

     # Versión 2
    modelo_dt_2 = DecisionTreeClassifier(max_depth=10, min_samples_split=5, random_state=42)
    modelo_dt_2.fit(x_train_std, y_train)
    y_preds_dt_2 = modelo_dt_2.predict(x_test_std)
    accuracy_dt_2 = accuracy_score(y_test, y_preds_dt_2)

    # Versión 3
    modelo_dt_3 = DecisionTreeClassifier(max_depth=5, min_samples_split=10, random_state=0)
    modelo_dt_3.fit(x_train_std, y_train)
    y_preds_dt_3 = modelo_dt_3.predict(x_test_std)
    accuracy_dt_3 = accuracy_score(y_test, y_preds_dt_3)

    # Versión 4
    modelo_dt_4 = DecisionTreeClassifier(max_depth=15, min_samples_split=4, random_state=1)
    modelo_dt_4.fit(x_train_std, y_train)
    y_preds_dt_4 = modelo_dt_4.predict(x_test_std)
    accuracy_dt_4 = accuracy_score(y_test, y_preds_dt_4)

    # Comparar resultados
    resultados = {
        'RandomForest_v1': accuracy_rf,
        'RandomForest_v2': accuracy_rf_2,
        'RandomForest_v3': accuracy_rf_3,
        'RandomForest_v4': accuracy_rf_4,
        'LogisticRegression_v1': accuracy_rl,
        'LogisticRegression_v2': accuracy_rl_2,
        'LogisticRegression_v3': accuracy_rl_3,
        'LogisticRegression_v4': accuracy_rl_4,
        'SVC_v1': accuracy_svc,
        'SVC_v2': accuracy_svc_2,
        'SVC_v3': accuracy_svc_3,
        'SVC_v4': accuracy_svc_4,
        'KNeighbors_v1': accuracy_knn,
        'KNeighbors_v2': accuracy_knn_2,
        'KNeighbors_v3': accuracy_knn_3,
        'KNeighbors_v4': accuracy_knn_4,
        'DecisionTree_v1': accuracy_dt,
        'DecisionTree_v2': accuracy_dt_2,
        'DecisionTree_v3': accuracy_dt_3,
        'DecisionTree_v4': accuracy_dt_4
    }

    print("Resultados de precisión por modelo:")
    for modelo, accuracy in resultados.items():
        print(f"{modelo}: {accuracy:.4f}")

    # Guardamos el modelo para producción


    with open('../models/random_forest_v1.pkl', 'wb') as f:
        pickle.dump(modelo_rf_2,f)

    #Aqui se guarda el modelo que haya dado los mejores resultados, el mejor fue el RF v2
