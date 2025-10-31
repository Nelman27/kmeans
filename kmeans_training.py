import os
import json
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

# Cargar variables de entorno desde un archivo .env
load_dotenv()


file_path = os.getenv('LOG_FILE_PATH', 'data.json')

# Carga de datos desde un archivo JSON
def load_logs(file_path):
    "Carga los registros de un archivo JSON."
    print(f"Cargando registros desde {file_path} . . .")
    logs = []
    # Intentamos primero parsear como JSON (array de objetos).
    # Si falla, intentamos leer como NDJSON (un objeto JSON por línea).
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)  # devuelve lista de dicts
            if isinstance(data, list):
                logs = list(data)
            else:
                logs = [data]
    except (json.JSONDecodeError, ValueError):
        # Fallback NDJSON
        logs = []
        with open(file_path, 'r') as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                try:
                    logs.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"Advertencia: Línea no válida en el archivo JSON, se omitirá: {line}")
                    continue

    print(f"Se cargaron {len(logs)} registros.")
    return logs

# Ingeniería de Features
def engineer_features(logs):
    "Realiza la ingenería de features en los registros cargados."
    print("Iniciando la ingeniería de features . . .")
    
    # Convertimos la lista de dicts a DataFrame
    db = pd.DataFrame(logs)

    # Trabajamos sobre una copia para no modificar el original si se necesita
    df = db.copy()

    # Creación de nuevas features
    # Feature 1: Numérica (status). Si falta, asumimos 200
    if 'status' in df.columns:
        df['status'] = df['status'].fillna(200).astype(int)
    else:
        df['status'] = 200

    # Feature 2 y 3: Categórica (level y service)
    if 'level' in df.columns:
        df['level'] = df['level'].fillna('unknown')
    else:
        df['level'] = 'unknown'

    if 'service' in df.columns:
        df['service'] = df['service'].fillna('unknown')
    else:
        df['service'] = 'unknown'

    # Basada en texto msg: flags binarios (0/1)
    if 'msg' in df.columns:
        df['has_timeout'] = df['msg'].apply(lambda x: 1 if isinstance(x, str) and 'timeout' in x.lower() else 0)
        df['has_db_error'] = df['msg'].apply(lambda x: 1 if isinstance(x, str) and 'db' in x.lower() else 0)
    else:
        df['has_timeout'] = 0
        df['has_db_error'] = 0

    print("DataFrame con features crudas primeras 5 filas:")
    print(df.head())
    
    # Codificación (Encoding) 
    # K-Means no entiende "ERROR" o "INFO". Debemos convertirlos a números.
    # Usamos One-Hot Encoding, que crea nuevas columnas:
    # level_ERROR | level_INFO | level_WARN
    #      1      |      0     |      0
    
    categorical_cols = ['level', 'service']
    # Compatibilidad con versiones antiguas de sklearn
    try:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    except TypeError:
        encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')

    encoded_cols = encoder.fit_transform(df[categorical_cols])
    encoded_df = pd.DataFrame(encoded_cols, columns=encoder.get_feature_names_out(categorical_cols))
    
    #Entrenamos el encoder y transformamos las columnas categóricas
    print("Aplicando One-Hot Encoding a las columnas categóricas . . .")
    print(encoded_df.columns.tolist())
    
    #Ensamble final de l vector de features
    
    #Seleccionamos los features numéricos y binarios que creamos
    numeric_df = df[['status', 'has_timeout', 'has_db_error']]

    # Resetear índices
    numeric_df = numeric_df.reset_index(drop=True)
    encoded_df = encoded_df.reset_index(drop=True)

    # Ensamble final de características
    final_features = pd.concat([numeric_df, encoded_df], axis=1)

    # Escalado (StandardScaler)
    scaler = StandardScaler()
    features_scaled_array = scaler.fit_transform(final_features)

    print(f"\nVector de Features final ensamblado y escalado. Forma: {features_scaled_array.shape}")

    # Devolvemos el DataFrame enriquecido y el array escalado para el modelo
    return df, features_scaled_array


# Entrenar kmeas 
def train_kmeans(features_array, num_clusters=3):
    "Entrenar el modelo K-Means con las features proporcionadas."
    print(f"Entrenando K-Means con {num_clusters} clusters . . .")
    # Crear e entrenar el modelo
    # n_init=10 es una elección razonable; en sklearn 1.4+ el valor por defecto cambió
    kmeans = KMeans(n_clusters=num_clusters, n_init=10, random_state=42)
    print("Entrenando el modelo . . .")
    kmeans.fit(features_array)
    print("¡Entrenamiento completo!")
    return kmeans


# Analizar resultados con PCA
def analyze_clusters(original_df, features_array, kmeans_model):
    """
    Añade las etiquetas del cluster al DataFrame original e imprime un resumen de lo que define a cada cluster.
    """
    print("\n--- Análisis de Clusters ---")
    
    # Obtener la etiqueta de cada cluster para cada punto de los log
    labels = kmeans_model.labels_
    
    # Añadir las etiquetas al DataFrame original para interpretar los clusters
    original_df['cluster'] = labels
    
    
    # Interpretación 
    # ¿Qué define a cada cluster?
    # Agrupamos por cluster y calculamos la media de las features numéricas
    # y el conteo de las features categóricas.
    
    pd.set_option('display.max_columns', None)  # Mostrar todas las columnas
    pd.set_option('display.width', 1000)    # Ancho máximo para evitar cortes
    
    #Analizar features numéricas (como: status y nuestras flags de texto)
    numeric_analysis = original_df.groupby('cluster')[['status', 'has_timeout', 'has_db_error']].mean()
    print("\nAnálisis de features numéricas por cluster:")
    print(numeric_analysis)
    
    #Analizar features categóricas (level y service)
    categorical_cols = original_df.groupby('cluster')[['level', 'service']].value_counts().unstack(fill_value=0)
    print("\nAnálisis de features categóricas por cluster:")
    print(categorical_cols)
    
    # Conclusión humana reporte
    print("\n--- Conclusión (Interpretación Humana) ---")
    for i in range(kmeans_model.n_clusters):
        print(f"\nCluster {i}:")
        # Imprimir las 10 primeras filas de este cluster para inspección visual
        print(original_df[original_df['cluster'] == i].head(10))
        
        print("\n(Observa la salida de arriba para ver los patrones. Por ejemplo:)")
        print("Cluster 0 podrían ser 'Errores de DB en api-pagos'")
        print("Cluster 1 podrían ser 'Logins exitosos en api-usuarios'")
        print("Cluster 2 podrían ser 'Timeouts en api-externa'")
    
    
    # Visualización 
    # No podemos graficar en 10+ dimensiones.
    # Usamos PCA para "reducir" las dimensiones a 2 (X, Y)
    # preservando la mayor cantidad de información posible.
    
    print("\nGenerando visualización 2D (cluster_plot.png)...")
    pca = PCA(n_components=2, random_state=42)
    features_2d = pca.fit_transform(features_array)
    
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap='viridis', alpha=0.7)
    
    # Graficar los centros (lo que el modelo piensa que es el "centro" de cada cluster)
    centroids_2d = pca.transform(kmeans_model.cluster_centers_)
    plt.scatter(centroids_2d[:, 0], centroids_2d[:, 1], c='red', marker='X', s=200, label='Centroides')
    
    plt.title('Visualización de Clusters de Logs (reducido a 2D con PCA)')
    plt.xlabel('Componente Principal 1')
    plt.ylabel('Componente Principal 2')
    plt.legend()
    plt.grid(True)
    plt.savefig('cluster_plot.png')
    print("Gráfico guardado en 'cluster_plot.png'. ¡Ábrelo para ver los grupos!")


# --- Bloque Principal de Ejecución ---
if __name__ == "__main__":
    file_path = 'data/data.json'
    # 1. Cargar
    # ¡AQUÍ ESTÁ LA CORRECCIÓN!
    # ANTES: logs_data = load_logs('sample_logs.jsonl')
    # AHORA: Usamos la variable 'file_path' que definiste en la línea 16
    logs_data = load_logs(file_path)
    
    if logs_data:
        # 2. Transformar
        df_original, features_vector = engineer_features(logs_data)
        
        # 3. Entrenar
        # Elegimos 3 clusters porque sabemos que hay 3 patrones
        # (Login, Error DB, Timeout). En el mundo real,
        # usarías una técnica como el "Método del Codo" para encontrar el número óptimo.
        model = train_kmeans(features_vector, num_clusters=3)
        
        # 4. Analizar
        analyze_clusters(df_original, features_vector, model)
    else:
        print("No se cargaron logs. Revisa el archivo 'sample_logs.jsonl'.")