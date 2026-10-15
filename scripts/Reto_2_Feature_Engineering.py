# %% [markdown]
# # Reto 2: Feature Engineering y Preparacion de Datos para Clustering
# 
# **Asignatura / Contexto:** Mercedes-Benz Hackathon - Clustering the Star  
# **Reto:** Reto 2 (Seleccion de variables, creacion de features derivadas y pipeline de transformacion)  
# **Dataset de entrada:** `data/datos_reto_mercedes_benz_malackathon.csv` (~302.000 registros x 39 columnas)  
# 
# ---
# 
# ## Objetivos del Reto 2
# 
# 1. **Seleccion justificada de variables:** Evaluar que informacion es util para segmentar clientes y descartar columnas redundantes o metadatos irrelevantes.
# 2. **Creacion de features derivadas:** Disenar y calcular nuevas variables con valor de negocio (antiguedad del vehiculo, frecuencia de taller, receptividad comercial, etc.).
# 3. **Tratamiento de variables categoricas:** Aplicar una codificacion adecuada evitando la dispersion dimensional masiva (maldicion de la dimensionalidad).
# 4. **Tratamiento de variables numericas y nulos:** Imputacion justificada y escalado robusto frente a valores atipicos.
# 5. **Pipeline de transformacion Scikit-Learn:** Encapsular el preprocesamiento en un flujo modular, reproducible y exportar los datasets listos para el algoritmo de clustering.

# %%
# ==============================================================================
# 1. CONFIGURACION E IMPORTS
# ==============================================================================
import os
import warnings
from datetime import datetime
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, RobustScaler

warnings.filterwarnings('ignore')

# Semilla fija para reproducibilidad
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Fecha de corte de referencia para calcular antiguedades
REFERENCE_DATE = pd.Timestamp('2026-10-15')

# Directorios de trabajo
DATA_DIR = "data"
FIGURES_DIR = "figures"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

print("[INFO] Librerias importadas y entorno configurado correctamente.")

# %% [markdown]
# ---
# ## 2. Carga del Dataset y Auditoria Inicial
# 
# Cargamos el archivo maestro del hackathon para iniciar el proceso de transformacion.

# %%
DATA_PATH = os.path.join(DATA_DIR, "datos_reto_mercedes_benz_malackathon.csv")

print(f"[INFO] Leyendo dataset desde: {DATA_PATH} ...")
df_raw = pd.read_csv(DATA_PATH, low_memory=False)
print(f"[INFO] Dimensiones iniciales: {df_raw.shape[0]:,} filas x {df_raw.shape[1]} columnas.")

# %% [markdown]
# ---
# ## 3. Seleccion y Descarte Justificado de Variables
# 
# Para realizar un clustering basado en distancia euclidiana (como K-Means), no podemos introducir variables a ciegas:
# 
# - **Identificadores y claves tecnicas:** Columnas como `Unnamed: 0` o `zipcode` no aportan significado analitico de patron de cliente. El codigo postal anadiria excesiva granularidad y ya disponemos de `country_code` y `population`.
# - **Multicolinealidad perfecta y redundancia:** 
#   - `number_of_households` tiene correlacion lineal de 1.00 con `population`. Mantener ambas duplicaria artificialmente el peso del tamano del municipio.
#   - `purchasing_power_per_household_eur` y `purchasing_power_millions_eur` superan el 0.96 de correlacion con `purchasing_power_per_inhabitant_eur`. Se mantiene solo esta ultima por ser la mas representativa del poder adquisitivo individual.
# - **Variables tecnicas de telemetria redundantes:** `speed_average` y `trip_distance_average` se descartan a favor de las medianas (`speed_median`, `trip_distance_median`), que son mucho mas robustas frente a anomalías de grabacion en los sensores. Columnas como `speed_mode` y `trip_distance_mode` contienen texto sucio.
# - **Variables con excesiva nulidad sin interpretabilidad:** `fuel_type` contiene un 59% de nulos y duplica el concepto recogido en `engine_concept`.

# %%
COLUMNS_TO_DROP = [
    'Unnamed: 0',                          # Identificador de fila autogenerado
    'zipcode',                             # Granularidad excesiva
    'number_of_households',                # Correlacion 1.00 con population
    'purchasing_power_per_household_eur',   # Redundante con purchasing_power_per_inhabitant_eur
    'purchasing_power_millions_eur',        # Redundante con purchasing_power_per_inhabitant_eur
    'fuel_type',                           # 59% nulos y redundante con engine_concept
    'speed_mode',                          # Valores de texto sin formato uniforme
    'trip_distance_mode',                  # Valores de texto sin formato uniforme
    'speed_average',                       # Mas sensible a outliers que la mediana
    'trip_distance_average',               # Mas sensible a outliers que la mediana
    'speed_all_counter',                   # Contador tecnico interno sin valor de negocio
    'trip_distance_all_counter',           # Contador tecnico interno sin valor de negocio
]

def drop_unnecessary_columns(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    existing = [c for c in cols if c in df.columns]
    df_clean = df.drop(columns=existing)
    print(f"[INFO] Columnas eliminadas: {len(existing)}")
    print(f"[INFO] Dimensiones resultantes: {df_clean.shape[0]:,} filas x {df_clean.shape[1]} columnas.")
    return df_clean

df_filtered = drop_unnecessary_columns(df_raw.copy(), COLUMNS_TO_DROP)

# %% [markdown]
# ---
# ## 4. Creacion de Features Derivadas (Feature Engineering)
# 
# Siguiendo el enunciado del reto, construimos variables calculadas a partir del conocimiento del negocio de automocion:
# 
# 1. `vehicle_age_years`: Antiguedad en anos a partir de `date_of_first_registration`. Permite identificar clientes con vehiculos nuevos vs. clientes en ciclo de renovacion.
# 2. `estimated_itv_passes`: Numero estimado de inspecciones tecnicas obligatorias pasadas, basandonos en la regulacion europea (primera a los 4 anos, posteriores cada 2 anos).
# 3. `workshop_frequency_yearly`: Tasa anualizada de visitas a taller oficial por ano de relacion del cliente con la marca. Evita penalizar a clientes recientes.
# 4. `marketing_receptivity_score`: Indice acumulado de autorizaciones de contacto comercial (telefono, e-post y correo tradicional), con valores de 0 a 3.
# 5. `has_telemetry_active`: Variable binaria que indica si el vehiculo transmite datos telematicos. El 60% de nulos en telemetria refleja que el cliente no tiene activa la app/conectividad, siendo en si mismo un indicador clave de comportamiento digital.

# %%
def create_derived_features(df: pd.DataFrame, ref_date: pd.Timestamp) -> pd.DataFrame:
    df_out = df.copy()
    
    # 1. Antiguedad del vehiculo en anos
    reg_date = pd.to_datetime(df_out['date_of_first_registration'], errors='coerce')
    age_days = (ref_date - reg_date).dt.days
    df_out['vehicle_age_years'] = (age_days / 365.25).clip(lower=0).round(2)
    # Imputar posibles nulos de fecha con la mediana general
    median_age = df_out['vehicle_age_years'].median()
    df_out['vehicle_age_years'] = df_out['vehicle_age_years'].fillna(median_age)
    
    # 2. Inspecciones tecnicas estimadas (ITV europea: primera a los 4 anos, luego cada 2)
    df_out['estimated_itv_passes'] = np.where(
        df_out['vehicle_age_years'] < 4,
        0,
        1 + ((df_out['vehicle_age_years'] - 4) / 2).astype(int)
    )
    
    # 3. Frecuencia anualizada de visitas a taller
    # Sumamos 0.1 anos para evitar divisiones por cero en clientes muy nuevos
    client_years = (df_out['months_as_mb_customer'].fillna(0) / 12.0) + 0.1
    visits = df_out['workshop_visits_warranty_goodwill'].fillna(0)
    df_out['workshop_frequency_yearly'] = (visits / client_years).round(4)
    
    # 4. Score de receptividad a marketing (0 a 3)
    consent_fields = [
        'consent_phone_communication_channel',
        'consent_epost_communication_channel',
        'consent_post_communication_channel'
    ]
    df_out['marketing_receptivity_score'] = df_out[consent_fields].fillna(0).sum(axis=1).astype(int)
    
    # 5. Flag de telemetria activa
    df_out['has_telemetry_active'] = (~df_out['trip_distance_median'].isnull()).astype(int)
    
    print("[INFO] Features derivadas creadas con exito:")
    print(f"  - vehicle_age_years: rango [{df_out['vehicle_age_years'].min():.1f}, {df_out['vehicle_age_years'].max():.1f}] anos.")
    print(f"  - estimated_itv_passes: valores posibles {sorted(df_out['estimated_itv_passes'].unique())}.")
    print(f"  - workshop_frequency_yearly: mediana = {df_out['workshop_frequency_yearly'].median():.2f} visitas/ano.")
    print(f"  - marketing_receptivity_score: distribucion = {dict(df_out['marketing_receptivity_score'].value_counts())}.")
    print(f"  - has_telemetry_active: {df_out['has_telemetry_active'].mean()*100:.1f}% clientes con conexion activa.")
    
    return df_out

df_features_all = create_derived_features(df_filtered, REFERENCE_DATE)

# %% [markdown]
# ---
# ## 5. Tratamiento de Variables Categoricas y Estructuracion
# 
# Si aplicaramos One-Hot Encoding a variables como `vehicle_type` (1.436 categorias) o `model_series` (88 categorias), creariamos un espacio de mas de 1.500 columnas con una densidad casi nula (maldicion de la dimensionalidad). En esas condiciones, las distancias euclidianas pierden todo significado.
# 
# **Solucion adoptada:**
# 1. `age_range`: Es una variable categorica ordinal con orden intrinseco (`<25`, `25-34`, `35-44`, `45-54`, `55-64`, `65+`). La codificamos numericamente del 0 al 5 mediante `OrdinalEncoder`.
# 2. `is_current_vehicle`: Se asegura como valor binario entero (0 o 1).
# 3. **Separacion de Metadatos:** Las variables descriptivas de vehiculo (`model_series`, `model_class`, `engine_concept`, etc.) se extraen a un DataFrame de metadatos. De esta forma, el algoritmo de clustering se entrena unicamente sobre dimensiones numericas compactas, y posteriormente se cruzan los perfiles resultantes con los metadatos para el analisis de negocio.
# 4. **Macro-Segmentacion:** Se divide el dataset en 4 grupos naturales segun pais (`country_code`: CH vs AT) y tipo de cliente (`role_type`: B2C vs B2B).

# %%
AGE_ORDER = ['<25', '25-34', '35-44', '45-54', '55-64', '65+']

def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    df_out = df.copy()
    
    # Imputar age_range faltante con la moda antes del encoding
    mode_age = df_out['age_range'].mode()[0]
    df_out['age_range'] = df_out['age_range'].fillna(mode_age)
    
    encoder = OrdinalEncoder(categories=[AGE_ORDER], handle_unknown='use_encoded_value', unknown_value=-1)
    df_out['age_range_encoded'] = encoder.fit_transform(df_out[['age_range']]).astype(int)
    
    # Binarizar is_current_vehicle
    if 'is_current_vehicle' in df_out.columns:
        df_out['is_current_vehicle'] = df_out['is_current_vehicle'].fillna(1).astype(int)
        
    print("[INFO] Codificacion ordinal de age_range completada (0 a 5).")
    return df_out

df_encoded = encode_categorical_features(df_features_all)

# %%
# Definicion de columnas de metadatos cualitativos para cruce posterior
METADATA_COLS = [
    'client_id', 'vehicle_id',
    'model_class', 'vehicle_type', 'model_series', 'engine_concept',
    'gender_derived', 'role', 'finance_leasing_status', 'product_group',
    'consent_phone_communication_channel',
    'consent_epost_communication_channel',
    'consent_post_communication_channel',
    'country_code', 'role_type',
    'date_of_first_registration', 'is_current_vehicle'
]

# %% [markdown]
# ---
# ## 6. Tratamiento de Variables Numericas y Valores Nulos
# 
# Las variables de telemetria (`trip_distance_median`, `speed_median`, `fuel_consumption_average`, `operating_hours_total`) presentan un 60% de valores nulos porque el sensor no estaba activo.
# 
# Para permitir que el algoritmo procese estos registros sin distorsionar los datos reales:
# 1. Se imputa la **mediana agrupada** por segmento (`role_type`, `country_code` y `model_class`).
# 2. Si algun grupo carece de observaciones, se utiliza la mediana global como respaldo.
# 3. La variable `has_telemetry_active` permite al modelo distinguir perfectamente si un cliente aporto datos reales o valores neutros imputados.

# %%
TELEMETRY_COLS = [
    'trip_distance_median',
    'speed_median',
    'fuel_consumption_average',
    'operating_hours_total'
]

def impute_telemetry_features(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    df_out = df.copy()
    group_keys = ['role_type', 'country_code', 'model_class']
    
    for col in cols:
        group_median = df_out.groupby(group_keys)[col].transform('median')
        df_out[col] = df_out[col].fillna(group_median)
        # Fallback a mediana global
        df_out[col] = df_out[col].fillna(df_out[col].median())
        
    print("[INFO] Variables de telemetria imputadas con mediana agrupada.")
    return df_out

df_imputed = impute_telemetry_features(df_encoded, TELEMETRY_COLS)

# Imputar posibles nulos residuales en variables socioeconomicas con mediana agrupada por pais y modelo
for col in ['purchasing_power_per_inhabitant_eur', 'population']:
    if col in df_imputed.columns:
        df_imputed[col] = df_imputed[col].fillna(df_imputed.groupby(['country_code', 'model_class'])[col].transform('median'))
        df_imputed[col] = df_imputed[col].fillna(df_imputed[col].median())

if 'months_as_mb_customer' in df_imputed.columns:
    df_imputed['months_as_mb_customer'] = df_imputed['months_as_mb_customer'].fillna(df_imputed['months_as_mb_customer'].median())

# %% [markdown]
# ---
# ## 7. Seleccion Final de Features para el Modelo
# 
# Consolidamos las 16 features numericas y ordinales que alimentaran el algoritmo de clustering:

# %%
FEATURES_FOR_CLUSTERING = [
    # Demografia
    'age_range_encoded',
    
    # Vehiculo y uso temporal
    'vehicle_age_years',
    'estimated_itv_passes',
    'is_current_vehicle',
    
    # Fidelidad y relacion con la marca
    'months_as_mb_customer',
    'currently_connected_vehicles',
    'workshop_frequency_yearly',
    'workshop_visits_warranty_goodwill',
    
    # Disposicion comercial
    'marketing_receptivity_score',
    
    # Entorno socioeconomico
    'purchasing_power_per_inhabitant_eur',
    'population',
    
    # Patron digital y conduccion
    'has_telemetry_active',
    'trip_distance_median',
    'speed_median',
    'fuel_consumption_average',
    'operating_hours_total'
]

print(f"[INFO] Total de features seleccionadas para clustering: {len(FEATURES_FOR_CLUSTERING)}")
for idx, f in enumerate(FEATURES_FOR_CLUSTERING, 1):
    print(f"  {idx:02d}. {f}")

# %% [markdown]
# ---
# ## 8. Pipeline de Transformacion y Escalado Robusto
# 
# ### ¿Por que RobustScaler y no StandardScaler?
# 
# En variables como `operating_hours_total` (con maximos que superan las 25.000 horas) o `population`, existen valores extremos tipicos de distribuciones con colas pesadas.
# - `StandardScaler` utiliza media y varianza, las cuales se ven fuertemente sesgadas por estos outliers, comprimiendo la variacion del 95% restante de los clientes.
# - `RobustScaler` utiliza la **mediana y el rango intercuartilico (IQR)**, escalando los datos de forma insensible a anomalías extremas.
# 
# Encapsulamos el escalador en un `Pipeline` modular de scikit-learn.

# %%
def create_preprocessing_pipeline() -> Pipeline:
    pipeline = Pipeline([
        ('scaler', RobustScaler())
    ])
    return pipeline

pipeline_example = create_preprocessing_pipeline()
print("[INFO] Pipeline de Scikit-Learn inicializado:")
print(pipeline_example)

# %% [markdown]
# ---
# ## 9. Particion en Macro-Segmentos y Exportacion de Artefactos
# 
# Particionamos los clientes en los 4 escenarios de negocio:
# - `CH_B2C`: Clientes particulares en Suiza
# - `AT_B2C`: Clientes particulares en Austria
# - `CH_B2B`: Empresas y flotas en Suiza
# - `AT_B2B`: Empresas y flotas en Austria
# 
# Para cada segmento, ajustamos y transformamos sus variables con el `RobustScaler`, guardando:
# - `{segmento}_features_scaled.csv`: Entrada directa para clustering.
# - `{segmento}_features_raw.csv`: Valores originales para interpretacion de medias/medianas.
# - `{segmento}_metadata.csv`: Identificadores y datos cualitativos.

# %%
def process_and_save_segments(df_data: pd.DataFrame, features: List[str], meta_cols: List[str], output_dir: str):
    existing_meta = [c for c in meta_cols if c in df_data.columns]
    df_meta = df_data[existing_meta].copy()
    
    segments = ['CH_B2C', 'AT_B2C', 'CH_B2B', 'AT_B2B']
    
    print("\n" + "=" * 70)
    print("PROCESAMIENTO Y EXPORTACION POR MACRO-SEGMENTO")
    print("=" * 70)
    
    for country in ['CH', 'AT']:
        for role in ['B2C', 'B2B']:
            seg_name = f"{country}_{role}"
            mask = (df_data['country_code'] == country) & (df_data['role_type'] == role)
            
            sub_raw = df_data.loc[mask, features].copy()
            sub_meta = df_meta.loc[mask].copy()
            
            # Ajuste y transformacion con el Pipeline de RobustScaler
            pipe = create_preprocessing_pipeline()
            scaled_array = pipe.fit_transform(sub_raw)
            sub_scaled = pd.DataFrame(scaled_array, index=sub_raw.index, columns=features)
            
            # Exportar CSVs
            raw_path = os.path.join(output_dir, f"{seg_name}_features_raw.csv")
            scaled_path = os.path.join(output_dir, f"{seg_name}_features_scaled.csv")
            meta_path = os.path.join(output_dir, f"{seg_name}_metadata.csv")
            
            sub_raw.to_csv(raw_path, index=True, lineterminator='\n')
            sub_scaled.to_csv(scaled_path, index=True, lineterminator='\n')
            sub_meta.to_csv(meta_path, index=True, lineterminator='\n')
            
            print(f"[OK] Segmento {seg_name}: {len(sub_raw):,d} filas.")
            print(f"     Raw:    {raw_path}")
            print(f"     Scaled: {scaled_path}")
            print(f"     Meta:   {meta_path}")

process_and_save_segments(df_imputed, FEATURES_FOR_CLUSTERING, METADATA_COLS, DATA_DIR)

# %% [markdown]
# ---
# ## 10. Resumen y Estado de Preparacion
# 
# El pipeline ha finalizado con exito. Todos los datos han sido transformados, escalados y particionados sin valores nulos residuales.
# El archivo `*_features_scaled.csv` de cada segmento constituye la entrada directa y valida para alimentar los algoritmos de clustering (K-Means, DBSCAN o Modelos de Mezcla Gaussiana).
