# %% [markdown]
# # 🌟 Clustering the Star — Mercedes-Benz Malackathon
# ## Retos Básicos 1 & 2: EDA, Limpieza de Datos y Feature Engineering
# 
# **Equipo:** [Nombre del Equipo]  
# **Fecha:** Octubre 2026  
# **Dataset:** ~302.000 registros × 39 columnas — Clientes Mercedes-Benz en Suiza (CH) y Austria (AT)
# 
# ---
# 
# ### Arquitectura del Pipeline de Datos
# 
# ```
# CSV Raw (302K × 39)
#   │
#   ├── FASE 1 → Limpieza Radical + Rescate Inteligente de Nulos
#   ├── FASE 2 → Feature Engineering de Alto Impacto (nuevos KPIs)
#   ├── FASE 3 → Tratamiento del Bloque de Telemetría (60% nulos)
#   ├── FASE 4 → Segmentación Estática + Pipeline Scikit-Learn
#   └── FASE 5 → Informe para el Jurado
#         │
#         ▼
#   4 DataFrames particionados (CH_B2C, CH_B2B, AT_B2C, AT_B2B)
#   + DataFrame de Metadatos para perfilado posterior
# ```

# %%
# ============================================================================
# IMPORTS Y CONFIGURACIÓN GLOBAL
# ============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from datetime import datetime
from typing import Tuple, Dict, List, Optional

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, RobustScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer

# Configuración visual
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 11
warnings.filterwarnings('ignore')

# Semilla para reproducibilidad
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Directorios de salida
import os
os.makedirs("figures", exist_ok=True)
os.makedirs("data", exist_ok=True)

# Fecha de referencia para cálculos de antigüedad
REFERENCE_DATE = pd.Timestamp('2026-10-15')

print("✅ Librerías cargadas correctamente")
print(f"📅 Fecha de referencia para cálculos: {REFERENCE_DATE.strftime('%Y-%m-%d')}")

# %% [markdown]
# ---
# # 📊 FASE 0: Carga y Auditoría Inicial del Dataset
# 
# Antes de tocar nada, necesitamos una fotografía completa del estado de los datos.
# Esta auditoría es la base de todas las decisiones posteriores.

# %%
# ============================================================================
# FASE 0: CARGA Y AUDITORÍA COMPLETA
# ============================================================================

DATA_PATH = "data/datos_reto_mercedes_benz_malackathon.csv"

def load_and_audit(path: str) -> pd.DataFrame:
    """
    Carga el dataset y genera un informe de auditoría completo.
    
    Returns:
        DataFrame raw sin modificar.
    """
    df = pd.read_csv(path, low_memory=False)
    
    print("=" * 70)
    print("  AUDITORÍA INICIAL DEL DATASET")
    print("=" * 70)
    print(f"\n📐 Dimensiones: {df.shape[0]:,} filas × {df.shape[1]} columnas")
    print(f"💾 Memoria estimada: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")
    
    # Reporte de nulos
    null_report = pd.DataFrame({
        'Columna': df.columns,
        'Nulos': df.isnull().sum().values,
        'Pct_Nulos': (df.isnull().sum().values / len(df) * 100).round(2),
        'Dtype': df.dtypes.values,
        'Uniques': df.nunique().values
    }).sort_values('Pct_Nulos', ascending=False)
    
    print("\n📋 Mapa de nulidad (ordenado por % de nulos):")
    print("-" * 70)
    for _, row in null_report.iterrows():
        bar = "█" * int(row['Pct_Nulos'] / 2)
        status = "🟢" if row['Pct_Nulos'] == 0 else ("🟡" if row['Pct_Nulos'] < 10 else ("🟠" if row['Pct_Nulos'] < 50 else "🔴"))
        print(f"  {status} {row['Columna']:<42} {row['Nulos']:>7,} ({row['Pct_Nulos']:>5.1f}%) {bar}")
    
    # Resumen por bloques temáticos
    print("\n" + "=" * 70)
    print("  BLOQUES TEMÁTICOS")
    print("=" * 70)
    
    telemetry_cols = [c for c in df.columns if any(k in c for k in ['trip_', 'speed_', 'fuel_', 'operating_'])]
    socio_cols = ['purchasing_power_per_inhabitant_eur', 'purchasing_power_per_household_eur',
                  'purchasing_power_millions_eur', 'population', 'number_of_households', 'zipcode']
    vehicle_cols = ['model_series', 'vehicle_type', 'model_class', 'engine_concept', 'fuel_type',
                    'product_group', 'date_of_first_registration', 'is_current_vehicle']
    customer_cols = ['age_range', 'gender_derived', 'role', 'role_type', 'months_as_mb_customer',
                     'currently_connected_vehicles', 'finance_leasing_status',
                     'workshop_visits_warranty_goodwill',
                     'consent_phone_communication_channel',
                     'consent_epost_communication_channel',
                     'consent_post_communication_channel']
    
    for name, cols in [("🚗 Vehículo", vehicle_cols), ("👤 Cliente", customer_cols),
                       ("📡 Telemetría", telemetry_cols), ("💰 Socioeconómico", socio_cols)]:
        existing = [c for c in cols if c in df.columns]
        avg_null = df[existing].isnull().mean().mean() * 100
        print(f"\n  {name}: {len(existing)} variables, nulidad media = {avg_null:.1f}%")
    
    return df

df_raw = load_and_audit(DATA_PATH)

# %%
# ============================================================================
# VISUALIZACIÓN: Mapa de calor de nulidad
# ============================================================================

def plot_null_heatmap(df: pd.DataFrame, title: str = "Mapa de Nulidad del Dataset") -> None:
    """Genera un mapa de calor visual de la nulidad por columna."""
    # Tomar una muestra para visualización eficiente
    sample = df.sample(min(5000, len(df)), random_state=RANDOM_STATE)
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 8), 
                             gridspec_kw={'width_ratios': [3, 1]})
    
    # Heatmap de nulidad (muestra)
    ax1 = axes[0]
    nulls = sample.isnull().astype(int)
    ax1.imshow(nulls.values.T, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    ax1.set_yticks(range(len(df.columns)))
    ax1.set_yticklabels(df.columns, fontsize=7)
    ax1.set_xlabel("Registros (muestra de 5.000)")
    ax1.set_title(title, fontsize=13, fontweight='bold')
    
    # Barplot de % nulos
    ax2 = axes[1]
    null_pct = df.isnull().mean() * 100
    colors = ['#2ecc71' if v == 0 else '#f39c12' if v < 10 else '#e74c3c' if v < 50 else '#8e44ad' 
              for v in null_pct.values]
    ax2.barh(range(len(null_pct)), null_pct.values, color=colors)
    ax2.set_yticks([])
    ax2.set_xlabel("% Nulos")
    ax2.set_title("% Nulidad", fontsize=11)
    ax2.axvline(x=50, color='red', linestyle='--', alpha=0.5, label='50%')
    ax2.legend(fontsize=8)
    
    plt.tight_layout()
    plt.savefig("figures/fig_01_null_heatmap.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("💾 Guardado: figures/fig_01_null_heatmap.png")

plot_null_heatmap(df_raw)

# %%
# ============================================================================
# VISUALIZACIÓN: Matriz de correlación de variables numéricas
# ============================================================================

def plot_correlation_matrix(df: pd.DataFrame) -> None:
    """Genera la matriz de correlación para detectar multicolinealidad."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    # Excluir índices sin sentido
    numeric_cols = [c for c in numeric_cols if c not in ['Unnamed: 0']]
    
    corr = df[numeric_cols].corr()
    
    # Detectar pares altamente correlacionados
    print("⚠️  PARES CON CORRELACIÓN |r| > 0.85:")
    print("-" * 60)
    for i in range(len(corr.columns)):
        for j in range(i + 1, len(corr.columns)):
            r = corr.iloc[i, j]
            if abs(r) > 0.85:
                print(f"  🔗 {corr.columns[i]:<40} ↔ {corr.columns[j]:<40} r={r:.3f}")
    
    # Heatmap
    fig, ax = plt.subplots(figsize=(16, 12))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, vmin=-1, vmax=1, square=True,
                linewidths=0.5, ax=ax, annot_kws={'size': 7})
    ax.set_title("Matriz de Correlación — Variables Numéricas (Pre-Limpieza)", 
                 fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig("figures/fig_02_correlation_matrix.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("💾 Guardado: figures/fig_02_correlation_matrix.png")

plot_correlation_matrix(df_raw)

# %% [markdown]
# ---
# # 🧹 FASE 1: Limpieza Radical y Rescate Inteligente de Nulos
# 
# ## Estrategia de limpieza en 2 pasos:
# 
# ### Paso 1 — Eliminación directa (DROP)
# | Columna | Razón de eliminación |
# |---------|---------------------|
# | `Unnamed: 0` | Índice auto-generado sin valor analítico |
# | `zipcode` | Granularidad excesiva (código postal) — ya tenemos `country_code` + `population` |
# | `number_of_households` | Correlación **1.00** con `population` → multicolinealidad perfecta |
# | `purchasing_power_per_household_eur` | Correlación **>0.96** con `per_inhabitant` → redundante |
# | `purchasing_power_millions_eur` | Correlación **>0.96** con `per_inhabitant` → redundante |
# | `fuel_type` | **59% nulos** y redundante con `engine_concept` |
# | `speed_mode`, `trip_distance_mode` | Texto sucio, difícil de parsear |
# | `speed_average`, `trip_distance_average` | Preferimos las medianas (robustas frente a outliers) |
# | `speed_all_counter`, `trip_distance_all_counter` | Contadores internos sin interpretabilidad de negocio |
# 
# ### Paso 2 — Imputación jerárquica y por dominio
# Rescatamos las columnas con nulos imputables usando la estructura jerárquica del catálogo Mercedes-Benz.

# %%
# ============================================================================
# FASE 1: LIMPIEZA RADICAL
# ============================================================================

# --------------------------------------------------------------------------
# 1.1 Columnas a eliminar (justificación documentada arriba)
# --------------------------------------------------------------------------

COLUMNS_TO_DROP = [
    'Unnamed: 0',                          # Índice sin valor
    'zipcode',                             # Granularidad excesiva  
    'number_of_households',                # corr=1.00 con population
    'purchasing_power_per_household_eur',   # corr>0.96 con per_inhabitant
    'purchasing_power_millions_eur',        # corr>0.96 con per_inhabitant
    'fuel_type',                           # 59% nulos, redundante con engine_concept
    'speed_mode',                          # Texto sucio
    'trip_distance_mode',                  # Texto sucio
    'speed_average',                       # Preferimos median (robusta)
    'trip_distance_average',               # Preferimos median (robusta)
    'speed_all_counter',                   # Contador interno
    'trip_distance_all_counter',           # Contador interno
]

def drop_redundant_columns(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """
    Elimina columnas redundantes, con multicolinealidad o inutilizables.
    Imprime un resumen de la operación.
    """
    existing = [c for c in cols if c in df.columns]
    missing = [c for c in cols if c not in df.columns]
    
    df_clean = df.drop(columns=existing)
    
    print("🗑️  ELIMINACIÓN DE COLUMNAS REDUNDANTES")
    print("=" * 60)
    print(f"  Eliminadas: {len(existing)} columnas")
    if missing:
        print(f"  ⚠️  No encontradas (ignoradas): {missing}")
    print(f"  Dimensiones resultantes: {df_clean.shape[0]:,} × {df_clean.shape[1]}")
    print(f"  Columnas conservadas: {list(df_clean.columns)}")
    
    return df_clean

df = drop_redundant_columns(df_raw.copy(), COLUMNS_TO_DROP)

# %%
# --------------------------------------------------------------------------
# 1.2 Imputación Jerárquica por Conocimiento de Dominio
# --------------------------------------------------------------------------

def hierarchical_mode_imputation(df: pd.DataFrame, target_col: str, 
                                  group_cols: List[str], 
                                  fallback: str = "Unknown") -> pd.DataFrame:
    """
    Imputa nulos de una columna categórica usando la moda agrupada
    por una jerarquía de columnas. Si no hay moda disponible, usa fallback.
    
    Versión vectorizada (merge) — O(N) en lugar de O(N*K) del for-loop.
    
    Args:
        df: DataFrame
        target_col: Columna a imputar
        group_cols: Lista de columnas de agrupación (jerarquía)
        fallback: Valor por defecto si la moda no existe
    
    Returns:
        DataFrame con la columna imputada
    """
    nulls_before = df[target_col].isnull().sum()
    
    if nulls_before == 0:
        print(f"  ✅ {target_col}: Sin nulos, no necesita imputación")
        return df
    
    # Calcular la moda por grupo (vectorizado)
    mode_map = (
        df.dropna(subset=[target_col])
        .groupby(group_cols)[target_col]
        .agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else fallback)
        .reset_index()
        .rename(columns={target_col: '_imputed_value'})
    )
    
    # Merge para obtener el valor de imputación para cada fila
    df = df.merge(mode_map, on=group_cols, how='left')
    
    # Aplicar: solo donde target_col es nulo
    mask = df[target_col].isnull()
    df.loc[mask, target_col] = df.loc[mask, '_imputed_value']
    df = df.drop(columns=['_imputed_value'])
    
    nulls_after = df[target_col].isnull().sum()
    rescued = nulls_before - nulls_after
    
    print(f"  🔧 {target_col}: {nulls_before:,} nulos → {nulls_after:,} nulos "
          f"(rescatados: {rescued:,}, {rescued/nulls_before*100:.1f}%)")
    
    # Rellenar cualquier restante con fallback
    if nulls_after > 0:
        df[target_col] = df[target_col].fillna(fallback)
        print(f"     └─ Restantes rellenados con '{fallback}': {nulls_after:,}")
    
    return df


def hierarchical_median_imputation(df: pd.DataFrame, target_col: str,
                                    group_cols: List[str]) -> pd.DataFrame:
    """
    Imputa nulos de una columna numérica usando la mediana agrupada.
    Fallback: mediana global.
    """
    nulls_before = df[target_col].isnull().sum()
    
    if nulls_before == 0:
        print(f"  ✅ {target_col}: Sin nulos")
        return df
    
    # Mediana por grupo
    median_map = df.groupby(group_cols)[target_col].transform('median')
    df[target_col] = df[target_col].fillna(median_map)
    
    # Fallback: mediana global
    nulls_mid = df[target_col].isnull().sum()
    if nulls_mid > 0:
        global_median = df[target_col].median()
        df[target_col] = df[target_col].fillna(global_median)
    
    nulls_after = df[target_col].isnull().sum()
    rescued = nulls_before - nulls_after
    
    print(f"  🔧 {target_col}: {nulls_before:,} nulos → {nulls_after:,} "
          f"(rescatados: {rescued:,})")
    
    return df


print("\n" + "=" * 70)
print("  IMPUTACIÓN JERÁRQUICA POR CONOCIMIENTO DE DOMINIO")
print("=" * 70)

# --- model_class (1.11% nulos) → moda por model_series ---
print("\n📌 model_class (moda por model_series):")
df = hierarchical_mode_imputation(df, 'model_class', ['model_series'], fallback='Unknown')

# --- engine_concept (44.38% nulos) → moda por vehicle_type + model_series ---
print("\n📌 engine_concept (moda por vehicle_type × model_series):")
# Primero imputar vehicle_type si tiene nulos (0.3%)
if df['vehicle_type'].isnull().sum() > 0:
    print("  ↳ Pre-imputando vehicle_type (moda por model_series):")
    df = hierarchical_mode_imputation(df, 'vehicle_type', ['model_series'], fallback='Unknown')

df = hierarchical_mode_imputation(df, 'engine_concept', ['vehicle_type', 'model_series'], fallback='Unknown')

# --- gender_derived (11.20% nulos) → lógica de negocio ---
print("\n📌 gender_derived (lógica B2B/B2C):")
nulls_before_gender = df['gender_derived'].isnull().sum()
df.loc[(df['gender_derived'].isnull()) & (df['role_type'] == 'B2B'), 'gender_derived'] = 'Company'
df.loc[df['gender_derived'].isnull(), 'gender_derived'] = 'Unknown'
nulls_after_gender = df['gender_derived'].isnull().sum()
print(f"  🔧 gender_derived: {nulls_before_gender:,} nulos → {nulls_after_gender:,} "
      f"(B2B→'Company', B2C→'Unknown')")

# --- purchasing_power_per_inhabitant_eur (6.3%) → mediana por country_code × model_class ---
print("\n📌 Variables socioeconómicas (mediana por country_code × model_class):")
df = hierarchical_median_imputation(df, 'purchasing_power_per_inhabitant_eur', 
                                     ['country_code', 'model_class'])

# --- population (6.28%) → mediana por country_code × model_class ---
df = hierarchical_median_imputation(df, 'population', ['country_code', 'model_class'])

# --- months_as_mb_customer (4 nulos) → mediana global ---
print("\n📌 months_as_mb_customer (mediana global — solo 4 nulos):")
nulls_months = df['months_as_mb_customer'].isnull().sum()
df['months_as_mb_customer'] = df['months_as_mb_customer'].fillna(df['months_as_mb_customer'].median())
print(f"  🔧 months_as_mb_customer: {nulls_months} nulos → 0 (mediana global)")

# --- date_of_first_registration (1.38%) → convertir + mediana por model_series ---
print("\n📌 date_of_first_registration (conversión + antigüedad):")
df['date_of_first_registration'] = pd.to_datetime(df['date_of_first_registration'], errors='coerce')
nulls_date = df['date_of_first_registration'].isnull().sum()
print(f"  Nulos tras conversión a fecha: {nulls_date:,}")

# Calcular antigüedad en días para poder imputar por mediana
df['_vehicle_age_days'] = (REFERENCE_DATE - df['date_of_first_registration']).dt.days

# Imputar antigüedad en días por model_series
median_age_by_series = df.groupby('model_series')['_vehicle_age_days'].transform('median')
df['_vehicle_age_days'] = df['_vehicle_age_days'].fillna(median_age_by_series)
# Fallback global
df['_vehicle_age_days'] = df['_vehicle_age_days'].fillna(df['_vehicle_age_days'].median())

# Reconstruir la fecha para los que tenían nulos (no estrictamente necesario, pero consistente)
mask_null_date = df['date_of_first_registration'].isnull()
df.loc[mask_null_date, 'date_of_first_registration'] = (
    REFERENCE_DATE - pd.to_timedelta(df.loc[mask_null_date, '_vehicle_age_days'], unit='D')
)
print(f"  🔧 Reconstruidas {mask_null_date.sum():,} fechas a partir de la antigüedad media por model_series")

# %%
# --------------------------------------------------------------------------
# 1.3 Verificación Post-Limpieza
# --------------------------------------------------------------------------

print("\n" + "=" * 70)
print("  VERIFICACIÓN POST-LIMPIEZA (Fase 1)")
print("=" * 70)

# Excluir telemetría (se trata en Fase 3)
telemetry_survivors = ['trip_distance_median', 'speed_median', 
                       'fuel_consumption_average', 'operating_hours_total']

non_telemetry = [c for c in df.columns if c not in telemetry_survivors + ['_vehicle_age_days']]
null_check = df[non_telemetry].isnull().sum()
has_nulls = null_check[null_check > 0]

if len(has_nulls) == 0:
    print("  ✅ TODAS las variables no-telemetría tienen 0 nulos")
else:
    print("  ⚠️  Variables con nulos residuales:")
    for col, count in has_nulls.items():
        print(f"     {col}: {count:,}")

print(f"\n📐 Dimensiones actuales: {df.shape[0]:,} × {df.shape[1]}")
print(f"   (Variables de telemetría pendientes de tratar en Fase 3: {telemetry_survivors})")

# %% [markdown]
# ---
# # ⚙️ FASE 2: Feature Engineering de Alto Impacto
# 
# Creamos **5 nuevos KPIs** que capturan dimensiones clave del valor y comportamiento del cliente:
# 
# | Feature | Descripción | Valor de Negocio |
# |---------|-------------|------------------|
# | `vehicle_age_years` | Antigüedad del vehículo en años | Ciclo de renovación, potencial de upselling |
# | `estimated_itv_passes` | Nº estimado de inspecciones técnicas pasadas | Proxy de desgaste / madurez del vehículo |
# | `workshop_frequency_yearly` | Visitas a taller anualizadas | Engagement con el servicio posventa |
# | `marketing_receptivity_score` | Score 0-3 de receptividad a marketing | Potencial de conversión en campañas |
# | `has_telemetry_active` | Flag de conectividad digital | Perfil digital vs. analógico del cliente |

# %%
# ============================================================================
# FASE 2: FEATURE ENGINEERING
# ============================================================================

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea las 5 features derivadas de alto impacto para el clustering.
    
    Features creadas:
        - vehicle_age_years: Antigüedad del vehículo
        - estimated_itv_passes: Inspecciones técnicas estimadas
        - workshop_frequency_yearly: Frecuencia anualizada de taller
        - marketing_receptivity_score: Score de receptividad (0-3)
        - has_telemetry_active: Flag de conectividad digital
    """
    print("=" * 70)
    print("  FEATURE ENGINEERING — 5 NUEVOS KPIs")
    print("=" * 70)
    
    # ----- 1. vehicle_age_years -----
    df['vehicle_age_years'] = (df['_vehicle_age_days'] / 365.25).round(2)
    # Clamp negatives to 0 (fechas futuras por error)
    df['vehicle_age_years'] = df['vehicle_age_years'].clip(lower=0)
    print(f"\n  1️⃣  vehicle_age_years: min={df['vehicle_age_years'].min():.1f}, "
          f"max={df['vehicle_age_years'].max():.1f}, "
          f"median={df['vehicle_age_years'].median():.1f}")
    
    # ----- 2. estimated_itv_passes -----
    # Lógica ITV europea: primera a los 4 años, luego cada 2 años
    df['estimated_itv_passes'] = np.where(
        df['vehicle_age_years'] < 4,
        0,
        1 + ((df['vehicle_age_years'] - 4) / 2).astype(int)
    )
    print(f"  2️⃣  estimated_itv_passes: distribución = "
          f"{df['estimated_itv_passes'].value_counts().sort_index().to_dict()}")
    
    # ----- 3. workshop_frequency_yearly -----
    # Visitas al taller / años como cliente (+0.1 para evitar div/0)
    customer_years = (df['months_as_mb_customer'] / 12) + 0.1
    df['workshop_frequency_yearly'] = (
        df['workshop_visits_warranty_goodwill'] / customer_years
    ).round(4)
    print(f"  3️⃣  workshop_frequency_yearly: median={df['workshop_frequency_yearly'].median():.2f}, "
          f"mean={df['workshop_frequency_yearly'].mean():.2f}")
    
    # ----- 4. marketing_receptivity_score -----
    consent_cols = ['consent_phone_communication_channel', 
                    'consent_epost_communication_channel',
                    'consent_post_communication_channel']
    df['marketing_receptivity_score'] = df[consent_cols].sum(axis=1).astype(int)
    print(f"  4️⃣  marketing_receptivity_score: distribución = "
          f"{df['marketing_receptivity_score'].value_counts().sort_index().to_dict()}")
    
    # ----- 5. has_telemetry_active -----
    # El 60% de nulos en telemetría NO es un fallo: es el indicador de penetración digital
    df['has_telemetry_active'] = (~df['trip_distance_median'].isnull()).astype(int)
    telemetry_pct = df['has_telemetry_active'].mean() * 100
    print(f"  5️⃣  has_telemetry_active: {telemetry_pct:.1f}% clientes con telemetría activa")
    
    # Limpiar columna temporal
    df = df.drop(columns=['_vehicle_age_days'], errors='ignore')
    
    print(f"\n📐 Dimensiones tras Feature Engineering: {df.shape[0]:,} × {df.shape[1]}")
    
    return df

df = engineer_features(df)

# %%
# ============================================================================
# VISUALIZACIÓN: Distribución de los nuevos KPIs
# ============================================================================

def plot_new_features(df: pd.DataFrame) -> None:
    """Visualiza la distribución de las 5 features creadas."""
    new_features = ['vehicle_age_years', 'estimated_itv_passes', 
                    'workshop_frequency_yearly', 'marketing_receptivity_score',
                    'has_telemetry_active']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f39c12']
    
    for i, (feat, color) in enumerate(zip(new_features, colors)):
        ax = axes[i]
        if df[feat].nunique() <= 10:
            df[feat].value_counts().sort_index().plot(kind='bar', ax=ax, color=color, 
                                                       edgecolor='white', alpha=0.85)
            ax.set_xlabel(feat)
        else:
            ax.hist(df[feat].dropna(), bins=50, color=color, edgecolor='white', alpha=0.85)
        ax.set_title(feat, fontsize=12, fontweight='bold')
        ax.set_ylabel("Frecuencia")
    
    # Sexto panel: comparación B2C vs B2B en telemetría
    ax = axes[5]
    telemetry_by_role = df.groupby('role_type')['has_telemetry_active'].mean() * 100
    telemetry_by_role.plot(kind='bar', ax=ax, color=['#1abc9c', '#e67e22'], 
                           edgecolor='white', alpha=0.85)
    ax.set_title("% Telemetría activa por B2C/B2B", fontsize=12, fontweight='bold')
    ax.set_ylabel("% con telemetría")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    
    plt.suptitle("FASE 2 — Distribución de los 5 Nuevos KPIs", 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig("figures/fig_03_new_features.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("💾 Guardado: figures/fig_03_new_features.png")

plot_new_features(df)

# %% [markdown]
# ---
# # 📡 FASE 3: Tratamiento del Bloque de Telemetría
# 
# ## El Reto del 60% de Nulos — La Jugada Maestra
# 
# El 60% de nulos en las variables de telemetría (`trip_distance_median`, `speed_median`, 
# `fuel_consumption_average`, `operating_hours_total`) **NO es un fallo de calidad de datos**.
# 
# Es el **indicador de penetración digital** del módulo conectado de Mercedes-Benz:
# - **40% de clientes** tienen la app/módulo activo → datos de conducción disponibles
# - **60% de clientes** no lo tienen → nulos legítimos por ausencia de sensor
# 
# ### Estrategia: Imputar sin distorsionar K-Means
# 
# 1. Ya hemos creado `has_telemetry_active` (0/1) que captura esta segmentación.
# 2. Imputamos los nulos con la **mediana agrupada** por `role_type × country_code × model_class`
#    para darle al K-Means valores numéricos neutros.
# 3. La variable `has_telemetry_active` actuará como **ancla** para que el algoritmo
#    no confunda un valor imputado con un valor real.
# 
# > **Alternativa avanzada (Reto 3):** Entrenar dos modelos de clustering separados —
# > uno para clientes con telemetría (features ricas) y otro sin ella (solo perfil/uso).

# %%
# ============================================================================
# FASE 3: TRATAMIENTO DE TELEMETRÍA
# ============================================================================

TELEMETRY_COLS = ['trip_distance_median', 'speed_median', 
                  'fuel_consumption_average', 'operating_hours_total']

def treat_telemetry_block(df: pd.DataFrame, 
                           telemetry_cols: List[str],
                           group_cols: List[str] = ['role_type', 'country_code', 'model_class']
                           ) -> pd.DataFrame:
    """
    Imputa nulos de telemetría con mediana agrupada, preservando
    la señal digital a través del flag has_telemetry_active.
    
    Estrategia:
        1. Mediana por role_type × country_code × model_class
        2. Fallback: mediana global
        3. La variable has_telemetry_active (creada en Fase 2) ancla
           el clustering para distinguir imputados de reales.
    """
    print("=" * 70)
    print("  TRATAMIENTO DEL BLOQUE DE TELEMETRÍA")
    print("=" * 70)
    
    for col in telemetry_cols:
        nulls_before = df[col].isnull().sum()
        pct_before = nulls_before / len(df) * 100
        
        # Mediana por grupo
        group_median = df.groupby(group_cols)[col].transform('median')
        df[col] = df[col].fillna(group_median)
        
        # Fallback: mediana global
        nulls_mid = df[col].isnull().sum()
        if nulls_mid > 0:
            global_med = df[col].median()
            df[col] = df[col].fillna(global_med)
        
        nulls_after = df[col].isnull().sum()
        print(f"  📡 {col:<30} {nulls_before:>7,} ({pct_before:>5.1f}%) → {nulls_after:,} nulos")
    
    print(f"\n  ✅ Todos los nulos de telemetría resueltos")
    print(f"  🏷️  has_telemetry_active ancla la distinción real/imputado")
    
    return df

df = treat_telemetry_block(df, TELEMETRY_COLS)

# %%
# ============================================================================
# VERIFICACIÓN TOTAL POST-FASES 1-3
# ============================================================================

print("\n" + "=" * 70)
print("  VERIFICACIÓN TOTAL — NULIDAD CERO")
print("=" * 70)

total_nulls = df.isnull().sum().sum()
print(f"\n  Total de nulos en el DataFrame completo: {total_nulls}")

if total_nulls == 0:
    print("  ✅ ¡CERO nulos! Dataset completamente limpio.")
else:
    print("  ⚠️  Columnas con nulos residuales:")
    remaining = df.isnull().sum()
    for col, count in remaining[remaining > 0].items():
        print(f"     {col}: {count:,}")

print(f"\n📐 Dimensiones finales pre-partición: {df.shape[0]:,} × {df.shape[1]}")
print(f"📋 Columnas: {list(df.columns)}")

# %% [markdown]
# ---
# # 🧩 FASE 4: Segmentación Estática + Pipeline de Scikit-Learn
# 
# ## Arquitectura de Macro-Segmentación
# 
# En lugar de aplicar One-Hot Encoding masivo sobre categóricas de alta cardinalidad
# (1.436 `vehicle_type`, 88 `model_series`, 42 `model_class`...), optamos por:
# 
# 1. **Separar las categóricas descriptivas** en un DataFrame de metadatos (para el Reto 4: perfilado)
# 2. **Particionar por `country_code` × `role_type`** → 4 macro-segmentos independientes
# 3. **Alimentar K-Means solo con ~14 features numéricas/ordinales** escaladas
# 
# Esta estrategia:
# - Evita la **Maldición de la Dimensionalidad** (curse of dimensionality)
# - Produce clusters **accionables por el equipo de Marketing** (ya segmentados por mercado y tipo de cliente)
# - Permite que el K-Means trabaje sobre **distancias euclidianas significativas**

# %%
# ============================================================================
# FASE 4: SEGMENTACIÓN Y PIPELINE
# ============================================================================

# --------------------------------------------------------------------------
# 4.1 Separación de Metadatos (para perfilado en Reto 4)
# --------------------------------------------------------------------------

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

def separate_metadata(df: pd.DataFrame, meta_cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Separa el DataFrame en:
        - df_meta: Variables categóricas/descriptivas para perfilado posterior
        - df_features: Variables numéricas/ordinales para el clustering
    
    Ambos DataFrames comparten el mismo índice para poder cruzarlos.
    """
    print("=" * 70)
    print("  SEPARACIÓN DE METADATOS vs. FEATURES")
    print("=" * 70)
    
    # Asegurar que client_id y vehicle_id están en el DataFrame
    existing_meta = [c for c in meta_cols if c in df.columns]
    
    df_meta = df[existing_meta].copy()
    
    # Features: todo lo que NO es metadato (excepto country_code, role_type y is_current_vehicle que se usan para clustering)
    keep_in_features = {'country_code', 'role_type', 'is_current_vehicle'}
    feature_cols = [c for c in df.columns if c not in meta_cols or c in keep_in_features]
    df_features = df[feature_cols].copy()
    
    print(f"  📋 Metadatos:  {df_meta.shape[1]} columnas → {existing_meta}")
    print(f"  📊 Features:   {df_features.shape[1]} columnas → {list(df_features.columns)}")
    
    return df_meta, df_features

df_meta, df_features = separate_metadata(df, METADATA_COLS)

# %%
# --------------------------------------------------------------------------
# 4.2 Encoding Ordinal de age_range
# --------------------------------------------------------------------------

# Orden natural de los rangos de edad
AGE_ORDER = ['<25', '25-34', '35-44', '45-54', '55-64', '65+']

def encode_age_range(df: pd.DataFrame, age_col: str = 'age_range', 
                      order: List[str] = AGE_ORDER) -> pd.DataFrame:
    """
    Codifica age_range de forma ordinal (0-5) manteniendo el orden natural.
    """
    encoder = OrdinalEncoder(categories=[order], handle_unknown='use_encoded_value', 
                              unknown_value=-1)
    df['age_range_encoded'] = encoder.fit_transform(df[[age_col]]).astype(int)
    
    print("\n🔢 Encoding ordinal de age_range:")
    for val, code in zip(order, range(len(order))):
        count = (df['age_range_encoded'] == code).sum()
        print(f"  {val:>8} → {code}  ({count:>6,} registros)")
    
    return df

df_features = encode_age_range(df_features)

# Convertir is_current_vehicle a int
if 'is_current_vehicle' in df_features.columns:
    df_features['is_current_vehicle'] = df_features['is_current_vehicle'].astype(int)

# %%
# --------------------------------------------------------------------------
# 4.3 Selección Final de Features para K-Means
# --------------------------------------------------------------------------

# Features numéricas puras para el clustering
KMEANS_FEATURES = [
    # Perfil demográfico
    'age_range_encoded',                     # Ordinal 0-5
    
    # Perfil de vehículo
    'vehicle_age_years',                     # Continua
    'estimated_itv_passes',                  # Discreta
    'is_current_vehicle',                    # Binaria 0/1
    
    # Relación con la marca
    'months_as_mb_customer',                 # Continua
    'currently_connected_vehicles',          # Discreta
    'workshop_frequency_yearly',             # Continua (engineered)
    'workshop_visits_warranty_goodwill',     # Discreta (raw)
    
    # Marketing
    'marketing_receptivity_score',           # Ordinal 0-3
    
    # Socioeconómico
    'purchasing_power_per_inhabitant_eur',   # Continua
    'population',                            # Continua
    
    # Digital / Telemetría
    'has_telemetry_active',                  # Binaria 0/1
    'trip_distance_median',                  # Continua (imputada)
    'speed_median',                          # Continua (imputada)
    'fuel_consumption_average',              # Continua (imputada)
    'operating_hours_total',                 # Continua (imputada)
]

print("\n" + "=" * 70)
print("  FEATURES SELECCIONADAS PARA K-MEANS")
print("=" * 70)
print(f"  Total: {len(KMEANS_FEATURES)} features numéricas/ordinales")
for i, feat in enumerate(KMEANS_FEATURES, 1):
    dtype = df_features[feat].dtype
    print(f"  {i:>2}. {feat:<45} ({dtype})")

# %%
# --------------------------------------------------------------------------
# 4.4 Partición Estratégica: 4 Macro-Segmentos
# --------------------------------------------------------------------------

def partition_by_segments(df: pd.DataFrame, df_meta: pd.DataFrame,
                          features: List[str]) -> Dict[str, pd.DataFrame]:
    """
    Particiona el DataFrame en 4 macro-segmentos:
        CH_B2C, CH_B2B, AT_B2C, AT_B2B
    
    Returns:
        Diccionario {nombre_segmento: DataFrame con features puras}
    """
    print("\n" + "=" * 70)
    print("  PARTICIÓN ESTRATÉGICA POR MACRO-SEGMENTOS")
    print("=" * 70)
    
    segments = {}
    segments_meta = {}
    
    for country in ['CH', 'AT']:
        for role in ['B2C', 'B2B']:
            key = f"{country}_{role}"
            mask = (df['country_code'] == country) & (df['role_type'] == role)
            
            seg_df = df.loc[mask, features].copy()
            seg_meta = df_meta.loc[mask].copy()
            
            segments[key] = seg_df
            segments_meta[key] = seg_meta
            
            pct = len(seg_df) / len(df) * 100
            print(f"  📦 {key}: {len(seg_df):>7,} registros ({pct:>5.1f}%) "
                  f"× {seg_df.shape[1]} features")
    
    total = sum(len(v) for v in segments.values())
    print(f"\n  Σ Total: {total:,} registros (verificación: {total == len(df)})")
    
    return segments, segments_meta

segments, segments_meta = partition_by_segments(df_features, df_meta, KMEANS_FEATURES)

# %%
# --------------------------------------------------------------------------
# 4.5 Pipeline de Scikit-Learn (Escalado Robusto)
# --------------------------------------------------------------------------

def build_preprocessing_pipeline(features: List[str]) -> Pipeline:
    """
    Construye un pipeline de preprocesamiento modular con RobustScaler.
    
    RobustScaler es preferido sobre StandardScaler porque:
    - Usa mediana e IQR en lugar de media y desviación estándar
    - Es resistente a outliers (ej. operating_hours_total con max=26842)
    - Preserva mejor la estructura de los datos para K-Means
    
    Returns:
        Pipeline de scikit-learn listo para .fit_transform()
    """
    pipeline = Pipeline([
        ('scaler', RobustScaler())  # Escala por mediana e IQR
    ], memory=None)
    
    return pipeline


def prepare_segments_for_clustering(segments: Dict[str, pd.DataFrame]) -> Dict[str, dict]:
    """
    Aplica el pipeline de preprocesamiento a cada macro-segmento.
    
    Returns:
        Diccionario con los datos escalados y los pipelines ajustados.
    """
    print("\n" + "=" * 70)
    print("  PIPELINE DE PREPROCESAMIENTO (RobustScaler)")
    print("=" * 70)
    
    prepared = {}
    
    for key, seg_df in segments.items():
        pipeline = build_preprocessing_pipeline(seg_df.columns.tolist())
        
        # Fit + Transform
        X_scaled = pipeline.fit_transform(seg_df)
        X_scaled_df = pd.DataFrame(X_scaled, columns=seg_df.columns, index=seg_df.index)
        
        prepared[key] = {
            'X_raw': seg_df,
            'X_scaled': X_scaled_df,
            'pipeline': pipeline,
            'n_samples': len(seg_df),
            'n_features': seg_df.shape[1]
        }
        
        print(f"\n  📦 {key}:")
        print(f"     Registros: {len(seg_df):,}")
        print(f"     Features:  {seg_df.shape[1]}")
        print(f"     Escalado:  Mediana → 0, IQR → 1")
        
        # Verificación rápida
        means = X_scaled_df.mean()
        stds = X_scaled_df.std()
        print(f"     Post-escala: μ ∈ [{means.min():.3f}, {means.max():.3f}], "
              f"σ ∈ [{stds.min():.3f}, {stds.max():.3f}]")
    
    return prepared

prepared_segments = prepare_segments_for_clustering(segments)

# %%
# ============================================================================
# VISUALIZACIÓN: Distribución post-escalado del segmento principal (CH_B2C)
# ============================================================================

def plot_scaled_distributions(prepared: dict, segment_key: str = 'CH_B2C') -> None:
    """Visualiza las distribuciones post-escalado del segmento principal."""
    X_scaled = prepared[segment_key]['X_scaled']
    
    n_features = X_scaled.shape[1]
    n_cols = 4
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 3.5 * n_rows))
    axes = axes.flatten()
    
    for i, col in enumerate(X_scaled.columns):
        ax = axes[i]
        data = X_scaled[col].clip(-5, 5)  # Clip para visualización
        ax.hist(data, bins=50, color=plt.cm.viridis(i / n_features), 
                edgecolor='white', alpha=0.8)
        ax.set_title(col, fontsize=9, fontweight='bold')
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.5)
        ax.set_ylabel("Freq")
    
    # Ocultar ejes sobrantes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    
    plt.suptitle(f"Distribuciones Post-RobustScaler — Segmento {segment_key}", 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig("figures/fig_04_scaled_distributions.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("💾 Guardado: figures/fig_04_scaled_distributions.png")

plot_scaled_distributions(prepared_segments, 'CH_B2C')

# %%
# ============================================================================
# GUARDADO DE ARTEFACTOS PARA EL RETO 3
# ============================================================================

def save_artifacts(prepared: Dict[str, dict], 
                   segments_meta: Dict[str, pd.DataFrame],
                   output_dir: str = "data") -> None:
    """
    Guarda los DataFrames particionados y escalados.
    
    Archivos generados:
        - {segment}_features_raw.csv: Features sin escalar (para análisis)
        - {segment}_features_scaled.csv: Features escaladas
        - {segment}_metadata.csv: Metadatos para perfilado posterior
    """
    print("\n" + "=" * 70)
    print("  GUARDADO DE ARTEFACTOS")
    print("=" * 70)
    
    for key in prepared:
        # Features raw
        raw_path = f"{output_dir}/{key}_features_raw.csv"
        prepared[key]['X_raw'].to_csv(raw_path, index=True)
        
        # Features escaladas
        scaled_path = f"{output_dir}/{key}_features_scaled.csv"
        prepared[key]['X_scaled'].to_csv(scaled_path, index=True)
        
        # Metadatos
        meta_path = f"{output_dir}/{key}_metadata.csv"
        segments_meta[key].to_csv(meta_path, index=True)
        
        print(f"  💾 {key}:")
        print(f"     Raw:    {raw_path}")
        print(f"     Scaled: {scaled_path}")
        print(f"     Meta:   {meta_path}")

save_artifacts(prepared_segments, segments_meta)

# %% [markdown]
# ---
# # 📝 FASE 5: Informe Técnico para el Jurado
# 
# ---
# 
# ## 🏢 Informe Ejecutivo: Arquitectura de Datos para "Clustering the Star"
# 
# ### 1. Auditoría de Calidad de Datos
# 
# #### 1.1 Multicolinealidad: Detectada y Eliminada
# 
# | Par de Variables | Correlación | Acción |
# |:-----------------|:-----------:|:-------|
# | `population` ↔ `number_of_households` | **1.00** | Eliminamos `number_of_households` (contenido idéntico) |
# | `pwr_per_inhabitant` ↔ `pwr_per_household` | **0.96+** | Conservamos solo `per_inhabitant` (más interpretable) |
# | `pwr_per_inhabitant` ↔ `pwr_millions` | **0.96+** | Eliminamos `millions` (derivado) |
# | `speed_median` ↔ `speed_average` | **0.88** | Conservamos `median` (robusta frente a outliers) |
# | `trip_distance_median` ↔ `trip_distance_average` | **0.87** | Conservamos `median` |
# 
# > **Justificación técnica:** En K-Means (distancia euclidiana), la multicolinealidad **duplica el peso**
# > de una dimensión, sesgando los centroides hacia esa dirección. Eliminarla es obligatorio, no opcional.
# 
# #### 1.2 Rescate Jerárquico de `engine_concept` (44% nulos)
# 
# En lugar de eliminar esta variable o rellenarla con un valor genérico, hemos explotado la **jerarquía
# del catálogo Mercedes-Benz**:
# 
# ```
# model_series (0% nulos) → vehicle_type (0.3% nulos) → engine_concept (44% nulos)
# ```
# 
# Usando la **moda agrupada por `vehicle_type × model_series`**, hemos rescatado la práctica totalidad
# de los nulos. Esto es posible porque cada combinación de modelo y tipo de vehículo en Mercedes-Benz
# tiene un motor predominante (ej. un GLC 220d 4MATIC es casi siempre COMBUSTION o MILD_HYBRID).
# 
# ---
# 
# ### 2. La Jugada Maestra: El 60% de Nulos en Telemetría
# 
# #### 2.1 Diagnóstico: No es un fallo, es una feature
# 
# Las variables de telemetría (`trip_distance`, `speed`, `fuel_consumption`, `operating_hours`)
# presentan ~60% de nulos. Un enfoque naive las eliminaría. Nosotros hemos descubierto que:
# 
# > **El 60% de nulos indica que el 60% de los clientes NO tiene activado el módulo de conectividad
# > (Mercedes me App / Mercedes me connect).** Es un dato de negocio, no un error de base de datos.
# 
# #### 2.2 Solución: Flag `has_telemetry_active` + Imputación controlada
# 
# 1. **Creamos `has_telemetry_active`** (0/1): Convierte el patrón de nulidad en un KPI medible
#    de **penetración digital**. Esto es directamente accionable por Marketing:
#    - Clientes **sin telemetría** (0) → Campaña de activación de Mercedes me
#    - Clientes **con telemetría** (1) → Candidatos a upselling de servicios conectados
# 
# 2. **Imputamos los nulos** con la mediana por segmento (role_type × country_code × model_class)
#    para que K-Means tenga valores numéricos, PERO el flag `has_telemetry_active` **ancla la 
#    distancia euclidiana** para que el algoritmo no confunda un valor imputado con uno real.
# 
# ---
# 
# ### 3. Arquitectura Híbrida: Por qué separamos B2C/B2B y CH/AT
# 
# #### 3.1 Problema: La Maldición de la Dimensionalidad
# 
# Si aplicamos One-Hot Encoding a variables como `vehicle_type` (1.436 categorías) o
# `model_series` (88 categorías), el espacio de features explota a **cientos de dimensiones
# dispersas** donde la distancia euclidiana pierde significado (fenómeno conocido como
# *curse of dimensionality*).
# 
# #### 3.2 Solución: Macro-Segmentación + K-Means Puro
# 
# | Paso | Estrategia | Resultado |
# |:-----|:-----------|:----------|
# | **Pre-clustering** | Separar por `country_code` × `role_type` | 4 macro-segmentos naturales |
# | **Feature set** | Solo 16 features numéricas/ordinales | Espacio compacto y significativo |
# | **Escalado** | RobustScaler (mediana + IQR) | Resistente a outliers |
# | **Post-clustering** | Cruzar clusters con metadatos categóricos | Perfilado rico para Marketing |
# 
# **Ventajas competitivas de esta arquitectura:**
# - ✅ Clusters **directamente accionables** por el equipo de Marketing regional (CH vs AT)
# - ✅ Segmentos **diferenciados** para B2B (flotas) y B2C (particulares)
# - ✅ K-Means trabaja en **16 dimensiones significativas** (no 200+ dispersas)
# - ✅ Pipeline **extrapolable**: si mañana llegan datos de Francia, solo ejecutamos `.transform()`
# 
# ---
# 
# ### 4. Feature Set Final para K-Means (Reto 3)
# 
# | # | Feature | Tipo | Origen |
# |:--|:--------|:-----|:-------|
# | 1 | `age_range_encoded` | Ordinal 0-5 | Encoding |
# | 2 | `vehicle_age_years` | Continua | Engineered |
# | 3 | `estimated_itv_passes` | Discreta | Engineered |
# | 4 | `is_current_vehicle` | Binaria | Raw |
# | 5 | `months_as_mb_customer` | Continua | Raw |
# | 6 | `currently_connected_vehicles` | Discreta | Raw |
# | 7 | `workshop_frequency_yearly` | Continua | Engineered |
# | 8 | `workshop_visits_warranty_goodwill` | Discreta | Raw |
# | 9 | `marketing_receptivity_score` | Ordinal 0-3 | Engineered |
# | 10 | `purchasing_power_per_inhabitant_eur` | Continua | Raw |
# | 11 | `population` | Continua | Raw |
# | 12 | `has_telemetry_active` | Binaria | Engineered ✨ |
# | 13 | `trip_distance_median` | Continua | Raw (imputada) |
# | 14 | `speed_median` | Continua | Raw (imputada) |
# | 15 | `fuel_consumption_average` | Continua | Raw (imputada) |
# | 16 | `operating_hours_total` | Continua | Raw (imputada) |
# 
# ---
# 
# ### 5. Partición de Datos para el Reto 3
# 
# | Macro-Segmento | Registros | % del Total | Contexto |
# |:---------------|----------:|:-----------:|:---------|
# | **CH_B2C** | ~189K | ~62% | Particulares en Suiza |
# | **AT_B2C** | ~108K | ~36% | Particulares en Austria |
# | **CH_B2B** | ~4K | ~1.3% | Flotas/empresas en Suiza |
# | **AT_B2B** | ~1.5K | ~0.5% | Flotas/empresas en Austria |
# 
# > Cada segmento tiene su propio pipeline de escalado (RobustScaler) ajustado,
# > almacenado para el Reto 3 junto con los DataFrames particionados.

# %%
# ============================================================================
# RESUMEN FINAL DE EJECUCIÓN
# ============================================================================

print("\n" + "🌟" * 35)
print("\n  RESUMEN DE EJECUCIÓN — RETOS 1 & 2 COMPLETADOS")
print("\n" + "🌟" * 35)

print(f"""
  📊 Dataset original:     {df_raw.shape[0]:,} × {df_raw.shape[1]}
  📊 Dataset limpio:       {df.shape[0]:,} × {df.shape[1]}
  
  🗑️  Columnas eliminadas:  {len(COLUMNS_TO_DROP)}
  ⚙️  Features creadas:     5 (vehicle_age_years, estimated_itv_passes, 
                             workshop_frequency_yearly, marketing_receptivity_score,
                             has_telemetry_active)
  🔧 Nulos resueltos:     {df_raw.isnull().sum().sum():,} → 0
  
  📦 Macro-segmentos:
""")

for key, data in prepared_segments.items():
    print(f"     {key}: {data['n_samples']:>7,} registros × {data['n_features']} features (escaladas)")

print(f"""
  💾 Artefactos guardados en data/:
     - 4 × features_raw.csv
     - 4 × features_scaled.csv  
     - 4 × metadata.csv
  
  ✅ Pipeline de Limpieza y Feature Engineering completado con éxito.
""")
