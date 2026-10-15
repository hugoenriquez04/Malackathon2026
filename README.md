# 🌟 Mercedes-Benz Hackathon: "Clustering the Star"
### Workspace Oficial del Proyecto — Retos 1 y 2 Completados

Bienvenido al espacio de trabajo del proyecto **Mercedes-Benz Hackathon**. Este repositorio contiene todo el pipeline de Data Science y Machine Learning para preparar y segmentar la base de clientes de **Suiza (CH)** y **Austria (AT)**, tanto particulares (**B2C**) como empresas (**B2B**), cubriendo más de **302.000 clientes**.

---

## 🗂️ Estructura del Workspace

El workspace está organizado en módulos claros para máxima reproducibilidad y facilidad de navegación:

```
mercedes-hackathon/
│
├── mercedes-benz-hackathon.code-workspace  ← Abrir este archivo en el IDE como Workspace
├── README.md                               ← Este documento guía
│
├── 📓 notebooks/                           ← Notebooks Jupyter nativos ejecutables
│   └── Retos_1_2_EDA_FeatureEngineering.ipynb
│
├── 🐍 scripts/                             ← Scripts Python ejecutables directamente
│   ├── Retos_1_2_EDA_FeatureEngineering.py
│   └── convert_to_ipynb.py
│
├── 📄 reports/                             ← Informes ejecutivos y técnicos para el jurado
│   └── INFORME_EJECUTIVO_RETOS_1_Y_2.md
│
├── 📊 figures/                             ← Visualizaciones de alta resolución generadas
│   ├── fig_01_null_heatmap.png             (Matriz de nulidad antes de limpieza)
│   ├── fig_02_correlation_matrix.png       (Detección de multicolinealidad)
│   ├── fig_03_new_features.png             (Distribución de los 5 nuevos KPIs)
│   └── fig_04_scaled_distributions.png     (Features tras RobustScaler)
│
└── 💾 data/                                ← Datasets brutos y procesados
    ├── datos_reto_mercedes_benz_malackathon.csv  (Dataset maestro ~100 MB)
    │
    ├── CH_B2C_features_raw.csv & _scaled.csv & _metadata.csv
    ├── CH_B2B_features_raw.csv & _scaled.csv & _metadata.csv
    ├── AT_B2C_features_raw.csv & _scaled.csv & _metadata.csv
    └── AT_B2B_features_raw.csv & _scaled.csv & _metadata.csv
```

---

## 🚀 Resumen del Pipeline

### Reto 1: EDA y Limpieza de Datos
- **2.273.248 valores nulos eliminados (0 nulos residuales)**.
- **Multicolinealidad perfecta** eliminada: `number_of_households` ($r=1.00$ con población) y variables redundantes de poder adquisitivo ($r>0.96$).
- **Imputación jerárquica de motorización (`engine_concept`)**: Se recuperaron **36.614 registros** utilizando la combinación `model_series` + `vehicle_type`.
- **Ancla Digital**: Creación de `has_telemetry_active` para transformar el 60% de nulos en telemetría en un diferenciador clave de comportamiento digital.

### Reto 2: Feature Engineering & Escalado
- Creación de 5 KPIs:
  1. `vehicle_age_years`
  2. `estimated_itv_passes`
  3. `workshop_frequency_yearly`
  4. `marketing_receptivity_score`
  5. `is_current_vehicle`
- Partición en 4 macro-segmentos para evitar la maldición de la dimensionalidad.
- Escalado no paramétrico con `RobustScaler` (resistente a outliers extremos).

---

## 🛠️ Cómo Abrir y Usar el Workspace

### En Antigravity IDE / VS Code:
1. En el menú superior: **File → Open Workspace from File...**
2. Selecciona: `C:\Users\javic\.gemini\antigravity-ide\scratch\mercedes-hackathon\mercedes-benz-hackathon.code-workspace`
3. O abre directamente la carpeta: `C:\Users\javic\.gemini\antigravity-ide\scratch\mercedes-hackathon`

### Ejecución de Scripts:
```bash
# Ejecutar pipeline de Limpieza y Feature Engineering (Retos 1 y 2)
python scripts/Retos_1_2_EDA_FeatureEngineering.py
```
