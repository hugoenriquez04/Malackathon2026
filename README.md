# 🌟 Mercedes-Benz Hackathon: "Clustering the Star"
### Pipeline Completo de Ciencia de Datos y Segmentación de Clientes

Repositorio oficial del proyecto **Mercedes-Benz Hackathon (Malackathon 2026)**. Contiene el pipeline íntegro de Data Science y Machine Learning para procesar, limpiar, enriquecer y segmentar la base de clientes de **Suiza (CH)** y **Austria (AT)**, tanto particulares (**B2C**) como corporativos (**B2B**), cubriendo más de **302.000 clientes**.

---

## 🗂️ Estructura del Proyecto

```
Malackathon/
│
├── 📓 notebooks/                               ← Suite completa de 7 Notebooks Jupyter
│   ├── 01_EDA.ipynb                            (Reto 1: Diagnóstico de calidad, imputación y limpieza)
│   ├── 02_Feature_Engineering.ipynb            (Reto 2: Creación de KPIs, particionado y escalado)
│   ├── 03_KMeans.ipynb                         (Reto 3: Segmentación base y selección óptima de K)
│   ├── 04_Interpretacion_Segmentos.ipynb       (Reto 4: Arquetipos de clientes e hipótesis de campañas)
│   ├── 05_Comparativa_Algoritmos.ipynb         (Reto 5: Benchmark con GMM, Agglomerative y DBSCAN)
│   ├── 06_PCA.ipynb                            (Reto 6: Reducción dimensional y explicabilidad)
│   └── 07_Jackknife.ipynb                      (Reto 7: Análisis de estabilidad y candidatos para piloto)
│
├── 🐍 scripts/                                 ← Pipelines ejecutables y utilidades auxiliares
│   ├── Retos_1_2_EDA_FeatureEngineering.py    (Pipeline consolidado Retos 1 y 2)
│   ├── Reto_2_Feature_Engineering.py          (Pipeline modular de ingeniería de variables)
│   ├── execute_notebook.py                     (Runner automatizado con captura de métricas/gráficos)
│   └── convert_to_ipynb.py                     (Conversor de scripts con celdas a .ipynb)
│
├── 📄 reports/                                 ← Informes ejecutivos y técnicos detallados
│   ├── INFORME_EJECUTIVO_RETOS_1_Y_2.md
│   └── INFORME_RETO_2_FEATURE_ENGINEERING.md
│
├── 📊 figures/                                 ← Gráficos y matrices generadas en alta resolución
│   ├── fig_01_null_heatmap.png
│   ├── fig_02_correlation_matrix.png
│   ├── fig_03_new_features.png
│   ├── fig_04_scaled_distributions.png
│   ├── fig_05_correlation_matrix_scaled.png
│   └── fig_06_correlation_matrix_final_14.png
│
├── 💾 data/                                    ← Datasets brutos y procesados (*.csv)
│   ├── datos_reto_mercedes_benz_malackathon.csv (Dataset maestro)
│   ├── reto1_dataset_limpio.csv                 (Dataset limpio tras Reto 1)
│   └── [AT|CH]_[B2B|B2C]_[raw|scaled|metadata].csv
│
├── mercedes-benz-hackathon.code-workspace      ← Configuración de workspace para VS Code / Cursor
├── requirements.txt                            ← Dependencias del entorno de Python
├── .gitignore                                  ← Reglas de exclusión de Git
└── README.md                                   ← Documentación principal del proyecto
```

---

## 🚀 Resumen del Pipeline Metodológico

1. **Reto 1 — EDA y Calidad de Datos (`01_EDA.ipynb`)**:
   - Tratamiento integral de **2.273.248 valores nulos (0 nulos residuales)**.
   - Eliminación de multicolinealidad perfecta y variables redundantes.
   - Imputación jerárquica de `engine_concept` recuperando más de 36.600 registros.
   - Creación del indicador `has_telemetry_active` para monetizar la huella digital.

2. **Reto 2 — Feature Engineering y Escalado (`02_Feature_Engineering.ipynb`)**:
   - Creación de 5 KPIs de negocio (`vehicle_age_years`, `estimated_itv_passes`, `workshop_frequency_yearly`, `marketing_receptivity_score`, `is_current_vehicle`).
   - Partición estratificada en 4 macro-segmentos (`CH_B2C`, `CH_B2B`, `AT_B2C`, `AT_B2B`).
   - Normalización robusta no paramétrica con `RobustScaler`.

3. **Retos 3 & 4 — Clustering K-Means e Interpretación (`03_KMeans.ipynb`, `04_Interpretacion_Segmentos.ipynb`)**:
   - Búsqueda sistemática de $K$ mediante Silhouette Score, Davies-Bouldin y Elbow Method.
   - Caracterización de arquetipos de clientes y diseño de hipótesis de campañas de marketing dirigidas.

4. **Retos 5, 6 & 7 — Benchmark, PCA y Robustez (`05_Comparativa_Algoritmos.ipynb`, `06_PCA.ipynb`, `07_Jackknife.ipynb`)**:
   - Comparación cruzada de algoritmos de agrupamiento (K-Means vs GMM vs Jerárquico vs DBSCAN).
   - Análisis de componentes principales (PCA) para comprender los vectores directores de la varianza.
   - Validación cruzada y estabilidad mediante remuestreo Jackknife para seleccionar clientes piloto.

---

## 🛠️ Instalación y Puesta en Marcha

### 1. Clonar el repositorio
```bash
git clone https://github.com/javicamcor/Malackathon.git
cd Malackathon
```

### 2. Crear y activar entorno virtual
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Instalar dependencias
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Ejecución de los Notebooks
Puedes abrir el workspace en VS Code / Cursor abriendo el archivo `mercedes-benz-hackathon.code-workspace` o ejecutar los notebooks directamente:
```bash
jupyter notebook notebooks/
```
