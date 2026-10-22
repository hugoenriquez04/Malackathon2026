# Mercedes-Benz Hackathon: "Clustering the Star"

Customer segmentation project developed for the **Clustering the Star** challenge by **Mercedes-Benz Group Services Madrid** during **Malackathon 2026**.

The notebooks contain the complete workflow, from exploratory analysis and feature engineering to clustering, interpretation, dimensionality reduction, and stability analysis.

## Participants

- Javier Campos Córcoles
- Hugo Enríquez Jiménez
- Alberto Gallego Tamayo
- Jose Manuel Urbano Olmo

## Project Structure

```text
Malackathon/
├── figures/                               # Generated visualisations
│   ├── fig_01_null_heatmap.png
│   ├── fig_02_correlation_matrix.png
│   ├── fig_03_new_features.png
│   ├── fig_04_scaled_distributions.png
│   ├── fig_05_correlation_matrix_scaled.png
│   └── fig_06_correlation_matrix_final_14.png
├── notebooks/                             # Analysis and modelling notebooks
│   ├── 01_EDA.ipynb
│   ├── 02_Feature_Engineering.ipynb
│   ├── 03_KMeans.ipynb
│   ├── 04_Interpretacion_Segmentos.ipynb
│   ├── 05_Comparativa_Algoritmos.ipynb
│   ├── 06_PCA.ipynb
│   └── 07_Jackknife.ipynb
├── .gitignore
├── mercedes-benz-hackathon.code-workspace
├── requirements.txt
└── README.md
```

## Analysis Workflow

The notebooks are numbered in execution order:

1. **Exploratory Analysis (`01_EDA.ipynb`)**
   Assesses quality, missing values, distributions, correlations, and redundant variables.
2. **Feature Engineering (`02_Feature_Engineering.ipynb`)**
   Creates business-oriented KPIs, prepares the CH/AT and B2C/B2B segments, and scales the modelling features.
3. **K-Means Clustering (`03_KMeans.ipynb`)**
   Evaluates candidate values of *K* using clustering quality metrics and creates the main customer segments.
4. **Segment Interpretation (`04_Interpretacion_Segmentos.ipynb`)**
   Profiles the resulting customer archetypes and develops marketing campaign hypotheses.
5. **Algorithm Comparison (`05_Comparativa_Algoritmos.ipynb`)**
   Compares K-Means with Gaussian Mixture Models, agglomerative clustering, and DBSCAN.
6. **Principal Component Analysis (`06_PCA.ipynb`)**
   Reduces dimensionality and analyses the main directions of variance.
7. **Jackknife Stability Analysis (`07_Jackknife.ipynb`)**
   Evaluates segment stability and identifies suitable candidates for a pilot campaign.

## Setup

Create a Python virtual environment, activate it, and install the project dependencies:

```bash
python -m venv .venv
```

On Windows:

```powershell
.venv\Scripts\Activate.ps1
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

Open the project in VS Code with `mercedes-benz-hackathon.code-workspace` and run the notebooks in numerical order. The required Python packages are listed in `requirements.txt`, including pandas, NumPy, SciPy, scikit-learn, Matplotlib, Seaborn, Jupyter, and IPython Kernel.
