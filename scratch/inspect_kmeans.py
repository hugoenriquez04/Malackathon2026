import json, sys
sys.stdout.reconfigure(encoding='utf-8')

with open('notebooks/03_KMeans.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, c in enumerate(nb['cells']):
    src = ''.join(c.get('source', []))
    if any(k in src.lower() for k in ['criterio', 'seleccion', 'silhouette', 'elbow', 'codo', 'davies', 'calinski', 'justific', 'prueba', 'decision', 'k=']):
        print(f"=== CELL {i} ({c['cell_type']}) ===")
        print(src[:600])
        print("---")
