import json
import sys
sys.stdout.reconfigure(encoding='utf-8')

with open('notebooks/04_Interpretacion_Segmentos.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Inspect cell 31 outputs in detail
c31 = nb['cells'][31]
for out in c31.get('outputs', []):
    if 'data' in out:
        for k, v in out['data'].items():
            if k == 'text/plain' or k == 'text/markdown':
                print(f"--- OUTPUT {k} ---")
                print(''.join(v))
