import json
import sys
sys.stdout.reconfigure(encoding='utf-8')


with open('notebooks/04_Interpretacion_Segmentos.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, c in enumerate(nb['cells']):
    src = ''.join(c.get('source', []))
    print(f"=== CELL {i} ({c['cell_type']}) ===")
    print(src[:600])
    for out in c.get('outputs', []):
        if 'text' in out:
            print("OUTPUT:", ''.join(out['text'])[:300])
        elif 'data' in out and 'text/plain' in out['data']:
            print("DATA PLAIN:", ''.join(out['data']['text/plain'])[:300])
