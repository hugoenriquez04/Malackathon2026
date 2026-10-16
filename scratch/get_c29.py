import json
import sys
sys.stdout.reconfigure(encoding='utf-8')

with open('notebooks/04_Interpretacion_Segmentos.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

c29 = nb['cells'][29]
for out in c29.get('outputs', []):
    if 'text' in out:
        print("".join(out['text']))
    if 'data' in out and 'text/plain' in out['data']:
        print("".join(out['data']['text/plain']))
