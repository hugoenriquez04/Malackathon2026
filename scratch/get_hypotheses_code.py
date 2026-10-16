import json
import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', 1000)

with open('notebooks/04_Interpretacion_Segmentos.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Let's inspect the code of function metadata_profiles and hypotheses in cell 9 and cell 10
print("=== CELL 9 ===")
print("".join(nb['cells'][9]['source']))
print("=== CELL 10 ===")
print("".join(nb['cells'][10]['source']))
