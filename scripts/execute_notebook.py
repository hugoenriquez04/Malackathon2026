import json
import sys
import io
import os

def run_notebook(nb_path: str):
    print(f"[INFO] Ejecutando notebook: {nb_path} ...")
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    global_env = {}
    exec_count = 1

    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            source = "".join(cell['source'])
            stdout_buf = io.StringIO()
            old_stdout = sys.stdout
            sys.stdout = stdout_buf
            
            try:
                exec(source, global_env)
            except Exception as e:
                sys.stdout = old_stdout
                print(f"[ERROR] Error en celda {i+1}: {e}")
                raise
            finally:
                sys.stdout = old_stdout

            out_text = stdout_buf.getvalue()
            cell['execution_count'] = exec_count
            cell['outputs'] = []
            
            if out_text:
                cell['outputs'].append({
                    "name": "stdout",
                    "output_type": "stream",
                    "text": out_text.splitlines(keepends=True)
                })
            
            print(f"  [OK] Celda {exec_count} ejecutada. Caracteres de salida: {len(out_text)}")
            exec_count += 1

    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print(f"[OK] Notebook guardado con todas sus salidas de ejecucion: {nb_path}")

if __name__ == '__main__':
    target = sys.argv[1] if len(sys.argv) > 1 else 'notebooks/Reto_2_Feature_Engineering.ipynb'
    run_notebook(target)
