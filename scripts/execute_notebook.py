import json
import sys
import io
import os
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def run_notebook(nb_path: str):
    print(f"[INFO] Ejecutando notebook unificado con capturador de texto y graficos: {nb_path} ...")
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
            
            captured_images = []
            
            def custom_show(*args, **kwargs):
                for fignum in plt.get_fignums():
                    fig = plt.figure(fignum)
                    buf = io.BytesIO()
                    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
                    buf.seek(0)
                    b64_str = base64.b64encode(buf.read()).decode('ascii')
                    captured_images.append(b64_str)
                plt.close('all')

            try:
                # Si matplotlib ya fue importado, parchear show para capturar los graficos
                if 'plt' in global_env:
                    global_env['plt'].show = custom_show
                exec(source, global_env)
                if 'plt' in global_env:
                    global_env['plt'].show = custom_show
                    # Si quedaron figuras sin llamar a show
                    custom_show()
            except Exception as e:
                sys.stdout = old_stdout
                print(f"[ERROR] Error en celda {i+1}: {e}")
                raise
            finally:
                sys.stdout = old_stdout

            out_text = stdout_buf.getvalue()
            cell['execution_count'] = exec_count
            cell['outputs'] = []
            
            # 1. Salidas de texto (stdout)
            if out_text:
                cell['outputs'].append({
                    "name": "stdout",
                    "output_type": "stream",
                    "text": out_text.splitlines(keepends=True)
                })
            
            # 2. Salidas graficas incrustadas (display_data)
            for img_b64 in captured_images:
                cell['outputs'].append({
                    "output_type": "display_data",
                    "data": {
                        "image/png": img_b64,
                        "text/plain": ["<Figure size 1400x600 with Axes>"]
                    },
                    "metadata": {}
                })
            
            graficos_count = len(captured_images)
            print(f"  [OK] Celda {exec_count} ejecutada. Caracteres texto: {len(out_text)}, Graficos incrustados: {graficos_count}")
            exec_count += 1

    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print(f"[OK] Notebook completamente ejecutado y guardado con textos y graficos incrustados: {nb_path}")

if __name__ == '__main__':
    target = sys.argv[1] if len(sys.argv) > 1 else 'notebooks/Reto_2_Feature_Engineering.ipynb'
    run_notebook(target)
