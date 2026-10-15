"""
Conversor de script .py con celdas # %% a formato Jupyter .ipynb
"""
import json
import sys

def py_to_ipynb(input_path: str, output_path: str):
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split by cell markers
    lines = content.split('\n')
    cells = []
    current_cell_lines = []
    current_cell_type = 'code'
    
    for line in lines:
        if line.strip().startswith('# %% [markdown]'):
            # Save previous cell
            if current_cell_lines:
                cells.append((current_cell_type, current_cell_lines))
            current_cell_lines = []
            current_cell_type = 'markdown'
        elif line.strip() == '# %%':
            # Save previous cell
            if current_cell_lines:
                cells.append((current_cell_type, current_cell_lines))
            current_cell_lines = []
            current_cell_type = 'code'
        else:
            if current_cell_type == 'markdown':
                # Strip leading "# " from markdown lines
                if line.startswith('# '):
                    current_cell_lines.append(line[2:])
                elif line == '#':
                    current_cell_lines.append('')
                else:
                    current_cell_lines.append(line)
            else:
                current_cell_lines.append(line)
    
    # Don't forget the last cell
    if current_cell_lines:
        cells.append((current_cell_type, current_cell_lines))
    
    # Build notebook structure
    nb_cells = []
    for cell_type, cell_lines in cells:
        # Remove leading/trailing empty lines
        while cell_lines and cell_lines[0].strip() == '':
            cell_lines = cell_lines[1:]
        while cell_lines and cell_lines[-1].strip() == '':
            cell_lines = cell_lines[:-1]
        
        if not cell_lines:
            continue
        
        source = [line + '\n' for line in cell_lines]
        if source:
            source[-1] = source[-1].rstrip('\n')  # Last line without trailing newline
        
        cell = {
            "cell_type": cell_type,
            "metadata": {},
            "source": source
        }
        
        if cell_type == 'code':
            cell["execution_count"] = None
            cell["outputs"] = []
        
        nb_cells.append(cell)
    
    notebook = {
        "cells": nb_cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.11.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
    
    print(f"[OK] Notebook generado: {output_path}")
    print(f"   Celdas: {len(nb_cells)} ({sum(1 for c in nb_cells if c['cell_type']=='code')} code, "
          f"{sum(1 for c in nb_cells if c['cell_type']=='markdown')} markdown)")

if __name__ == '__main__':
    from pathlib import Path
    
    script_dir = Path(__file__).resolve().parent
    root_dir = script_dir.parent
    
    input_py = sys.argv[1] if len(sys.argv) > 1 else str(script_dir / 'Retos_1_2_EDA_FeatureEngineering.py')
    output_nb = sys.argv[2] if len(sys.argv) > 2 else str(root_dir / 'notebooks' / 'Retos_1_2_EDA_FeatureEngineering.ipynb')
    
    py_to_ipynb(input_py, output_nb)
