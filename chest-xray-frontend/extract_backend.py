import json

notebook_path = "orginal1st (9).ipynb"
output_path = "backend.py"

try:
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    code_cells = []
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = "".join(cell['source'])
            code_cells.append(source)

    full_code = "\n\n".join(code_cells)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(full_code)

    print(f"Successfully extracted code to {output_path}")

except Exception as e:
    print(f"Error extracting code: {e}")
