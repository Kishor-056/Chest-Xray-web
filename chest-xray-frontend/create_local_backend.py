import os

input_file = "backend.py"
output_file = "local_backend.py"

with open(input_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()

start_idx = -1
for i, line in enumerate(lines):
    if "🎯 COMPLETE CHEST X-RAY API SERVER - PRODUCTION READY" in line:
        start_idx = i - 1 # start from the triple quote above it
        if '"""' not in lines[start_idx]:
            start_idx = i
        break

if start_idx != -1:
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(lines[start_idx:])
    print(f"Created {output_file} from line {start_idx}")
else:
    print("Could not find the start marker.")
