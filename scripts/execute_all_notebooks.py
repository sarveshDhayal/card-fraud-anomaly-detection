"""Execute all 11 notebooks in-process and populate cell outputs."""

import os
import sys

# Ensure repository root is on sys.path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import glob
from scripts.execute_notebook_inprocess import execute_notebook

notebook_files = sorted(glob.glob("notebooks/[0-1][0-9]_*.ipynb"))
print(f"Found {len(notebook_files)} modular notebooks to execute.")

for nb_file in notebook_files:
    print(f"\n---> Executing {nb_file}...")
    try:
        execute_notebook(nb_file, nb_file)
    except Exception as e:
        print(f"Error in {nb_file}: {e}")

print("\nAll modular notebooks executed successfully!")
