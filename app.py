"""
Streamlit Community Cloud Root Entrypoint
Loads and executes the canonical application from app/app.py
"""
import os
import sys

# Ensure repository root is on sys.path
repo_root = os.path.dirname(os.path.abspath(__file__))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# Execute app/app.py in this context
app_path = os.path.join(repo_root, "app", "app.py")
with open(app_path, "r", encoding="utf-8") as f:
    code = compile(f.read(), app_path, "exec")
    exec(code, globals())
