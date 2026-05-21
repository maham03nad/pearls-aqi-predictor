"""
Root Streamlit entrypoint.

Streamlit Cloud runs this file by default.
It forwards execution to the actual dashboard file:
    streamlit.app/app.py
"""

import runpy
from pathlib import Path


APP_PATH = Path(__file__).parent / "streamlit.app" / "app.py"

if not APP_PATH.exists():
    raise FileNotFoundError(f"Dashboard file not found: {APP_PATH}")

runpy.run_path(str(APP_PATH), run_name="__main__")