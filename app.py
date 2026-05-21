"""
Root Streamlit entrypoint for Streamlit Cloud.
The actual dashboard code is inside streamlit.app/app.py.
"""

import sys
from pathlib import Path

APP_DIR = Path(__file__).parent / "streamlit.app"

if not APP_DIR.exists():
    raise FileNotFoundError(f"Dashboard folder not found: {APP_DIR}")

sys.path.insert(0, str(APP_DIR))

from app import main

main()