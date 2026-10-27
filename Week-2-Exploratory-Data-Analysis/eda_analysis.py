"""Expected Week 2 entry point; delegates to the existing Titanic EDA implementation."""
from pathlib import Path
import runpy

runpy.run_path(str(Path(__file__).with_name("titanic_eda.py")), run_name="__main__")
