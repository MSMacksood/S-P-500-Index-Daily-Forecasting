"""Ensures the project root is importable as `src.*` regardless of how
pytest / a script / a notebook is invoked (belt-and-suspenders alongside the
pyproject.toml `pythonpath` setting)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
