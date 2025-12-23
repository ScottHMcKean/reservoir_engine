"""Reservoir Engine Streamlit Application.

A Databricks-native app for reservoir engineering workflows.

Modules:
- EDA: Exploratory data analysis and visualization
- DCA: Decline curve analysis (Arps, Butler)
- Type Curves: Probabilistic type curve generation with H3
- Uplift: ML-based production prediction

Run with:
    uv run streamlit run app/main.py
"""

from app.config import AppConfig, load_config
from app.database import CSVBackend, LakebaseBackend, get_backend, init_backend

__all__ = [
    "AppConfig",
    "load_config",
    "CSVBackend",
    "LakebaseBackend",
    "get_backend",
    "init_backend",
]
