"""Uplift module - ML-based production prediction and mapping."""

from reservoir_engine.uplift.features import (
    build_feature_matrix,
    calculate_eur_targets,
    create_spatial_features,
)
from reservoir_engine.uplift.inference import create_prediction_grid, predict_on_grid
from reservoir_engine.uplift.models import evaluate_model, train_production_model

__all__ = [
    # Features
    "build_feature_matrix",
    "calculate_eur_targets",
    "create_spatial_features",
    # Models
    "train_production_model",
    "evaluate_model",
    # Inference
    "predict_on_grid",
    "create_prediction_grid",
]

