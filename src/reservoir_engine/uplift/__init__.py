"""Uplift module - ML-based production prediction and mapping with H3."""

from reservoir_engine.uplift.features import (
    build_feature_matrix,
    calculate_eur_targets,
    create_spatial_features_h3,
)
from reservoir_engine.uplift.inference import (
    create_prediction_grid_h3,
    export_h3_hexagons_geojson,
    export_to_geojson,
    predict_on_grid,
)
from reservoir_engine.uplift.models import evaluate_model, train_production_model

__all__ = [
    # Features
    "build_feature_matrix",
    "calculate_eur_targets",
    "create_spatial_features_h3",
    # Models
    "train_production_model",
    "evaluate_model",
    # Inference
    "predict_on_grid",
    "create_prediction_grid_h3",
    "export_to_geojson",
    "export_h3_hexagons_geojson",
]
