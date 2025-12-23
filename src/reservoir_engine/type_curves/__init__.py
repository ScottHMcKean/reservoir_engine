"""Type Curves module - Probabilistic type curve generation with H3 spatial indexing."""

from reservoir_engine.type_curves.aggregation import align_production, normalize_production
from reservoir_engine.type_curves.probabilistic import (
    fit_type_curve,
    generate_type_curve,
    sample_type_curve,
)
from reservoir_engine.type_curves.selection import (
    DEFAULT_RESOLUTION,
    add_h3_index,
    filter_wells_by_attributes,
    get_h3_index,
    get_h3_neighbors,
    get_h3_resolution_for_radius,
    wells_in_bbox,
    wells_in_h3_cells,
    wells_in_h3_ring,
    wells_in_polygon_h3,
)

__all__ = [
    # H3 Selection
    "get_h3_index",
    "add_h3_index",
    "wells_in_h3_cells",
    "wells_in_h3_ring",
    "wells_in_polygon_h3",
    "wells_in_bbox",
    "get_h3_neighbors",
    "get_h3_resolution_for_radius",
    "filter_wells_by_attributes",
    "DEFAULT_RESOLUTION",
    # Aggregation
    "normalize_production",
    "align_production",
    # Probabilistic
    "generate_type_curve",
    "fit_type_curve",
    "sample_type_curve",
]
