"""Tests for probabilistic type curve generation."""

import numpy as np
import pandas as pd
import pytest

from reservoir_engine.type_curves.aggregation import align_production
from reservoir_engine.type_curves.probabilistic import (
    generate_type_curve,
    sample_type_curve,
)


class TestGenerateTypeCurve:
    """Tests for generate_type_curve function."""

    def test_generates_percentiles(self, sample_production_df):
        # First align the data
        aligned = align_production(sample_production_df)

        type_curve = generate_type_curve(
            aligned,
            percentiles=[10, 50, 90],
            min_wells_per_month=2,
        )

        assert "p10" in type_curve.columns
        assert "p50" in type_curve.columns
        assert "p90" in type_curve.columns

    def test_p10_less_than_p90(self, sample_production_df):
        aligned = align_production(sample_production_df)
        type_curve = generate_type_curve(aligned, min_wells_per_month=2)

        # P10 should be less than P90 (lower percentile = lower value)
        assert (type_curve["p10"] <= type_curve["p90"]).all()


class TestSampleTypeCurve:
    """Tests for sample_type_curve function."""

    def test_generates_samples(self, sample_production_df):
        aligned = align_production(sample_production_df)
        samples = sample_type_curve(aligned, n_samples=100, method="bootstrap")

        assert "sample_id" in samples.columns
        assert samples["sample_id"].nunique() == 100

