"""Tests for uplift feature engineering functions."""

import pytest

from reservoir_engine.uplift.features import (
    build_feature_matrix,
    calculate_eur_targets,
    create_spatial_features,
)


class TestBuildFeatureMatrix:
    """Tests for build_feature_matrix function."""

    def test_builds_from_production(self, sample_wells_df, sample_production_df):
        features = build_feature_matrix(
            sample_wells_df,
            production_df=sample_production_df,
        )

        assert "cum_oil" in features.columns
        assert len(features) == 3  # One row per well

    def test_builds_from_all_sources(
        self, sample_wells_df, sample_production_df, sample_completions_df
    ):
        features = build_feature_matrix(
            sample_wells_df,
            production_df=sample_production_df,
            completions_df=sample_completions_df,
        )

        assert "cum_oil" in features.columns
        assert "stage_count" in features.columns


class TestCalculateEurTargets:
    """Tests for calculate_eur_targets function."""

    def test_calculates_multiple_horizons(self, sample_production_df):
        targets = calculate_eur_targets(
            sample_production_df,
            months=[6, 12, 24],
        )

        assert "eur_6m" in targets.columns
        assert "eur_12m" in targets.columns
        assert "eur_24m" in targets.columns

    def test_eur_increases_with_time(self, sample_production_df):
        targets = calculate_eur_targets(
            sample_production_df,
            months=[6, 12, 24],
        )

        # EUR at 24 months should be >= EUR at 12 months
        assert (targets["eur_24m"] >= targets["eur_12m"]).all()


class TestCreateSpatialFeatures:
    """Tests for create_spatial_features function."""

    def test_creates_density_features(self, sample_wells_df):
        spatial = create_spatial_features(sample_wells_df)

        assert "nearest_well_dist" in spatial.columns
        assert "well_density" in spatial.columns

