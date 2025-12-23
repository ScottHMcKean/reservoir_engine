"""Tests for uplift feature engineering functions."""

import pytest

from reservoir_engine.uplift.features import (
    build_feature_matrix,
    calculate_eur_targets,
    create_spatial_features_h3,
)


class TestBuildFeatureMatrix:
    """Tests for build_feature_matrix function."""

    def test_builds_from_production(
        self, sample_well_header_df, sample_production_monthly_df
    ):
        features = build_feature_matrix(
            sample_well_header_df,
            production_df=sample_production_monthly_df,
        )

        assert "cum_oil" in features.columns
        assert len(features) == 3  # One row per well

    def test_builds_from_all_sources(
        self,
        sample_well_header_df,
        sample_production_monthly_df,
        sample_completions_df,
    ):
        features = build_feature_matrix(
            sample_well_header_df,
            production_df=sample_production_monthly_df,
            completions_df=sample_completions_df,
        )

        assert "cum_oil" in features.columns
        assert "completion_count" in features.columns

    def test_includes_header_features(self, sample_well_header_df):
        features = build_feature_matrix(sample_well_header_df)

        assert "surface_latitude" in features.columns
        assert "surface_longitude" in features.columns
        assert "drill_td" in features.columns


class TestCalculateEurTargets:
    """Tests for calculate_eur_targets function."""

    def test_calculates_multiple_horizons(self, sample_production_monthly_df):
        targets = calculate_eur_targets(
            sample_production_monthly_df,
            months=[6, 12, 24],
        )

        assert "eur_6m" in targets.columns
        assert "eur_12m" in targets.columns
        assert "eur_24m" in targets.columns

    def test_eur_increases_with_time(self, sample_production_monthly_df):
        targets = calculate_eur_targets(
            sample_production_monthly_df,
            months=[6, 12, 24],
        )

        # EUR at 24 months should be >= EUR at 12 months
        assert (targets["eur_24m"] >= targets["eur_12m"]).all()


class TestCreateSpatialFeaturesH3:
    """Tests for create_spatial_features_h3 function."""

    def test_creates_h3_features(self, sample_well_header_df):
        spatial = create_spatial_features_h3(sample_well_header_df)

        assert "h3_index" in spatial.columns
        assert "wells_in_cell" in spatial.columns
        assert "h3_parent" in spatial.columns
        assert "wells_in_region" in spatial.columns
