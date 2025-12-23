"""Uplift Analysis Page.

ML-based production prediction and spatial analysis.
"""

import sys
from pathlib import Path

# Ensure project root is in path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from app.database import get_backend


def render():
    """Render the Uplift Analysis page."""
    st.title("Uplift Analysis")
    st.markdown("Machine learning for production prediction and optimization")

    # Tabs
    tab1, tab2, tab3 = st.tabs(["Feature Engineering", "Model Training", "Spatial Prediction"])

    with tab1:
        render_feature_engineering()

    with tab2:
        render_model_training()

    with tab3:
        render_spatial_prediction()


def render_feature_engineering():
    """Render feature engineering interface."""
    from reservoir_engine.uplift import (
        build_feature_matrix,
        calculate_eur_targets,
        create_spatial_features_h3,
    )

    st.subheader("Feature Engineering")

    backend = get_backend()

    st.markdown(
        "Build a feature matrix combining well header, production, and completion data."
    )

    # Data sources
    st.markdown("#### Available Data Sources")

    col1, col2, col3 = st.columns(3)

    with col1:
        wells = backend.get_well_header()
        st.metric("Well Headers", len(wells))
        st.checkbox("Include Well Header", value=True, key="use_wells")

    with col2:
        # Sample production for count
        prod_sample = backend.get_production_monthly(wells["uwi"].head(10).tolist())
        st.metric("Production Records", f"~{len(prod_sample):,}")
        st.checkbox("Include Production", value=True, key="use_production")

    with col3:
        completions = backend.get_completions(wells["uwi"].head(10).tolist())
        st.metric("Completions", f"~{len(completions):,}")
        st.checkbox("Include Completions", value=True, key="use_completions")

    # Build features
    n_wells = st.slider("Number of Wells to Analyze", 10, 500, 100, 10)

    if st.button("Build Feature Matrix", type="primary"):
        with st.spinner("Building feature matrix..."):
            try:
                well_subset = wells.head(n_wells)
                uwis = well_subset["uwi"].tolist()

                production = backend.get_production_monthly(uwis) if st.session_state.get("use_production", True) else None
                completions_df = backend.get_completions(uwis) if st.session_state.get("use_completions", True) else None

                features = build_feature_matrix(
                    well_subset,
                    production_df=production,
                    completions_df=completions_df,
                )

                # Add spatial features
                if "surface_latitude" in features.columns:
                    spatial = create_spatial_features_h3(well_subset)
                    features = features.merge(spatial, on="uwi", how="left")

                # Calculate targets
                if production is not None and not production.empty:
                    targets = calculate_eur_targets(production, months=[12, 24, 60])
                    features = features.merge(targets, on="uwi", how="left")

                st.session_state.feature_matrix = features

                st.success(f"Built feature matrix: {features.shape[0]} wells, {features.shape[1]} features")

            except Exception as e:
                st.error(f"Error building features: {e}")

    # Display features
    if "feature_matrix" in st.session_state:
        features = st.session_state.feature_matrix

        st.markdown("#### Feature Matrix Preview")

        # Feature summary
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Features", features.shape[1])

        with col2:
            numeric_cols = features.select_dtypes(include=[np.number]).columns
            st.metric("Numeric Features", len(numeric_cols))

        with col3:
            missing = features.isnull().sum().sum() / (features.shape[0] * features.shape[1]) * 100
            st.metric("Missing Data", f"{missing:.1f}%")

        with st.expander("View Features"):
            st.dataframe(features.head(20), use_container_width=True, hide_index=True)

        # Feature distributions
        st.markdown("#### Feature Distributions")

        numeric_features = features.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_features:
            selected_feature = st.selectbox("Select Feature", numeric_features)

            fig = px.histogram(
                features,
                x=selected_feature,
                nbins=30,
                color_discrete_sequence=["#FF3621"],
            )
            st.plotly_chart(fig, use_container_width=True)


def render_model_training():
    """Render model training interface."""
    from reservoir_engine.uplift import train_production_model

    st.subheader("Model Training")

    if "feature_matrix" not in st.session_state:
        st.info("First build a feature matrix in the 'Feature Engineering' tab")
        return

    features = st.session_state.feature_matrix

    # Target selection
    target_cols = [c for c in features.columns if c.startswith("eur_") or c.startswith("cum_")]

    if not target_cols:
        st.warning("No target columns found. Ensure production data was included.")
        return

    col1, col2 = st.columns(2)

    with col1:
        target = st.selectbox("Target Variable", target_cols)

    with col2:
        model_type = st.selectbox("Model Type", ["xgboost", "random_forest", "linear"])

    # Feature selection
    feature_cols = [c for c in features.columns if c not in target_cols + ["uwi"]]
    numeric_features = features[feature_cols].select_dtypes(include=[np.number]).columns.tolist()

    st.markdown(f"**Available features:** {len(numeric_features)}")

    # Training
    test_size = st.slider("Test Set Size", 0.1, 0.4, 0.2, 0.05)

    if st.button("Train Model", type="primary"):
        with st.spinner("Training model..."):
            try:
                # Prepare data
                train_df = features.dropna(subset=[target])
                train_df = train_df.dropna(subset=numeric_features, thresh=len(numeric_features) // 2)

                if len(train_df) < 20:
                    st.error("Not enough data for training")
                    return

                # Train
                model, metrics, predictions = train_production_model(
                    train_df,
                    target_column=target,
                    feature_columns=numeric_features,
                    model_type=model_type,
                    test_size=test_size,
                )

                st.session_state.trained_model = model
                st.session_state.model_metrics = metrics
                st.session_state.model_predictions = predictions
                st.session_state.model_features = numeric_features

                st.success("Model trained successfully!")

            except Exception as e:
                st.error(f"Error training model: {e}")

    # Display results
    if "model_metrics" in st.session_state:
        metrics = st.session_state.model_metrics

        st.markdown("#### Model Performance")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("R² Score", f"{metrics.get('r2', 0):.3f}")

        with col2:
            st.metric("RMSE", f"{metrics.get('rmse', 0):,.0f}")

        with col3:
            st.metric("MAE", f"{metrics.get('mae', 0):,.0f}")

        # Predictions vs Actual
        predictions = st.session_state.model_predictions

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=predictions["actual"],
                y=predictions["predicted"],
                mode="markers",
                marker=dict(color="#FF3621", size=8, opacity=0.6),
            )
        )

        # Add 1:1 line
        min_val = min(predictions["actual"].min(), predictions["predicted"].min())
        max_val = max(predictions["actual"].max(), predictions["predicted"].max())
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode="lines",
                line=dict(color="#6E6E6E", dash="dash"),
                name="1:1 Line",
            )
        )

        fig.update_layout(
            title="Predictions vs Actual",
            xaxis_title="Actual",
            yaxis_title="Predicted",
        )

        st.plotly_chart(fig, use_container_width=True)

        # Feature importance
        if hasattr(st.session_state.trained_model, "feature_importances_"):
            st.markdown("#### Feature Importance")

            importance = pd.DataFrame(
                {
                    "feature": st.session_state.model_features,
                    "importance": st.session_state.trained_model.feature_importances_,
                }
            ).sort_values("importance", ascending=True).tail(15)

            fig = px.bar(
                importance,
                x="importance",
                y="feature",
                orientation="h",
                color_discrete_sequence=["#FF3621"],
            )
            st.plotly_chart(fig, use_container_width=True)


def render_spatial_prediction():
    """Render spatial prediction interface."""
    from reservoir_engine.uplift import create_prediction_grid_h3

    st.subheader("Spatial Prediction")

    if "trained_model" not in st.session_state:
        st.info("First train a model in the 'Model Training' tab")
        return

    backend = get_backend()
    wells = backend.get_well_header()

    st.markdown("Generate predictions on an H3 grid for visualization.")

    # Grid parameters
    col1, col2 = st.columns(2)

    with col1:
        if "surface_longitude" in wells.columns:
            valid = wells.dropna(subset=["surface_longitude", "surface_latitude"])
            min_lon = st.number_input(
                "Min Longitude",
                value=float(valid["surface_longitude"].min()),
            )
            max_lon = st.number_input(
                "Max Longitude",
                value=float(valid["surface_longitude"].max()),
            )

    with col2:
        if "surface_latitude" in wells.columns:
            min_lat = st.number_input(
                "Min Latitude",
                value=float(valid["surface_latitude"].min()),
            )
            max_lat = st.number_input(
                "Max Latitude",
                value=float(valid["surface_latitude"].max()),
            )

    resolution = st.selectbox("H3 Resolution", [6, 7, 8], index=1)

    if st.button("Generate Prediction Grid", type="primary"):
        with st.spinner("Generating predictions..."):
            try:
                # Create grid
                grid = create_prediction_grid_h3(min_lon, min_lat, max_lon, max_lat, resolution)

                st.session_state.prediction_grid = grid
                st.success(f"Created grid with {len(grid)} cells")

            except Exception as e:
                st.error(f"Error: {e}")

    # Display grid
    if "prediction_grid" in st.session_state:
        grid = st.session_state.prediction_grid

        st.markdown("#### Prediction Grid")

        # For now, just show the grid (actual predictions would require feature interpolation)
        fig = px.scatter_mapbox(
            grid,
            lat="latitude",
            lon="longitude",
            hover_data=["h3_index"],
            zoom=7,
            height=500,
            color_discrete_sequence=["#FF3621"],
        )
        fig.update_layout(mapbox_style="carto-positron")
        fig.update_traces(marker=dict(size=10, opacity=0.7))

        st.plotly_chart(fig, use_container_width=True)

        st.info(
            "Note: Full spatial prediction requires feature interpolation from nearby wells. "
            "This visualization shows the H3 grid structure."
        )

