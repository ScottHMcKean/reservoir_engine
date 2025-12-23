"""Type Curves Page.

Generate probabilistic type curves using H3 spatial selection.
"""

import sys
from pathlib import Path

# Ensure project root is in path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import streamlit as st
import plotly.graph_objects as go

from app.database import get_backend


def render():
    """Render the Type Curves page."""
    st.title("Type Curve Generation")
    st.markdown("Generate probabilistic type curves from analog wells")

    # Tabs
    tab1, tab2, tab3 = st.tabs(["Well Selection", "Type Curve Analysis", "Export"])

    with tab1:
        render_well_selection()

    with tab2:
        render_type_curve_analysis()

    with tab3:
        render_export()


def render_well_selection():
    """Render H3-based well selection."""
    from reservoir_engine.type_curves import (
        wells_in_bbox,
        wells_in_h3_ring,
        filter_wells_by_attributes,
    )

    st.subheader("Select Analog Wells")

    backend = get_backend()
    wells = backend.get_well_header()

    if wells.empty:
        st.warning("No well data available")
        return

    # Selection method
    method = st.radio(
        "Selection Method",
        ["Bounding Box", "H3 Ring (Radius)", "Attribute Filter"],
        horizontal=True,
    )

    selected_uwis = []

    if method == "Bounding Box":
        st.markdown("#### Define Bounding Box")

        # Get data extent
        if "surface_longitude" in wells.columns and "surface_latitude" in wells.columns:
            valid_wells = wells.dropna(subset=["surface_longitude", "surface_latitude"])

            col1, col2 = st.columns(2)

            with col1:
                min_lon = st.number_input(
                    "Min Longitude",
                    value=float(valid_wells["surface_longitude"].min()),
                    format="%.4f",
                )
                min_lat = st.number_input(
                    "Min Latitude",
                    value=float(valid_wells["surface_latitude"].min()),
                    format="%.4f",
                )

            with col2:
                max_lon = st.number_input(
                    "Max Longitude",
                    value=float(valid_wells["surface_longitude"].max()),
                    format="%.4f",
                )
                max_lat = st.number_input(
                    "Max Latitude",
                    value=float(valid_wells["surface_latitude"].max()),
                    format="%.4f",
                )

            if st.button("Select Wells in Box", type="primary"):
                selected_uwis = wells_in_bbox(
                    wells, min_lon, min_lat, max_lon, max_lat
                )
                st.session_state.selected_uwis = selected_uwis

    elif method == "H3 Ring (Radius)":
        st.markdown("#### Select by H3 Ring")

        col1, col2, col3 = st.columns(3)

        with col1:
            center_lat = st.number_input(
                "Center Latitude",
                value=54.3,
                format="%.4f",
            )

        with col2:
            center_lon = st.number_input(
                "Center Longitude",
                value=-116.7,
                format="%.4f",
            )

        with col3:
            k_rings = st.slider("K-Rings (radius)", 1, 10, 3)
            resolution = st.selectbox("H3 Resolution", [7, 8, 9], index=1)

        if st.button("Select Wells in Ring", type="primary"):
            try:
                selected_uwis = wells_in_h3_ring(
                    wells,
                    center_lat,
                    center_lon,
                    k_rings=k_rings,
                    resolution=resolution,
                )
                st.session_state.selected_uwis = selected_uwis
            except Exception as e:
                st.error(f"Error: {e}")

    else:  # Attribute Filter
        st.markdown("#### Filter by Attributes")

        col1, col2 = st.columns(2)

        with col1:
            hole_direction = st.selectbox(
                "Well Type",
                ["All", "HORIZONTAL", "VERTICAL", "DIRECTIONAL"],
            )
            hole_direction = None if hole_direction == "All" else hole_direction

        with col2:
            if "play_type" in wells.columns:
                play_types = ["All"] + wells["play_type"].dropna().unique().tolist()
                play_type = st.selectbox("Play Type", play_types)
                play_type = None if play_type == "All" else play_type
            else:
                play_type = None

        if "lateral_length" in wells.columns:
            min_lateral = st.slider(
                "Min Lateral Length (m)",
                0,
                5000,
                0,
                100,
            )
            min_lateral = min_lateral if min_lateral > 0 else None
        else:
            min_lateral = None

        if st.button("Apply Filter", type="primary"):
            selected_uwis = filter_wells_by_attributes(
                wells,
                hole_direction=hole_direction,
                play_type=play_type,
                min_lateral_length=min_lateral,
            )
            st.session_state.selected_uwis = selected_uwis

    # Display selection results
    if "selected_uwis" in st.session_state and st.session_state.selected_uwis:
        selected = st.session_state.selected_uwis

        st.success(f"Selected {len(selected)} wells")

        # Map visualization
        selected_wells = wells[wells["uwi"].isin(selected)]

        if "surface_latitude" in selected_wells.columns:
            import plotly.express as px

            fig = px.scatter_mapbox(
                selected_wells.dropna(subset=["surface_latitude", "surface_longitude"]),
                lat="surface_latitude",
                lon="surface_longitude",
                hover_name="uwi",
                zoom=8,
                height=400,
                color_discrete_sequence=["#FF3621"],
            )
            fig.update_layout(mapbox_style="carto-positron")
            st.plotly_chart(fig, use_container_width=True)

        # Selection summary
        with st.expander("View Selected Wells"):
            display_cols = ["uwi", "hole_direction", "target_formation", "lateral_length"]
            display_cols = [c for c in display_cols if c in selected_wells.columns]
            st.dataframe(selected_wells[display_cols], use_container_width=True, hide_index=True)


def render_type_curve_analysis():
    """Render type curve generation and analysis."""
    from reservoir_engine.type_curves import (
        align_production,
        generate_type_curve,
    )

    st.subheader("Generate Type Curve")

    if "selected_uwis" not in st.session_state or not st.session_state.selected_uwis:
        st.info("First select wells from the 'Well Selection' tab")
        return

    selected_uwis = st.session_state.selected_uwis
    st.markdown(f"**Analyzing {len(selected_uwis)} wells**")

    backend = get_backend()

    # Load production data
    with st.spinner("Loading production data..."):
        production = backend.get_production_monthly(selected_uwis[:100])  # Limit for performance

    if production.empty:
        st.warning("No production data for selected wells")
        return

    # Configuration
    col1, col2, col3 = st.columns(3)

    with col1:
        phase = st.selectbox("Production Phase", ["oil", "gas"], key="tc_phase")

    with col2:
        align_by = st.selectbox(
            "Align By",
            ["first_production", "peak_production"],
        )

    with col3:
        percentiles = st.multiselect(
            "Percentiles",
            [10, 25, 50, 75, 90],
            default=[10, 50, 90],
        )

    min_wells = st.slider("Min Wells per Month", 2, 10, 3)

    if st.button("Generate Type Curve", type="primary"):
        with st.spinner("Generating type curve..."):
            try:
                # Align production
                aligned = align_production(
                    production,
                    align_by=align_by,
                    rate_column=phase,
                )

                # Generate type curve
                type_curve = generate_type_curve(
                    aligned,
                    percentiles=percentiles,
                    rate_column=phase,
                    min_wells_per_month=min_wells,
                )

                st.session_state.type_curve = type_curve
                st.session_state.aligned_production = aligned

            except Exception as e:
                st.error(f"Error generating type curve: {e}")
                return

    # Display results
    if "type_curve" in st.session_state:
        type_curve = st.session_state.type_curve
        aligned = st.session_state.aligned_production

        st.subheader("Type Curve Results")

        # Plot
        fig = go.Figure()

        # Add percentile lines
        colors = {"p10": "#6E6E6E", "p50": "#FF3621", "p90": "#1B1B1B"}

        for col in type_curve.columns:
            if col.startswith("p"):
                color = colors.get(col, "#6E6E6E")
                fig.add_trace(
                    go.Scatter(
                        x=type_curve["month"],
                        y=type_curve[col],
                        mode="lines",
                        name=col.upper(),
                        line=dict(color=color, width=2 if col == "p50" else 1),
                    )
                )

        # Add individual wells as faint lines
        if st.checkbox("Show Individual Wells", value=False):
            for uwi in aligned["uwi"].unique()[:20]:
                uwi_data = aligned[aligned["uwi"] == uwi].sort_values("month")
                fig.add_trace(
                    go.Scatter(
                        x=uwi_data["month"],
                        y=uwi_data[phase],
                        mode="lines",
                        name=uwi[-10:],
                        line=dict(color="#CCCCCC", width=0.5),
                        showlegend=False,
                    )
                )

        fig.update_layout(
            title="Probabilistic Type Curve",
            xaxis_title="Months on Production",
            yaxis_title=f"{phase.title()} Rate",
            yaxis_type="log" if st.checkbox("Log Scale", value=True, key="tc_log") else "linear",
            hovermode="x unified",
        )

        st.plotly_chart(fig, use_container_width=True)

        # Statistics
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Wells in Analysis", aligned["uwi"].nunique())

        with col2:
            st.metric("Months Analyzed", type_curve["month"].max())

        with col3:
            if "p50" in type_curve.columns:
                p50_12 = type_curve[type_curve["month"] <= 12]["p50"].sum()
                st.metric("P50 12-Month EUR", f"{p50_12:,.0f}")


def render_export():
    """Render export options."""
    st.subheader("Export Type Curves")

    if "type_curve" not in st.session_state:
        st.info("Generate a type curve first")
        return

    type_curve = st.session_state.type_curve

    st.markdown("#### Download Options")

    # CSV export
    csv = type_curve.to_csv(index=False)
    st.download_button(
        "Download Type Curve (CSV)",
        csv,
        "type_curve.csv",
        "text/csv",
    )

    # Display data
    with st.expander("View Type Curve Data"):
        st.dataframe(type_curve, use_container_width=True, hide_index=True)

