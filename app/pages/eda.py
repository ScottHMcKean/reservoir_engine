"""Exploratory Data Analysis (EDA) Page.

Provides visualization and statistics for production data.
"""

import sys
from pathlib import Path

# Ensure project root is in path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from app.database import get_backend


def render():
    """Render the EDA page."""
    st.title("Exploratory Data Analysis")
    st.markdown("Analyze production data, identify outliers, and explore trends")

    backend = get_backend()

    # Tabs for different analyses
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Well Map", "Production Trends", "Well Statistics", "Outlier Detection"])

    with tab1:
        render_overview(backend)

    with tab2:
        render_well_map(backend)

    with tab3:
        render_production_trends(backend)

    with tab4:
        render_well_statistics(backend)

    with tab5:
        render_outlier_detection(backend)


def render_overview(backend):
    """Render overview statistics."""
    st.subheader("Data Overview")

    wells = backend.get_well_header()
    production = backend.get_production_monthly()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Wells", f"{len(wells):,}")

    with col2:
        if "hole_direction" in wells.columns:
            hz = (wells["hole_direction"] == "HORIZONTAL").sum()
            st.metric("Horizontal Wells", f"{hz:,}")
        else:
            st.metric("Horizontal Wells", "N/A")

    with col3:
        st.metric("Production Records", f"{len(production):,}")

    with col4:
        if "uwi" in production.columns:
            active = production["uwi"].nunique()
            st.metric("Wells with Production", f"{active:,}")

    # Well type distribution
    st.subheader("Well Distribution")

    col1, col2 = st.columns(2)

    with col1:
        if "hole_direction" in wells.columns:
            type_counts = wells["hole_direction"].value_counts()
            fig = px.pie(
                values=type_counts.values,
                names=type_counts.index,
                title="Well Type Distribution",
                color_discrete_sequence=["#FF3621", "#1B1B1B", "#6E6E6E", "#F5F5F5"],
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        if "current_status" in wells.columns:
            status_counts = wells["current_status"].value_counts().head(10)
            fig = px.bar(
                x=status_counts.values,
                y=status_counts.index,
                orientation="h",
                title="Top 10 Well Status",
                color_discrete_sequence=["#FF3621"],
            )
            fig.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, use_container_width=True)

    # Map of wells
    st.subheader("Well Locations")
    if "surface_latitude" in wells.columns and "surface_longitude" in wells.columns:
        map_df = wells.dropna(subset=["surface_latitude", "surface_longitude"])

        if len(map_df) > 0:
            fig = px.scatter_mapbox(
                map_df,
                lat="surface_latitude",
                lon="surface_longitude",
                hover_name="uwi" if "uwi" in map_df.columns else None,
                color="hole_direction" if "hole_direction" in map_df.columns else None,
                zoom=7,
                height=500,
                color_discrete_sequence=["#FF3621", "#1B1B1B", "#6E6E6E"],
            )
            fig.update_layout(mapbox_style="carto-positron")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Location data not available")


def render_well_map(backend):
    """Render well map with horizontal sections (optimized for speed)."""
    st.subheader("Well Locations and Horizontal Sections")

    wells = backend.get_well_header()

    if wells.empty:
        st.warning("No well data available")
        return

    # Filter wells with location data
    map_wells = wells.dropna(subset=["surface_latitude", "surface_longitude"])

    if map_wells.empty:
        st.warning("No wells with location data")
        return

    # Filter options
    col1, col2, col3 = st.columns(3)

    with col1:
        if "hole_direction" in map_wells.columns:
            directions = ["All"] + map_wells["hole_direction"].dropna().unique().tolist()
            selected_direction = st.selectbox("Well Type", directions, index=0)
            if selected_direction != "All":
                map_wells = map_wells[map_wells["hole_direction"] == selected_direction]

    with col2:
        if "current_status" in map_wells.columns:
            statuses = ["All"] + map_wells["current_status"].dropna().unique().tolist()[:10]
            selected_status = st.selectbox("Status", statuses)
            if selected_status != "All":
                map_wells = map_wells[map_wells["current_status"] == selected_status]

    with col3:
        max_wells = st.slider("Max Wells to Display", 100, 5000, 1000, 100)
        if len(map_wells) > max_wells:
            map_wells = map_wells.head(max_wells)

    st.markdown(f"**Displaying {len(map_wells):,} wells**")

    # Check if we have both surface and BH locations
    has_bh = (
        "bh_latitude" in map_wells.columns
        and "bh_longitude" in map_wells.columns
        and map_wells["bh_latitude"].notna().any()
    )

    show_sticks = has_bh and st.checkbox("Show Well Sticks (Surface to BH)", value=True)

    # Build map figure
    fig = go.Figure()

    if show_sticks:
        # Optimized: Build all well sticks as a single trace using None separators
        trajectory_wells = map_wells.dropna(subset=["bh_latitude", "bh_longitude"])

        if not trajectory_wells.empty:
            # Build coordinate arrays with None separators between wells
            lats = []
            lons = []

            for _, well in trajectory_wells.iterrows():
                # Surface point -> BH point -> None (to break the line)
                lats.extend([well["surface_latitude"], well["bh_latitude"], None])
                lons.extend([well["surface_longitude"], well["bh_longitude"], None])

            # Single trace for ALL well sticks (much faster than one trace per well)
            fig.add_trace(
                go.Scattermapbox(
                    lat=lats,
                    lon=lons,
                    mode="lines",
                    line=dict(width=2, color="#FF3621"),
                    name="Well Sticks",
                    hoverinfo="skip",
                )
            )

            # Add BH points as single trace
            fig.add_trace(
                go.Scattermapbox(
                    lat=trajectory_wells["bh_latitude"].tolist(),
                    lon=trajectory_wells["bh_longitude"].tolist(),
                    mode="markers",
                    marker=dict(size=5, color="#1B1B1B"),
                    name="Bottom Hole",
                    hovertemplate="<b>BH:</b> %{customdata}<extra></extra>",
                    customdata=trajectory_wells["uwi"].tolist(),
                )
            )

    # Add surface locations as single trace per type (efficient)
    if "hole_direction" in map_wells.columns:
        for direction in map_wells["hole_direction"].dropna().unique():
            subset = map_wells[map_wells["hole_direction"] == direction]
            color = "#FF3621" if direction == "HORIZONTAL" else "#1B1B1B" if direction == "VERTICAL" else "#6E6E6E"
            fig.add_trace(
                go.Scattermapbox(
                    lat=subset["surface_latitude"].tolist(),
                    lon=subset["surface_longitude"].tolist(),
                    mode="markers",
                    marker=dict(size=7, color=color),
                    name=f"Surface ({direction})",
                    hovertemplate="<b>%{customdata}</b><extra></extra>",
                    customdata=subset["uwi"].tolist(),
                )
            )
    else:
        fig.add_trace(
            go.Scattermapbox(
                lat=map_wells["surface_latitude"].tolist(),
                lon=map_wells["surface_longitude"].tolist(),
                mode="markers",
                marker=dict(size=7, color="#FF3621"),
                name="Surface",
                hovertemplate="<b>%{customdata}</b><extra></extra>",
                customdata=map_wells["uwi"].tolist(),
            )
        )

    # Update layout
    center_lat = map_wells["surface_latitude"].mean()
    center_lon = map_wells["surface_longitude"].mean()

    fig.update_layout(
        mapbox=dict(
            style="carto-positron",
            center=dict(lat=center_lat, lon=center_lon),
            zoom=8,
        ),
        height=600,
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255,255,255,0.8)",
        ),
    )

    st.plotly_chart(fig, use_container_width=True)

    # Well count by type
    if "hole_direction" in map_wells.columns:
        col1, col2, col3 = st.columns(3)
        with col1:
            hz = (map_wells["hole_direction"] == "HORIZONTAL").sum()
            st.metric("Horizontal", f"{hz:,}")
        with col2:
            vt = (map_wells["hole_direction"] == "VERTICAL").sum()
            st.metric("Vertical", f"{vt:,}")
        with col3:
            other = len(map_wells) - hz - vt
            st.metric("Other", f"{other:,}")


def render_production_trends(backend):
    """Render production trend analysis."""
    st.subheader("Production Trends")

    wells = backend.get_well_header()
    uwis = wells["uwi"].head(50).tolist() if "uwi" in wells.columns else []

    # Well selector
    selected_uwis = st.multiselect(
        "Select Wells to Analyze",
        options=uwis,
        default=uwis[:3] if len(uwis) >= 3 else uwis,
        max_selections=10,
    )

    if not selected_uwis:
        st.info("Select wells to view production trends")
        return

    production = backend.get_production_monthly(selected_uwis)

    if production.empty:
        st.warning("No production data available for selected wells")
        return

    # Phase selector
    phase = st.selectbox("Production Phase", ["oil", "gas", "water"], index=0)

    if phase not in production.columns:
        st.warning(f"Column '{phase}' not found in production data")
        return

    # Time series plot
    fig = go.Figure()

    for uwi in selected_uwis:
        uwi_data = production[production["uwi"] == uwi].sort_values("production_date")
        if not uwi_data.empty:
            fig.add_trace(
                go.Scatter(
                    x=uwi_data["production_date"],
                    y=uwi_data[phase],
                    mode="lines+markers",
                    name=uwi[-15:],  # Truncate UWI for readability
                    marker=dict(size=4),
                )
            )

    fig.update_layout(
        title=f"{phase.title()} Production Over Time",
        xaxis_title="Date",
        yaxis_title=f"{phase.title()} (M3)" if phase in ["oil", "water"] else f"{phase.title()} (E3M3)",
        yaxis_type="log" if st.checkbox("Log Scale", value=True) else "linear",
        hovermode="x unified",
    )

    st.plotly_chart(fig, use_container_width=True)

    # Cumulative production
    st.subheader("Cumulative Production")

    cum_col = f"cum_{phase}"
    if cum_col in production.columns:
        fig2 = go.Figure()

        for uwi in selected_uwis:
            uwi_data = production[production["uwi"] == uwi].sort_values("production_date")
            if not uwi_data.empty:
                fig2.add_trace(
                    go.Scatter(
                        x=uwi_data["production_date"],
                        y=uwi_data[cum_col],
                        mode="lines",
                        name=uwi[-15:],
                        fill="tozeroy",
                    )
                )

        fig2.update_layout(
            title=f"Cumulative {phase.title()} Production",
            xaxis_title="Date",
            yaxis_title=f"Cumulative {phase.title()}",
        )

        st.plotly_chart(fig2, use_container_width=True)


def render_well_statistics(backend):
    """Render well-level statistics."""
    st.subheader("Well Statistics")

    summary = backend.get_production_summary()

    if summary.empty:
        st.info("No production summary data available")
        return

    # Top producers
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Top Oil Producers (Cumulative)")
        if "cum_oil" in summary.columns:
            top_oil = summary.nlargest(10, "cum_oil")[["uwi", "cum_oil"]]
            top_oil["cum_oil"] = top_oil["cum_oil"].apply(lambda x: f"{x:,.0f}")
            st.dataframe(top_oil, use_container_width=True, hide_index=True)

    with col2:
        st.markdown("#### Top Gas Producers (Cumulative)")
        if "cum_gas" in summary.columns:
            top_gas = summary.nlargest(10, "cum_gas")[["uwi", "cum_gas"]]
            top_gas["cum_gas"] = top_gas["cum_gas"].apply(lambda x: f"{x:,.0f}")
            st.dataframe(top_gas, use_container_width=True, hide_index=True)

    # Distribution plots
    st.markdown("#### Production Distributions")

    col1, col2 = st.columns(2)

    with col1:
        if "first_3_month_boe" in summary.columns:
            fig = px.histogram(
                summary,
                x="first_3_month_boe",
                nbins=50,
                title="First 3 Month BOE Distribution",
                color_discrete_sequence=["#FF3621"],
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        if "cum_boe" in summary.columns:
            fig = px.histogram(
                summary,
                x="cum_boe",
                nbins=50,
                title="Cumulative BOE Distribution",
                color_discrete_sequence=["#1B1B1B"],
            )
            st.plotly_chart(fig, use_container_width=True)


def render_outlier_detection(backend):
    """Render outlier detection analysis."""
    st.subheader("Outlier Detection")

    from reservoir_engine.eda.analysis import detect_outliers

    wells = backend.get_well_header()
    uwis = wells["uwi"].head(100).tolist() if "uwi" in wells.columns else []

    production = backend.get_production_monthly(uwis[:50] if len(uwis) > 50 else uwis)

    if production.empty:
        st.info("No production data available")
        return

    col1, col2 = st.columns(2)

    with col1:
        phase = st.selectbox("Analyze Phase", ["oil", "gas", "water"], key="outlier_phase")

    with col2:
        method = st.selectbox("Detection Method", ["iqr", "zscore", "mad"])

    if phase not in production.columns:
        st.warning(f"Column '{phase}' not found")
        return

    # Detect outliers
    result = detect_outliers(production, phase, method=method)

    outliers = result[result["is_outlier"]]
    non_outliers = result[~result["is_outlier"]]

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Total Records", f"{len(result):,}")

    with col2:
        pct = len(outliers) / len(result) * 100 if len(result) > 0 else 0
        st.metric("Outliers Detected", f"{len(outliers):,} ({pct:.1f}%)")

    # Visualization
    fig = go.Figure()

    fig.add_trace(
        go.Box(
            y=non_outliers[phase],
            name="Normal",
            marker_color="#1B1B1B",
        )
    )

    if len(outliers) > 0:
        fig.add_trace(
            go.Scatter(
                y=outliers[phase],
                x=["Normal"] * len(outliers),
                mode="markers",
                name="Outliers",
                marker=dict(color="#FF3621", size=8),
            )
        )

    fig.update_layout(
        title=f"{phase.title()} Distribution with Outliers",
        yaxis_title=phase.title(),
    )

    st.plotly_chart(fig, use_container_width=True)

    # Show outlier details
    if len(outliers) > 0:
        st.markdown("#### Outlier Details")
        display_cols = ["uwi", "production_date", phase]
        display_cols = [c for c in display_cols if c in outliers.columns]
        st.dataframe(outliers[display_cols].head(20), use_container_width=True, hide_index=True)

