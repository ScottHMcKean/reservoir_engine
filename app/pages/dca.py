"""Decline Curve Analysis (DCA) Page.

Fit and forecast production using Arps and Butler models.
"""

import sys
from pathlib import Path

# Ensure project root is in path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from app.database import get_backend


def render():
    """Render the DCA page."""
    st.title("Decline Curve Analysis")
    st.markdown("Fit decline curves and forecast production")

    # Tabs for different models
    tab1, tab2, tab3 = st.tabs(["Arps Analysis", "Butler (SAGD)", "Batch Forecast"])

    with tab1:
        render_arps_analysis()

    with tab2:
        render_butler_analysis()

    with tab3:
        render_batch_forecast()


def render_arps_analysis():
    """Render Arps decline curve analysis."""
    from reservoir_engine.dca import fit_arps, arps_forecast, calculate_eur

    backend = get_backend()
    wells = backend.get_well_header()

    st.subheader("Arps Hyperbolic Decline")

    col1, col2 = st.columns([2, 1])

    with col1:
        # Well selector
        uwi_options = wells["uwi"].tolist() if "uwi" in wells.columns else []
        selected_uwi = st.selectbox(
            "Select Well",
            options=uwi_options[:100],  # Limit for performance
            index=0 if uwi_options else None,
        )

    with col2:
        fit_method = st.selectbox(
            "Fit Method",
            options=["auto", "hyperbolic", "harmonic", "exponential"],
            index=0,
        )

    if not selected_uwi:
        st.info("Select a well to analyze")
        return

    production = backend.get_production_monthly([selected_uwi])

    if production.empty:
        st.warning("No production data for selected well")
        return

    # Fit controls
    col1, col2, col3 = st.columns(3)

    with col1:
        forecast_months = st.number_input("Forecast Months", 60, 600, 360, 12)

    with col2:
        economic_limit = st.number_input("Economic Limit (M3/month)", 0.1, 100.0, 5.0, 0.5)

    with col3:
        phase = st.selectbox("Phase", ["oil", "gas"], index=0)

    # Check if required columns exist
    if phase not in production.columns:
        st.error(f"Column '{phase}' not found in production data")
        return

    if "production_date" not in production.columns:
        st.error("Column 'production_date' not found")
        return

    # Fit button
    if st.button("Fit Decline Curve", type="primary"):
        with st.spinner("Fitting decline curve..."):
            try:
                params = fit_arps(
                    production,
                    uwi=selected_uwi,
                    rate_column=phase,
                    method=fit_method,
                )

                # Store in session state
                st.session_state.arps_params = params
                st.session_state.arps_uwi = selected_uwi

                st.success("Decline curve fitted successfully!")

            except Exception as e:
                st.error(f"Error fitting curve: {e}")
                return

    # Display results if available
    if "arps_params" in st.session_state and st.session_state.get("arps_uwi") == selected_uwi:
        params = st.session_state.arps_params

        # Parameter display
        st.subheader("Fitted Parameters")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("qi (Initial Rate)", f"{params.qi:.1f}")
        with col2:
            st.metric("Di (Decline Rate)", f"{params.di:.4f}")
        with col3:
            st.metric("b (Exponent)", f"{params.b:.2f}")
        with col4:
            eur = calculate_eur(params, economic_limit)
            st.metric("EUR", f"{eur:,.0f}")

        # Generate forecast
        forecast = arps_forecast(params, months=forecast_months, economic_limit=economic_limit)

        # Plot
        fig = go.Figure()

        # Historical data
        well_data = production[production["uwi"] == selected_uwi].sort_values("production_date")
        fig.add_trace(
            go.Scatter(
                x=well_data["production_date"],
                y=well_data[phase],
                mode="markers",
                name="Historical",
                marker=dict(color="#1B1B1B", size=6),
            )
        )

        # Forecast
        fig.add_trace(
            go.Scatter(
                x=forecast["date"],
                y=forecast["rate"],
                mode="lines",
                name="Forecast",
                line=dict(color="#FF3621", width=2),
            )
        )

        # Economic limit line
        fig.add_hline(
            y=economic_limit,
            line_dash="dash",
            line_color="#6E6E6E",
            annotation_text="Economic Limit",
        )

        fig.update_layout(
            title=f"Decline Curve - {selected_uwi[-15:]}",
            xaxis_title="Date",
            yaxis_title=f"{phase.title()} Rate",
            yaxis_type="log",
            hovermode="x unified",
        )

        st.plotly_chart(fig, use_container_width=True)

        # Forecast table
        with st.expander("View Forecast Data"):
            st.dataframe(forecast.head(60), use_container_width=True, hide_index=True)


def render_butler_analysis():
    """Render Butler SAGD analysis."""
    from reservoir_engine.dca import butler_forecast, ButlerParams

    st.subheader("Butler SAGD Model")

    st.info(
        "The Butler model is designed for SAGD thermal operations. "
        "It accounts for the ramp-up period before peak production."
    )

    backend = get_backend()
    wells = backend.get_well_header()

    # Filter for potential thermal wells
    uwi_options = wells["uwi"].tolist() if "uwi" in wells.columns else []

    selected_uwi = st.selectbox(
        "Select Well",
        options=uwi_options[:50],
        key="butler_uwi",
    )

    if not selected_uwi:
        st.info("Select a well to analyze")
        return

    production = backend.get_production_monthly([selected_uwi])

    if production.empty:
        st.warning("No production data for selected well")
        return

    # Butler parameters (manual input since we may not have steam data)
    st.markdown("#### Model Parameters")
    col1, col2, col3 = st.columns(3)

    with col1:
        q_peak = st.number_input("Peak Rate (M3/month)", 10.0, 1000.0, 100.0, 10.0)

    with col2:
        tau = st.number_input("Time to Peak (months)", 3, 36, 12, 1)

    with col3:
        m = st.number_input("Decline Parameter", 0.01, 0.5, 0.1, 0.01)

    sor = st.slider("Steam-Oil Ratio", 1.0, 10.0, 3.0, 0.5)

    # Create params
    params = ButlerParams(q_peak=q_peak, m=m, tau=tau, sor=sor)

    # Forecast
    forecast_months = st.number_input("Forecast Months", 60, 480, 240, 12, key="butler_months")

    forecast = butler_forecast(params, months=forecast_months, include_sor=True)

    # Plot
    fig = go.Figure()

    # Historical
    well_data = production[production["uwi"] == selected_uwi].sort_values("production_date")
    if "oil" in well_data.columns:
        fig.add_trace(
            go.Scatter(
                x=well_data["production_date"],
                y=well_data["oil"],
                mode="markers",
                name="Historical",
                marker=dict(color="#1B1B1B", size=6),
            )
        )

    # Forecast
    fig.add_trace(
        go.Scatter(
            x=forecast["date"],
            y=forecast["rate"],
            mode="lines",
            name="Butler Forecast",
            line=dict(color="#FF3621", width=2),
        )
    )

    fig.update_layout(
        title=f"Butler SAGD Model - {selected_uwi[-15:]}",
        xaxis_title="Date",
        yaxis_title="Oil Rate (M3/month)",
        hovermode="x unified",
    )

    st.plotly_chart(fig, use_container_width=True)

    # Steam requirements
    if "steam_required" in forecast.columns:
        st.subheader("Steam Requirements")
        fig2 = go.Figure()
        fig2.add_trace(
            go.Scatter(
                x=forecast["date"],
                y=forecast["steam_required"],
                mode="lines",
                fill="tozeroy",
                name="Steam Required",
                line=dict(color="#6E6E6E"),
            )
        )
        fig2.update_layout(
            title="Projected Steam Requirements",
            xaxis_title="Date",
            yaxis_title="Steam (M3)",
        )
        st.plotly_chart(fig2, use_container_width=True)


def render_batch_forecast():
    """Render batch forecasting for multiple wells."""
    from reservoir_engine.dca import fit_arps, calculate_eur

    st.subheader("Batch Decline Curve Fitting")

    backend = get_backend()
    wells = backend.get_well_header()

    st.markdown(
        "Fit decline curves to multiple wells and export results. "
        "This is useful for portfolio analysis and reserve estimation."
    )

    # Well selection
    uwi_options = wells["uwi"].tolist() if "uwi" in wells.columns else []

    col1, col2 = st.columns(2)

    with col1:
        n_wells = st.number_input("Number of Wells", 5, 100, 10, 5)

    with col2:
        fit_method = st.selectbox("Fit Method", ["auto", "hyperbolic"], key="batch_method")

    selected_uwis = uwi_options[:n_wells]

    economic_limit = st.number_input("Economic Limit", 0.1, 100.0, 5.0, key="batch_econ")

    if st.button("Run Batch Analysis", type="primary"):
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, uwi in enumerate(selected_uwis):
            status_text.text(f"Processing {uwi}...")
            progress_bar.progress((i + 1) / len(selected_uwis))

            try:
                production = backend.get_production_monthly([uwi])
                if production.empty or "oil" not in production.columns:
                    continue

                params = fit_arps(production, uwi=uwi, method=fit_method)
                eur = calculate_eur(params, economic_limit)

                results.append(
                    {
                        "uwi": uwi,
                        "qi": params.qi,
                        "di": params.di,
                        "b": params.b,
                        "eur": eur,
                        "status": "Success",
                    }
                )
            except Exception as e:
                results.append(
                    {
                        "uwi": uwi,
                        "qi": None,
                        "di": None,
                        "b": None,
                        "eur": None,
                        "status": f"Error: {str(e)[:50]}",
                    }
                )

        progress_bar.empty()
        status_text.empty()

        if results:
            results_df = pd.DataFrame(results)
            st.session_state.batch_results = results_df

            st.success(f"Processed {len(results)} wells")

    # Display results
    if "batch_results" in st.session_state:
        results_df = st.session_state.batch_results

        col1, col2, col3 = st.columns(3)

        successful = results_df[results_df["status"] == "Success"]

        with col1:
            st.metric("Wells Processed", len(results_df))

        with col2:
            st.metric("Successful Fits", len(successful))

        with col3:
            if len(successful) > 0 and "eur" in successful.columns:
                total_eur = successful["eur"].sum()
                st.metric("Total EUR", f"{total_eur:,.0f}")

        st.dataframe(results_df, use_container_width=True, hide_index=True)

        # Download button
        csv = results_df.to_csv(index=False)
        st.download_button(
            "Download Results (CSV)",
            csv,
            "dca_results.csv",
            "text/csv",
        )

