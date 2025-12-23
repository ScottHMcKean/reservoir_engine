"""Visualization functions for reservoir data.

All plot functions return figure objects that can be displayed
in both Jupyter notebooks and Dash/Streamlit applications.
"""

from typing import Any, Literal

import pandas as pd


def plot_production_timeseries(
    df: pd.DataFrame,
    uwi: str | list[str] | None = None,
    phase: Literal["oil", "gas", "water", "all"] = "oil",
    date_column: str = "date",
    use_plotly: bool = True,
) -> Any:
    """Plot production time series for one or more wells.

    Args:
        df: DataFrame with production data
        uwi: Well identifier(s) to plot. If None, plots all wells.
        phase: Which phase to plot - oil, gas, water, or all
        date_column: Name of date column
        use_plotly: If True, returns Plotly figure; else Matplotlib

    Returns:
        Plotly Figure or Matplotlib Figure

    Example:
        >>> fig = plot_production_timeseries(df, uwi="WELL-001", phase="oil")
        >>> fig.show()  # In notebook
    """
    # Filter by UWI if provided
    plot_df = df.copy()
    if uwi is not None:
        if isinstance(uwi, str):
            uwi = [uwi]
        plot_df = plot_df[plot_df["uwi"].isin(uwi)]

    phase_columns = {
        "oil": ["oil_bbl"],
        "gas": ["gas_mcf"],
        "water": ["water_bbl"],
        "all": ["oil_bbl", "gas_mcf", "water_bbl"],
    }
    columns_to_plot = phase_columns[phase]

    if use_plotly:
        import plotly.express as px
        import plotly.graph_objects as go

        fig = go.Figure()

        for col in columns_to_plot:
            if col not in plot_df.columns:
                continue

            for well_uwi in plot_df["uwi"].unique():
                well_data = plot_df[plot_df["uwi"] == well_uwi].sort_values(date_column)
                fig.add_trace(
                    go.Scatter(
                        x=well_data[date_column],
                        y=well_data[col],
                        mode="lines",
                        name=f"{well_uwi} - {col}",
                    )
                )

        fig.update_layout(
            title="Production Time Series",
            xaxis_title="Date",
            yaxis_title="Production",
            hovermode="x unified",
        )
        return fig
    else:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 6))

        for col in columns_to_plot:
            if col not in plot_df.columns:
                continue

            for well_uwi in plot_df["uwi"].unique():
                well_data = plot_df[plot_df["uwi"] == well_uwi].sort_values(date_column)
                ax.plot(well_data[date_column], well_data[col], label=f"{well_uwi} - {col}")

        ax.set_xlabel("Date")
        ax.set_ylabel("Production")
        ax.set_title("Production Time Series")
        ax.legend()
        return fig


def plot_decline_curve(
    df: pd.DataFrame,
    uwi: str,
    forecast_df: pd.DataFrame | None = None,
    phase: Literal["oil", "gas"] = "oil",
    log_scale: bool = True,
    use_plotly: bool = True,
) -> Any:
    """Plot production decline curve with optional forecast overlay.

    Args:
        df: DataFrame with historical production data
        uwi: Well identifier
        forecast_df: Optional DataFrame with forecast data (date, rate columns)
        phase: Which phase to plot
        log_scale: If True, use log scale for y-axis
        use_plotly: If True, returns Plotly figure; else Matplotlib

    Returns:
        Plotly Figure or Matplotlib Figure
    """
    rate_col = f"{phase}_bbl" if phase == "oil" else f"{phase}_mcf"

    well_df = df[df["uwi"] == uwi].copy()
    well_df = well_df.sort_values("date")

    if use_plotly:
        import plotly.graph_objects as go

        fig = go.Figure()

        # Historical data
        fig.add_trace(
            go.Scatter(
                x=well_df["date"],
                y=well_df[rate_col],
                mode="markers",
                name="Historical",
                marker=dict(size=6),
            )
        )

        # Forecast if provided
        if forecast_df is not None:
            fig.add_trace(
                go.Scatter(
                    x=forecast_df["date"],
                    y=forecast_df["rate"],
                    mode="lines",
                    name="Forecast",
                    line=dict(color="red", dash="dash"),
                )
            )

        fig.update_layout(
            title=f"Decline Curve - {uwi}",
            xaxis_title="Date",
            yaxis_title=f"{phase.title()} Rate",
            yaxis_type="log" if log_scale else "linear",
        )
        return fig
    else:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.scatter(well_df["date"], well_df[rate_col], label="Historical", s=20)

        if forecast_df is not None:
            ax.plot(
                forecast_df["date"],
                forecast_df["rate"],
                "r--",
                label="Forecast",
            )

        ax.set_xlabel("Date")
        ax.set_ylabel(f"{phase.title()} Rate")
        ax.set_title(f"Decline Curve - {uwi}")
        if log_scale:
            ax.set_yscale("log")
        ax.legend()
        return fig


def plot_production_heatmap(
    df: pd.DataFrame,
    value_column: str = "oil_bbl",
    date_column: str = "date",
    use_plotly: bool = True,
) -> Any:
    """Create a heatmap of production values by well and time.

    Args:
        df: DataFrame with production data
        value_column: Column to use for heatmap values
        date_column: Name of date column
        use_plotly: If True, returns Plotly figure; else Matplotlib

    Returns:
        Plotly Figure or Matplotlib Figure
    """
    # Pivot data for heatmap
    pivot_df = df.pivot_table(
        values=value_column,
        index="uwi",
        columns=date_column,
        aggfunc="sum",
        fill_value=0,
    )

    if use_plotly:
        import plotly.express as px

        fig = px.imshow(
            pivot_df,
            labels=dict(x="Date", y="Well", color=value_column),
            title=f"Production Heatmap - {value_column}",
            aspect="auto",
        )
        return fig
    else:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(14, 8))
        im = ax.imshow(pivot_df.values, aspect="auto", cmap="YlOrRd")
        ax.set_xlabel("Date")
        ax.set_ylabel("Well")
        ax.set_title(f"Production Heatmap - {value_column}")
        plt.colorbar(im, ax=ax)
        return fig


def plot_rate_cumulative(
    df: pd.DataFrame,
    uwi: str,
    phase: Literal["oil", "gas"] = "oil",
    use_plotly: bool = True,
) -> Any:
    """Plot rate vs cumulative production (diagnostic plot).

    Args:
        df: DataFrame with production data
        uwi: Well identifier
        phase: Which phase to plot
        use_plotly: If True, returns Plotly figure; else Matplotlib

    Returns:
        Plotly Figure or Matplotlib Figure
    """
    rate_col = f"{phase}_bbl" if phase == "oil" else f"{phase}_mcf"

    well_df = df[df["uwi"] == uwi].copy()
    well_df = well_df.sort_values("date")
    well_df["cumulative"] = well_df[rate_col].cumsum()

    if use_plotly:
        import plotly.express as px

        fig = px.scatter(
            well_df,
            x="cumulative",
            y=rate_col,
            title=f"Rate vs Cumulative - {uwi}",
            labels={"cumulative": f"Cumulative {phase.title()}", rate_col: "Rate"},
        )
        fig.update_yaxes(type="log")
        return fig
    else:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(well_df["cumulative"], well_df[rate_col], s=20)
        ax.set_xlabel(f"Cumulative {phase.title()}")
        ax.set_ylabel("Rate")
        ax.set_title(f"Rate vs Cumulative - {uwi}")
        ax.set_yscale("log")
        return fig

