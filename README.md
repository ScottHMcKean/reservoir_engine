# Reservoir Engine

A modular, Databricks-native framework for reservoir engineering workflows. Built for simplicity, testability, and extensibility.

## Overview

Reservoir Engine provides a clean, composable toolkit for decline curve analysis, type curve generation, and production forecasting. It leverages Spark for distributed computation and Lakebase (Postgres) for persistent storage, while maintaining a simple functional API that works identically in notebooks and applications.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              DATA SOURCES                                    │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────────────────┐│
│  │ Production  │ │   Surveys   │ │ Completions │ │ Steam Volumes (Thermal) ││
│  │    (IHS)    │ │    (IHS)    │ │    (IHS)    │ │         (IHS)           ││
│  └──────┬──────┘ └──────┬──────┘ └──────┬──────┘ └────────────┬────────────┘│
└─────────┼───────────────┼───────────────┼────────────────────┼──────────────┘
          │               │               │                    │
          ▼               ▼               ▼                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DATA LAYER (data/)                                 │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ loaders.py - Unified data ingestion from IHS, CSV, Delta, Postgres     │ │
│  │ schemas.py - Pydantic models for validation & type safety              │ │
│  │ storage.py - Delta Lake & Lakebase (Postgres) read/write utilities     │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                    │                                         │
│                    ┌───────────────┼───────────────┐                        │
│                    ▼               ▼               ▼                        │
│           ┌────────────┐   ┌────────────┐   ┌────────────┐                  │
│           │   Delta    │   │  Lakebase  │   │  In-Memory │                  │
│           │   Tables   │◄─►│ (Postgres) │◄─►│  DataFrames│                  │
│           └────────────┘   └────────────┘   └────────────┘                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        ▼                           ▼                           ▼
┌───────────────┐          ┌───────────────┐          ┌───────────────┐
│  EDA Module   │          │  DCA Module   │          │  Type Curves  │
│    (eda/)     │          │    (dca/)     │          │(type_curves/) │
├───────────────┤          ├───────────────┤          ├───────────────┤
│• analysis.py  │          │• models.py    │          │• selection.py │
│  - statistics │          │  - Arps       │          │  - AOI picker │
│  - trends     │          │  - Butler     │          │  - polygon    │
│  - outliers   │          │• autofit.py   │          │• aggregation  │
│• plots.py     │          │  - curve fit  │          │  - normalize  │
│  - time series│          │  - optimize   │          │  - align      │
│  - scatter    │          │• forecast.py  │          │• probabilistic│
│  - heatmaps   │          │  - EUR calc   │          │  - P10/50/90  │
└───────┬───────┘          └───────┬───────┘          └───────┬───────┘
        │                          │                          │
        └──────────────────────────┼──────────────────────────┘
                                   ▼
                        ┌───────────────────┐
                        │   Uplift Module   │
                        │     (uplift/)     │
                        ├───────────────────┤
                        │• features.py      │
                        │  - engineering    │
                        │  - spatial        │
                        │• models.py        │
                        │  - ML training    │
                        │  - validation     │
                        │• inference.py     │
                        │  - predictions    │
                        │  - mapping        │
                        └───────────────────┘
```

## Data Flow

### 1. Ingestion Pipeline

```
IHS Data Feeds
     │
     ├── Production: Monthly oil/gas/water volumes by well
     ├── Surveys: Directional survey data (MD, TVD, inclination, azimuth)
     ├── Completions: Perforations, stages, proppant, fluid volumes
     └── Steam (Thermal): Injection volumes, pressures, temperatures
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│ data.loaders.load_ihs_production(catalog, schema, table)    │
│ data.loaders.load_ihs_surveys(catalog, schema, table)       │
│ data.loaders.load_ihs_completions(catalog, schema, table)   │
│ data.loaders.load_ihs_steam(catalog, schema, table)         │
└─────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│ data.schemas - Validation & Normalization                   │
│ • ProductionRecord: uwi, date, oil, gas, water, days_on     │
│ • SurveyRecord: uwi, md, tvd, inclination, azimuth          │
│ • CompletionRecord: uwi, stage, perf_top, perf_btm, ...     │
│ • SteamRecord: uwi, date, steam_injected, pressure, temp    │
└─────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│ data.storage - Persistence Layer                            │
│ • to_delta(df, path) / from_delta(path)                     │
│ • to_postgres(df, table) / from_postgres(table)             │
│ • sync_delta_to_postgres(delta_path, pg_table)              │
└─────────────────────────────────────────────────────────────┘
```

### 2. Analysis Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ EXPLORATION (eda/)                                                          │
│                                                                             │
│   production_df = load_production(...)                                      │
│         │                                                                   │
│         ├─► eda.analysis.calculate_statistics(df)                           │
│         │       → mean, std, percentiles by well/field                      │
│         │                                                                   │
│         ├─► eda.analysis.detect_outliers(df, method="iqr")                  │
│         │       → flagged records for review                                │
│         │                                                                   │
│         └─► eda.plots.production_timeseries(df, groupby="well")             │
│                 → matplotlib/plotly figure (reusable in notebook & app)     │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ DECLINE CURVE ANALYSIS (dca/)                                               │
│                                                                             │
│   # Unconventional: Arps Hyperbolic/Harmonic                                │
│   params = dca.autofit.fit_arps(production_df, uwi="WELL-001")              │
│         → ArpsParams(qi, di, b, t_switch)                                   │
│                                                                             │
│   forecast = dca.forecast.arps_forecast(params, months=360)                 │
│         → DataFrame[date, rate, cumulative]                                 │
│                                                                             │
│   # Thermal: Butler SAGD                                                    │
│   params = dca.autofit.fit_butler(production_df, steam_df, uwi="PAD-01")    │
│         → ButlerParams(m, phi, So, k, ...)                                  │
│                                                                             │
│   forecast = dca.forecast.butler_forecast(params, months=240)               │
│         → DataFrame[date, rate, sor, cumulative]                            │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ TYPE CURVES (type_curves/)                                                  │
│                                                                             │
│   # Select wells by AOI or polygon                                          │
│   wells = type_curves.selection.wells_in_polygon(polygon_geojson)           │
│         → list[str] (UWIs)                                                  │
│                                                                             │
│   # Normalize and align production                                          │
│   aligned = type_curves.aggregation.normalize_production(                   │
│       production_df,                                                        │
│       wells=wells,                                                          │
│       align_by="first_production"                                           │
│   )                                                                         │
│                                                                             │
│   # Generate probabilistic type curve                                       │
│   type_curve = type_curves.probabilistic.generate_type_curve(               │
│       aligned,                                                              │
│       percentiles=[10, 50, 90]                                              │
│   )                                                                         │
│         → DataFrame[month, p10, p50, p90]                                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ UPLIFT ANALYSIS (uplift/)                                                   │
│                                                                             │
│   # Feature engineering                                                     │
│   features = uplift.features.build_feature_matrix(                          │
│       production_df, completions_df, surveys_df                             │
│   )                                                                         │
│         → DataFrame with engineered features                                │
│                                                                             │
│   # Train model                                                             │
│   model = uplift.models.train_production_model(                             │
│       features,                                                             │
│       target="eur_180",                                                     │
│       model_type="xgboost"                                                  │
│   )                                                                         │
│                                                                             │
│   # Spatial inference                                                       │
│   predictions = uplift.inference.predict_on_grid(                           │
│       model,                                                                │
│       grid_df,                                                              │
│       resolution_m=500                                                      │
│   )                                                                         │
│         → GeoDataFrame with predicted values                                │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Key Data Models

### Production Record
```python
@dataclass
class ProductionRecord:
    uwi: str              # Unique Well Identifier
    date: date            # Production month
    oil_bbl: float        # Oil production (barrels)
    gas_mcf: float        # Gas production (MCF)
    water_bbl: float      # Water production (barrels)
    days_on: int          # Producing days in month
```

### Arps Parameters (Unconventional)
```python
@dataclass
class ArpsParams:
    qi: float             # Initial rate (bbl/day or mcf/day)
    di: float             # Initial decline rate (1/month)
    b: float              # Hyperbolic exponent (0 ≤ b ≤ 2)
    t_switch: float       # Time to switch to exponential (months)
    d_min: float          # Minimum decline rate (terminal)
```

### Butler Parameters (Thermal SAGD)
```python
@dataclass
class ButlerParams:
    m: float              # Drainage rate coefficient
    phi: float            # Porosity
    So: float             # Initial oil saturation
    k: float              # Permeability (mD)
    h: float              # Pay thickness (m)
    L: float              # Well pair length (m)
```

## Module Design Principles

1. **Functions over Classes**: Prefer pure functions with clear inputs/outputs
2. **Composition over Inheritance**: Build complex behavior by composing simple functions
3. **Loose Coupling**: Modules communicate via well-defined data structures
4. **Single Responsibility**: Each function does one thing well
5. **Testability**: All functions are side-effect free where possible

## Project Structure

```
reservoir_engine/
├── README.md
├── pyproject.toml
├── src/
│   └── reservoir_engine/
│       ├── __init__.py
│       ├── config.py              # Configuration management
│       ├── data/
│       │   ├── __init__.py
│       │   ├── loaders.py         # Data loading from various sources
│       │   ├── schemas.py         # Pydantic data models
│       │   └── storage.py         # Delta/Postgres storage utilities
│       ├── eda/
│       │   ├── __init__.py
│       │   ├── analysis.py        # Statistical analysis functions
│       │   └── plots.py           # Visualization functions
│       ├── dca/
│       │   ├── __init__.py
│       │   ├── models.py          # Decline curve models
│       │   ├── autofit.py         # Curve fitting optimization
│       │   └── forecast.py        # Production forecasting
│       ├── type_curves/
│       │   ├── __init__.py
│       │   ├── selection.py       # AOI/polygon well selection
│       │   ├── aggregation.py     # Production normalization
│       │   └── probabilistic.py   # P10/P50/P90 generation
│       └── uplift/
│           ├── __init__.py
│           ├── features.py        # Feature engineering
│           ├── models.py          # ML model training
│           └── inference.py       # Spatial prediction
└── tests/
    ├── conftest.py                # Shared fixtures
    ├── test_data/
    ├── test_eda/
    ├── test_dca/
    ├── test_type_curves/
    └── test_uplift/
```

## Quick Start

```python
from reservoir_engine.data import loaders, storage
from reservoir_engine.dca import autofit, forecast
from reservoir_engine.type_curves import selection, probabilistic

# Load production data
production = loaders.load_from_delta("catalog.schema.production")

# Fit decline curve for a single well
params = autofit.fit_arps(production, uwi="WELL-001")

# Generate 30-year forecast
fcst = forecast.arps_forecast(params, months=360)

# Generate type curve for an area
wells = selection.wells_in_polygon(my_polygon)
type_curve = probabilistic.generate_type_curve(production, wells)
```

## Installation

```bash
# Development
uv sync --all-extras

# Production (Databricks)
uv sync
```

## Running Tests

```bash
uv run pytest tests/ -v
```

## Comparison with Commercial Tools

| Feature | Reservoir Engine | ComboCurve | Enersight | EVA Turing |
|---------|-----------------|------------|-----------|------------|
| Open Source | ✅ | ❌ | ❌ | ❌ |
| Databricks Native | ✅ | ❌ | ❌ | ❌ |
| Arps DCA | ✅ | ✅ | ✅ | ✅ |
| Butler (Thermal) | ✅ | ❌ | ✅ | ❌ |
| ML Type Curves | ✅ | ✅ | ❌ | ✅ |
| Modular/Composable | ✅ | ❌ | ❌ | ❌ |
| Notebook-First | ✅ | ❌ | ❌ | ❌ |

## License

See LICENSE file.

