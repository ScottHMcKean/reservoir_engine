# Reservoir Engine

A modular, Databricks-native framework for reservoir engineering workflows. Built for simplicity, testability, and extensibility.

## Key Features

- **Ruthlessly Simple**: Pure functions, explicit data contracts, no magic
- **H3 Spatial Indexing**: Consistent hexagonal grid for all geospatial operations
- **IHS-Native Schemas**: Direct compatibility with IHS/Enverus data formats
- **Databricks-Ready**: Spark and Lakebase (Postgres) integration
- **Comprehensive Tests**: 55 tests covering core functionality

## Quick Start

```python
from reservoir_engine.data import load_well_header_csv, load_production_monthly_csv
from reservoir_engine.dca import fit_arps, arps_forecast
from reservoir_engine.type_curves import align_production, generate_type_curve

# Load data
wells = load_well_header_csv("data/well_header.csv")
production = load_production_monthly_csv("data/well_production_monthly.csv")

# Fit decline curve
params = fit_arps(production, uwi="100010106021W500")
forecast = arps_forecast(params, months=360)

# Generate type curve
aligned = align_production(production)
type_curve = generate_type_curve(aligned, percentiles=[10, 50, 90])
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              DATA LAYER                                      │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │ loaders.py - CSV, Delta Lake, Postgres with IHS column mappings        ││
│  │ schemas.py - Pydantic models: WellHeader, ProductionMonthly, etc.      ││
│  │ storage.py - Delta/Postgres persistence                                 ││
│  └─────────────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        ▼                           ▼                           ▼
┌───────────────┐          ┌───────────────┐          ┌───────────────┐
│  EDA Module   │          │  DCA Module   │          │  Type Curves  │
│               │          │               │          │  (H3-based)   │
├───────────────┤          ├───────────────┤          ├───────────────┤
│• statistics   │          │• Arps model   │          │• H3 selection │
│• outliers     │          │• Butler SAGD  │          │• normalization│
│• plots        │          │• autofit      │          │• P10/50/90    │
└───────┬───────┘          └───────┬───────┘          └───────┬───────┘
        │                          │                          │
        └──────────────────────────┼──────────────────────────┘
                                   ▼
                        ┌───────────────────┐
                        │   Uplift Module   │
                        │   (H3 + ML)       │
                        ├───────────────────┤
                        │• feature matrix   │
                        │• XGBoost/RF       │
                        │• H3 grid predict  │
                        └───────────────────┘
```

## Modules

### Data (`reservoir_engine.data`)

Load, validate, and store reservoir data.

```python
# CSV loading (development)
wells = load_well_header_csv("data/well_header.csv")
production = load_production_monthly_csv("data/well_production_monthly.csv")

# Delta Lake (Databricks)
wells = load_well_header_delta("catalog", "schema", "well_header")
```

**Schemas:**
- `WellHeader` - Master well info (location, depths, formation)
- `ProductionMonthly` - Monthly oil/gas/water volumes
- `ProductionSummary` - Pre-calculated EUR metrics
- `SurveyStation` - Directional survey points
- `Completion` - Completion intervals and treatments

### EDA (`reservoir_engine.eda`)

Exploratory data analysis and visualization.

```python
from reservoir_engine.eda import calculate_statistics, detect_outliers, production_summary

stats = calculate_statistics(production, value_column="oil", group_by="uwi")
outliers = detect_outliers(production, "oil", method="iqr")
summary = production_summary(production)
```

### DCA (`reservoir_engine.dca`)

Decline curve analysis for production forecasting.

**Arps Hyperbolic (Unconventional):**
```python
from reservoir_engine.dca import fit_arps, arps_forecast, calculate_eur

params = fit_arps(production, uwi="100010106021W500", method="auto")
# ArpsParams(qi=150.0, di=0.12, b=0.85)

forecast = arps_forecast(params, months=360)
eur = calculate_eur(params, economic_limit=5.0)
```

**Butler SAGD (Thermal):**
```python
from reservoir_engine.dca import fit_butler, butler_forecast

params = fit_butler(production, steam_df, uwi="PAD-01")
forecast = butler_forecast(params, months=240, include_sor=True)
```

**Gas-First DCA with CGR-Based Condensate:**
```python
from reservoir_engine.dca import (
    DCAConfig, CGRConfig,
    prepare_time_series, fit_gas, fit_cgr,
    forecast_all, fit_and_forecast
)

# Option 1: Step-by-step
config = DCAConfig(
    num_segments=1,
    terminal_decline_year=0.06,  # 6% annual terminal decline
)
ts = prepare_time_series(production_df, gas_col="gas", cond_col="condensate")
gas_params = fit_gas(ts, config)
cgr_params = fit_cgr(ts, gas_params)
forecast = forecast_all(np.arange(360), gas_params, cgr_params)

# Option 2: One-liner
gas_params, cgr_params, forecast = fit_and_forecast(
    production_df,
    forecast_months=360,
    gas_col="gas",
    cond_col="condensate",
)
```

**Gas-First DCA Features:**
- Fixed 30.4-day normalized months (calendar-agnostic fitting)
- Multi-segment Arps with shared terminal exponential decline
- CGR model: condensate derived from gas via log-linear CGR(qg)
- Shared segment boundaries across phases
- Log-rate least squares objective for robust fitting

**Butler SAGD DCA (Thermal):**
```python
from reservoir_engine.dca import (
    SAGDConfig,
    prepare_sagd_series, fit_sagd_butler,
    forecast_sagd, calculate_sagd_sor,
    fit_and_forecast_sagd
)

# Two-stage Butler: rising (sqrt(t)) -> spreading (1/sqrt(t))
config = SAGDConfig(
    model="butler_2stage",
    use_terminal_decline=True,
    terminal_decline_year=0.06,
)
ts = prepare_sagd_series(production_df, oil_col="oil", steam_col="steam_volume")
params = fit_sagd_butler(ts, config)
forecast = forecast_sagd(np.arange(240), params)
sor_metrics = calculate_sagd_sor(ts, params)

# Multi-segment Butler for complex SAGD behavior
config_multiseg = SAGDConfig(
    model="butler_multiseg",
    num_segments=3,
    segment_boundaries=[0, 12, 36],
)
```

**SAGD Butler Features:**
- Two-stage model: rising phase (q ∝ t^α) and spreading phase (q ∝ t^(-α))
- Multi-segment power-law for complex pad behavior
- Continuity enforced at segment boundaries
- Shared terminal exponential decline
- SOR (Steam-Oil Ratio) calculation

### Architecture Mirror

Both engines share the same structure:

| Aspect     | Gas-First Arps              | SAGD Butler                  |
|------------|-----------------------------|------------------------------|
| Time axis  | Normalized months           | Normalized months            |
| Segments   | Multi-segment Arps          | Multi-segment power-law      |
| Boundaries | Shared across phases        | Shared across segments       |
| Terminal   | Exponential D<sub>f</sub>   | Exponential D<sub>f</sub>    |
| Fitting    | Log-rate least squares      | Log-rate least squares       |
| Calendar   | I/O layer only              | I/O layer only               |


### Type Curves (`reservoir_engine.type_curves`)

Probabilistic type curve generation using H3 spatial indexing.

```python
from reservoir_engine.type_curves import (
    wells_in_h3_ring,
    wells_in_polygon_h3,
    align_production,
    generate_type_curve,
)

# Select wells using H3 (5km radius at resolution 7)
wells = wells_in_h3_ring(wells_df, center_lat=54.5, center_lon=-116.5, k_rings=3)

# Or by polygon
polygon = [(54.0, -117.0), (54.0, -116.0), (55.0, -116.0), (55.0, -117.0)]
wells = wells_in_polygon_h3(wells_df, polygon, resolution=8)

# Generate type curve
aligned = align_production(production, wells=wells, align_by="first_production")
type_curve = generate_type_curve(aligned, percentiles=[10, 50, 90])
```

### Uplift (`reservoir_engine.uplift`)

ML-based production prediction with H3 spatial features.

```python
from reservoir_engine.uplift import (
    build_feature_matrix,
    calculate_eur_targets,
    train_production_model,
    create_prediction_grid_h3,
    predict_on_grid,
)

# Build features
features = build_feature_matrix(wells, production, completions)
targets = calculate_eur_targets(production, months=[12, 24, 60])

# Train model
model, metrics, preds = train_production_model(
    features.merge(targets, on="uwi"),
    target_column="eur_12m",
    model_type="xgboost"
)

# Predict on H3 grid
grid = create_prediction_grid_h3(-117, 54, -116, 55, resolution=8)
predictions = predict_on_grid(model, grid, feature_columns, wells)
```

## H3 Spatial Indexing

All geospatial operations use [H3](https://h3geo.org/) hexagonal indexing:

| Resolution | Edge Length | Area | Use Case |
|------------|-------------|------|----------|
| 4 | ~22 km | ~1,770 km² | Regional analysis |
| 7 | ~1.2 km | ~5.16 km² | Field-level |
| 9 | ~174 m | ~0.1 km² | Well-level (default) |
| 10 | ~66 m | ~0.015 km² | Precise location |

**Why H3?**
- Consistent hexagonal cells (no projection distortion)
- Hierarchical (zoom in/out seamlessly)
- Native Databricks/Spark support
- Fast spatial queries

## Data Requirements

See [docs/DATA_REQUIREMENTS.md](docs/DATA_REQUIREMENTS.md) for detailed schema documentation.

**Available Tables:**
- well_header (~9,409 wells)
- production_monthly (~782K records)
- production_summary (~5,735 wells)
- survey_stations (~730K points)
- completions (~342K records)

**Missing Tables (for full functionality):**
- steam_injection (required for Butler/SAGD model)
- well_economics (required for NPV analysis)
- frac_treatment_details (partial - missing proppant/fluid data)

## Installation

```bash
# Development (all dependencies)
uv sync --all-extras

# Production (minimal for Databricks)
uv sync

# App only (Streamlit)
uv sync --extra app
```

## Running the App

The Streamlit app provides a Databricks-themed UI for all modules:

```bash
# Run locally (test mode - uses CSV data)
uv run streamlit run app/main.py

# Access at http://localhost:8501
```

### App Pages

| Page | Description |
|------|-------------|
| **EDA** | Production statistics, outlier detection, well maps |
| **DCA** | Arps hyperbolic and Butler SAGD fitting, batch forecasting |
| **Type Curves** | H3-based well selection, P10/50/90 type curves |
| **Uplift** | Feature engineering, ML training, spatial prediction |

### App Architecture

```
app/
├── main.py           # Entry point, navigation, theming
├── config.py         # AppConfig (test/prod modes)
├── database.py       # CSVBackend (test) / LakebaseBackend (prod)
└── pages/
    ├── eda.py        # EDA visualizations
    ├── dca.py        # Decline curve analysis
    ├── type_curves.py # Type curve generation
    └── uplift.py     # ML-based prediction
```

### Test Mode vs Prod Mode

| Mode | Data Source | Use Case |
|------|-------------|----------|
| **test** | Local CSV files in `data/` | Development, demos |
| **prod** | Lakebase (Postgres) on Databricks | Production deployment |

Configure via `config.yaml`:
```yaml
app:
  mode: test  # or prod
  lakebase_instance: my-instance  # for prod mode
```

## Testing

```bash
# Run all tests
uv run pytest tests/ -v

# Run specific module tests
uv run pytest tests/test_dca/ -v
```

**Current status:** 55 tests, all passing

## Project Structure

```
reservoir_engine/
├── src/reservoir_engine/
│   ├── config.py              # Configuration
│   ├── data/
│   │   ├── loaders.py         # IHS-aware data loading
│   │   ├── schemas.py         # Pydantic data models
│   │   └── storage.py         # Delta/Postgres persistence
│   ├── eda/
│   │   ├── analysis.py        # Statistics, outliers
│   │   └── plots.py           # Plotly/Matplotlib
│   ├── dca/
│   │   ├── models.py          # Arps, Butler equations
│   │   ├── autofit.py         # Curve fitting
│   │   └── forecast.py        # Forecasting, EUR
│   ├── type_curves/
│   │   ├── selection.py       # H3-based well selection
│   │   ├── aggregation.py     # Normalization, alignment
│   │   └── probabilistic.py   # P10/P50/P90 generation
│   └── uplift/
│       ├── features.py        # Feature engineering
│       ├── models.py          # ML training
│       └── inference.py       # H3 grid prediction
├── tests/                      # 55 pytest tests
├── docs/
│   ├── MODULARITY_REVIEW.md   # Architecture assessment
│   └── DATA_REQUIREMENTS.md   # Schema documentation
└── pyproject.toml
```

## Design Principles

1. **Functions over Classes** - Pure functions with clear inputs/outputs
2. **Explicit Data Contracts** - Pydantic schemas for all data structures
3. **Loose Coupling** - Modules communicate via DataFrames
4. **Single Responsibility** - Each function does one thing well
5. **H3 for All Spatial** - Consistent hexagonal grid everywhere

## Modularity Review

See [docs/MODULARITY_REVIEW.md](docs/MODULARITY_REVIEW.md) for detailed analysis.

**Key findings:**
- No circular dependencies
- Easy to add new DCA models
- Easy to swap data providers
- Fast test suite (2.4s)

## Comparison with Commercial Tools

| Feature | Reservoir Engine | ComboCurve | Enersight | EVA Turing |
|---------|-----------------|------------|-----------|------------|
| Open Source | Yes | No | No | No |
| Databricks Native | Yes | No | No | No |
| H3 Spatial | Yes | No | No | No |
| Arps DCA | Yes | Yes | Yes | Yes |
| Butler (Thermal) | Yes | No | Yes | No |
| ML Type Curves | Yes | Yes | No | Yes |
| Modular/Composable | Yes | No | No | No |

## License

See LICENSE file.
