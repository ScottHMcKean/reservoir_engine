# Data Requirements

## Available Tables (from IHS)

Based on the data in `data/`, the following tables are available:

| Table | Rows | Description |
|-------|------|-------------|
| `well_header.csv` | ~9,409 | Master well information (location, status, depths, formations) |
| `well_production_monthly.csv` | ~782,201 | Monthly production volumes (oil, gas, water, BOE) |
| `well_production_summary.csv` | ~5,735 | Pre-calculated EUR metrics and statistics |
| `well_directional_survey_stations.csv` | ~729,913 | Directional survey points (MD, TVD, inclination, azimuth) |
| `well_completions.csv` | ~341,992 | Completion intervals and treatments |
| `uwis_area_of_interest.csv` | ~9,409 | UWI filter list for study area |

---

## Missing Tables (Required for Full Functionality)

### 1. Steam Injection Data  MISSING

**Required for:** Butler SAGD model, thermal asset analysis

**Expected schema:**
```
uwi                 STRING    -- Unique Well Identifier (injector)
injection_date      DATE      -- Injection month
steam_tonnes        FLOAT     -- Steam injected (tonnes)
steam_mmbtu         FLOAT     -- Steam energy (MMBTU)
pressure_kpa        FLOAT     -- Injection pressure (kPa)
temperature_c       FLOAT     -- Steam temperature (°C)
cum_steam           FLOAT     -- Cumulative steam injected
```

**Workaround:** Butler model can be fitted without steam data (uses default SOR), but steam-oil ratio calculations will be estimates.

**Source:** IHS typically provides this in `well_injection_monthly` or similar.

---

### 2. Well Economics Data  MISSING

**Required for:** Economic limit calculations, NPV analysis

**Expected schema:**
```
uwi                 STRING    -- Unique Well Identifier
date                DATE      -- Effective date
oil_price           FLOAT     -- Oil price ($/bbl)
gas_price           FLOAT     -- Gas price ($/mcf)
opex_per_boe        FLOAT     -- Operating cost ($/BOE)
royalty_rate        FLOAT     -- Royalty percentage
```

**Workaround:** Use hardcoded economic assumptions or user-provided parameters.

---

### 3. Formation/Zone Data  MISSING (Optional)

**Required for:** Multi-zone analysis, stacked pay evaluations

**Expected schema:**
```
uwi                 STRING    -- Unique Well Identifier
formation           STRING    -- Formation name
zone                STRING    -- Zone identifier
top_depth           FLOAT     -- Top of zone (m)
base_depth          FLOAT     -- Base of zone (m)
net_pay             FLOAT     -- Net pay thickness (m)
porosity            FLOAT     -- Average porosity
water_saturation    FLOAT     -- Water saturation
```

**Note:** Some formation data exists in `well_header` and `well_completions`.

---

### 4. Frac Treatment Details  PARTIAL

**Required for:** Proppant intensity analysis, treatment optimization

**Currently available in `well_completions`:**
- `perf_shots` - Number of perforations
- `top_depth`, `base_depth` - Interval depths
- `completion_type` - FRACTURED, JET PERFORATION, etc.

**Missing fields:**
```
proppant_tonnes     FLOAT     -- Proppant mass (tonnes)
proppant_type       STRING    -- Proppant type (sand, ceramic, etc.)
fluid_m3            FLOAT     -- Fluid volume (m³)
fluid_type          STRING    -- Fluid system (slickwater, gel, etc.)
max_rate            FLOAT     -- Max pump rate (m³/min)
max_pressure        FLOAT     -- Max treating pressure (kPa)
stages              INT       -- Number of frac stages
clusters_per_stage  INT       -- Clusters per stage
```

**Source:** IHS `well_treatment_detail` or `well_frac_data`.

---

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              AVAILABLE DATA                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────────┐  │
│  │  well_header    │  │  production_    │  │  production_summary         │  │
│  │  (9.4K wells)   │  │  monthly        │  │  (5.7K wells)               │  │
│  │                 │  │  (782K records) │  │                             │  │
│  │  • location     │  │                 │  │  • EUR metrics              │  │
│  │  • depths       │  │  • oil/gas/wtr  │  │  • IP metrics               │  │
│  │  • formation    │  │  • BOE          │  │  • peak/last dates          │  │
│  │  • status       │  │  • cumulative   │  │                             │  │
│  └────────┬────────┘  └────────┬────────┘  └─────────────┬───────────────┘  │
│           │                    │                         │                  │
│           └────────────────────┼─────────────────────────┘                  │
│                                │                                            │
│  ┌─────────────────┐  ┌────────┴────────┐                                   │
│  │  survey_        │  │  completions    │                                   │
│  │  stations       │  │  (342K records) │                                   │
│  │  (730K points)  │  │                 │                                   │
│  │                 │  │  • intervals    │                                   │
│  │  • MD/TVD       │  │  • perf shots   │                                   │
│  │  • inclination  │  │  • formation    │                                   │
│  │  • lat/lon      │  │                 │                                   │
│  └─────────────────┘  └─────────────────┘                                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              MISSING DATA                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────────┐  │
│  │  steam_         │  │  well_          │  │  frac_treatment_            │  │
│  │  injection      │  │  economics      │  │  details                    │  │
│  │   MISSING     │  │   MISSING     │  │   PARTIAL                 │  │
│  │                 │  │                 │  │                             │  │
│  │  Needed for:    │  │  Needed for:    │  │  Needed for:                │  │
│  │  • SAGD/Butler  │  │  • NPV analysis │  │  • Proppant analysis        │  │
│  │  • Thermal ops  │  │  • Econ limits  │  │  • Treatment optimization   │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Module Dependencies on Data

| Module | Required Tables | Optional Tables |
|--------|-----------------|-----------------|
| **EDA** | production_monthly | well_header, production_summary |
| **DCA (Arps)** | production_monthly | well_header |
| **DCA (Butler)** | production_monthly | steam_injection  |
| **Type Curves** | production_monthly, well_header | completions, survey_stations |
| **Uplift ML** | production_monthly, well_header | completions, survey_stations, production_summary |

---

## Recommended Data Onboarding Order

1. **Phase 1 (Core):**
   - well_header
   - production_monthly
   - uwis_area_of_interest

2. **Phase 2 (Enhanced):**
   - production_summary
   - completions
   - survey_stations

3. **Phase 3 (Advanced):**
   - steam_injection (for thermal)
   - frac_treatment_details (for completion optimization)
   - well_economics (for NPV)

---

## Loading Data

### From CSV (Development)
```python
from reservoir_engine.data import (
    load_well_header_csv,
    load_production_monthly_csv,
    load_completions_csv,
)

wells = load_well_header_csv("data/well_header.csv")
production = load_production_monthly_csv("data/well_production_monthly.csv")
completions = load_completions_csv("data/well_completions.csv")
```

### From Delta Lake (Databricks)
```python
from reservoir_engine.data import (
    load_well_header_delta,
    load_production_monthly_delta,
)

wells = load_well_header_delta("catalog", "schema", "well_header")
production = load_production_monthly_delta("catalog", "schema", "production_monthly")
```

