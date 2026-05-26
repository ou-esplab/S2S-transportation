# S2S_transportation

Analysis of the relationship between precipitation anomalies and fatal motor vehicle crashes in the continental United States (1981–2023), with emphasis on winter (DJF) season variability driven by ENSO and atmospheric weather regimes.

## Scientific Background

Precipitation is a leading environmental factor in fatal traffic crashes. This project quantifies how large-scale climate signals — specifically ENSO (El Niño–Southern Oscillation) phase and synoptic-scale weather regimes — modulate both precipitation and crash rates at the state and national level. The analysis focuses on December–January–February (DJF) when precipitation-related crash risk is highest.

Key climate drivers examined:
- **ENSO** — Niño 3.4 SST anomalies (monthly); El Niño and La Niña phases produce distinct precipitation patterns across the US
- **Weather regimes** — four recurring synoptic patterns over North America identified by K-means clustering of ERA5 500 hPa geopotential height (DJF 1981–2019)

## Environment Setup

```bash
conda env create -f environment.yml
conda activate s2stransportation
```

Key dependencies: Python ≥ 3.10, pandas 2.2.3, xarray, Dask, Cartopy, GeoPandas, SciPy, Matplotlib.

## Directory Structure

```
S2S_transportation/
├── environment.yml                     # Conda environment specification
├── README.md                           # This file
├── data/                               # Symlink to external data directory (not in repo)
├── figs/                               # Publication figures (Figure2.png – Figure10.png)
└── src/
    ├── MakePrecipStates.py             # STEP 1: Generate state-level precip anomalies
    ├── MakeDailyMonthlyCombinedDatabases.ipynb  # STEP 2: Build combined crash+climate databases
    ├── Journal_Figs-REDO.ipynb         # STEP 3: Generate publication figures
    ├── plot_utils.py                   # Plotting helper functions (used by Step 3)
    └── archive/                        # Scripts not in current workflow (see below)
```

## Data Sources

| Dataset | Description | Source |
|---------|-------------|--------|
| FARS | Fatality Analysis Reporting System; fatal crash records 1975–2023; filtered to precipitation-related crashes in CONUS | NHTSA |
| CHIRPS | Climate Hazards Group InfraRed Precipitation with Station data; daily gridded precipitation at 0.25° resolution | UCSB CHG |
| Niño 3.4 SST | Monthly SST anomalies in Niño 3.4 region; ENSO phase indicator | NOAA |
| ERA5 composites | 500 hPa geopotential height DJF composites by weather regime cluster | ECMWF/ERA5 |
| NHTS 2009 | Vehicle miles traveled (VMT) by state; used to normalize crash counts | FHWA |

External data files are stored under `data/` (symlinked; not tracked in git):
- `data/FARS/FARSNO2UPDATED_FIXED.csv` — processed FARS crash records
- `data/precip/state_precip_chirps.csv` — output of Step 1
- `data/ENSO/ENSOMONTHLY.csv` — monthly Niño 3.4 index
- `data/wxregimes/kmeans_4cluster_DJF_1981-2019_NA.nc` — weather regime assignments
- `data/wxregimes/era5_cluster_comp_z_na_DJF1981-2019.nc` — ERA5 500 hPa composites
- `data/combined_databases/` — outputs of Step 2 (inputs to Step 3)
- `data/nhts2009transferabilityvtvmtbystate.csv` — VMT by state

## Workflow

### Step 1 — Generate state-level precipitation anomalies

```bash
python src/MakePrecipStates.py
```

Reads CHIRPS daily gridded precipitation (1981–2023), computes a triangular-smoothed daily climatology, calculates anomalies, and spatially averages to each CONUS state using rasterized state polygons. Parallelized with Dask.

**Output:** `data/precip/state_precip_chirps.csv`

### Step 2 — Build combined crash + climate databases

Run all cells in `src/MakeDailyMonthlyCombinedDatabases.ipynb`.

- Loads FARS crash records; filters to precipitation-related codes, CONUS states, 1981–2023 (~178 k records)
- Merges with state precip anomalies (Step 1), monthly ENSO index, and DJF weather regime assignments
- Detrends crash counts (linear regression on year)
- Aggregates to four summary tables

**Outputs** in `data/combined_databases/`:
| File | Rows | Description |
|------|------|-------------|
| `database_daily_summary_national.csv` | ~15,700 | Daily national crash and climate variables |
| `database_daily_summary_state.csv` | ~754,000 | Daily per-state crash and climate variables |
| `database_monthly_summary_national.csv` | ~516 | Monthly national aggregates |
| `database_monthly_summary_state.csv` | varies | Monthly per-state aggregates |

### Step 3 — Generate publication figures

Run all cells in `src/Journal_Figs-REDO.ipynb`.

Reads the four combined databases plus ERA5 composites and NHTS VMT data. Calls helper functions from `src/plot_utils.py`.

**Outputs** in `figs/`:
| Figure | Description |
|--------|-------------|
| Figure 2 | El Niño / La Niña state precipitation anomaly maps |
| Figure 3 | Weather regime 500 hPa composites |
| Figure 4 | Weather regime precipitation anomaly composites by state |
| Figure 5 | National fatal crashes and climatology time series |
| Figure 6 | DJF state crash anomalies normalized by VMT |
| Figure 7 | Monthly crash anomaly distributions |
| Figure 8 | State-level R² maps (precip–crash; ENSO precip–crash) |
| Figure 9 | CONUS crash rates by weather regime |
| Figure 10 | State crash rates by weather regime |

## Known Issues

**`src/MakePrecipStates.py` line 213** — Fixed: the original code attempted `df.drop(columns=['dayofyear', ...])` but `dayofyear` is not a column of the merged DataFrame at that point (it is only a groupby dimension in the climatology calculation). The fix uses `errors='ignore'` so the drop is safe regardless of which columns are present.

## Archived Scripts

Scripts in `src/archive/` are not part of the current workflow but are preserved for reference:

| File | Description |
|------|-------------|
| `MakePrecipSWEStates_monthly.py` | Alternative approach computing monthly DJF-only precipitation anomalies (1981–2019); outputs to a different path than the active script and is not used by the current pipeline |
| `MakePrecipSWEStates_wxregimes.py` | Computes precipitation anomaly composites by weather regime; the output is read in Step 2 but is not propagated into the final figure database |
| `processSWE_daily.ipynb` | Experimental processing of snow water equivalent (SWE) data; deactivated because SWE records are too limited for the full analysis period |
| `Journal_Figs-REDO2.ipynb` | Later version of the figures notebook (superseded by `Journal_Figs-REDO.ipynb`) |
| `Journal_Figs.ipynb` | Original figures notebook (superseded by `Journal_Figs-REDO.ipynb`) |
