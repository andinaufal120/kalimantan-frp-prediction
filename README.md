# Kalimantan FRP Prediction

Machine learning pipeline for predicting Fire Radiative Power (FRP) from satellite and meteorological data across Kalimantan, Indonesia (2022–2024).

## Overview

FRP measures the intensity of active fire detections from satellite sensors. This project models log1p-transformed FRP for 59,155 fire detections across Kalimantan, spanning the strong 2023 El Niño-driven fire peak (Aug–Oct 2023) and an anomalous 2024 dry season that produced elevated fire activity under neutral ENSO conditions. The pipeline covers exploratory analysis and feature selection, baseline model comparison (Random Forest vs. XGBoost) under a time-aware cross-validation scheme, an AutoGluon ensemble benchmark, and an interpretability/production playbook.

## Project Structure

| Path | Description |
|---|---|
| `eda.ipynb` | Exploratory analysis, outlier characterization, feature selection |
| `model.ipynb` | RF & XGBoost baselines with custom sliding-window CV |
| `playbook.ipynb` | Interpretability, geographic mapping, error analysis |
| `autogluon.ipynb` | AutoGluon ensemble benchmark (separate environment) |
| `Dataset_ML.csv` | Processed dataset used for modeling |
| `data.csv` | Raw dataset (earlier version) |
| `artifacts/` | Generated figures, reports, and saved models |
| `requirements.txt` | Dependencies for the main environment |

## Pipeline

Run the notebooks in this order:

1. **[`eda.ipynb`](https://colab.research.google.com/github/andinaufal120/kalimantan-frp-prediction/blob/main/eda.ipynb)** — Characterizes the FRP target (raw vs. log1p), temporal and spatial structure, and outliers (p99 ≈ 52.6 MW, concentrated in Aug–Sep 2023). Uses Spearman correlation to narrow 32 candidate features down to 14.
2. **[`model.ipynb`](https://colab.research.google.com/github/andinaufal120/kalimantan-frp-prediction/blob/main/model.ipynb)** — Trains Random Forest and XGBoost baselines using a custom sliding-window cross-validation (12-month train / 3-month test / 2-month step, 11 folds) so evaluation respects time ordering and isolates the El Niño peak and 2024 anomaly folds.
3. **[`playbook.ipynb`](https://colab.research.google.com/github/andinaufal120/kalimantan-frp-prediction/blob/main/playbook.ipynb)** — Interpretability and production-facing analysis: feature importance, geographic visualization (Cartopy), error distribution by percentile, and alternative CV strategies.
4. **`autogluon.ipynb`** — Benchmarks an AutoGluon ensemble (`medium_quality` preset, 600s/fold) against the RF/XGBoost baselines using the same fold structure. Runs in its own environment (see Setup).

## Setup

The project uses two separate virtual environments because AutoGluon's dependency requirements conflict with the main modeling stack (scikit-learn, XGBoost, Cartopy).

**Main environment** (for `eda.ipynb`, `model.ipynb`, `playbook.ipynb`):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**AutoGluon environment** (for `autogluon.ipynb` only):

```bash
python -m venv .venv-gluon
source .venv-gluon/bin/activate
pip install autogluon
```

## Data

`Dataset_ML.csv` is the processed dataset (59,155 rows, 46 columns) used for modeling. The target is `frp_log1p` (log1p-transformed FRP, raw range ~0.1–359 MW). Features span several groups:

- **Vegetation**: EVI, NDVI age, NDWI
- **Temperature/moisture**: land surface temperature (day/night), vapor pressure deficit
- **Meteorology**: 2m temperature/dewpoint, wind components, hourly precipitation, solar radiation
- **Temporal**: circular encodings for hour-of-day and day-of-year
- **Land cover**: land cover type, peat type class (none/peat/deep peat)
- **Spatial**: distance to river/road/settlement, population density, forest clearance index

After Spearman-based feature selection in `eda.ipynb`, the final 14 modeling features are:
`EVI`, `ndvi_age_days`, `lst_day`, `lst_night`, `land_cover_type`, `peat_type_class`, `ssrd`, `tp_hourly`, `hour_sin`, `hour_cos`, `doy_sin`, `doy_cos`, `vpd_kpa`, `wind_speed_m_s`.

`data.csv` is an earlier, less processed version of the dataset.

## Results

Per-fold metrics (mean ± std across 11 folds), evaluated on log1p-transformed FRP:

| Model | RMSE (log1p) | MAE (log1p) | MAE (MW) | R² |
|---|---|---|---|---|
| Random Forest | 0.5646 ± 0.0927 | 0.4259 ± 0.0673 | 3.75 ± 1.57 | 0.4085 ± 0.1071 |
| XGBoost | 0.5845 ± 0.0891 | 0.4391 ± 0.0689 | 3.83 ± 1.58 | 0.3661 ± 0.0994 |
| AutoGluon | 0.5836 ± 0.0913 | 0.4415 ± 0.0693 | 3.87 ± 1.58 | 0.3687 ± 0.1004 |

**Key findings:**
- Random Forest performs best overall and is the most stable across folds.
- The 2023 El Niño peak folds (Aug–Oct 2023) are the hardest to predict (RMSE ~0.64 vs. ~0.41 in quiet periods).
- The 2024 anomaly folds are out-of-distribution: models trained primarily on El Niño-driven fire behavior struggle to generalize to the neutral-ENSO 2024 dry season.

## Artifacts

- `artifacts/figures/` — Generated plots (FRP distributions, temporal trends, outlier analysis, Spearman correlation matrix, CV fold layout, per-fold metrics, geographic FRP map).
- `artifacts/reports/` — AutoGluon per-fold results and leaderboards (`ag_results.csv`, `ag_leaderboard.csv`).
- `artifacts/models/` — Saved AutoGluon predictor.

## License

MIT — see [LICENSE](LICENSE).
