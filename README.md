# Master Minds Challenge

This repository contains a tabular machine-learning pipeline for airport movement prediction.

## Participants

- Lancelot TARIOT CAMILLE
- Romain POIRRIER
- Maïa JOUENNE
- Sang NGUYEN

At a high level, the project:

1. downloads a raw movement dataset from BigQuery when needed,
2. rebuilds a richer modeling dataset from it,
3. searches hyperparameters for the supported regressors,
4. trains models and exports prediction files, validation subsets, test-set files, and plots.

The most important convention in the project is this:

- `data/original_dataset` or `data/original_dataset.csv` is the raw local snapshot,
- `data/main_dataset.csv` is the derived dataset used for modeling.

The pipeline first checks for a local `data/original_dataset`, then `data/original_dataset.csv`, before downloading from BigQuery.
`main_dataset.csv` is safe to regenerate.

## Table Of Contents

- [Pipeline](#pipeline)
- [How To Run](#how-to-run)
- [Data And Environment](#data-and-environment)
- [What Each Script Does](#what-each-script-does)
- [Targets](#targets)
- [Feature Families](#feature-families)
- [Main Files And Folders](#main-files-and-folders)
- [Models](#models)
- [What You May Want To Change](#what-you-may-want-to-change)
- [Practical Note](#practical-note)

## Pipeline

![Pipeline diagram](images/pipeline_diagram.png)

## How To Run

The repository is currently documented around Python 3.13.

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Then run the main pipeline in this order:

```bash
python update_datasets.py
python 1_find_best_hyperparameters.py
python 2_individual_predictions.py
```

If you only want to regenerate plots from saved validation prediction files:

```bash
python 3_regenerate_prediction_plots.py
```

`3_regenerate_prediction_plots.py` also supports:

- `--predictions-dir`
- `--pattern`
- `--show`
- `--max-points`

## Data And Environment

`update_datasets.py` calls `load_dotenv()` automatically, so a local `.env` file is supported.

BigQuery access is only needed when neither `data/original_dataset` nor `data/original_dataset.csv` exists locally.
For that download step, the code expects:

- `PROJECT_ID`
- `DATASET_ID`
- `TABLE_ID`

It also supports either of these optional credential-path variables:

- `SERVICE_ACCOUNT_KEY_PATH`
- `YOUR_SERVICE_ACCOUNT_KEY_PATH`

If one of those two variables is set, the script forwards it to `GOOGLE_APPLICATION_CREDENTIALS`.

The dataset build also uses external reference data:

- `data/weather/weather.csv`
  If this file is missing, the script downloads weather history and forecast data from Open-Meteo.
- `data/holidays/openholidays_2023_2027.csv`
  Used for foreign public-holiday and school-holiday features.
  If it is missing, those foreign holiday features fall back to default values.
- `data/holidays/upply-airports.csv`
  Used for airport coordinates and route-distance features.
  If it is missing, `flight_distance_km` falls back to missing values.
- `data/world_events/clean_airports.parquet`
  Used for airport-to-country mapping and route-country features.
- `data/world_events/smoothed_risk_events.parquet`
  Used for world-event risk features.

The two `data/world_events/*.parquet` files are optional at runtime.
If `clean_airports.parquet` is absent, country and route-derived features become partially unavailable.
If `smoothed_risk_events.parquet` is absent, world-event risk features fall back to neutral default values.

## What Each Script Does

[`update_datasets.py`](./update_datasets.py)

This script is the dataset builder. It:

- reuses a local `data/original_dataset` or `data/original_dataset.csv` when present,
- only downloads from BigQuery when no local raw snapshot exists,
- refreshes `data/weather/weather.csv` if it is missing,
- loads holiday, airport, and optional local world-event reference data,
- adds weather, holiday, route, world-event, and engineered features,
- writes the result to `data/main_dataset.csv`.

[`1_find_best_hyperparameters.py`](./1_find_best_hyperparameters.py)

This script loads `data/main_dataset.csv`, prepares the training data, and runs Bayesian hyperparameter search.

The best results are written to:

- `hyperparameters.json`

[`2_individual_predictions.py`](./2_individual_predictions.py)

This script loads the dataset and tuned hyperparameters, trains the enabled models, and writes prediction artifacts.

For each supported model, it can write:

- `<ModelName>_preds.csv`
- `<ModelName>_validation_preds.csv`
- `<ModelName>_preds_test_set.csv`
- `<ModelName>_validation_preds.png`
- `<ModelName>_validation_preds_hourly_profile.png`
- `<ModelName>_validation_preds_pmr.png`

All of those files are written under `predictions/<ModelName>/`.

[`3_regenerate_prediction_plots.py`](./3_regenerate_prediction_plots.py)

This script rebuilds validation plots and hourly-profile plots from saved validation prediction CSV files.

## Targets

The pipeline predicts two targets.

### `NbPaxTotal`

Main passenger-count target.

### `PMR`

Secondary target derived from:

- `OzionPHMRPaxArrival`
- `OzionPHMRPaxDeparture`

The value is chosen according to movement direction, with a fallback for anomalous rows.

## Feature Families

The derived dataset includes several feature groups.

### Weather

- `precipitation_sum`
- `rain_sum`
- `snowfall_sum`
- `windspeed_10m_max`

### Route And Geography

- `origin_country`
- `dest_country`
- `is_domestic`
- `is_route_domestic`
- `has_stopover`
- `flight_distance_km`

### Calendar And Holidays

- `day_of_week`
- `is_weekend`
- `season`
- `is_fr_public_holiday`
- `is_bridge_day`
- `is_fr_school_holiday_zone_a`
- `is_fr_school_holiday_zone_b`
- `is_fr_school_holiday_zone_c`
- `is_first_last_day_of_school_holiday`
- `is_origin_public_holiday`
- `is_origin_school_holiday`
- `is_dest_public_holiday`
- `is_dest_school_holiday`
- `is_any_day_off`
- `days_until_next_day_off`
- `days_until_next_workday`

### World Events

- `risk_score`
- `nbDaysSinceRiskStarted`
- `nbDaysSinceSafeStarted`

### Time Encoding

- `month_of_year`
- `hour_of_day`
- `hour_sin`
- `hour_cos`
- `day_of_week_sin`
- `day_of_week_cos`
- `month_sin`
- `month_cos`

### Passenger History

- `pax_lag_1`
- `pax_lag_7`
- `pax_lag_30`
- `pax_rolling_7`
- `pax_ewm_7`
- `pax_diff_1`
- `pax_diff_7`

### Model-Time Daily Forecast Features

These features are generated inside the model pipeline by the Prophet-based transformer:

- `daily_seats`
- `daily_forecast_pax`
- `daily_forecast_load_factor`
- `relative_seat_share_day`
- `forecast_pax_per_seat_day`

## Main Files And Folders

[`pipeline_config.py`](./pipeline_config.py)

Shared project configuration:

- dataset paths,
- model paths,
- common column names,
- feature-group names,
- plotting limits,
- reference values.

[`movement_model_utils.py`](./movement_model_utils.py)

Shared modeling utilities:

- dataset loading and cleaning,
- PMR target derivation,
- runtime config loading from `model_configs/*.json`,
- date-based train and validation splitting,
- model pipeline construction,
- prediction export,
- plotting.

[`model_configs`](./model_configs)

Model-facing JSON configuration for:

- feature selection,
- row filters,
- prediction overrides.

See [`model_configs/README.md`](./model_configs/README.md) for the supported JSON schema.

[`predictions`](./predictions)

Per-model prediction outputs and generated plots.

[`data`](./data)

Main working data folder.

The most important files inside it are:

- `data/original_dataset` or `data/original_dataset.csv`
- `data/main_dataset.csv`
- `data/weather/weather.csv`
- `data/holidays/openholidays_2023_2027.csv`
- `data/holidays/upply-airports.csv`
- `data/world_events/clean_airports.parquet` if available
- `data/world_events/smoothed_risk_events.parquet` if available

## Models

The hyperparameter search currently includes:

- `XGBRegressor`
- `HistGradientBoostingRegressor`
- `LGBMRegressor`

`2_individual_predictions.py` currently instantiates only those same three regressors from `hyperparameters.json`.

## What You May Want To Change

If you want to change train, validation, or test windows:

- edit `PredictionWindowConfig` in [`2_individual_predictions.py`](./2_individual_predictions.py)

If you want to change hyperparameter-search limits:

- edit `SearchConfig` in [`1_find_best_hyperparameters.py`](./1_find_best_hyperparameters.py)

If you want to change feature selection, row filters, or prediction overrides:

- edit files in [`model_configs`](./model_configs)

If you want to rebuild the modeling dataset from the raw snapshot:

- run `python update_datasets.py`

## Practical Note

If results look stale or inconsistent, the usual reset is:

1. keep your local `data/original_dataset` or `data/original_dataset.csv` if you want to preserve the raw snapshot,
2. delete `data/main_dataset.csv`,
3. optionally delete `data/weather/weather.csv` if you want a fresh weather pull,
4. rerun `python update_datasets.py`,
5. rerun tuning or predictions if needed.
