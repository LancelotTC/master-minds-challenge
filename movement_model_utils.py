from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, TypeVar

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from tqdm.auto import tqdm

MAIN_DATASET_PATH = Path("data") / "main_dataset.csv"
ID_COLUMN = "IdMovement"
ROW_ID_COLUMN = "row_number"
TARGET_COLUMN = "NbPaxTotal"
PREDICTION_COLUMN = f"{TARGET_COLUMN}Prediction"
PREDICTION_OUTPUT_DIR = Path("predictions")
PROJECT_ROOT = Path(__file__).resolve().parent
HYPERPARAMETERS_RESULTS_PATH = PROJECT_ROOT / "hyperparameters.json"
MAX_PLOT_POINTS = 80_000
NULL_LIKE_STRINGS = {"", "NULL", "NONE", "NAN", "NAT"}
PREDICTION_MODE_MISSING_TARGET = "missing_target"
PREDICTION_MODE_KNOWN_TARGET = "known_target"

BASE_FEATURE_COLUMNS = [
    "IdAircraftType",
    "IdBusinessUnitType",
    "IdBusContactType",
    "airlineOACICode",
    "AirportPrevious",
    "ServiceCode",
    "FlightNumberNormalized",
    "LTScheduledDatetime",
    "Direction",
    "SysTerminal",
    "NbOfSeats",
    "day_of_week",
    "is_weekend",
    "season",
    "dest_country",
    "is_fr_public_holiday",
    "is_fr_school_holiday_zone_a",
    "is_dest_public_holiday",
    "is_dest_school_holiday",
    "precipitation_sum",
    "rain_sum",
    "snowfall_sum",
    "windspeed_10m_max",
]

ENGINEERED_FEATURE_COLUMNS = [
    "month_of_year",
    "hour_of_day",
    "hour_sin",
    "hour_cos",
    "day_of_week_sin",
    "day_of_week_cos",
    "month_sin",
    "month_cos",
    "is_any_day_off",
    "days_until_next_day_off",
    "days_until_next_workday",
]

FEATURE_COLUMNS = [
    *BASE_FEATURE_COLUMNS,
    *ENGINEERED_FEATURE_COLUMNS,
]

CATEGORICAL_FEATURE_COLUMNS = {
    "IdAircraftType",
    "airlineOACICode",
    "AirportPrevious",
    "ServiceCode",
    "FlightNumberNormalized",
    "Direction",
    "SysTerminal",
    "season",
    "dest_country",
}

NUMERIC_FEATURE_COLUMNS = set(FEATURE_COLUMNS) - CATEGORICAL_FEATURE_COLUMNS
REQUIRED_COLUMNS = [ID_COLUMN, TARGET_COLUMN, *FEATURE_COLUMNS]
ProgressResult = TypeVar("ProgressResult")


def load_hyperparameter_results(path: str | Path = HYPERPARAMETERS_RESULTS_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def clean_model_params(params: dict[str, object]) -> dict[str, object]:
    return {str(key).removeprefix("model__"): value for key, value in params.items()}


def run_progress_step(
    progress_bar,
    label: str,
    action: Callable[..., ProgressResult],
    /,
    *args,
    **kwargs,
) -> ProgressResult:
    if progress_bar is not None:
        progress_bar.set_postfix_str(label)

    result = action(*args, **kwargs)

    if progress_bar is not None:
        progress_bar.update(1)

    return result


def load_main_dataset_dataframe(
    limit: int | None = None,
    dataset_path: str | Path = MAIN_DATASET_PATH,
) -> pd.DataFrame:
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    dataframe = pd.read_csv(
        dataset_path,
        usecols=lambda column_name: column_name in REQUIRED_COLUMNS,
        nrows=limit,
        low_memory=False,
    )

    missing_columns = [column_name for column_name in REQUIRED_COLUMNS if column_name not in dataframe.columns]
    if missing_columns:
        raise RuntimeError(f"Missing required columns in {dataset_path}: {missing_columns}")

    if ROW_ID_COLUMN in dataframe.columns:
        raise RuntimeError(f"Column '{ROW_ID_COLUMN}' is reserved for internal use. Rename it in the dataset.")

    dataframe.insert(0, ROW_ID_COLUMN, np.arange(1, len(dataframe) + 1))
    return clean_dataframe(dataframe)


def clean_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    cleaned = dataframe.copy()

    for column_name in cleaned.columns:
        cleaned[column_name] = replace_null_like_values(cleaned[column_name])

    cleaned[ROW_ID_COLUMN] = pd.to_numeric(cleaned[ROW_ID_COLUMN], errors="coerce")
    cleaned[ID_COLUMN] = to_object_string_series(cleaned[ID_COLUMN])
    cleaned[TARGET_COLUMN] = pd.to_numeric(cleaned[TARGET_COLUMN], errors="coerce")

    for column_name in FEATURE_COLUMNS:
        if column_name in CATEGORICAL_FEATURE_COLUMNS:
            cleaned[column_name] = to_object_string_series(cleaned[column_name])
        elif column_name == "LTScheduledDatetime":
            cleaned[column_name] = pd.to_datetime(cleaned[column_name], errors="coerce")
        else:
            cleaned[column_name] = pd.to_numeric(cleaned[column_name], errors="coerce")

    object_columns = cleaned.select_dtypes(include=["object"]).columns
    for column_name in object_columns:
        cleaned[column_name] = cleaned[column_name].replace({pd.NA: np.nan})

    return cleaned


def replace_null_like_values(series: pd.Series) -> pd.Series:
    if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
        return series

    as_string = series.astype("string").str.strip()
    null_mask = as_string.isna() | as_string.str.upper().isin(NULL_LIKE_STRINGS)
    return series.mask(null_mask, other=np.nan)


def to_object_string_series(series: pd.Series) -> pd.Series:
    string_values = series.astype("string")
    object_values = string_values.astype("object")
    return object_values.where(pd.notna(object_values), np.nan)


def load_training_and_prediction_frames(
    limit: int | None = None,
    prediction_mode: str = PREDICTION_MODE_MISSING_TARGET,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame]:
    if prediction_mode not in {PREDICTION_MODE_MISSING_TARGET, PREDICTION_MODE_KNOWN_TARGET}:
        raise ValueError(
            "prediction_mode must be 'missing_target' or 'known_target'."
        )

    raw_dataframe = load_main_dataset_dataframe(limit=limit)
    feature_dataframe = build_feature_dataframe(raw_dataframe)
    target = pd.to_numeric(raw_dataframe[TARGET_COLUMN], errors="coerce")
    complete_feature_mask = pd.Series(True, index=feature_dataframe.index)
    discarded_feature_mask = ~complete_feature_mask

    print_dataset_debug_summary(raw_dataframe, target, complete_feature_mask)
    if discarded_feature_mask.any():
        print_discarded_rows_debug(
            raw_dataframe,
            feature_dataframe,
            discarded_feature_mask,
        )

    training_mask = target.notna()
    training_features = feature_dataframe.loc[training_mask, FEATURE_COLUMNS].copy()
    training_target = target.loc[training_mask].astype(float)

    prediction_identifier_columns = [ROW_ID_COLUMN, ID_COLUMN]
    if prediction_mode == PREDICTION_MODE_KNOWN_TARGET:
        prediction_mask = training_mask
        prediction_identifier_columns.append(TARGET_COLUMN)
    else:
        prediction_mask = target.isna()

    prediction_identifiers = raw_dataframe.loc[prediction_mask, prediction_identifier_columns].copy()
    prediction_features = feature_dataframe.loc[
        prediction_mask,
        FEATURE_COLUMNS,
    ].copy()

    return (
        training_features,
        training_target,
        prediction_features,
        prediction_identifiers,
    )


def build_feature_dataframe(raw_dataframe: pd.DataFrame) -> pd.DataFrame:
    features = raw_dataframe[FEATURE_COLUMNS].copy()
    scheduled = pd.to_datetime(features["LTScheduledDatetime"], errors="coerce")
    datetime_values = scheduled.astype("int64", copy=False)
    features["LTScheduledDatetime"] = pd.Series(
        np.where(scheduled.notna(), datetime_values / 1_000_000_000, np.nan),
        index=raw_dataframe.index,
    )

    for column_name in CATEGORICAL_FEATURE_COLUMNS:
        features[column_name] = to_object_string_series(features[column_name])

    for column_name in NUMERIC_FEATURE_COLUMNS:
        features[column_name] = pd.to_numeric(features[column_name], errors="coerce")

    object_columns = features.select_dtypes(include=["object"]).columns
    for column_name in object_columns:
        features[column_name] = features[column_name].replace({pd.NA: np.nan})

    return features


def print_dataset_debug_summary(
    raw_dataframe: pd.DataFrame,
    target: pd.Series,
    complete_feature_mask: pd.Series,
) -> None:
    return
    total_rows = len(raw_dataframe)
    complete_rows = int(complete_feature_mask.sum())
    incomplete_rows = total_rows - complete_rows
    target_present_rows = int(target.notna().sum())
    prediction_candidate_rows = int(target.isna().sum())

    print(
        "Dataset debug:"
        f" total_rows={total_rows}"
        f" complete_feature_rows={complete_rows}"
        f" discarded_for_missing_features={incomplete_rows}"
        f" target_present_rows={target_present_rows}"
        f" target_missing_rows={prediction_candidate_rows}"
    )


def print_discarded_rows_debug(
    raw_dataframe: pd.DataFrame,
    feature_dataframe: pd.DataFrame,
    discarded_feature_mask: pd.Series,
) -> None:
    return
    discarded_features = feature_dataframe.loc[discarded_feature_mask, FEATURE_COLUMNS]
    discarded_raw_rows = raw_dataframe.loc[discarded_feature_mask]

    print("\nDiscarded rows because at least one kept feature is null:")
    for row_index in discarded_features.index:
        missing_columns = discarded_features.columns[discarded_features.loc[row_index].isna()].tolist()
        row_number = discarded_raw_rows.at[row_index, ROW_ID_COLUMN]
        movement_id = (
            discarded_raw_rows.at[row_index, ID_COLUMN]
            if ID_COLUMN in discarded_raw_rows.columns
            else "N/A"
        )
        target_value = discarded_raw_rows.at[row_index, TARGET_COLUMN]
        print(
            f"row_number={row_number} "
            f"IdMovement={movement_id} "
            f"{TARGET_COLUMN}={target_value} "
            f"missing_features={missing_columns}"
        )


def make_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    categorical_columns = list(CATEGORICAL_FEATURE_COLUMNS)
    numeric_columns = [column_name for column_name in FEATURE_COLUMNS if column_name not in categorical_columns]

    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                numeric_columns,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OrdinalEncoder(
                                handle_unknown="use_encoded_value",
                                unknown_value=-1,
                            ),
                        ),
                    ]
                ),
                categorical_columns,
            ),
        ],
        remainder="drop",
    )


def write_predictions(
    predictions: np.ndarray | list[float],
    identifiers: pd.DataFrame,
    filename_stem: str,
) -> Path:
    output = identifiers.copy()
    clipped_predictions = np.clip(np.rint(np.asarray(predictions)), 0, None).astype(int)
    output[PREDICTION_COLUMN] = clipped_predictions

    model_folder_name = get_model_folder_name(filename_stem)
    model_output_dir = PREDICTION_OUTPUT_DIR / model_folder_name
    model_output_dir.mkdir(parents=True, exist_ok=True)
    output_path = model_output_dir / f"{filename_stem}.csv"
    output.to_csv(output_path, index=False)
    return output_path


def _prepare_prediction_plot_data(dataframe: pd.DataFrame) -> pd.DataFrame:
    paired = pd.DataFrame(
        {
            TARGET_COLUMN: pd.to_numeric(dataframe[TARGET_COLUMN], errors="coerce"),
            PREDICTION_COLUMN: pd.to_numeric(dataframe[PREDICTION_COLUMN], errors="coerce"),
        }
    ).dropna()
    return paired


def plot_prediction_results(
    predictions_file: str | Path,
    show: bool = False,
    max_points: int = MAX_PLOT_POINTS,
) -> Path | None:
    predictions_file = Path(predictions_file)
    dataframe = pd.read_csv(predictions_file)

    if TARGET_COLUMN not in dataframe.columns or PREDICTION_COLUMN not in dataframe.columns:
        print(
            f"Skipping plot for {predictions_file}: "
            f"requires both {TARGET_COLUMN} and {PREDICTION_COLUMN}."
        )
        return None

    plot_data = _prepare_prediction_plot_data(dataframe)
    if plot_data.empty:
        print(f"Skipping plot for {predictions_file}: no plottable values found.")
        return None

    if max_points > 0 and len(plot_data) > max_points:
        plot_data = plot_data.sample(n=max_points, random_state=42)

    import matplotlib.pyplot as plt

    actual_values = plot_data[TARGET_COLUMN].to_numpy(dtype=float)
    predicted_values = plot_data[PREDICTION_COLUMN].to_numpy(dtype=float)
    residual_values = predicted_values - actual_values

    axis_min = min(0.0, float(np.nanmin([actual_values.min(), predicted_values.min()])))
    axis_max = float(np.nanmax([actual_values.max(), predicted_values.max()]))
    if not np.isfinite(axis_min) or not np.isfinite(axis_max):
        print(f"Skipping plot for {predictions_file}: no finite values found.")
        return None
    if axis_max <= axis_min:
        axis_max = axis_min + 1.0

    line_x = np.linspace(axis_min, axis_max, 300)

    figure, axes = plt.subplots(1, 2, figsize=(14, 6))

    density_main = axes[0].hexbin(
        actual_values,
        predicted_values,
        gridsize=65,
        mincnt=1,
        bins="log",
        cmap="viridis",
    )
    axes[0].plot(line_x, line_x, color="white", linewidth=1.5, label="Ideal")
    axes[0].plot(line_x, line_x * 1.1, color="orange", linestyle="--", linewidth=1.2, label="+/-10%")
    axes[0].plot(line_x, line_x * 0.9, color="orange", linestyle="--", linewidth=1.2)
    axes[0].fill_between(line_x, line_x * 0.9, line_x * 1.1, color="orange", alpha=0.08)
    axes[0].set_xlim(axis_min, axis_max)
    axes[0].set_ylim(axis_min, axis_max)
    axes[0].set_xlabel(TARGET_COLUMN)
    axes[0].set_ylabel(PREDICTION_COLUMN)
    axes[0].set_title("Predicted vs actual")
    axes[0].legend(loc="upper left")
    figure.colorbar(density_main, ax=axes[0], label="log10(count)")

    density_residual = axes[1].hexbin(
        actual_values,
        residual_values,
        gridsize=65,
        mincnt=1,
        bins="log",
        cmap="magma",
    )
    axes[1].axhline(0.0, color="white", linewidth=1.5)
    axes[1].set_xlabel(TARGET_COLUMN)
    axes[1].set_ylabel(f"{PREDICTION_COLUMN} - {TARGET_COLUMN}")
    axes[1].set_title("Residuals")
    figure.colorbar(density_residual, ax=axes[1], label="log10(count)")

    figure.suptitle(f"{predictions_file.stem} (n={len(plot_data):,})")
    figure.tight_layout(rect=(0, 0, 1, 0.96))

    plot_path = predictions_file.with_suffix(".png")
    figure.savefig(plot_path, dpi=160, bbox_inches="tight")
    print(f"Plot written to {plot_path}")

    if show:
        plt.show()

    plt.close(figure)
    return plot_path


def regenerate_prediction_plots(
    predictions_root: str | Path = PREDICTION_OUTPUT_DIR,
    pattern: str = "*_preds.csv",
    show: bool = False,
    max_points: int = MAX_PLOT_POINTS,
) -> list[Path]:
    predictions_root = Path(predictions_root)
    if not predictions_root.exists():
        raise FileNotFoundError(f"Predictions directory not found: {predictions_root}")

    prediction_files = sorted(predictions_root.rglob(pattern))
    if not prediction_files:
        print(f"No prediction CSV files found in {predictions_root} with pattern '{pattern}'.")
        return []

    generated_plots: list[Path] = []
    for prediction_file in tqdm(prediction_files, desc="Regenerating plots", unit="file"):
        plot_path = plot_prediction_results(prediction_file, show=show, max_points=max_points)
        if plot_path is not None:
            generated_plots.append(plot_path)

    print(f"Regenerated {len(generated_plots)} plot(s) from {len(prediction_files)} prediction file(s).")
    return generated_plots


def get_model_folder_name(filename_stem: str) -> str:
    if filename_stem.endswith("_preds"):
        return filename_stem.removesuffix("_preds")
    return filename_stem
