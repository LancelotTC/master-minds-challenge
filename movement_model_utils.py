from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, TypeVar

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import TimeSeriesSplit
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
MODEL_CONFIG_DIR = PROJECT_ROOT / "model_configs"
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
    "origin_country",
    "dest_country",
    "is_domestic",
    "is_route_domestic",
    "has_stopover",
    "is_fr_public_holiday",
    "is_bridge_day",
    "is_fr_school_holiday_zone_a",
    "is_fr_school_holiday_zone_b",
    "is_fr_school_holiday_zone_c",
    "is_first_last_day_of_school_holiday",
    "is_origin_public_holiday",
    "is_origin_school_holiday",
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
    "origin_country",
    "dest_country",
    "day_of_week",
    "is_weekend",
    "is_domestic",
    "is_route_domestic",
    "has_stopover",
    "is_fr_public_holiday",
    "is_bridge_day",
    "is_fr_school_holiday_zone_a",
    "is_fr_school_holiday_zone_b",
    "is_fr_school_holiday_zone_c",
    "is_first_last_day_of_school_holiday",
    "is_origin_public_holiday",
    "is_origin_school_holiday",
    "is_dest_public_holiday",
    "is_dest_school_holiday",
    "is_any_day_off",
}

NUMERIC_FEATURE_COLUMNS = set(FEATURE_COLUMNS) - CATEGORICAL_FEATURE_COLUMNS
REQUIRED_COLUMNS = [ID_COLUMN, TARGET_COLUMN, *FEATURE_COLUMNS]
ProgressResult = TypeVar("ProgressResult")


@dataclass(frozen=True)
class ModelRuntimeConfig:
    feature_columns: tuple[str, ...]
    categorical_feature_columns: tuple[str, ...]
    row_filters: tuple[dict[str, Any], ...]
    prediction_overrides: tuple[dict[str, Any], ...]
    condition_columns: tuple[str, ...]
    prediction_override_columns: tuple[str, ...]
    required_columns: tuple[str, ...]


def _deduplicate_preserve_order(values: list[str] | tuple[str, ...] | set[str]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values))


def _extract_condition_columns(condition: dict[str, Any] | None) -> list[str]:
    if not condition:
        return []

    columns: list[str] = []
    for key in ("all", "any"):
        for child in condition.get(key, []):
            columns.extend(_extract_condition_columns(child))

    if "not" in condition:
        columns.extend(_extract_condition_columns(condition["not"]))

    column_name = condition.get("column")
    if column_name:
        columns.append(str(column_name))

    return _deduplicate_preserve_order(columns)


def _extract_rule_columns(rules: list[dict[str, Any]]) -> list[str]:
    columns: list[str] = []
    for rule in rules:
        columns.extend(_extract_condition_columns(rule.get("when")))
    return _deduplicate_preserve_order(columns)


def _extract_prediction_override_action_columns(prediction_overrides: list[dict[str, Any]]) -> list[str]:
    columns: list[str] = []
    for rule in prediction_overrides:
        for key in ("set_prediction_from_column", "clip_upper_column", "clip_lower_column"):
            column_name = rule.get(key)
            if column_name:
                columns.append(str(column_name))
    return _deduplicate_preserve_order(columns)


def _merge_string_list_setting(
    current_values: list[str],
    payload: dict[str, Any],
    exact_key: str,
    add_key: str,
    remove_key: str,
) -> list[str]:
    merged_values = list(current_values)

    if exact_key in payload:
        merged_values = _deduplicate_preserve_order(payload.get(exact_key, []))

    for value in payload.get(add_key, []):
        if value not in merged_values:
            merged_values.append(value)

    remove_values = set(payload.get(remove_key, []))
    if remove_values:
        merged_values = [value for value in merged_values if value not in remove_values]

    return merged_values


def _strip_json_comments(raw_text: str) -> str:
    stripped_characters: list[str] = []
    in_string = False
    escape_next = False
    index = 0

    while index < len(raw_text):
        character = raw_text[index]
        next_character = raw_text[index + 1] if index + 1 < len(raw_text) else ""

        if in_string:
            stripped_characters.append(character)
            if escape_next:
                escape_next = False
            elif character == "\\":
                escape_next = True
            elif character == '"':
                in_string = False
            index += 1
            continue

        if character == '"':
            in_string = True
            stripped_characters.append(character)
            index += 1
            continue

        if character == "/" and next_character == "/":
            index += 2
            while index < len(raw_text) and raw_text[index] not in "\r\n":
                index += 1
            continue

        if character == "/" and next_character == "*":
            index += 2
            while index + 1 < len(raw_text) and not (raw_text[index] == "*" and raw_text[index + 1] == "/"):
                index += 1
            index += 2
            continue

        stripped_characters.append(character)
        index += 1

    return "".join(stripped_characters)


def _strip_trailing_json_commas(raw_text: str) -> str:
    return re.sub(r",(\s*[}\]])", r"\1", raw_text)


def load_json_config(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path)
    raw_text = config_path.read_text(encoding="utf-8")
    normalized_text = _strip_trailing_json_commas(_strip_json_comments(raw_text))

    try:
        return json.loads(normalized_text)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Invalid config JSON in {config_path} at line {error.lineno} column {error.colno}: {error.msg}"
        ) from error


@lru_cache(maxsize=1)
def load_model_runtime_config(config_dir: str | Path = MODEL_CONFIG_DIR) -> ModelRuntimeConfig:
    config_dir = Path(config_dir)

    feature_columns = list(FEATURE_COLUMNS)
    categorical_feature_columns = _deduplicate_preserve_order(CATEGORICAL_FEATURE_COLUMNS)
    row_filters: list[dict[str, Any]] = []
    prediction_overrides: list[dict[str, Any]] = []

    if config_dir.exists():
        for config_path in sorted(config_dir.glob("*.json")):
            payload = load_json_config(config_path)

            if not payload.get("enabled", True):
                continue

            feature_columns = _merge_string_list_setting(
                feature_columns,
                payload,
                "feature_columns",
                "add_feature_columns",
                "remove_feature_columns",
            )
            categorical_feature_columns = _merge_string_list_setting(
                categorical_feature_columns,
                payload,
                "categorical_feature_columns",
                "add_categorical_feature_columns",
                "remove_categorical_feature_columns",
            )
            row_filters.extend(payload.get("row_filters", []))
            prediction_overrides.extend(payload.get("prediction_overrides", []))

    categorical_feature_columns = [
        column_name for column_name in categorical_feature_columns if column_name in feature_columns
    ]
    row_filter_columns = _extract_rule_columns(row_filters)
    prediction_override_columns = _deduplicate_preserve_order(
        [
            *_extract_rule_columns(prediction_overrides),
            *_extract_prediction_override_action_columns(prediction_overrides),
        ]
    )
    required_columns = _deduplicate_preserve_order(
        [ID_COLUMN, TARGET_COLUMN, *feature_columns, *row_filter_columns, *prediction_override_columns]
    )

    return ModelRuntimeConfig(
        feature_columns=tuple(feature_columns),
        categorical_feature_columns=tuple(categorical_feature_columns),
        row_filters=tuple(row_filters),
        prediction_overrides=tuple(prediction_overrides),
        condition_columns=tuple(_deduplicate_preserve_order([*row_filter_columns, *prediction_override_columns])),
        prediction_override_columns=tuple(prediction_override_columns),
        required_columns=tuple(required_columns),
    )


def _coerce_condition_value(series: pd.Series, value: Any) -> Any:
    if value is None:
        return None

    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(value, errors="coerce")

    if pd.api.types.is_numeric_dtype(series):
        coerced = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        return coerced.item() if hasattr(coerced, "item") else coerced

    return value


def _evaluate_leaf_condition(dataframe: pd.DataFrame, condition: dict[str, Any]) -> pd.Series:
    column_name = condition.get("column")
    if not column_name:
        raise ValueError(f"Condition is missing 'column': {condition}")
    if column_name not in dataframe.columns:
        raise RuntimeError(f"Condition references unknown column '{column_name}'.")

    operator = str(condition.get("operator", "eq")).lower()
    series = dataframe[column_name]
    value = condition.get("value")

    if operator in {"is_null", "null"}:
        return series.isna()
    if operator in {"not_null", "is_not_null"}:
        return series.notna()

    if operator in {"in", "not_in"}:
        raw_values = value if isinstance(value, list) else [value]
        coerced_values = [_coerce_condition_value(series, item) for item in raw_values]
        mask = series.isin(coerced_values)
        return ~mask if operator == "not_in" else mask

    coerced_value = _coerce_condition_value(series, value)
    operation_map = {
        "eq": series.eq,
        "==": series.eq,
        "ne": series.ne,
        "!=": series.ne,
        "gt": series.gt,
        ">": series.gt,
        "gte": series.ge,
        ">=": series.ge,
        "lt": series.lt,
        "<": series.lt,
        "lte": series.le,
        "<=": series.le,
    }
    if operator not in operation_map:
        raise ValueError(f"Unsupported condition operator '{operator}'.")
    return operation_map[operator](coerced_value).fillna(False)


def evaluate_condition(dataframe: pd.DataFrame, condition: dict[str, Any] | None) -> pd.Series:
    if condition is None:
        return pd.Series(True, index=dataframe.index)

    if "all" in condition:
        mask = pd.Series(True, index=dataframe.index)
        for child in condition["all"]:
            mask &= evaluate_condition(dataframe, child)
        return mask

    if "any" in condition:
        mask = pd.Series(False, index=dataframe.index)
        for child in condition["any"]:
            mask |= evaluate_condition(dataframe, child)
        return mask

    if "not" in condition:
        return ~evaluate_condition(dataframe, condition["not"])

    return _evaluate_leaf_condition(dataframe, condition)


def apply_row_filters(dataframe: pd.DataFrame, row_filters: tuple[dict[str, Any], ...]) -> pd.DataFrame:
    if not row_filters:
        return dataframe

    keep_rules = [rule for rule in row_filters if str(rule.get("action", "drop")).lower() == "keep"]
    drop_rules = [rule for rule in row_filters if str(rule.get("action", "drop")).lower() != "keep"]

    if keep_rules:
        keep_mask = pd.Series(False, index=dataframe.index)
        for rule in keep_rules:
            keep_mask |= evaluate_condition(dataframe, rule.get("when"))
    else:
        keep_mask = pd.Series(True, index=dataframe.index)

    for rule in drop_rules:
        keep_mask &= ~evaluate_condition(dataframe, rule.get("when"))

    filtered = dataframe.loc[keep_mask].copy()
    dropped_rows = len(dataframe) - len(filtered)
    if dropped_rows:
        print(f"Applied row filters: kept {len(filtered):,}/{len(dataframe):,} rows.")

    return filtered


def apply_prediction_overrides(
    predictions: np.ndarray | list[float],
    context_dataframe: pd.DataFrame,
    prediction_overrides: tuple[dict[str, Any], ...],
) -> np.ndarray:
    adjusted_predictions = np.asarray(predictions, dtype=float).copy()
    if not prediction_overrides:
        return adjusted_predictions

    for rule in prediction_overrides:
        supported_override_keys = {
            "set_prediction",
            "set_prediction_from_column",
            "clip_upper_column",
            "clip_lower_column",
        }
        if not any(key in rule for key in supported_override_keys):
            continue

        rule_mask = evaluate_condition(context_dataframe, rule.get("when")).to_numpy(dtype=bool, copy=False)
        if not np.any(rule_mask):
            continue

        rule_name = rule.get("name", "unnamed_override")
        affected_rows = int(rule_mask.sum())

        if "set_prediction" in rule:
            adjusted_predictions[rule_mask] = float(rule["set_prediction"])
            print(f"Applied prediction override '{rule_name}' to {affected_rows:,} row(s).")
            continue

        if "set_prediction_from_column" in rule:
            column_name = str(rule["set_prediction_from_column"])
            if column_name not in context_dataframe.columns:
                raise RuntimeError(
                    f"Prediction override '{rule_name}' references unknown column '{column_name}'."
                )
            column_values = pd.to_numeric(context_dataframe[column_name], errors="coerce").to_numpy(dtype=float)
            valid_mask = rule_mask & np.isfinite(column_values)
            adjusted_predictions[valid_mask] = column_values[valid_mask]
            print(f"Applied prediction override '{rule_name}' to {int(valid_mask.sum()):,} row(s).")
            continue

        if "clip_upper_column" in rule:
            column_name = str(rule["clip_upper_column"])
            if column_name not in context_dataframe.columns:
                raise RuntimeError(
                    f"Prediction override '{rule_name}' references unknown column '{column_name}'."
                )
            column_values = pd.to_numeric(context_dataframe[column_name], errors="coerce").to_numpy(dtype=float)
            valid_mask = rule_mask & np.isfinite(column_values)
            adjusted_predictions[valid_mask] = np.minimum(
                adjusted_predictions[valid_mask],
                column_values[valid_mask],
            )
            print(f"Applied prediction override '{rule_name}' to {int(valid_mask.sum()):,} row(s).")
            continue

        if "clip_lower_column" in rule:
            column_name = str(rule["clip_lower_column"])
            if column_name not in context_dataframe.columns:
                raise RuntimeError(
                    f"Prediction override '{rule_name}' references unknown column '{column_name}'."
                )
            column_values = pd.to_numeric(context_dataframe[column_name], errors="coerce").to_numpy(dtype=float)
            valid_mask = rule_mask & np.isfinite(column_values)
            adjusted_predictions[valid_mask] = np.maximum(
                adjusted_predictions[valid_mask],
                column_values[valid_mask],
            )
            print(f"Applied prediction override '{rule_name}' to {int(valid_mask.sum()):,} row(s).")
            continue

        print(f"Applied prediction override '{rule_name}' to {int(rule_mask.sum()):,} row(s).")

    return adjusted_predictions


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


def get_feature_datetimes(features: pd.DataFrame) -> pd.Series:
    if "LTScheduledDatetime" not in features.columns:
        raise RuntimeError("Date-based splitting requires the 'LTScheduledDatetime' feature column.")

    raw_values = pd.to_numeric(features["LTScheduledDatetime"], errors="coerce")
    finite_values = raw_values[np.isfinite(raw_values)]
    if finite_values.empty:
        return pd.to_datetime(raw_values, unit="s", errors="coerce")

    median_abs_value = float(np.median(np.abs(finite_values)))
    if median_abs_value >= 1e17:
        return pd.to_datetime(raw_values, unit="ns", errors="coerce")
    if median_abs_value >= 1e14:
        return pd.to_datetime(raw_values, unit="us", errors="coerce")
    if median_abs_value >= 1e11:
        return pd.to_datetime(raw_values, unit="ms", errors="coerce")
    if median_abs_value >= 1e8:
        return pd.to_datetime(raw_values, unit="s", errors="coerce")

    return pd.to_datetime(raw_values * 1000.0, unit="s", errors="coerce")


def sort_features_and_target_by_datetime(
    features: pd.DataFrame,
    target: pd.Series,
) -> tuple[pd.DataFrame, pd.Series]:
    scheduled_datetimes = get_feature_datetimes(features)
    ordered_index = (
        pd.DataFrame(
            {
                "scheduled_datetime": scheduled_datetimes,
                "original_position": np.arange(len(features)),
            },
            index=features.index,
        )
        .sort_values(["scheduled_datetime", "original_position"], na_position="last")
        .index
    )
    return features.loc[ordered_index].copy(), target.loc[ordered_index].copy()


def split_train_validation_by_date(
    features: pd.DataFrame,
    target: pd.Series,
    validation_fraction: float = 0.05,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1.")

    sorted_features, sorted_target = sort_features_and_target_by_datetime(features, target)
    scheduled_dates = get_feature_datetimes(sorted_features).dt.normalize()
    unique_dates = pd.Index(scheduled_dates.dropna().unique()).sort_values()

    if len(unique_dates) < 2:
        raise RuntimeError("Date-based validation split requires at least two distinct scheduled dates.")

    validation_date_count = max(1, int(np.ceil(len(unique_dates) * validation_fraction)))
    training_date_count = len(unique_dates) - validation_date_count
    if training_date_count < 1:
        training_date_count = len(unique_dates) - 1
        validation_date_count = 1

    training_dates = set(unique_dates[:training_date_count].tolist())
    validation_dates = set(unique_dates[training_date_count:].tolist())

    training_mask = scheduled_dates.isin(training_dates) | scheduled_dates.isna()
    validation_mask = scheduled_dates.isin(validation_dates)

    if not validation_mask.any():
        raise RuntimeError("Date-based validation split produced an empty validation set.")

    return (
        sorted_features.loc[training_mask].copy(),
        sorted_features.loc[validation_mask].copy(),
        sorted_target.loc[training_mask].copy(),
        sorted_target.loc[validation_mask].copy(),
    )


def build_datewise_cv_splits(
    features: pd.DataFrame,
    n_splits: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    if n_splits < 1:
        raise ValueError("n_splits must be at least 1 for date-based cross-validation.")

    scheduled_dates = get_feature_datetimes(features).dt.normalize()
    unique_dates = pd.Index(scheduled_dates.dropna().unique()).sort_values()
    if len(unique_dates) < 2:
        raise RuntimeError(
            "Date-based CV requires at least two distinct scheduled dates."
        )

    splits: list[tuple[np.ndarray, np.ndarray]] = []
    effective_splits = min(n_splits, len(unique_dates) - 1)

    if effective_splits == 1:
        train_dates = set(unique_dates[:-1].tolist())
        validation_dates = {unique_dates[-1]}
        train_mask = (scheduled_dates.isin(train_dates) | scheduled_dates.isna()).to_numpy(dtype=bool, copy=False)
        validation_mask = scheduled_dates.isin(validation_dates).to_numpy(dtype=bool, copy=False)
        train_indices = np.flatnonzero(train_mask)
        validation_indices = np.flatnonzero(validation_mask)
        if len(train_indices) == 0 or len(validation_indices) == 0:
            raise RuntimeError("Date-based CV could not produce a non-empty train/validation split.")
        return [(train_indices, validation_indices)]

    splitter = TimeSeriesSplit(n_splits=effective_splits)

    for train_date_indices, validation_date_indices in splitter.split(unique_dates):
        train_dates = set(unique_dates[train_date_indices].tolist())
        validation_dates = set(unique_dates[validation_date_indices].tolist())

        train_mask = (scheduled_dates.isin(train_dates) | scheduled_dates.isna()).to_numpy(dtype=bool, copy=False)
        validation_mask = scheduled_dates.isin(validation_dates).to_numpy(dtype=bool, copy=False)

        train_indices = np.flatnonzero(train_mask)
        validation_indices = np.flatnonzero(validation_mask)
        if len(train_indices) == 0 or len(validation_indices) == 0:
            continue

        splits.append((train_indices, validation_indices))

    if len(splits) != effective_splits:
        raise RuntimeError("Date-based CV could not produce the expected number of non-empty splits.")

    return splits


def load_main_dataset_dataframe(
    limit: int | None = None,
    dataset_path: str | Path = MAIN_DATASET_PATH,
    runtime_config: ModelRuntimeConfig | None = None,
) -> pd.DataFrame:
    runtime_config = runtime_config or load_model_runtime_config()
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    required_columns = set(runtime_config.required_columns)
    dataframe = pd.read_csv(
        dataset_path,
        usecols=lambda column_name: column_name in required_columns,
        nrows=limit,
        low_memory=False,
    )

    missing_columns = [column_name for column_name in runtime_config.required_columns if column_name not in dataframe.columns]
    if missing_columns:
        raise RuntimeError(f"Missing required columns in {dataset_path}: {missing_columns}")

    if ROW_ID_COLUMN in dataframe.columns:
        raise RuntimeError(f"Column '{ROW_ID_COLUMN}' is reserved for internal use. Rename it in the dataset.")

    dataframe.insert(0, ROW_ID_COLUMN, np.arange(1, len(dataframe) + 1))
    return clean_dataframe(dataframe, runtime_config=runtime_config)


def clean_dataframe(
    dataframe: pd.DataFrame,
    runtime_config: ModelRuntimeConfig | None = None,
) -> pd.DataFrame:
    runtime_config = runtime_config or load_model_runtime_config()
    cleaned = dataframe.copy()

    for column_name in cleaned.columns:
        cleaned[column_name] = replace_null_like_values(cleaned[column_name])

    cleaned[ROW_ID_COLUMN] = pd.to_numeric(cleaned[ROW_ID_COLUMN], errors="coerce")
    cleaned[ID_COLUMN] = to_object_string_series(cleaned[ID_COLUMN])
    cleaned[TARGET_COLUMN] = pd.to_numeric(cleaned[TARGET_COLUMN], errors="coerce")

    for column_name in runtime_config.feature_columns:
        if column_name not in cleaned.columns:
            continue
        if column_name in runtime_config.categorical_feature_columns:
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
        raise ValueError("prediction_mode must be 'missing_target' or 'known_target'.")

    runtime_config = load_model_runtime_config()
    raw_dataframe = load_main_dataset_dataframe(limit=limit, runtime_config=runtime_config)
    raw_dataframe = apply_row_filters(raw_dataframe, runtime_config.row_filters)
    feature_dataframe = build_feature_dataframe(raw_dataframe, runtime_config=runtime_config)
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
    training_features = feature_dataframe.loc[training_mask, list(runtime_config.feature_columns)].copy()
    training_target = target.loc[training_mask].astype(float)

    prediction_identifier_columns = _deduplicate_preserve_order(
        [ROW_ID_COLUMN, ID_COLUMN, *runtime_config.prediction_override_columns]
    )
    if prediction_mode == PREDICTION_MODE_KNOWN_TARGET:
        prediction_mask = training_mask
        prediction_identifier_columns.append(TARGET_COLUMN)
    else:
        prediction_mask = target.isna()
    prediction_identifier_columns = _deduplicate_preserve_order(prediction_identifier_columns)

    prediction_identifiers = raw_dataframe.loc[prediction_mask, prediction_identifier_columns].copy()
    prediction_features = feature_dataframe.loc[
        prediction_mask,
        list(runtime_config.feature_columns),
    ].copy()

    return (
        training_features,
        training_target,
        prediction_features,
        prediction_identifiers,
    )


def build_feature_dataframe(
    raw_dataframe: pd.DataFrame,
    runtime_config: ModelRuntimeConfig | None = None,
) -> pd.DataFrame:
    runtime_config = runtime_config or load_model_runtime_config()
    feature_columns = list(runtime_config.feature_columns)
    categorical_feature_columns = set(runtime_config.categorical_feature_columns)

    features = raw_dataframe[feature_columns].copy()
    scheduled = pd.to_datetime(features["LTScheduledDatetime"], errors="coerce")
    datetime_values = scheduled.astype("int64", copy=False)
    features["LTScheduledDatetime"] = pd.Series(
        np.where(scheduled.notna(), datetime_values / 1_000_000_000, np.nan),
        index=raw_dataframe.index,
    )

    for column_name in categorical_feature_columns:
        if column_name in features.columns:
            features[column_name] = to_object_string_series(features[column_name])

    for column_name in feature_columns:
        if column_name in categorical_feature_columns:
            continue
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


def print_discarded_rows_debug(
    raw_dataframe: pd.DataFrame,
    feature_dataframe: pd.DataFrame,
    discarded_feature_mask: pd.Series,
) -> None:
    return


def make_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    runtime_config = load_model_runtime_config()
    categorical_columns = [
        column_name for column_name in runtime_config.categorical_feature_columns if column_name in features.columns
    ]
    numeric_columns = [column_name for column_name in features.columns if column_name not in categorical_columns]

    preprocessor = ColumnTransformer(
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
    preprocessor.set_output(transform="default")
    return preprocessor


def write_predictions(
    predictions: np.ndarray | list[float],
    identifiers: pd.DataFrame,
    filename_stem: str,
) -> Path:
    runtime_config = load_model_runtime_config()
    output = identifiers.copy()
    adjusted_predictions = apply_prediction_overrides(
        predictions,
        output,
        runtime_config.prediction_overrides,
    )
    clipped_predictions = np.clip(np.rint(np.asarray(adjusted_predictions)), 0, None).astype(int)
    output_columns = [column_name for column_name in [ROW_ID_COLUMN, ID_COLUMN, TARGET_COLUMN] if column_name in output.columns]
    output = output[output_columns].copy()
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
        print(f"Skipping plot for {predictions_file}: " f"requires both {TARGET_COLUMN} and {PREDICTION_COLUMN}.")
        return None

    plot_data = _prepare_prediction_plot_data(dataframe)
    if plot_data.empty:
        print(f"Skipping plot for {predictions_file}: no plottable values found.")
        return None

    full_plot_data = plot_data.copy()
    actual_full = full_plot_data[TARGET_COLUMN].to_numpy(dtype=float)
    predicted_full = full_plot_data[PREDICTION_COLUMN].to_numpy(dtype=float)
    absolute_errors = np.abs(predicted_full - actual_full)
    mean_absolute_deviation = float(np.mean(absolute_errors))
    median_absolute_deviation = float(np.median(absolute_errors))
    actual_mean = float(np.mean(actual_full))
    actual_median = float(np.median(actual_full))
    predicted_mean = float(np.mean(predicted_full))
    predicted_median = float(np.median(predicted_full))

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

    stats_text = "\n".join(
        [
            f"MAE / MedAE: {mean_absolute_deviation:,.2f} / {median_absolute_deviation:,.2f}",
            f"Actual mean / med: {actual_mean:,.2f} / {actual_median:,.2f}",
            f"Pred mean / med: {predicted_mean:,.2f} / {predicted_median:,.2f}",
        ]
    )
    figure.text(
        0.5,
        0.01,
        stats_text,
        ha="center",
        va="bottom",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "alpha": 0.9, "edgecolor": "0.8"},
    )

    figure.suptitle(f"{predictions_file.stem} (n={len(plot_data):,})")
    figure.tight_layout(rect=(0, 0.08, 1, 0.96))

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
