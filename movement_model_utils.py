from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

DATABASE_PATH = Path("database.db")
SOURCE_TABLE = "mouvements_aero_insa"
ID_COLUMN = "IdMovement"
ROW_ID_COLUMN = "sqlite_rowid"
TARGET_COLUMN = "NbPax"
PREDICTION_COLUMN = "NbPaxPrediction"
PREDICTION_OUTPUT_DIR = Path("predictions")
NULL_LIKE_STRINGS = {"", "NULL", "NONE", "NAN", "NAT"}

# These columns are identifiers or direct passenger-count variants that would
# leak the target too strongly when predicting NbPax.
EXCLUDED_FEATURE_COLUMNS = {
    ID_COLUMN,
    "IdMovementVinci",
    "IdFarms",
    "NbPax",
    "NbPaxHTransit",
    "NbPaxTransit",
    "NbPaxConnecting",
    "NbPaxTotal",
    "FarmsNbPax",
    "FarmsNbPaxConnecting",
    "FarmsNbPaxHTransit",
    "FarmsNbPaxTransit",
    "FarmsNbPaxTotal",
    "FarmsNbPaxPHMR",
    "FarmsNbPaxAssisting",
    "FarmsNbPaxExpected",
    "InvoiceNbPayingPax",
    "InvoiceNbPaxConnecting",
    "InvoiceNbPaxTransit",
    "InvoiceNbNonPayingPax",
    "InvoiceNbPaxHTransit",
    "InvoiceNbPaxTotal",
}


@contextmanager
def open_database(
    database_path: str | Path = DATABASE_PATH,
) -> Iterator[tuple[object, object]]:
    database_path = str(database_path)

    try:
        from sqlite_manager import SqliteManager
    except ImportError:
        connection = sqlite3.connect(database_path)
        try:
            yield connection.cursor(), connection
        finally:
            connection.close()
        return

    with SqliteManager(database_path, True) as (db, connection):
        yield db, connection


def load_hyperparameter_results(path: str | Path = "hyperparameters.json") -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def clean_model_params(params: dict[str, object]) -> dict[str, object]:
    return {str(key).removeprefix("model__"): value for key, value in params.items()}


def get_table_schema(
    table_name: str = SOURCE_TABLE,
    database_path: str | Path = DATABASE_PATH,
) -> dict[str, str]:
    with open_database(database_path) as (_, connection):
        rows = connection.execute(f"PRAGMA table_info('{table_name}')").fetchall()

    return {str(row[1]): str(row[2]).upper() or "STRING" for row in rows}


def load_movements_dataframe(
    where_clause: str | None = None,
    limit: int | None = None,
    include_rowid: bool = False,
    table_name: str = SOURCE_TABLE,
    database_path: str | Path = DATABASE_PATH,
) -> pd.DataFrame:
    select_prefix = f"rowid AS {ROW_ID_COLUMN}, " if include_rowid else ""
    query = f"SELECT {select_prefix}* FROM {table_name}"

    if where_clause:
        query = f"{query} WHERE {where_clause}"
    if limit is not None:
        query = f"{query} LIMIT {int(limit)}"

    with open_database(database_path) as (_, connection):
        dataframe = pd.read_sql_query(query, connection)

    return clean_dataframe(dataframe, get_table_schema(table_name, database_path))


def clean_dataframe(
    dataframe: pd.DataFrame,
    schema: dict[str, str],
) -> pd.DataFrame:
    cleaned = dataframe.copy()

    for column_name in cleaned.columns:
        cleaned[column_name] = replace_null_like_values(cleaned[column_name])

    for column_name, sql_type in schema.items():
        if column_name not in cleaned.columns:
            continue
        cleaned[column_name] = coerce_series_to_schema_type(
            cleaned[column_name],
            sql_type,
        )

    object_columns = cleaned.select_dtypes(include=["object"]).columns
    for column_name in object_columns:
        cleaned[column_name] = cleaned[column_name].replace({pd.NA: np.nan})

    return cleaned


def replace_null_like_values(series: pd.Series) -> pd.Series:
    if not (
        pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)
    ):
        return series

    as_string = series.astype("string").str.strip()
    null_mask = as_string.isna() | as_string.str.upper().isin(NULL_LIKE_STRINGS)
    return series.mask(null_mask, other=np.nan)


def coerce_series_to_schema_type(series: pd.Series, sql_type: str) -> pd.Series:
    if sql_type in {"INTEGER", "FLOAT"}:
        return pd.to_numeric(series, errors="coerce")

    if sql_type == "BOOLEAN":
        return pd.to_numeric(
            series.replace(
                {
                    True: 1,
                    False: 0,
                    "true": 1,
                    "false": 0,
                    "True": 1,
                    "False": 0,
                    "oui": 1,
                    "non": 0,
                    "Oui": 1,
                    "Non": 0,
                }
            ),
            errors="coerce",
        )

    if sql_type in {"DATETIME", "DATE"}:
        datetimes = pd.to_datetime(series, errors="coerce")
        datetime_ints = datetimes.astype("int64", copy=False)
        return pd.Series(
            np.where(datetimes.notna(), datetime_ints / 1_000_000_000, np.nan),
            index=series.index,
        )

    if sql_type == "TIME":
        timedeltas = pd.to_timedelta(series.astype("string"), errors="coerce")
        return timedeltas.dt.total_seconds()

    string_values = series.astype("string").replace({pd.NA: np.nan})
    return string_values.astype("object")


def build_feature_columns(dataframe: pd.DataFrame) -> list[str]:
    candidate_columns = [
        column_name
        for column_name in dataframe.columns
        if column_name not in EXCLUDED_FEATURE_COLUMNS
        and column_name != ROW_ID_COLUMN
    ]

    return [
        column_name
        for column_name in candidate_columns
        if dataframe[column_name].notna().any()
        and dataframe[column_name].nunique(dropna=True) > 1
    ]


def load_training_and_prediction_frames(
    limit: int | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame]:
    dataframe = load_movements_dataframe(include_rowid=True, limit=limit)
    target = pd.to_numeric(dataframe[TARGET_COLUMN], errors="coerce")
    feature_columns = build_feature_columns(dataframe)

    training_mask = target.notna()
    training_features = dataframe.loc[training_mask, feature_columns].copy()
    training_target = target.loc[training_mask].astype(float)
    prediction_identifiers = dataframe.loc[
        ~training_mask,
        [ROW_ID_COLUMN, ID_COLUMN],
    ].copy()
    prediction_features = dataframe.loc[~training_mask, feature_columns].copy()

    return (
        training_features,
        training_target,
        prediction_features,
        prediction_identifiers,
    )


def make_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    categorical_columns = list(
        features.select_dtypes(include=["object", "string", "category"]).columns
    )
    numeric_columns = [
        column_name
        for column_name in features.columns
        if column_name not in categorical_columns
    ]

    transformers = []
    if numeric_columns:
        transformers.append(
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                numeric_columns,
            )
        )
    if categorical_columns:
        transformers.append(
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
            )
        )

    return ColumnTransformer(transformers=transformers, remainder="drop")


def write_predictions(
    predictions: np.ndarray | list[float],
    identifiers: pd.DataFrame,
    filename_stem: str,
) -> Path:
    output = identifiers.copy()
    clipped_predictions = np.clip(np.rint(np.asarray(predictions)), 0, None).astype(int)
    output[PREDICTION_COLUMN] = clipped_predictions

    PREDICTION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PREDICTION_OUTPUT_DIR / f"{filename_stem}.csv"
    output.to_csv(output_path, index=False)
    return output_path
