from pathlib import Path
from typing import Optional

from catboost import CatBoostRegressor
from codecarbon import track_emissions
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.base import clone
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from tqdm.auto import tqdm

from movement_model_utils import (
    build_prediction_output,
    build_test_prediction_output,
    clean_model_params,
    get_feature_datetimes,
    get_model_folder_name,
    get_prediction_column_name,
    load_hyperparameter_results,
    load_training_and_prediction_frames,
    PMR_TARGET_COLUMN,
    PREDICTION_OUTPUT_DIR,
    ROW_ID_COLUMN,
    ID_COLUMN,
    make_model_pipeline,
    plot_prediction_results,
    run_progress_step,
    sort_features_and_target_by_datetime,
    split_train_validation_by_date,
    TARGET_COLUMN,
)

TRAINING_START_DATE = "2023-01-01"
TRAINING_END_DATE = "2026-03-24"
VALIDATION_START_DATE = None
VALIDATION_END_DATE = None
TEST_START_DATE = "2026-03-25"
TEST_END_DATE = "2026-03-31"


def params_without(params: dict[str, object], *excluded_keys: str) -> dict[str, object]:
    excluded = set(excluded_keys)
    return {key: value for key, value in params.items() if key not in excluded}


def build_prediction_pipeline(model, features):
    return make_model_pipeline(model, features)


def split_training_data_by_datetime(
    features,
    target,
    training_start_date: str,
    training_end_date: str,
    validation_start_date: str | None,
    validation_end_date: str | None,
):
    ordered_features, ordered_target = sort_features_and_target_by_datetime(features, target)
    if validation_start_date is None and validation_end_date is None:
        scheduled_dates = get_feature_datetimes(ordered_features).dt.normalize()
        training_start = pd.to_datetime(training_start_date).normalize()
        training_end = pd.to_datetime(training_end_date).normalize()
        training_mask = scheduled_dates.between(training_start, training_end, inclusive="both") | scheduled_dates.isna()

        if not training_mask.any():
            raise RuntimeError("Date-based training split produced an empty training set.")

        print("Date split -> " f"train: {training_start.date()} to {training_end.date()} | " "validation: disabled")
        return (
            ordered_features.loc[training_mask].copy(),
            None,
            ordered_target.loc[training_mask].copy(),
            None,
        )

    return split_train_validation_by_date(
        ordered_features,
        ordered_target,
        training_start_date=training_start_date,
        training_end_date=training_end_date,
        validation_start_date=validation_start_date,
        validation_end_date=validation_end_date,
    )


def get_predictions_from_datetime_split(
    model,
    features,
    target,
    prediction_features,
    training_start_date: str,
    training_end_date: str,
    validation_start_date: str | None,
    validation_end_date: str | None,
    step_progress=None,
):
    X_train, X_val, y_train, y_val = split_training_data_by_datetime(
        features,
        target,
        training_start_date=training_start_date,
        training_end_date=training_end_date,
        validation_start_date=validation_start_date,
        validation_end_date=validation_end_date,
    )
    pipeline = build_prediction_pipeline(model, X_train)

    run_progress_step(step_progress, "fit", pipeline.fit, X_train, y_train)
    validation_predictions = None
    validation_r2 = None
    validation_mae = None
    if X_val is not None and y_val is not None:
        validation_predictions = run_progress_step(step_progress, "val_predict", pipeline.predict, X_val)
        validation_r2 = r2_score(y_val, validation_predictions)
        validation_mae = mean_absolute_error(y_val, validation_predictions)
    prediction_rows = run_progress_step(step_progress, "predict", pipeline.predict, prediction_features)

    return (
        prediction_rows,
        validation_predictions,
        validation_r2,
        validation_mae,
    )


def write_validation_prediction_subset(
    full_predictions_path: str | Path,
    validation_start_date: str,
    validation_end_date: str,
):
    full_predictions_path = Path(full_predictions_path)
    validation_output_path = full_predictions_path.with_name(
        full_predictions_path.stem.replace("_preds", "_validation_preds") + full_predictions_path.suffix
    )

    full_predictions = pd.read_csv(full_predictions_path)
    if "LTScheduledDatetime" not in full_predictions.columns:
        raise RuntimeError("Validation prediction export requires 'LTScheduledDatetime' in the full prediction CSV.")

    validation_start = pd.to_datetime(validation_start_date).normalize()
    validation_end = pd.to_datetime(validation_end_date).normalize()
    scheduled = pd.to_datetime(full_predictions["LTScheduledDatetime"], errors="coerce").dt.normalize()

    validation_predictions = full_predictions.loc[
        scheduled.between(validation_start, validation_end, inclusive="both")
    ].copy()
    validation_predictions.to_csv(validation_output_path, index=False)
    return validation_output_path, len(validation_predictions)


def write_prediction_dataframe(
    dataframe: pd.DataFrame,
    filename_stem: str,
) -> Path:
    model_folder_name = get_model_folder_name(filename_stem)
    model_output_dir = PREDICTION_OUTPUT_DIR / model_folder_name
    model_output_dir.mkdir(parents=True, exist_ok=True)
    output_path = model_output_dir / f"{filename_stem}.csv"
    dataframe.to_csv(output_path, index=False)
    return output_path


def write_test_prediction_dataframe(
    dataframe: pd.DataFrame,
    filename_stem: str,
) -> Path:
    model_folder_name = get_model_folder_name(filename_stem)
    model_output_dir = PREDICTION_OUTPUT_DIR / model_folder_name
    model_output_dir.mkdir(parents=True, exist_ok=True)
    output_path = model_output_dir / f"{filename_stem}_test_set.csv"
    export_columns = [
        column_name
        for column_name in dataframe.columns
        if column_name not in {ROW_ID_COLUMN, ID_COLUMN, TARGET_COLUMN, PMR_TARGET_COLUMN}
    ]
    dataframe[export_columns].to_csv(output_path, index=False)
    return output_path


def merge_prediction_outputs(base_output: pd.DataFrame, extra_output: pd.DataFrame) -> pd.DataFrame:
    merge_keys = [
        column_name
        for column_name in [ROW_ID_COLUMN, ID_COLUMN, "FlightNumberNormalized", "LTScheduledDatetime"]
        if column_name in base_output.columns and column_name in extra_output.columns
    ]
    extra_value_columns = [column_name for column_name in extra_output.columns if column_name not in merge_keys]
    return base_output.merge(extra_output[merge_keys + extra_value_columns], on=merge_keys, how="left")


def merge_test_outputs(base_output: pd.DataFrame, extra_output: pd.DataFrame) -> pd.DataFrame:
    if ROW_ID_COLUMN in base_output.columns and ROW_ID_COLUMN in extra_output.columns:
        extra_value_columns = [
            column_name
            for column_name in extra_output.columns
            if column_name not in {ROW_ID_COLUMN, "FlightNumberNormalized", "LTScheduledDatetime"}
        ]
        return base_output.merge(extra_output[[ROW_ID_COLUMN] + extra_value_columns], on=ROW_ID_COLUMN, how="left")

    merge_keys = ["FlightNumberNormalized", "LTScheduledDatetime"]
    extra_value_columns = [column_name for column_name in extra_output.columns if column_name not in merge_keys]
    return base_output.merge(extra_output[merge_keys + extra_value_columns], on=merge_keys, how="left")


def build_enabled_regressors(results: dict[str, dict[str, object]]) -> dict[str, object]:
    regressors = {}

    if "XGBRegressor" in results:
        best_params = clean_model_params(results["XGBRegressor"]["best_params"])
        regressors["XGBRegressor"] = XGBRegressor(**best_params)

    # if "LGBMRegressor" in results:
    #     regressors["LGBMRegressor"] = LGBMRegressor(
    #         random_state=42,
    #         objective="mae",
    #         verbosity=-1,
    #         force_col_wise=True,
    #         **clean_model_params(results["LGBMRegressor"]["best_params"]),
    #     )

    # if "DecisionTreeRegressor" in results:
    #     regressors["DecisionTreeRegressor"] = DecisionTreeRegressor(
    #         **clean_model_params(results["DecisionTreeRegressor"]["best_params"])
    #     )
    # if "RandomForestRegressor" in results:
    #     regressors["RandomForestRegressor"] = RandomForestRegressor(
    #         **clean_model_params(results["RandomForestRegressor"]["best_params"])
    #     )
    # if "ExtraTreesRegressor" in results:
    #     regressors["ExtraTreesRegressor"] = ExtraTreesRegressor(
    #         **clean_model_params(results["ExtraTreesRegressor"]["best_params"])
    #     )
    # if "CatBoostRegressor" in results:
    #     regressors["CatBoostRegressor"] = CatBoostRegressor(
    #         loss_function="MAE",
    #         verbose=False,
    #         random_seed=42,
    #         allow_writing_files=False,
    #         **params_without(
    #             clean_model_params(results["CatBoostRegressor"]["best_params"]),
    #             "loss_function",
    #             "verbose",
    #             "random_seed",
    #             "allow_writing_files",
    #         ),
    #     )
    # if "GradientBoostingRegressor" in results:
    #     regressors["GradientBoostingRegressor"] = GradientBoostingRegressor(
    #         **clean_model_params(results["GradientBoostingRegressor"]["best_params"])
    #     )
    # if "HistGradientBoostingRegressor" in results:
    #     regressors["HistGradientBoostingRegressor"] = HistGradientBoostingRegressor(
    #         **clean_model_params(results["HistGradientBoostingRegressor"]["best_params"])
    #     )

    return regressors


def validate_split_date_constants() -> None:
    required_dates = {
        "TRAINING_START_DATE": TRAINING_START_DATE,
        "TRAINING_END_DATE": TRAINING_END_DATE,
    }
    missing_names = [name for name, value in required_dates.items() if not value]
    if missing_names:
        raise RuntimeError(
            "Set the date range constants at the top of 2_individual_predictions.py before running: "
            + ", ".join(missing_names)
        )
    if (VALIDATION_START_DATE is None) != (VALIDATION_END_DATE is None):
        raise RuntimeError("Set both VALIDATION_START_DATE and VALIDATION_END_DATE, or set both to None.")
    if VALIDATION_START_DATE is not None and pd.to_datetime(VALIDATION_START_DATE) > pd.to_datetime(
        VALIDATION_END_DATE
    ):
        raise RuntimeError("VALIDATION_START_DATE must be on or before VALIDATION_END_DATE.")
    if TEST_START_DATE is None and TEST_END_DATE is not None:
        pd.to_datetime(TEST_END_DATE)
    if TEST_START_DATE is not None:
        pd.to_datetime(TEST_START_DATE)
    if TEST_START_DATE is not None and TEST_END_DATE is not None:
        if pd.to_datetime(TEST_START_DATE) > pd.to_datetime(TEST_END_DATE):
            raise RuntimeError("TEST_START_DATE must be on or before TEST_END_DATE.")


@track_emissions()
def main():
    validate_split_date_constants()
    training_features, training_target, prediction_features, prediction_ids = load_training_and_prediction_frames(
        target_column=TARGET_COLUMN,
        prediction_selection_column=TARGET_COLUMN,
    )
    pmr_training_features, pmr_training_target, pmr_prediction_features, pmr_prediction_ids = (
        load_training_and_prediction_frames(
            target_column=PMR_TARGET_COLUMN,
            prediction_selection_column=TARGET_COLUMN,
        )
    )
    results = load_hyperparameter_results()
    regressors = build_enabled_regressors(results)

    if not regressors:
        raise RuntimeError("No enabled regressors found in hyperparameters.json.")

    has_test_range = TEST_START_DATE is not None or TEST_END_DATE is not None
    has_validation_range = VALIDATION_START_DATE is not None and VALIDATION_END_DATE is not None

    for name, model in tqdm(regressors.items(), total=len(regressors), desc="Regressors", unit="model"):
        step_total = (
            11 if (has_validation_range and has_test_range) else 10 if (has_validation_range or has_test_range) else 8
        )
        with tqdm(total=step_total, desc=f"{name}", unit="step", leave=False) as step_progress:
            predictions, validation_predictions, validation_r2, validation_mae = get_predictions_from_datetime_split(
                clone(model),
                training_features,
                training_target,
                prediction_features,
                training_start_date=TRAINING_START_DATE,
                training_end_date=TRAINING_END_DATE,
                validation_start_date=VALIDATION_START_DATE,
                validation_end_date=VALIDATION_END_DATE,
                step_progress=step_progress,
            )
            pmr_predictions, _, pmr_validation_r2, pmr_validation_mae = get_predictions_from_datetime_split(
                clone(model),
                pmr_training_features,
                pmr_training_target,
                pmr_prediction_features,
                training_start_date=TRAINING_START_DATE,
                training_end_date=TRAINING_END_DATE,
                validation_start_date=VALIDATION_START_DATE,
                validation_end_date=VALIDATION_END_DATE,
                step_progress=step_progress,
            )

            pax_output = build_prediction_output(
                predictions,
                prediction_ids,
                target_column=TARGET_COLUMN,
            )
            pmr_output = build_prediction_output(
                pmr_predictions,
                pmr_prediction_ids,
                target_column=PMR_TARGET_COLUMN,
                prediction_column=get_prediction_column_name(PMR_TARGET_COLUMN),
            )
            combined_output = merge_prediction_outputs(pax_output, pmr_output)

            output_path = run_progress_step(
                step_progress,
                "write_csv",
                write_prediction_dataframe,
                combined_output,
                f"{name}_preds",
            )
            validation_output_path = None
            if has_validation_range:
                validation_output_path = run_progress_step(
                    step_progress,
                    "write_val_csv",
                    write_validation_prediction_subset,
                    output_path,
                    VALIDATION_START_DATE,
                    VALIDATION_END_DATE,
                )
                if validation_predictions is None:
                    raise RuntimeError("Validation predictions were expected but not produced.")
                if validation_output_path[1] != len(validation_predictions):
                    raise RuntimeError(
                        "Validation prediction row mismatch: "
                        f"{validation_output_path[1]:,} rows in validation subset vs "
                        f"{len(validation_predictions):,} validation predictions."
                    )
                validation_output_path = validation_output_path[0]
            test_output_path = None
            if has_test_range:
                pax_test_output = build_test_prediction_output(
                    predictions,
                    prediction_ids,
                    TEST_START_DATE,
                    TEST_END_DATE,
                    target_column=TARGET_COLUMN,
                    exported_prediction_column_name="Predicted NbPaxTotal",
                )
                pmr_test_output = build_test_prediction_output(
                    pmr_predictions,
                    pmr_prediction_ids,
                    TEST_START_DATE,
                    TEST_END_DATE,
                    target_column=PMR_TARGET_COLUMN,
                    prediction_column=get_prediction_column_name(PMR_TARGET_COLUMN),
                    exported_prediction_column_name="Predicted PMR",
                )
                combined_test_output = merge_test_outputs(pax_test_output, pmr_test_output)
                test_output_path = run_progress_step(
                    step_progress,
                    "write_test_csv",
                    write_test_prediction_dataframe,
                    combined_test_output,
                    f"{name}_preds",
                )
            if has_validation_range and validation_output_path is not None:
                run_progress_step(step_progress, "plot", plot_prediction_results, validation_output_path)
                run_progress_step(
                    step_progress,
                    "plot_pmr",
                    plot_prediction_results,
                    validation_output_path,
                    target_column=PMR_TARGET_COLUMN,
                    prediction_column=get_prediction_column_name(PMR_TARGET_COLUMN),
                    output_suffix="_pmr",
                )

        if has_validation_range and validation_r2 is not None and validation_mae is not None:
            print(
                f"\033[34m{f'{name} validation R2: {validation_r2:.4f} | validation MAE: {validation_mae:.4f}'}\033[0m"
            )
        if has_validation_range and pmr_validation_r2 is not None and pmr_validation_mae is not None:
            print(
                f"\033[34m{f'{name} PMR validation R2: {pmr_validation_r2:.4f} | validation MAE: {pmr_validation_mae:.4f}'}\033[0m"
            )
        print(f"Predictions written to {output_path}")
        if validation_output_path is not None:
            print(f"Validation predictions and validation-only plot source written to {validation_output_path}")
        if test_output_path is not None:
            print(f"Test-set predictions written to {test_output_path}")


if __name__ == "__main__":
    main()
