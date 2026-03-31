from pathlib import Path
from typing import Optional

from catboost import CatBoostRegressor
from codecarbon import track_emissions
import pandas as pd

from lightgbm import LGBMRegressor
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
    clean_model_params,
    load_hyperparameter_results,
    load_training_and_prediction_frames,
    PREDICTION_MODE_MISSING_TARGET,
    PREDICTION_MODE_KNOWN_TARGET,
    make_model_pipeline,
    plot_prediction_results,
    run_progress_step,
    sort_features_and_target_by_datetime,
    split_train_validation_by_date,
    write_predictions,
)

PREDICTION_MODE = PREDICTION_MODE_KNOWN_TARGET
TRAINING_START_DATE = "2023-01-01"
TRAINING_END_DATE = "2025-12-31"
VALIDATION_START_DATE = "2026-03-17"
VALIDATION_END_DATE = "2026-03-24"


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
    validation_start_date: str,
    validation_end_date: str,
):
    ordered_features, ordered_target = sort_features_and_target_by_datetime(features, target)
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
    validation_start_date: str,
    validation_end_date: str,
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
    validation_predictions = run_progress_step(step_progress, "val_predict", pipeline.predict, X_val)
    prediction_rows = run_progress_step(step_progress, "predict", pipeline.predict, prediction_features)

    return (
        prediction_rows,
        validation_predictions,
        r2_score(y_val, validation_predictions),
        mean_absolute_error(y_val, validation_predictions),
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
        "VALIDATION_START_DATE": VALIDATION_START_DATE,
        "VALIDATION_END_DATE": VALIDATION_END_DATE,
    }
    missing_names = [name for name, value in required_dates.items() if not value]
    if missing_names:
        raise RuntimeError(
            "Set the date range constants at the top of 2_individual_predictions.py before running: "
            + ", ".join(missing_names)
        )


@track_emissions()
def main():
    validate_split_date_constants()
    training_features, training_target, prediction_features, prediction_ids = load_training_and_prediction_frames(
        prediction_mode=PREDICTION_MODE
    )
    results = load_hyperparameter_results()
    regressors = build_enabled_regressors(results)

    if not regressors:
        raise RuntimeError("No enabled regressors found in hyperparameters.json.")

    for name, model in tqdm(regressors.items(), total=len(regressors), desc="Regressors", unit="model"):
        with tqdm(total=6, desc=f"{name}", unit="step", leave=False) as step_progress:
            predictions, validation_predictions, validation_r2, validation_mae = get_predictions_from_datetime_split(
                model,
                training_features,
                training_target,
                prediction_features,
                training_start_date=TRAINING_START_DATE,
                training_end_date=TRAINING_END_DATE,
                validation_start_date=VALIDATION_START_DATE,
                validation_end_date=VALIDATION_END_DATE,
                step_progress=step_progress,
            )
            output_path = run_progress_step(
                step_progress,
                "write_csv",
                write_predictions,
                predictions,
                prediction_ids,
                f"{name}_preds",
            )
            validation_output_path = run_progress_step(
                step_progress,
                "write_val_csv",
                write_validation_prediction_subset,
                output_path,
                VALIDATION_START_DATE,
                VALIDATION_END_DATE,
            )
            if validation_output_path[1] != len(validation_predictions):
                raise RuntimeError(
                    "Validation prediction row mismatch: "
                    f"{validation_output_path[1]:,} rows in validation subset vs "
                    f"{len(validation_predictions):,} validation predictions."
                )
            validation_output_path = validation_output_path[0]
            run_progress_step(step_progress, "plot", plot_prediction_results, validation_output_path)

        print(
            f"\033[34m{f"{name} validation R2: {validation_r2:.4f} | " f"validation MAE: {validation_mae:.4f}"}\033[0m"
        )
        print(f"Predictions written to {output_path}")
        print(f"Validation predictions and validation-only plot source written to {validation_output_path}")


if __name__ == "__main__":
    main()
