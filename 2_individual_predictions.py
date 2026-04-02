from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from codecarbon import track_emissions
from lightgbm import LGBMRegressor
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from tqdm.auto import tqdm
from xgboost import XGBRegressor

from movement_model_utils import (
    ID_COLUMN,
    PMR_TARGET_COLUMN,
    PREDICTION_OUTPUT_DIR,
    ROW_ID_COLUMN,
    TARGET_COLUMN,
    build_prediction_output,
    build_test_prediction_output,
    clean_model_params,
    get_feature_datetimes,
    get_model_folder_name,
    get_prediction_column_name,
    load_hyperparameter_results,
    load_training_and_prediction_frames,
    make_model_pipeline,
    plot_prediction_results,
    run_progress_step,
    sort_features_and_target_by_datetime,
    split_train_validation_by_date,
)


@dataclass(frozen=True, slots=True)
class PredictionWindowConfig:
    training_start_date: str = "2025-01-01"
    training_end_date: str = "2026-03-29"
    validation_start_date: str | None = "2026-04-01"
    validation_end_date: str | None = "2026-04-01"
    test_start_date: str | None = "2026-04-02"
    test_end_date: str | None = "2026-04-03"

    def has_validation_window(self) -> bool:
        return self.validation_start_date is not None and self.validation_end_date is not None

    def has_test_window(self) -> bool:
        return self.test_start_date is not None or self.test_end_date is not None

    def validate(self) -> None:
        if not self.training_start_date or not self.training_end_date:
            raise RuntimeError("Set training_start_date and training_end_date before running predictions.")
        if (self.validation_start_date is None) != (self.validation_end_date is None):
            raise RuntimeError("Set both validation dates, or set both to None.")
        if self.has_validation_window() and pd.to_datetime(self.validation_start_date) > pd.to_datetime(
            self.validation_end_date
        ):
            raise RuntimeError("validation_start_date must be on or before validation_end_date.")
        if self.test_start_date is not None:
            pd.to_datetime(self.test_start_date)
        if self.test_end_date is not None:
            pd.to_datetime(self.test_end_date)
        if self.test_start_date is not None and self.test_end_date is not None:
            if pd.to_datetime(self.test_start_date) > pd.to_datetime(self.test_end_date):
                raise RuntimeError("test_start_date must be on or before test_end_date.")


class PredictionFileWriter:
    @staticmethod
    def _output_dir(filename_stem: str) -> Path:
        model_output_dir = PREDICTION_OUTPUT_DIR / get_model_folder_name(filename_stem)
        model_output_dir.mkdir(parents=True, exist_ok=True)
        return model_output_dir

    @classmethod
    def write_full_predictions(cls, dataframe: pd.DataFrame, filename_stem: str) -> Path:
        output_path = cls._output_dir(filename_stem) / f"{filename_stem}.csv"
        dataframe.to_csv(output_path, index=False)
        return output_path

    @classmethod
    def write_test_predictions(cls, dataframe: pd.DataFrame, filename_stem: str) -> Path:
        output_path = cls._output_dir(filename_stem) / f"{filename_stem}_test_set.csv"
        export_columns = [
            column_name
            for column_name in dataframe.columns
            if column_name not in {ROW_ID_COLUMN, ID_COLUMN, TARGET_COLUMN, PMR_TARGET_COLUMN}
        ]
        dataframe[export_columns].to_csv(output_path, index=False)
        return output_path

    @staticmethod
    def write_validation_subset(
        full_predictions_path: str | Path,
        validation_start_date: str,
        validation_end_date: str,
    ) -> tuple[Path, int]:
        full_predictions_path = Path(full_predictions_path)
        output_path = full_predictions_path.with_name(
            full_predictions_path.stem.replace("_preds", "_validation_preds") + full_predictions_path.suffix
        )

        full_predictions = pd.read_csv(full_predictions_path)
        if "LTScheduledDatetime" not in full_predictions.columns:
            raise RuntimeError("Validation export requires 'LTScheduledDatetime' in the prediction CSV.")

        validation_start = pd.to_datetime(validation_start_date).normalize()
        validation_end = pd.to_datetime(validation_end_date).normalize()
        scheduled = pd.to_datetime(full_predictions["LTScheduledDatetime"], errors="coerce").dt.normalize()
        validation_predictions = full_predictions.loc[
            scheduled.between(validation_start, validation_end, inclusive="both")
        ].copy()
        validation_predictions.to_csv(output_path, index=False)
        return output_path, len(validation_predictions)


class IndividualPredictionRunner:
    def __init__(self, config: PredictionWindowConfig):
        self.config = config

    @staticmethod
    def build_enabled_regressors(results: dict[str, dict[str, object]]) -> dict[str, object]:
        regressors: dict[str, object] = {}
        if "XGBRegressor" in results:
            regressors["XGBRegressor"] = XGBRegressor(**clean_model_params(results["XGBRegressor"]["best_params"]))
        if "HistGradientBoostingRegressor" in results:
            regressors["HistGradientBoostingRegressor"] = HistGradientBoostingRegressor(
                **clean_model_params(results["HistGradientBoostingRegressor"]["best_params"])
            )
        if "LGBMRegressor" in results:
            regressors["LGBMRegressor"] = LGBMRegressor(
                random_state=42,
                objective="mae",
                verbosity=-1,
                force_col_wise=True,
                **clean_model_params(results["LGBMRegressor"]["best_params"]),
            )
        return regressors

    @staticmethod
    def merge_prediction_outputs(base_output: pd.DataFrame, extra_output: pd.DataFrame) -> pd.DataFrame:
        merge_keys = [
            column_name
            for column_name in [ROW_ID_COLUMN, ID_COLUMN, "FlightNumberNormalized", "LTScheduledDatetime"]
            if column_name in base_output.columns and column_name in extra_output.columns
        ]
        extra_value_columns = [column_name for column_name in extra_output.columns if column_name not in merge_keys]
        return base_output.merge(extra_output[merge_keys + extra_value_columns], on=merge_keys, how="left")

    @staticmethod
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

    @staticmethod
    def split_training_data(
        features: pd.DataFrame,
        target: pd.Series,
        config: PredictionWindowConfig,
    ) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.Series, pd.Series | None]:
        ordered_features, ordered_target = sort_features_and_target_by_datetime(features, target)
        if not config.has_validation_window():
            scheduled_dates = get_feature_datetimes(ordered_features).dt.normalize()
            training_start = pd.to_datetime(config.training_start_date).normalize()
            training_end = pd.to_datetime(config.training_end_date).normalize()
            training_mask = (
                scheduled_dates.between(training_start, training_end, inclusive="both") | scheduled_dates.isna()
            )
            if not training_mask.any():
                raise RuntimeError("Date-based training split produced an empty training set.")
            return (
                ordered_features.loc[training_mask].copy(),
                None,
                ordered_target.loc[training_mask].copy(),
                None,
            )

        return split_train_validation_by_date(
            ordered_features,
            ordered_target,
            training_start_date=config.training_start_date,
            training_end_date=config.training_end_date,
            validation_start_date=config.validation_start_date,
            validation_end_date=config.validation_end_date,
        )

    @classmethod
    def fit_and_predict(
        cls,
        model,
        features: pd.DataFrame,
        target: pd.Series,
        prediction_features: pd.DataFrame,
        config: PredictionWindowConfig,
        step_progress=None,
    ) -> tuple:
        X_train, X_val, y_train, y_val = cls.split_training_data(features, target, config)
        pipeline = make_model_pipeline(model, X_train)

        run_progress_step(step_progress, "fit", pipeline.fit, X_train, y_train)

        validation_predictions = None
        validation_r2 = None
        validation_mae = None
        if X_val is not None and y_val is not None:
            validation_predictions = run_progress_step(step_progress, "val_predict", pipeline.predict, X_val)
            validation_r2 = r2_score(y_val, validation_predictions)
            validation_mae = mean_absolute_error(y_val, validation_predictions)

        prediction_rows = run_progress_step(step_progress, "predict", pipeline.predict, prediction_features)
        return prediction_rows, validation_predictions, validation_r2, validation_mae

    def run(self) -> None:
        self.config.validate()
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

        regressors = self.build_enabled_regressors(load_hyperparameter_results())
        if not regressors:
            raise RuntimeError("No enabled regressors found in hyperparameters.json.")

        for name, model in tqdm(regressors.items(), total=len(regressors), desc="Regressors", unit="model"):
            step_total = 8
            if self.config.has_validation_window():
                step_total += 2
            if self.config.has_test_window():
                step_total += 1

            with tqdm(total=step_total, desc=name, unit="step", leave=False) as step_progress:
                predictions, validation_predictions, validation_r2, validation_mae = self.fit_and_predict(
                    clone(model),
                    training_features,
                    training_target,
                    prediction_features,
                    self.config,
                    step_progress=step_progress,
                )
                pmr_predictions, _, pmr_validation_r2, pmr_validation_mae = self.fit_and_predict(
                    clone(model),
                    pmr_training_features,
                    pmr_training_target,
                    pmr_prediction_features,
                    self.config,
                    step_progress=step_progress,
                )

                pax_output = build_prediction_output(predictions, prediction_ids, target_column=TARGET_COLUMN)
                pmr_output = build_prediction_output(
                    pmr_predictions,
                    pmr_prediction_ids,
                    target_column=PMR_TARGET_COLUMN,
                    prediction_column=get_prediction_column_name(PMR_TARGET_COLUMN),
                )
                combined_output = self.merge_prediction_outputs(pax_output, pmr_output)
                output_path = run_progress_step(
                    step_progress,
                    "write_csv",
                    PredictionFileWriter.write_full_predictions,
                    combined_output,
                    f"{name}_preds",
                )

                validation_output_path = None
                if self.config.has_validation_window():
                    if validation_predictions is None:
                        raise RuntimeError("Validation predictions were expected but not produced.")
                    validation_output_path = run_progress_step(
                        step_progress,
                        "write_val_csv",
                        PredictionFileWriter.write_validation_subset,
                        output_path,
                        self.config.validation_start_date,
                        self.config.validation_end_date,
                    )
                    if validation_output_path[1] != len(validation_predictions):
                        raise RuntimeError(
                            "Validation prediction row mismatch: "
                            f"{validation_output_path[1]:,} rows in validation subset vs "
                            f"{len(validation_predictions):,} validation predictions."
                        )
                    validation_output_path = validation_output_path[0]

                test_output_path = None
                if self.config.has_test_window():
                    pax_test_output = build_test_prediction_output(
                        predictions,
                        prediction_ids,
                        self.config.test_start_date,
                        self.config.test_end_date,
                        target_column=TARGET_COLUMN,
                        exported_prediction_column_name="Predicted NbPaxTotal",
                    )
                    pmr_test_output = build_test_prediction_output(
                        pmr_predictions,
                        pmr_prediction_ids,
                        self.config.test_start_date,
                        self.config.test_end_date,
                        target_column=PMR_TARGET_COLUMN,
                        prediction_column=get_prediction_column_name(PMR_TARGET_COLUMN),
                        exported_prediction_column_name="Predicted PMR",
                    )
                    combined_test_output = self.merge_test_outputs(pax_test_output, pmr_test_output)
                    test_output_path = run_progress_step(
                        step_progress,
                        "write_test_csv",
                        PredictionFileWriter.write_test_predictions,
                        combined_test_output,
                        f"{name}_preds",
                    )

                if validation_output_path is not None:
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

            if validation_r2 is not None and validation_mae is not None:
                print(f"{name} validation R2: {validation_r2:.4f} | validation MAE: {validation_mae:.4f}")
            if pmr_validation_r2 is not None and pmr_validation_mae is not None:
                print(f"{name} PMR validation R2: {pmr_validation_r2:.4f} | validation MAE: {pmr_validation_mae:.4f}")
            print(f"Predictions written to {output_path}")
            if validation_output_path is not None:
                print(f"Validation predictions written to {validation_output_path}")
            if test_output_path is not None:
                print(f"Test-set predictions written to {test_output_path}")


@track_emissions()
def main() -> None:
    IndividualPredictionRunner(PredictionWindowConfig()).run()


if __name__ == "__main__":
    main()
