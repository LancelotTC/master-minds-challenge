import json
import os
from dataclasses import dataclass
from pathlib import Path

from codecarbon import track_emissions
from lightgbm import LGBMRegressor
from skopt import BayesSearchCV
from skopt.space import Categorical, Integer, Real
from sklearn.ensemble import HistGradientBoostingRegressor
from tqdm.auto import tqdm
from xgboost import XGBRegressor

from movement_model_utils import (
    build_datewise_cv_splits,
    load_training_and_prediction_frames,
    make_model_pipeline,
    sort_features_and_target_by_datetime,
)
from pipeline_config import PIPELINE_CONFIG


@dataclass(frozen=True, slots=True)
class SearchConfig:
    random_seed: int = 42
    training_dataset_size: int | None = 100
    search_iterations: int = 2
    cv_folds: int = 2
    jobs: int = 11
    rounds: int = 1
    results_path: Path = PIPELINE_CONFIG.model_paths.hyperparameters_results


class ResultsStore:
    @staticmethod
    def load(results_path: Path) -> dict[str, object]:
        if not results_path.exists():
            return {}

        try:
            with open(results_path, "r", encoding="utf-8") as file:
                return json.load(file)
        except json.JSONDecodeError:
            return {}

    @staticmethod
    def save(results_path: Path, results: dict[str, object]) -> None:
        results_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = results_path.with_suffix(f"{results_path.suffix}.tmp")
        try:
            with open(temp_path, "w", encoding="utf-8") as file:
                json.dump(results, file, indent=4)
            os.replace(temp_path, results_path)
        finally:
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)


class HyperparameterSearchRunner:
    def __init__(self, config: SearchConfig):
        self.config = config

    @staticmethod
    def integer_range(lower: int, upper: int) -> Integer:
        return Integer(lower, upper if lower < upper else lower + 1)

    def prepare_training_data(self) -> tuple:
        features, target, _, _ = load_training_and_prediction_frames()
        features, target = sort_features_and_target_by_datetime(features, target)

        max_rows = self.config.training_dataset_size
        if max_rows is None or len(features) <= max_rows:
            return features, target
        if max_rows <= 0:
            raise ValueError("training_dataset_size must be positive when set.")

        trimmed_features = features.tail(max_rows).copy()
        trimmed_target = target.loc[trimmed_features.index].copy()
        return trimmed_features, trimmed_target

    def build_models(self, features) -> dict[str, tuple]:
        return {
            "XGBRegressor": (
                make_model_pipeline(
                    XGBRegressor(random_state=self.config.random_seed),
                    features,
                ),
                {
                    "model__n_estimators": self.integer_range(200, 4000),
                    "model__learning_rate": Real(0.003, 0.3, prior="log-uniform"),
                    "model__max_depth": self.integer_range(2, 16),
                    "model__max_leaves": self.integer_range(16, 1024),
                    "model__min_child_weight": self.integer_range(1, 80),
                    "model__subsample": Real(0.5, 1.0),
                    "model__colsample_bytree": Real(0.4, 1.0),
                    "model__colsample_bylevel": Real(0.4, 1.0),
                    "model__colsample_bynode": Real(0.4, 1.0),
                    "model__gamma": Real(1e-8, 10.0, prior="log-uniform"),
                    "model__reg_lambda": Real(1e-4, 100.0, prior="log-uniform"),
                    "model__reg_alpha": Real(1e-4, 100.0, prior="log-uniform"),
                    "model__max_bin": self.integer_range(64, 512),
                    "model__grow_policy": Categorical(["depthwise", "lossguide"]),
                    "model__objective": Categorical(["reg:absoluteerror"]),
                    "model__eval_metric": Categorical(["mae"]),
                    "model__tree_method": Categorical(["hist"]),
                },
            ),
            "HistGradientBoostingRegressor": (
                make_model_pipeline(
                    HistGradientBoostingRegressor(random_state=self.config.random_seed),
                    features,
                ),
                {
                    "model__learning_rate": Real(0.003, 0.3, prior="log-uniform"),
                    "model__max_depth": self.integer_range(2, 20),
                    "model__max_iter": self.integer_range(200, 4000),
                    "model__max_leaf_nodes": self.integer_range(15, 255),
                    "model__l2_regularization": Real(1e-8, 100.0, prior="log-uniform"),
                    "model__min_samples_leaf": self.integer_range(5, 500),
                    "model__max_bins": Integer(64, 255),
                    "model__early_stopping": Categorical([False]),
                    "model__loss": Categorical(["absolute_error"]),
                },
            ),
            "LGBMRegressor": (
                make_model_pipeline(
                    LGBMRegressor(
                        random_state=self.config.random_seed,
                        objective="mae",
                        verbosity=-1,
                        force_col_wise=True,
                    ),
                    features,
                ),
                {
                    "model__n_estimators": self.integer_range(200, 4000),
                    "model__learning_rate": Real(0.003, 0.3, prior="log-uniform"),
                    "model__num_leaves": self.integer_range(16, 1024),
                    "model__max_depth": self.integer_range(2, 16),
                    "model__min_child_samples": self.integer_range(5, 500),
                    "model__min_child_weight": Real(1e-4, 100.0, prior="log-uniform"),
                    "model__subsample": Real(0.5, 1.0),
                    "model__subsample_freq": self.integer_range(0, 7),
                    "model__colsample_bytree": Real(0.4, 1.0),
                    "model__min_split_gain": Real(1e-8, 10.0, prior="log-uniform"),
                    "model__reg_lambda": Real(1e-4, 100.0, prior="log-uniform"),
                    "model__reg_alpha": Real(1e-4, 100.0, prior="log-uniform"),
                    "model__max_bin": self.integer_range(63, 511),
                },
            ),
        }

    @staticmethod
    def _make_progress_callback(progress_bar):
        def callback(_result):
            if progress_bar.n < progress_bar.total:
                progress_bar.update(1)
            return False

        return callback

    def run(self) -> dict[str, object]:
        features, target = self.prepare_training_data()
        models = self.build_models(features)
        results = ResultsStore.load(self.config.results_path)
        cv_splits = build_datewise_cv_splits(features, self.config.cv_folds)
        effective_training_rows = len(features)

        print(f"Training rows: {len(features)}")
        print(f"Feature columns: {len(features.columns)}")

        for round_index in tqdm(range(1, self.config.rounds + 1), desc="Hyperparameter rounds", unit="round"):
            for model_name, (pipeline, search_space) in tqdm(
                list(models.items()),
                desc=f"Round {round_index}/{self.config.rounds}",
                unit="model",
                leave=False,
            ):
                search = BayesSearchCV(
                    pipeline,
                    search_space,
                    n_iter=self.config.search_iterations,
                    cv=cv_splits,
                    scoring="neg_mean_absolute_error",
                    n_jobs=self.config.jobs,
                    random_state=self.config.random_seed + round_index,
                    verbose=0,
                )

                with tqdm(total=self.config.search_iterations, desc=f"{model_name} search", unit="iter", leave=False) as search_bar:
                    search.fit(features, target, callback=self._make_progress_callback(search_bar))
                    if search_bar.n < search_bar.total:
                        search_bar.update(search_bar.total - search_bar.n)

                best_score = float(search.best_score_)
                best_params = {key.removeprefix("model__"): value for key, value in search.best_params_.items()}
                existing_result = results.get(model_name, {})
                previous_rows = existing_result.get("training_rows")
                previous_score = float(existing_result.get("best_score", -9999)) if previous_rows == effective_training_rows else -9999

                if best_score > previous_score:
                    results[model_name] = {
                        "best_score": best_score,
                        "best_params": best_params,
                        "training_rows": effective_training_rows,
                    }
                    ResultsStore.save(self.config.results_path, results)

        return results


@track_emissions()
def main() -> None:
    HyperparameterSearchRunner(SearchConfig()).run()


if __name__ == "__main__":
    main()
