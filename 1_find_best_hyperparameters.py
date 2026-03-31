import json
import os
from pathlib import Path

from catboost import CatBoostRegressor
from codecarbon import track_emissions
from lightgbm import LGBMRegressor
from skopt import BayesSearchCV
from skopt.space import Categorical, Integer, Real
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from tqdm.auto import tqdm

from movement_model_utils import (
    build_datewise_cv_splits,
    load_training_and_prediction_frames,
    make_model_pipeline,
    sort_features_and_target_by_datetime,
)

RANDOM_SEED = 42
TRAINING_DATASET_SIZE = 311_572
SEARCH_ITERATIONS = 100
CV_FOLDS = 5
# Use all CPUs. Each parallel job keeps its own data copy in memory.
JOBS = -1
RESULTS_FILENAME = "hyperparameters.json"
RESULTS_PATH = Path(__file__).resolve().parent / RESULTS_FILENAME


def integer_range(lower: int, upper: int) -> Integer:
    if lower >= upper:
        upper = lower + 1
    return Integer(lower, upper)


def resolve_effective_training_size(total_rows: int) -> int:
    if TRAINING_DATASET_SIZE is None:
        return total_rows
    return min(TRAINING_DATASET_SIZE, total_rows)


def prepare_data():
    features, target, _, _ = load_training_and_prediction_frames()
    features, target = sort_features_and_target_by_datetime(features, target)

    if TRAINING_DATASET_SIZE is None:
        return features, target

    if TRAINING_DATASET_SIZE <= 0:
        raise ValueError("TRAINING_DATASET_SIZE must be None or a positive integer.")

    effective_training_size = resolve_effective_training_size(len(features))
    if effective_training_size == len(features):
        return features, target

    recent_features = features.tail(effective_training_size).copy()
    recent_target = target.loc[recent_features.index].copy()

    return recent_features, recent_target


def build_models(features):
    return {
        # "XGBRegressor": (
        #     Pipeline(
        #         [
        #             ("preprocess", preprocessor),
        #             ("model", XGBRegressor(random_state=RANDOM_SEED)),
        #         ]
        #     ),
        #     {
        #         "model__n_estimators": integer_range(200, 4000),
        #         "model__learning_rate": Real(0.003, 0.3, prior="log-uniform"),
        #         "model__max_depth": integer_range(2, 16),
        #         "model__max_leaves": integer_range(16, 1024),
        #         "model__min_child_weight": integer_range(1, 80),
        #         "model__subsample": Real(0.5, 1.0),
        #         "model__colsample_bytree": Real(0.4, 1.0),
        #         "model__colsample_bylevel": Real(0.4, 1.0),
        #         "model__colsample_bynode": Real(0.4, 1.0),
        #         "model__gamma": Real(1e-8, 10.0, prior="log-uniform"),
        #         "model__reg_lambda": Real(1e-4, 100.0, prior="log-uniform"),
        #         "model__reg_alpha": Real(1e-4, 100.0, prior="log-uniform"),
        #         "model__max_bin": integer_range(64, 512),
        #         "model__grow_policy": Categorical(["depthwise", "lossguide"]),
        #         "model__objective": Categorical(["reg:absoluteerror"]),
        #         "model__eval_metric": Categorical(["mae"]),
        #         "model__tree_method": Categorical(["hist"]),
        #     },
        # ),
        # "HistGradientBoostingRegressor": (
        #     Pipeline(
        #         [
        #             ("preprocess", preprocessor),
        #             (
        #                 "model",
        #                 HistGradientBoostingRegressor(random_state=RANDOM_SEED),
        #             ),
        #         ]
        #     ),
        #     {
        #         "model__learning_rate": Real(0.003, 0.3, prior="log-uniform"),
        #         "model__max_depth": integer_range(2, 20),
        #         "model__max_iter": integer_range(200, 4000),
        #         "model__max_leaf_nodes": integer_range(15, 255),
        #         "model__l2_regularization": Real(1e-8, 100.0, prior="log-uniform"),
        #         "model__min_samples_leaf": integer_range(5, 500),
        #         "model__max_bins": Integer(64, 255),
        #         "model__early_stopping": Categorical([False]),
        #         "model__loss": Categorical(["absolute_error"]),
        #     },
        # ),
        "LGBMRegressor": (
            make_model_pipeline(
                LGBMRegressor(
                    random_state=RANDOM_SEED,
                    objective="mae",
                    verbosity=-1,
                    force_col_wise=True,
                ),
                features,
            ),
            {
                "model__n_estimators": integer_range(200, 4000),
                "model__learning_rate": Real(0.003, 0.3, prior="log-uniform"),
                "model__num_leaves": integer_range(16, 1024),
                "model__max_depth": integer_range(2, 16),
                "model__min_child_samples": integer_range(5, 500),
                "model__min_child_weight": Real(1e-4, 100.0, prior="log-uniform"),
                "model__subsample": Real(0.5, 1.0),
                "model__subsample_freq": integer_range(0, 7),
                "model__colsample_bytree": Real(0.4, 1.0),
                "model__min_split_gain": Real(1e-8, 10.0, prior="log-uniform"),
                "model__reg_lambda": Real(1e-4, 100.0, prior="log-uniform"),
                "model__reg_alpha": Real(1e-4, 100.0, prior="log-uniform"),
                "model__max_bin": integer_range(63, 511),
            },
        ),
    }


# lstm


def write_results_file(results_path: Path, results: dict[str, object]) -> None:
    results_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = results_path.with_suffix(f"{results_path.suffix}.tmp")

    try:
        with open(temp_path, "w", encoding="utf-8") as file:
            json.dump(results, file, indent=4)
        os.replace(temp_path, results_path)
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)


def _make_search_progress_callback(progress_bar):
    def _callback(_result):
        if progress_bar.n < progress_bar.total:
            progress_bar.update(1)
        return False

    return _callback


def test_regressors(features, target, n_rounds=3):
    features, target = sort_features_and_target_by_datetime(features, target)
    models = build_models(features)
    results_path = RESULTS_PATH
    results = {}
    effective_training_rows = len(features)
    cv_splits = build_datewise_cv_splits(features, CV_FOLDS)

    if results_path.exists():
        try:
            with open(results_path, "r", encoding="utf-8") as file:
                results = json.load(file)
        except json.JSONDecodeError:
            print("Warning: invalid JSON, starting fresh.")

    base_seed = RANDOM_SEED

    model_items = list(models.items())

    for round_idx in tqdm(range(1, n_rounds + 1), desc="Hyperparameter rounds", unit="round"):
        print(f"\n=== ROUND {round_idx}/{n_rounds} ===")
        for name, (model, params) in tqdm(
            model_items,
            desc=f"Round {round_idx}/{n_rounds}",
            unit="model",
            leave=False,
        ):
            search = BayesSearchCV(
                model,
                params,
                n_iter=SEARCH_ITERATIONS,
                cv=cv_splits,
                scoring="neg_mean_absolute_error",
                n_jobs=JOBS,
                random_state=base_seed + round_idx,
                verbose=0,
            )

            with tqdm(total=SEARCH_ITERATIONS, desc=f"{name} search", unit="iter", leave=False) as search_bar:
                search.fit(features, target, callback=_make_search_progress_callback(search_bar))
                if search_bar.n < search_bar.total:
                    search_bar.update(search_bar.total - search_bar.n)

            best_score = float(search.best_score_)
            best_params = {key.removeprefix("model__"): value for key, value in search.best_params_.items()}
            existing_result = results.get(name, {})
            previous_training_rows = existing_result.get("training_rows")
            same_training_rows = previous_training_rows == effective_training_rows
            previous_score = float(existing_result.get("best_score", -9999)) if same_training_rows else -9999

            if best_score > previous_score:
                print(f"Improved: {best_score:.4f} > {previous_score:.4f}")
                results[name] = {
                    "best_score": best_score,
                    "best_params": best_params,
                    "training_rows": effective_training_rows,
                }
                try:
                    write_results_file(results_path, results)
                except OSError as error:
                    raise OSError(
                        f"Failed to write hyperparameter results to {results_path}. "
                        f"Current working directory: {Path.cwd()}"
                    ) from error
            else:
                print(f"No improvement ({best_score:.4f} <= {previous_score:.4f})")

            print(f"Tried params: {best_params}")

    print(f"\nAll results saved to {results_path}")
    return results


@track_emissions()
def main():
    X, y = prepare_data()
    print(f"Training rows: {len(X)}")
    print(f"Feature columns: {len(X.columns)}")
    print("Using fixed wide BayesSearchCV search spaces.")
    test_regressors(X, y, n_rounds=1)


if __name__ == "__main__":
    main()
