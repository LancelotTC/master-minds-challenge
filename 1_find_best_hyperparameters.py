import json
import os

from catboost import CatBoostRegressor
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

from movement_model_utils import load_training_and_prediction_frames, make_preprocessor

RANDOM_SEED = 42
TRAINING_DATASET_SIZE = 5_000
TRAINING_DATASET_SUBSAMPLE_SEED = RANDOM_SEED
SEARCH_ITERATIONS = 10
CV_FOLDS = 2
JOBS = 11


def prepare_data():
    features, target, _, _ = load_training_and_prediction_frames()

    if TRAINING_DATASET_SIZE is not None:
        if TRAINING_DATASET_SIZE <= 0:
            raise ValueError("TRAINING_DATASET_SIZE must be None or a positive integer.")

        if TRAINING_DATASET_SIZE < len(features):
            sampled_indices = features.sample(
                n=TRAINING_DATASET_SIZE,
                random_state=TRAINING_DATASET_SUBSAMPLE_SEED,
            ).index
            features = features.loc[sampled_indices].copy()
            target = target.loc[sampled_indices].copy()

    return features, target


def build_models(features):
    preprocessor = make_preprocessor(features)

    return {
        "CatBoostRegressor": (
            Pipeline(
                [
                    ("preprocess", preprocessor),
                    (
                        "model",
                        CatBoostRegressor(
                            verbose=False,
                            random_seed=RANDOM_SEED,
                            allow_writing_files=False,
                        ),
                    ),
                ]
            ),
            {
                "model__depth": Integer(4, 10),
                "model__n_estimators": Integer(300, 1600),
                "model__learning_rate": Real(0.01, 0.15, prior="log-uniform"),
                "model__random_strength": Real(0.1, 3.0, prior="log-uniform"),
                "model__subsample": Real(0.65, 1.0),
                "model__rsm": Real(0.6, 1.0),
                "model__min_data_in_leaf": Integer(20, 400),
                "model__leaf_estimation_method": Categorical(["Gradient"]),
                "model__loss_function": Categorical(["MAE"]),
            },
        ),
        "XGBRegressor": (
            Pipeline(
                [
                    ("preprocess", preprocessor),
                    ("model", XGBRegressor(random_state=RANDOM_SEED)),
                ]
            ),
            {
                "model__n_estimators": Integer(300, 1800),
                "model__learning_rate": Real(0.01, 0.15, prior="log-uniform"),
                "model__max_depth": Integer(3, 8),
                "model__subsample": Real(0.65, 1.0),
                "model__colsample_bytree": Real(0.6, 1.0),
                "model__gamma": Real(0.0, 2.0),
                "model__reg_lambda": Real(0.1, 8.0, prior="log-uniform"),
                "model__min_child_weight": Integer(5, 80),
                "model__objective": Categorical(["reg:absoluteerror"]),
                "model__eval_metric": Categorical(["mae"]),
                "model__tree_method": Categorical(["hist"]),
            },
        ),
        "DecisionTreeRegressor": (
            Pipeline(
                [
                    ("preprocess", preprocessor),
                    ("model", DecisionTreeRegressor(random_state=RANDOM_SEED)),
                ]
            ),
            {
                "model__max_depth": Integer(4, 18),
                "model__criterion": Categorical(["absolute_error"]),
                "model__min_samples_split": Integer(20, 500),
                "model__min_samples_leaf": Integer(10, 200),
                "model__max_features": Categorical([None, "sqrt", "log2"]),
                "model__ccp_alpha": Real(1e-6, 1e-2, prior="log-uniform"),
            },
        ),
        "GradientBoostingRegressor": (
            Pipeline(
                [
                    ("preprocess", preprocessor),
                    ("model", GradientBoostingRegressor(random_state=RANDOM_SEED)),
                ]
            ),
            {
                "model__n_estimators": Integer(200, 1200),
                "model__learning_rate": Real(0.01, 0.12, prior="log-uniform"),
                "model__max_depth": Integer(2, 5),
                "model__subsample": Real(0.65, 1.0),
                "model__loss": Categorical(["absolute_error"]),
            },
        ),
        "RandomForestRegressor": (
            Pipeline(
                [
                    ("preprocess", preprocessor),
                    ("model", RandomForestRegressor(random_state=RANDOM_SEED)),
                ]
            ),
            {
                "model__n_estimators": Integer(300, 1800),
                "model__max_depth": Integer(6, 24),
                "model__min_samples_split": Integer(10, 200),
                "model__min_samples_leaf": Integer(5, 100),
                "model__max_features": Categorical(["sqrt", "log2", None]),
                "model__bootstrap": Categorical([True, False]),
                "model__criterion": Categorical(["absolute_error"]),
            },
        ),
        "ExtraTreesRegressor": (
            Pipeline(
                [
                    ("preprocess", preprocessor),
                    ("model", ExtraTreesRegressor(random_state=RANDOM_SEED)),
                ]
            ),
            {
                "model__n_estimators": Integer(300, 1800),
                "model__max_depth": Integer(6, 24),
                "model__min_samples_split": Integer(10, 300),
                "model__min_samples_leaf": Integer(5, 100),
                "model__max_features": Categorical(["sqrt", "log2", None]),
                "model__bootstrap": Categorical([True, False]),
                "model__criterion": Categorical(["absolute_error"]),
            },
        ),
        "HistGradientBoostingRegressor": (
            Pipeline(
                [
                    ("preprocess", preprocessor),
                    (
                        "model",
                        HistGradientBoostingRegressor(random_state=RANDOM_SEED),
                    ),
                ]
            ),
            {
                "model__learning_rate": Real(0.01, 0.15, prior="log-uniform"),
                "model__max_depth": Integer(3, 12),
                "model__max_iter": Integer(200, 1200),
                "model__l2_regularization": Real(1e-4, 5.0, prior="log-uniform"),
                "model__min_samples_leaf": Integer(20, 300),
                "model__max_bins": Integer(64, 255),
                "model__early_stopping": Categorical([False]),
                "model__loss": Categorical(["absolute_error"]),
            },
        ),
    }


# lstm


def test_regressors(features, target, n_rounds=3):
    models = build_models(features)
    results_path = "hyperparameters.json"
    results = {}

    if os.path.exists(results_path):
        try:
            with open(results_path, "r", encoding="utf-8") as file:
                results = json.load(file)
        except json.JSONDecodeError:
            print("Warning: invalid JSON, starting fresh.")

    base_seed = RANDOM_SEED

    for round_idx in range(1, n_rounds + 1):
        print(f"\n=== ROUND {round_idx}/{n_rounds} ===")
        for name, (model, params) in models.items():
            print(f"Testing {name}...")
            search = BayesSearchCV(
                model,
                params,
                n_iter=SEARCH_ITERATIONS,
                cv=CV_FOLDS,
                scoring="neg_mean_absolute_error",
                n_jobs=JOBS,
                random_state=base_seed + round_idx,
                verbose=1,
            )
            search.fit(features, target)

            best_score = float(search.best_score_)
            best_params = {key.removeprefix("model__"): value for key, value in search.best_params_.items()}
            previous_score = float(results.get(name, {}).get("best_score", -9999))

            if best_score > previous_score:
                print(f"Improved: {best_score:.4f} > {previous_score:.4f}")
                results[name] = {"best_score": best_score, "best_params": best_params}
                with open(results_path, "w", encoding="utf-8") as file:
                    json.dump(results, file, indent=4)
            else:
                print(f"No improvement ({best_score:.4f} <= {previous_score:.4f})")

            print(f"Tried params: {best_params}")

    print(f"\nAll results saved to {results_path}")
    return results


if __name__ == "__main__":
    X, y = prepare_data()
    print(f"Training rows: {len(X)}")
    print(f"Feature columns: {len(X.columns)}")
    test_regressors(X, y, n_rounds=1)
