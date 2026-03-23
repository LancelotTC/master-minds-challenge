import pandas as pd
import os, json, random
from skopt import BayesSearchCV
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from catboost import CatBoostRegressor
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeRegressor
from skopt.space import Real, Integer, Categorical
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
)

RANDOM_SEED = 42


def prepare_data(path):
    df = pd.read_csv(path, skipinitialspace=True)
    target_col = "NbPax"

    df = df.dropna(subset=[target_col])
    # df = df.drop(columns=["id"], errors="ignore")

    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


def make_preprocessor(X: pd.DataFrame, scale_numeric: bool = False):
    categorical_cols = [
        "brand",
        "model",
        "car_class",
        "range",
        "fuel_type",
        "hybrid",
        "grbx_type_ratios",
    ]
    numeric_cols = [c for c in X.columns if c not in categorical_cols]

    num_steps = [("imputer", SimpleImputer(strategy="mean"))]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))

    numeric_transformer = Pipeline(steps=num_steps)

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    return preprocessor


def test_regressors(X, y, n_rounds=3):
    preprocessor = make_preprocessor(X)

    models = {
        "CatBoostRegressor": (
            Pipeline(
                [
                    ("preprocess", make_preprocessor(X)),
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
                "model__depth": Integer(3, 12),
                "model__n_estimators": Integer(100, 3000),
                "model__learning_rate": Real(1e-4, 0.3, prior="log-uniform"),
                "model__random_strength": Real(0.1, 10.0, prior="log-uniform"),
                "model__subsample": Real(0.5, 1.0),
                "model__rsm": Real(0.5, 1.0),
                "model__min_data_in_leaf": Integer(1, 200),
                "model__leaf_estimation_method": Categorical(["Gradient"]),
                "model__loss_function": ["MAE"],
            },
        ),
        "XGBRegressor": (
            Pipeline(
                [
                    ("preprocess", preprocessor),
                    (
                        "model",
                        XGBRegressor(random_state=42),
                    ),
                ]
            ),
            {
                "model__n_estimators": Integer(100, 3000),
                "model__learning_rate": Real(1e-4, 0.3, prior="log-uniform"),
                "model__max_depth": Integer(2, 15),
                "model__subsample": Real(0.5, 1.0),
                "model__colsample_bytree": Real(0.5, 1.0),
                "model__gamma": Real(0.0, 5.0),
                "model__reg_lambda": Real(0.0, 10.0),
                "model__min_child_weight": Integer(1, 50),
                "model__objective": ["reg:absoluteerror"],
                "model__eval_metric": ["mae"],
                "model__tree_method": ["hist"],
            },
        ),
        "DecisionTreeRegressor": (
            Pipeline(
                [
                    ("preprocess", preprocessor),
                    ("model", DecisionTreeRegressor(random_state=42)),
                ]
            ),
            {
                "model__max_depth": Integer(1, 50),
                "model__criterion": Categorical(["absolute_error"]),
                "model__min_samples_split": Integer(2, 100),
                "model__min_samples_leaf": Integer(1, 50),
                "model__max_features": Categorical([None, "sqrt", "log2"]),
                "model__ccp_alpha": Real(0.0, 0.1),
            },
        ),
        "GradientBoostingRegressor": (
            Pipeline(
                [
                    ("preprocess", preprocessor),
                    ("model", GradientBoostingRegressor(random_state=42)),
                ]
            ),
            {
                "model__n_estimators": Integer(50, 2000),
                "model__learning_rate": Real(1e-4, 0.3, prior="log-uniform"),
                "model__max_depth": Integer(2, 10),
                "model__subsample": Real(0.5, 1.0),
                "model__loss": ["absolute_error"],
            },
        ),
        "RandomForestRegressor": (
            Pipeline(
                [
                    ("preprocess", preprocessor),
                    ("model", RandomForestRegressor(random_state=42)),
                ]
            ),
            {
                "model__n_estimators": Integer(100, 3000),
                "model__max_depth": Integer(2, 50),
                "model__min_samples_split": Integer(2, 50),
                "model__min_samples_leaf": Integer(1, 50),
                "model__max_features": Categorical(["sqrt", "log2", None]),
                "model__bootstrap": Categorical([True, False]),
                "model__criterion": Categorical(["absolute_error"]),
            },
        ),
        "ExtraTreesRegressor": (
            Pipeline(
                [
                    ("preprocess", preprocessor),
                    ("model", ExtraTreesRegressor(random_state=42)),
                ]
            ),
            {
                "model__n_estimators": Integer(100, 3000),
                "model__max_depth": Integer(2, 50),
                "model__min_samples_split": Integer(2, 100),
                "model__min_samples_leaf": Integer(1, 50),
                "model__max_features": Categorical(["sqrt", "log2", None]),
                "model__bootstrap": Categorical([True, False]),
                "model__criterion": ["absolute_error"],
            },
        ),
        "HistGradientBoostingRegressor": (
            Pipeline(
                [
                    ("preprocess", preprocessor),
                    ("model", HistGradientBoostingRegressor(random_state=42)),
                ]
            ),
            {
                "model__learning_rate": Real(1e-4, 0.3, prior="log-uniform"),
                "model__max_depth": Integer(2, 50),
                "model__max_iter": Integer(50, 2000),
                "model__l2_regularization": Real(0.0, 10.0),
                "model__min_samples_leaf": Integer(5, 200),
                "model__max_bins": Integer(32, 255),
                "model__early_stopping": Categorical([False]),
                "model__loss": ["absolute_error"],
            },
        ),
    }

    results_path = "hyperparameters.json"
    results: dict[str, dict] = {}
    if os.path.exists(results_path):
        try:
            with open(results_path, "r") as f:
                results = json.load(f)
        except json.JSONDecodeError:
            print("Warning: invalid JSON, starting fresh.")

    base_seed = random.randint(0, 10_000)

    for round_idx in range(1, n_rounds + 1):
        print(f"\n=== ROUND {round_idx}/{n_rounds} ===")
        for name, (model, params) in models.items():
            print(f"Testing {name}...")
            search = BayesSearchCV(
                model,
                params,
                n_iter=40,
                cv=3,
                scoring="neg_mean_absolute_error",
                n_jobs=10,
                random_state=base_seed + round_idx,
                verbose=1,
            )
            search.fit(X, y)
            best_score = search.best_score_
            best_params = search.best_params_

            prev_score = results.get(name, {}).get("best_score", -9999)
            if best_score > prev_score:
                print(f"↑ Improved: {best_score:.4f} > {prev_score:.4f}")
                results[name] = {"best_score": best_score, "best_params": best_params}
                with open(results_path, "w") as f:
                    json.dump(
                        {k.removeprefix("model__"): v for k, v in results.items()},
                        f,
                        indent=4,
                    )
                print(f"Tried params: {best_params}")
            else:
                print(f"↓ No improvement ({best_score:.4f} ≤ {prev_score:.4f})")
                print(f"Tried params: {best_params}")

    print(f"\nAll results saved to {results_path}")
    return results


X, y = prepare_data("datasets/train.csv")
test_regressors(X, y, n_rounds=1)
