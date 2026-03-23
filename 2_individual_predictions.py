import os, json
import pandas as pd
from pandas import DataFrame
from xgboost import XGBRegressor
from typing import Optional, cast
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from catboost import CatBoostRegressor
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
)


def write_preds(preds, filename: Optional[str] = None):
    start_id = 41257
    df = DataFrame({"id": range(start_id, start_id + len(preds)), "co2": preds})
    os.makedirs("datasets/predictions", exist_ok=True)
    df.to_csv(f"datasets/predictions/{filename or 'test_preds'}.csv", index=False)


class ColorString:
    BLUE = "blue"

    def __init__(self, string: str, fg: Optional[str] = None):
        color_code = {"blue": 34}.get(fg, 0)
        self.string = f"\033[{color_code}m{string}\033[0m"

    def __str__(self):
        return self.string


def make_preprocessor(X: pd.DataFrame):
    categorical_cols = [
        "brand",
        "model",
        "car_class",
        "range",
        "fuel_type",
        "hybrid",
        "grbx_type_ratios",
    ]
    categorical_cols = [c for c in categorical_cols if c in X.columns]
    numeric_cols = [c for c in X.columns if c not in categorical_cols]

    num_proc = Pipeline([("imputer", SimpleImputer(strategy="mean"))])
    cat_proc = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            ),
        ]
    )

    return ColumnTransformer(
        [
            ("num", num_proc, numeric_cols),
            ("cat", cat_proc, categorical_cols),
        ]
    )


def get_predictions(model, training_set: pd.DataFrame, test_set: pd.DataFrame):
    label_col = "co2"
    X = training_set.drop(columns=["id", label_col], errors="ignore")
    y = training_set[label_col]

    preprocessor = make_preprocessor(X)

    pipeline = Pipeline(
        [
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline.fit(X_train, y_train)
    preds_val = pipeline.predict(X_val)
    score = r2_score(y_val, preds_val)

    preds_test = pipeline.predict(test_set.drop(columns=["id"], errors="ignore"))
    return preds_test, score


if __name__ == "__main__":
    training_set = pd.read_csv("datasets/train.csv", skipinitialspace=True)
    real_test_set = pd.read_csv("datasets/test.csv", skipinitialspace=True)

    with open("hyperparameters.json") as f:
        results = json.load(f)

    regressors = {
        "DecisionTreeRegressor": DecisionTreeRegressor(
            **{
                cast(str, key).removeprefix("model__"): value
                for key, value in results["DecisionTreeRegressor"][
                    "best_params"
                ].items()
            }
        ),
        "RandomForestRegressor": RandomForestRegressor(
            **{
                cast(str, key).removeprefix("model__"): value
                for key, value in results["RandomForestRegressor"][
                    "best_params"
                ].items()
            }
        ),
        "ExtraTreesRegressor": ExtraTreesRegressor(
            **{
                cast(str, key).removeprefix("model__"): value
                for key, value in results["ExtraTreesRegressor"]["best_params"].items()
            }
        ),
        "CatBoostRegressor": CatBoostRegressor(
            loss_function="MAE"
            ** {
                cast(str, key).removeprefix("model__"): value
                for key, value in results["CatBoostRegressor"]["best_params"].items()
            }
        ),
        "XGBRegressor": XGBRegressor(
            **{
                cast(str, key).removeprefix("model__"): value
                for key, value in results["XGBRegressor"]["best_params"].items()
            }
        ),
        "GradientBoostingRegressor": GradientBoostingRegressor(
            **{
                cast(str, key).removeprefix("model__"): value
                for key, value in results["GradientBoostingRegressor"][
                    "best_params"
                ].items()
            }
        ),
        "HistGradientBoostingRegressor": HistGradientBoostingRegressor(
            **{
                cast(str, key).removeprefix("model__"): value
                for key, value in results["HistGradientBoostingRegressor"][
                    "best_params"
                ].items()
            }
        ),
    }

    for name, model in regressors.items():
        preds, score = get_predictions(model, training_set, real_test_set)
        write_preds(preds, f"{name}_preds")
        print(ColorString(f"{name} R² score: {score:.4f}", fg="blue"))
