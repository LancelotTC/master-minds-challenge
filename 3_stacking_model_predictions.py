import os, json
import pandas as pd
from pandas import DataFrame
from typing import cast, Optional
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import StackingRegressor
from sklearn.preprocessing import OrdinalEncoder

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor,
)


def write_preds(preds, filename: Optional[str] = None):
    start_id = 41257
    df = DataFrame({"id": range(start_id, start_id + len(preds)), "co2": preds})
    os.makedirs("datasets/predictions", exist_ok=True)
    df.to_csv(f"datasets/predictions/{filename or 'stacking_preds'}.csv", index=False)
    print(
        f"Predictions written to datasets/predictions/{filename or 'stacking_preds'}.csv"
    )


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


if __name__ == "__main__":
    training_set = pd.read_csv("datasets/train.csv", skipinitialspace=True)
    real_test_set = pd.read_csv("datasets/test.csv", skipinitialspace=True)
    label_col = "co2"

    X = training_set.drop(columns=["id", label_col], errors="ignore")
    y = training_set[label_col]
    X_test = real_test_set.drop(columns=["id"], errors="ignore")

    with open("hyperparameters.json") as f:
        results = json.load(f)

    def clean_params(name):
        return {
            cast(str, k).removeprefix("model__"): v
            for k, v in results[name]["best_params"].items()
        }

    base_models = [
        ("rf", RandomForestRegressor(**clean_params("RandomForestRegressor"))),
        ("et", ExtraTreesRegressor(**clean_params("ExtraTreesRegressor"))),
        ("gbr", GradientBoostingRegressor(**clean_params("GradientBoostingRegressor"))),
    ]

    meta_model = RidgeCV(alphas=[0.1, 1.0, 10.0])

    preprocessor = make_preprocessor(X)

    model = Pipeline(
        [
            ("preprocess", preprocessor),
            (
                "stack",
                StackingRegressor(
                    estimators=base_models,
                    final_estimator=meta_model,
                    n_jobs=10,
                ),
            ),
        ]
    )

    model.fit(X, y)
    preds_test = model.predict(X_test)
    write_preds(preds_test, "stacking_regressor_preds")
