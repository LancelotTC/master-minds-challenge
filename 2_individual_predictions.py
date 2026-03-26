from typing import Optional

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from movement_model_utils import (
    clean_model_params,
    load_hyperparameter_results,
    load_training_and_prediction_frames,
    make_preprocessor,
    PREDICTION_MODE_MISSING_TARGET,
    PREDICTION_MODE_KNOWN_TARGET,
    plot_prediction_results,
    write_predictions,
)

PREDICTION_MODE = PREDICTION_MODE_KNOWN_TARGET


class ColorString:
    BLUE = "blue"

    def __init__(self, string: str, fg: Optional[str] = None):
        color_code = {"blue": 34}.get(fg, 0)
        self.string = f"\033[{color_code}m{string}\033[0m"

    def __str__(self):
        return self.string


def params_without(params: dict[str, object], *excluded_keys: str) -> dict[str, object]:
    excluded = set(excluded_keys)
    return {key: value for key, value in params.items() if key not in excluded}


def get_predictions(model, features, target, prediction_features):
    pipeline = Pipeline(
        [
            ("preprocess", make_preprocessor(features)),
            ("model", model),
        ]
    )

    X_train, X_val, y_train, y_val = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
    )

    pipeline.fit(X_train, y_train)
    validation_predictions = pipeline.predict(X_val)
    prediction_rows = pipeline.predict(prediction_features)

    return (
        prediction_rows,
        r2_score(y_val, validation_predictions),
        mean_absolute_error(y_val, validation_predictions),
    )


if __name__ == "__main__":
    training_features, training_target, prediction_features, prediction_ids = load_training_and_prediction_frames(
        prediction_mode=PREDICTION_MODE
    )
    results = load_hyperparameter_results()

    regressors = {
        # "DecisionTreeRegressor": DecisionTreeRegressor(
        #     **clean_model_params(results["DecisionTreeRegressor"]["best_params"])
        # ),
        # "RandomForestRegressor": RandomForestRegressor(
        #     **clean_model_params(results["RandomForestRegressor"]["best_params"])
        # ),
        # "ExtraTreesRegressor": ExtraTreesRegressor(**clean_model_params(results["ExtraTreesRegressor"]["best_params"])),
        # "CatBoostRegressor": CatBoostRegressor(
        #     loss_function="MAE",
        #     verbose=False,
        #     random_seed=42,
        #     allow_writing_files=False,
        #     **params_without(
        #         clean_model_params(results["CatBoostRegressor"]["best_params"]),
        #         "loss_function",
        #         "verbose",
        #         "random_seed",
        #         "allow_writing_files",
        #     ),
        # ),
        # "XGBRegressor": XGBRegressor(**clean_model_params(results["XGBRegressor"]["best_params"])),
        "GradientBoostingRegressor": GradientBoostingRegressor(
            **clean_model_params(results["GradientBoostingRegressor"]["best_params"])
        ),
        # "HistGradientBoostingRegressor": HistGradientBoostingRegressor(
        #     **clean_model_params(results["HistGradientBoostingRegressor"]["best_params"])
        # ),
    }

    for name, model in regressors.items():
        predictions, validation_r2, validation_mae = get_predictions(
            model,
            training_features,
            training_target,
            prediction_features,
        )
        output_path = write_predictions(predictions, prediction_ids, f"{name}_preds")
        print(
            ColorString(
                (f"{name} validation R2: {validation_r2:.4f} | " f"validation MAE: {validation_mae:.4f}"),
                fg="blue",
            )
        )
        print(f"Predictions written to {output_path}")
        plot_prediction_results(output_path)
