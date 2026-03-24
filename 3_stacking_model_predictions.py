from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
    StackingRegressor,
)
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline

from movement_model_utils import (
    clean_model_params,
    load_hyperparameter_results,
    load_training_and_prediction_frames,
    make_preprocessor,
    write_predictions,
)


if __name__ == "__main__":
    training_features, training_target, prediction_features, prediction_ids = (
        load_training_and_prediction_frames()
    )
    results = load_hyperparameter_results()

    def params_for(model_name):
        return clean_model_params(results[model_name]["best_params"])

    base_models = [
        ("rf", RandomForestRegressor(**params_for("RandomForestRegressor"))),
        ("et", ExtraTreesRegressor(**params_for("ExtraTreesRegressor"))),
        ("gbr", GradientBoostingRegressor(**params_for("GradientBoostingRegressor"))),
    ]

    meta_model = RidgeCV(alphas=[0.1, 1.0, 10.0])
    stacked_model = Pipeline(
        [
            ("preprocess", make_preprocessor(training_features)),
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

    stacked_model.fit(training_features, training_target)
    predictions = stacked_model.predict(prediction_features)
    output_path = write_predictions(
        predictions,
        prediction_ids,
        "stacking_regressor_preds",
    )
    print(f"Predictions written to {output_path}")

