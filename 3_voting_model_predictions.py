from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
    VotingRegressor,
)
from sklearn.pipeline import Pipeline

from movement_model_utils import (
    clean_model_params,
    load_hyperparameter_results,
    load_training_and_prediction_frames,
    make_preprocessor,
    PREDICTION_MODE_MISSING_TARGET,
    plot_prediction_results,
    write_predictions,
)

EXPONENT = 10
PREDICTION_MODE = PREDICTION_MODE_MISSING_TARGET


if __name__ == "__main__":
    training_features, training_target, prediction_features, prediction_ids = (
        load_training_and_prediction_frames(prediction_mode=PREDICTION_MODE)
    )
    results = load_hyperparameter_results()

    def params_for(model_name):
        return clean_model_params(results[model_name]["best_params"])

    models = {
        "rf": RandomForestRegressor(**params_for("RandomForestRegressor"), verbose=1),
        "gb": GradientBoostingRegressor(**params_for("GradientBoostingRegressor")),
        "et": ExtraTreesRegressor(**params_for("ExtraTreesRegressor"), verbose=1),
    }
    score_names = [
        "RandomForestRegressor",
        "GradientBoostingRegressor",
        "ExtraTreesRegressor",
    ]
    raw_scores = [abs(float(results[name]["best_score"])) for name in score_names]

    inverse_errors = [1.0 / (score + 1e-9) for score in raw_scores]
    total_inverse_error = sum(inverse_errors)
    weights = [
        (score / total_inverse_error) ** EXPONENT for score in inverse_errors
    ]

    print("Model weights:", dict(zip(models.keys(), weights)))

    ensemble = Pipeline(
        [
            ("preprocess", make_preprocessor(training_features)),
            (
                "voter",
                VotingRegressor(
                    estimators=list(models.items()),
                    weights=weights,
                    n_jobs=10,
                ),
            ),
        ]
    )

    ensemble.fit(training_features, training_target)
    predictions = ensemble.predict(prediction_features)
    output_path = write_predictions(predictions, prediction_ids, "voting_regressor_preds")
    print(f"Predictions written to {output_path}")
    plot_prediction_results(output_path)

