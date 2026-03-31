from codecarbon import track_emissions

from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
    VotingRegressor,
)
from sklearn.pipeline import Pipeline
from tqdm.auto import tqdm

from movement_model_utils import (
    clean_model_params,
    load_hyperparameter_results,
    load_training_and_prediction_frames,
    make_model_pipeline,
    PREDICTION_MODE_MISSING_TARGET,
    plot_prediction_results,
    run_progress_step,
    write_predictions,
)

EXPONENT = 10
PREDICTION_MODE = PREDICTION_MODE_MISSING_TARGET


@track_emissions()
def main():
    with tqdm(total=9, desc="Voting workflow", unit="step") as progress:
        training_features, training_target, prediction_features, prediction_ids = run_progress_step(
            progress,
            "load_data",
            load_training_and_prediction_frames,
            prediction_mode=PREDICTION_MODE,
        )
        results = run_progress_step(progress, "load_hparams", load_hyperparameter_results)

        def params_for(model_name):
            return clean_model_params(results[model_name]["best_params"])

        def build_models():
            return {
                "rf": RandomForestRegressor(**params_for("RandomForestRegressor"), verbose=1),
                "gb": GradientBoostingRegressor(**params_for("GradientBoostingRegressor")),
                "et": ExtraTreesRegressor(**params_for("ExtraTreesRegressor"), verbose=1),
            }

        models = run_progress_step(progress, "build_models", build_models)
        score_names = [
            "RandomForestRegressor",
            "GradientBoostingRegressor",
            "ExtraTreesRegressor",
        ]

        def compute_weights():
            raw_scores = [abs(float(results[name]["best_score"])) for name in score_names]
            inverse_errors = [1.0 / (score + 1e-9) for score in raw_scores]
            total_inverse_error = sum(inverse_errors)
            return [(score / total_inverse_error) ** EXPONENT for score in inverse_errors]

        weights = run_progress_step(progress, "compute_weights", compute_weights)
        print("Model weights:", dict(zip(models.keys(), weights)))

        def build_ensemble():
            return make_model_pipeline(
                VotingRegressor(
                    estimators=list(models.items()),
                    weights=weights,
                    n_jobs=10,
                ),
                training_features,
            )

        ensemble = run_progress_step(progress, "build_pipeline", build_ensemble)
        run_progress_step(progress, "fit", ensemble.fit, training_features, training_target)
        predictions = run_progress_step(progress, "predict", ensemble.predict, prediction_features)
        output_path = run_progress_step(
            progress,
            "write_csv",
            write_predictions,
            predictions,
            prediction_ids,
            "voting_regressor_preds",
        )
        print(f"Predictions written to {output_path}")
        run_progress_step(progress, "plot", plot_prediction_results, output_path)


if __name__ == "__main__":
    main()
