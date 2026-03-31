from codecarbon import track_emissions

from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
    StackingRegressor,
)
from sklearn.linear_model import RidgeCV
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

PREDICTION_MODE = PREDICTION_MODE_MISSING_TARGET


@track_emissions()
def main():
    with tqdm(total=8, desc="Stacking workflow", unit="step") as progress:
        training_features, training_target, prediction_features, prediction_ids = run_progress_step(
            progress,
            "load_data",
            load_training_and_prediction_frames,
            prediction_mode=PREDICTION_MODE,
        )
        results = run_progress_step(progress, "load_hparams", load_hyperparameter_results)

        def params_for(model_name):
            return clean_model_params(results[model_name]["best_params"])

        def build_base_models():
            return [
                ("rf", RandomForestRegressor(**params_for("RandomForestRegressor"))),
                ("et", ExtraTreesRegressor(**params_for("ExtraTreesRegressor"))),
                ("gbr", GradientBoostingRegressor(**params_for("GradientBoostingRegressor"))),
            ]

        base_models = run_progress_step(progress, "build_models", build_base_models)
        meta_model = RidgeCV(alphas=[0.1, 1.0, 10.0])

        def build_stacked_model():
            return make_model_pipeline(
                StackingRegressor(
                    estimators=base_models,
                    final_estimator=meta_model,
                    n_jobs=10,
                ),
                training_features,
            )

        stacked_model = run_progress_step(progress, "build_pipeline", build_stacked_model)
        run_progress_step(progress, "fit", stacked_model.fit, training_features, training_target)
        predictions = run_progress_step(progress, "predict", stacked_model.predict, prediction_features)
        output_path = run_progress_step(
            progress,
            "write_csv",
            write_predictions,
            predictions,
            prediction_ids,
            "stacking_regressor_preds",
        )
        print(f"Predictions written to {output_path}")
        run_progress_step(progress, "plot", plot_prediction_results, output_path)


if __name__ == "__main__":
    main()
