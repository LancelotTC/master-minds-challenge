import argparse

from movement_model_utils import regenerate_prediction_plots


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Regenerate prediction plots from saved *_preds.csv files.")
    parser.add_argument(
        "--predictions-dir",
        default="predictions",
        help="Root folder containing prediction CSV files (default: predictions).",
    )
    parser.add_argument(
        "--pattern",
        default="*_preds.csv",
        help="Glob pattern used to find prediction CSV files recursively.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display figures interactively while generating PNG files.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=80000,
        help="Maximum paired points per figure (0 keeps all points).",
    )
    args = parser.parse_args()

    regenerate_prediction_plots(
        predictions_root=args.predictions_dir,
        pattern=args.pattern,
        show=args.show,
        max_points=args.max_points,
    )
