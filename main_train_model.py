import argparse

from models.train import TrainConfig, train_random_forest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RandomForest on Information Arrays.")
    parser.add_argument("--input", type=str, default="data/information_arrays.csv")
    parser.add_argument("--output-dir", type=str, default="data")

    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=0, help="0 means None (no limit).")
    parser.add_argument("--min-samples-leaf", type=int, default=1)

    # Key change: exclude tags by default
    parser.add_argument(
        "--include-tags",
        action="store_true",
        help="Include rule-based semantic tag_* features (may inflate metrics).",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = TrainConfig(
        test_size=args.test_size,
        random_state=args.seed,
        n_estimators=args.n_estimators,
        max_depth=None if args.max_depth == 0 else args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
    )

    train_random_forest(
        input_csv=args.input,
        output_dir=args.output_dir,
        include_tags=bool(args.include_tags),
        cfg=cfg,
    )


if __name__ == "__main__":
    main()
