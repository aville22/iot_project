import argparse
from pathlib import Path

from generators.vitals_simulator import VitalsConfig, generate_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic IoT vital signs dataset with realistic patient dynamics."
    )
    parser.add_argument(
        "--n-patients",
        type=int,
        default=150,
        help="Total number of patients to simulate.",
    )
    parser.add_argument(
        "--points-per-patient",
        type=int,
        default=300,
        help="Number of time points per patient.",
    )
    parser.add_argument(
        "--step-seconds",
        type=int,
        default=2,
        help="Time step between measurements (seconds).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/iot_vitals_stream.csv",
        help="Path to output CSV file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = VitalsConfig(
        n_points=args.points_per_patient,
        step_seconds=args.step_seconds,
    )

    print(
        f"Generating dataset | n_patients={args.n_patients}, "
        f"points_per_patient={args.points_per_patient}, "
        f"step={args.step_seconds}s"
    )

    dataset = generate_dataset(
        n_patients=args.n_patients,
        config=config,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataset.to_csv(output_path, index=False)

    print(f"Saved dataset to: {output_path.resolve()}")
    print(f"Rows: {len(dataset)}, columns: {list(dataset.columns)}")


if __name__ == "__main__":
    main()
