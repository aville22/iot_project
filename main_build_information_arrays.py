import argparse
from pathlib import Path

import pandas as pd

from stream.preprocessing import PreprocessConfig, preprocess_stream
from stream.windowing import WindowConfig, build_information_arrays


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Information Arrays from synthetic IoT vital signs stream."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/iot_vitals_stream.csv",
        help="Path to input stream CSV.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/information_arrays.csv",
        help="Path to output information arrays CSV.",
    )

    # Preprocessing options
    parser.add_argument("--fill-method", type=str, default="ffill_then_bfill")
    parser.add_argument("--smooth", action="store_true", help="Enable rolling smoothing.")
    parser.add_argument("--smooth-window", type=int, default=5)
    parser.add_argument("--normalize", action="store_true", help="Enable z-score normalization.")
    parser.add_argument("--normalization-mode", type=str, default="global_zscore")

    # Windowing options
    parser.add_argument("--window-seconds", type=int, default=30)
    parser.add_argument("--stride-seconds", type=int, default=10)
    parser.add_argument("--label-strategy", type=str, default="mode")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)

    p_cfg = PreprocessConfig(
        fill_method=args.fill_method,
        apply_smoothing=bool(args.smooth),
        smoothing_window=args.smooth_window,
        apply_normalization=bool(args.normalize),
        normalization_mode=args.normalization_mode,
    )

    w_cfg = WindowConfig(
        window_seconds=args.window_seconds,
        stride_seconds=args.stride_seconds,
        label_strategy=args.label_strategy,
    )

    print(f"Reading stream: {input_path.resolve()}")
    df_clean = preprocess_stream(df, config=p_cfg)

    print("Building information arrays...")
    info = build_information_arrays(df_clean, config=w_cfg)

    info.to_csv(output_path, index=False)

    print(f"Saved information arrays to: {output_path.resolve()}")
    print(f"Rows: {len(info)}, columns: {len(info.columns)}")


if __name__ == "__main__":
    main()
