#!/usr/bin/env python3
import argparse
import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split


def parse_args():
    p = argparse.ArgumentParser(description="Stratified train/test split for a CSV dataset")
    p.add_argument("--input", required=True, help="Path to input CSV file")
    p.add_argument("--label", default="class", help="Label column name (default: class)")
    p.add_argument("--test-size", type=float, default=0.2, help="Test set fraction (0-1), default 0.2")
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    p.add_argument("--train-out", default=None, help="Output CSV path for train set")
    p.add_argument("--test-out", default=None, help="Output CSV path for test set")
    p.add_argument("--encoding", default=None, help="Optional input file encoding (e.g., utf-8, latin1)")
    p.add_argument("--sep", default=",", help="CSV delimiter (default ,)")
    p.add_argument("--format", choices=["csv", "parquet", "both"], default="csv",
                   help="Output format: csv, parquet, or both (default: csv)")
    p.add_argument("--stats-out", default=None,
                   help="Optional path to save JSON summary of class distributions")
    return p.parse_args()


def summarize(label_series):
    counts = label_series.value_counts(dropna=False).sort_index()
    total = len(label_series)
    return {k: {"count": int(v), "ratio": (float(v) / total if total else 0.0)} for k, v in counts.items()}


def main():
    args = parse_args()

    in_path = args.input
    if not os.path.isfile(in_path):
        print(f"ERROR: Input file not found: {in_path}", file=sys.stderr)
        sys.exit(1)

    try:
        df = pd.read_csv(in_path, sep=args.sep, encoding=args.encoding)
    except Exception as e:
        print(f"ERROR: Failed to read CSV: {e}", file=sys.stderr)
        sys.exit(1)

    if args.label not in df.columns:
        print(f"ERROR: Label column '{args.label}' not found. Available columns: {list(df.columns)}", file=sys.stderr)
        sys.exit(1)

    y = df[args.label]

    # Basic checks for stratification feasibility
    class_counts = y.value_counts()
    too_small = class_counts[class_counts < 2]
    if len(too_small) > 0:
        print("ERROR: Stratified split needs at least 2 samples per class. Too small classes:", file=sys.stderr)
        for k, v in too_small.items():
            print(f"  - {k}: {v}", file=sys.stderr)
        print("Consider reducing test-size or merging rare classes.", file=sys.stderr)
        sys.exit(1)

    try:
        train_df, test_df = train_test_split(
            df,
            test_size=args.test_size,
            random_state=args.seed,
            stratify=y,
            shuffle=True,
        )
    except ValueError as e:
        print(f"ERROR: Stratified split failed: {e}", file=sys.stderr)
        sys.exit(1)

    # Determine output paths (base names without extension decisions)
    base_dir = os.path.dirname(os.path.abspath(in_path))
    base_name = os.path.splitext(os.path.basename(in_path))[0]
    # If user provided explicit outputs, respect them; else choose per format
    if args.train_out:
        train_out_csv = args.train_out
    else:
        train_out_csv = os.path.join(base_dir, f"{base_name}_train.csv")
    if args.test_out:
        test_out_csv = args.test_out
    else:
        test_out_csv = os.path.join(base_dir, f"{base_name}_test.csv")

    # Compose parquet default paths
    train_out_parquet = os.path.join(base_dir, f"{base_name}_train.parquet")
    test_out_parquet = os.path.join(base_dir, f"{base_name}_test.parquet")

    saved = {}
    if args.format in ("csv", "both"):
        train_df.to_csv(train_out_csv, index=False)
        test_df.to_csv(test_out_csv, index=False)
        saved["train_csv"] = train_out_csv
        saved["test_csv"] = test_out_csv

    if args.format in ("parquet", "both"):
        try:
            train_df.to_parquet(train_out_parquet, index=False)
            test_df.to_parquet(test_out_parquet, index=False)
        except Exception as e:
            print("ERROR: Writing parquet failed. Ensure 'pyarrow' or 'fastparquet' is installed.", file=sys.stderr)
            print(f"Details: {e}", file=sys.stderr)
            sys.exit(1)
        saved["train_parquet"] = train_out_parquet
        saved["test_parquet"] = test_out_parquet

    # Print summaries
    print("Split complete. Files:")
    for k in sorted(saved.keys()):
        path = saved[k]
        which = "Train" if "train" in k else "Test"
        print(f"  {which} ({k.split('_')[-1]}): {path}  (n={len(train_df) if which=='Train' else len(test_df)})")

    import json
    summary = {
        "original": summarize(y),
        "train": summarize(train_df[args.label]),
        "test": summarize(test_df[args.label]),
        "format": args.format,
        "outputs": saved,
    }
    print("\nClass distribution (count, ratio):")
    print(json.dumps(summary, indent=2))

    # Optionally save stats to JSON
    stats_path = args.stats_out or os.path.join(base_dir, f"{base_name}_split_stats.json")
    try:
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\nSaved stats to: {stats_path}")
    except Exception as e:
        print(f"WARNING: Failed to write stats JSON to {stats_path}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
