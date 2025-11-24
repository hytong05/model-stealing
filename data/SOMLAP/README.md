# SOMLAP dataset split

This workspace contains a helper to create balanced train/test splits from `SOMLAP DATASET.csv`.

## Quick start

Use the provided virtual environment path if available.

```bash
# Split with 20% test, stratified by the 'class' column
/home/hytong/Documents/model_extraction_malware/data/SOMLAP/.venv/bin/python split_dataset.py \
  --input "SOMLAP DATASET.csv" \
  --label class \
  --test-size 0.2 \
  --seed 42
```

Outputs:
- `SOMLAP DATASET_train.csv`
- `SOMLAP DATASET_test.csv`
 - `SOMLAP DATASET_split_stats.json` (class distribution summary)

## Options
- `--label`: Label column name (default: `class`).
- `--test-size`: Fraction for test set (default: 0.2).
- `--seed`: Random seed for reproducibility.
- `--encoding`: CSV encoding if needed.
- `--sep`: CSV delimiter (default `,`).
- `--train-out`, `--test-out`: Custom output paths.

## Parquet output
To write Parquet files instead of CSV:

```bash
/home/hytong/Documents/model_extraction_malware/data/SOMLAP/.venv/bin/python split_dataset.py \
  --input "SOMLAP DATASET.csv" \
  --label class \
  --test-size 0.2 \
  --seed 42 \
  --format parquet
```

Outputs:
- `SOMLAP DATASET_train.parquet`
- `SOMLAP DATASET_test.parquet`
- `SOMLAP DATASET_split_stats.json`

Use `--format both` to generate both CSV and Parquet.

Note: Parquet requires `pyarrow` (installed by this workspace setup). If needed manually:

```bash
/home/hytong/Documents/model_extraction_malware/data/SOMLAP/.venv/bin/python -m pip install pyarrow
```
