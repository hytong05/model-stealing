#!/usr/bin/env python3
"""
Offline/standalone label query utility.

Cho phép gửi một tập feature vectors tới target model (local weights)
để nhận về nhãn nhị phân, tách biệt hoàn toàn quá trình truy vấn và tấn công.

Ví dụ:

Local (không cần server):
    python scripts/oracle/query_labels.py \
        --input-path data/pool_features.npy \
        --output-path cache/pool_labels.npy \
        --model-type h5 \
        --model-path artifacts/targets/CEE.h5

Lưu ý: Khi dùng local Keras/Dual models, dữ liệu phải được scale giống như khi train attack (ví dụ RobustScaler).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.targets.oracle_client import LocalOracleClient


def load_features(input_path: Path, input_format: str, label_col: Optional[str]) -> np.ndarray:
    if input_format == "npy":
        return np.load(input_path)
    if input_format == "parquet":
        df = pd.read_parquet(input_path)
        if label_col and label_col in df.columns:
            df = df.drop(columns=[label_col])
        return df.values.astype(np.float32)
    raise ValueError(f"Input format không được hỗ trợ: {input_format}")


def save_labels(labels: np.ndarray, output_path: Path, output_format: str) -> None:
    if output_format == "npy":
        np.save(output_path, labels)
    elif output_format == "txt":
        np.savetxt(output_path, labels, fmt="%d")
    else:
        raise ValueError(f"Output format không được hỗ trợ: {output_format}")


def infer_format(path: Path, provided: Optional[str]) -> str:
    if provided and provided != "auto":
        return provided
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return "npy"
    if suffix == ".parquet":
        return "parquet"
    raise ValueError(
        "Không xác định được định dạng input. Vui lòng dùng --input-format (npy/parquet)."
    )


def build_oracle_client(args: argparse.Namespace):
    if not args.model_path or not args.model_type:
        raise ValueError("Cần cung cấp --model-path và --model-type.")
    return LocalOracleClient(
        model_type=args.model_type,
        model_path=args.model_path,
        normalization_stats_path=args.normalization_stats_path,
        threshold=args.threshold,
        feature_dim=args.feature_dim,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Module truy vấn nhãn target model độc lập")
    parser.add_argument("--input-path", required=True, help="Đường dẫn tới file features (.npy hoặc .parquet)")
    parser.add_argument("--input-format", choices=["auto", "npy", "parquet"], default="auto")
    parser.add_argument("--label-col", help="Tên cột label trong parquet (nếu muốn drop)")
    parser.add_argument("--output-path", required=True, help="File lưu nhãn (mặc định .npy)")
    parser.add_argument("--output-format", choices=["npy", "txt"], default="npy")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--model-path", required=True, help="Đường dẫn local tới model (h5 hoặc lgb)")
    parser.add_argument("--model-type", choices=["h5", "lgb"], required=True, help="Loại model")
    parser.add_argument("--normalization-stats-path", help="Stats .npz cho LightGBM")
    parser.add_argument("--feature-dim", type=int, help="Số đặc trưng (nếu muốn ép)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold binary cho local oracle")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_path).expanduser()
    if not input_path.exists():
        raise FileNotFoundError(f"Không tìm thấy input: {input_path}")

    output_path = Path(args.output_path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    input_format = infer_format(input_path, args.input_format)
    print(f"📂 Input: {input_path} ({input_format})")
    features = load_features(input_path, input_format, args.label_col)
    print(f"   Loaded features: {features.shape}")

    client = build_oracle_client(args)

    num_samples = features.shape[0]
    labels = np.zeros(num_samples, dtype=np.int32)

    print(f"\n🚀 Đang truy vấn oracle module (local weights)...")
    for start in range(0, num_samples, args.batch_size):
        end = min(start + args.batch_size, num_samples)
        batch = features[start:end]
        labels[start:end] = client.predict(batch)
        if (start // args.batch_size) % 50 == 0:
            print(f"   … processed {end}/{num_samples}")

    save_labels(labels, output_path, args.output_format)
    print(f"\n✅ Đã lưu nhãn vào {output_path} ({labels.shape[0]} samples)")


if __name__ == "__main__":
    main()

