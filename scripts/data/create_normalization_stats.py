"""
Script để tạo file normalization_stats.npz từ training data

Script này tính toán mean và std của các features từ training parquet file
và lưu vào file .npz để sử dụng với LightGBM models.

Usage:
    python scripts/data/create_normalization_stats.py \
        --train_parquet path/to/train.parquet \
        --output_path path/to/normalization_stats.npz \
        --sample_size 50000
"""
import argparse
import sys
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def get_feature_columns(parquet_path: str, label_col: str = "Label") -> list:
    """Lấy danh sách feature columns từ parquet file."""
    pq_file = pq.ParquetFile(parquet_path)
    return [name for name in pq_file.schema.names if name != label_col]


def compute_normalization_stats(
    parquet_path: str,
    feature_cols: list,
    label_col: str = "Label",
    sample_size: int = 50000,
    batch_size: int = 2048,
):
    """
    Tính mean và std trên sample từ train set.
    
    Args:
        parquet_path: Đường dẫn tới file parquet training data
        feature_cols: Danh sách tên các feature columns
        label_col: Tên label column
        sample_size: Số samples để tính stats (mặc định 50000)
        batch_size: Batch size để đọc parquet file
        
    Returns:
        feature_means: numpy array chứa mean của từng feature
        feature_stds: numpy array chứa std của từng feature
    """
    print(f"🔄 Đang tính normalization stats từ {parquet_path}...")
    print(f"   Sample size: {sample_size:,}")
    print(f"   Batch size: {batch_size:,}")
    print(f"   Number of features: {len(feature_cols)}")
    
    pq_file = pq.ParquetFile(parquet_path)
    all_means = []
    all_stds = []
    all_counts = []
    rows_processed = 0
    
    total_batches = (pq_file.metadata.num_rows + batch_size - 1) // batch_size
    batch_count = 0
    
    for batch in pq_file.iter_batches(
        batch_size=batch_size, columns=feature_cols + [label_col]
    ):
        if rows_processed >= sample_size:
            break
            
        batch_df = batch.to_pandas()
        X = batch_df[feature_cols].values.astype(np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        batch_mean = np.mean(X, axis=0)
        batch_std = np.std(X, axis=0, ddof=0)
        batch_count = X.shape[0]
        
        all_means.append(batch_mean * batch_count)
        all_stds.append(batch_std * batch_count)
        all_counts.append(batch_count)
        rows_processed += batch_count
        
        batch_count += 1
        if batch_count % 20 == 0:
            print(f"   ⏳ Đã xử lý {batch_count}/{total_batches} batches, {rows_processed:,}/{sample_size:,} samples...")
    
    total_count = sum(all_counts)
    overall_mean = sum(all_means) / total_count
    overall_std = sum(all_stds) / total_count
    overall_std = np.where(overall_std == 0, 1.0, overall_std)
    
    print(f"✅ Đã tính stats trên {rows_processed:,} samples")
    print(f"   Mean range: [{overall_mean.min():.4f}, {overall_mean.max():.4f}]")
    print(f"   Std range: [{overall_std.min():.4f}, {overall_std.max():.4f}]")
    
    return overall_mean.astype(np.float32), overall_std.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(
        description="Tạo file normalization_stats.npz từ training data"
    )
    parser.add_argument(
        "--train_parquet",
        type=str,
        default=None,
        help="Đường dẫn tới file train parquet. Mặc định: data/train_ember_2018_v2_features_label_other.parquet",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Đường dẫn để lưu file normalization_stats.npz. Mặc định: artifacts/targets/normalization_stats.npz",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=50000,
        help="Số samples để tính stats (mặc định: 50000)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2048,
        help="Batch size để đọc parquet (mặc định: 2048)",
    )
    parser.add_argument(
        "--label_col",
        type=str,
        default="Label",
        help="Tên label column (mặc định: 'Label')",
    )
    
    args = parser.parse_args()
    
    # Thiết lập đường dẫn mặc định
    if args.train_parquet is None:
        # Ưu tiên file mới trong ember_2018_v2
        train_parquet_new = PROJECT_ROOT / "data" / "ember_2018_v2" / "train" / "train_ember_2018_v2_features_label_other.parquet"
        train_parquet_old = PROJECT_ROOT / "data" / "train_ember_2018_v2_features_label_other.parquet"
        train_parquet = str(train_parquet_new if train_parquet_new.exists() else train_parquet_old)
    else:
        train_parquet = args.train_parquet
    
    if args.output_path is None:
        output_path = str(PROJECT_ROOT / "artifacts" / "targets" / "normalization_stats.npz")
    else:
        output_path = args.output_path
    
    # Kiểm tra file input
    if not Path(train_parquet).exists():
        raise FileNotFoundError(
            f"Không tìm thấy file training data: {train_parquet}"
        )
    
    print("=" * 80)
    print("TẠO FILE NORMALIZATION STATS")
    print("=" * 80)
    print(f"Input: {train_parquet}")
    print(f"Output: {output_path}")
    print("=" * 80)
    
    # Lấy feature columns
    print(f"\n🔄 Đang đọc feature columns...")
    feature_cols = get_feature_columns(train_parquet, args.label_col)
    print(f"✅ Tìm thấy {len(feature_cols)} features")
    
    # Tính normalization stats
    feature_means, feature_stds = compute_normalization_stats(
        train_parquet,
        feature_cols,
        args.label_col,
        sample_size=args.sample_size,
        batch_size=args.batch_size,
    )
    
    # Lưu file
    print(f"\n🔄 Đang lưu normalization stats vào {output_path}...")
    np.savez(
        output_path,
        feature_means=feature_means,
        feature_stds=feature_stds,
        feature_cols=np.array(feature_cols, dtype=object),
    )
    
    # Kiểm tra file đã được tạo
    output_file = Path(output_path)
    file_size = output_file.stat().st_size / (1024 * 1024)  # MB
    print(f"✅ Đã lưu normalization stats!")
    print(f"   File: {output_path}")
    print(f"   Size: {file_size:.2f} MB")
    print(f"   Features: {len(feature_cols)}")
    print(f"   Shape means: {feature_means.shape}")
    print(f"   Shape stds: {feature_stds.shape}")
    
    # Test load lại để đảm bảo
    print(f"\n🔄 Đang test load lại file...")
    test_stats = np.load(output_path, allow_pickle=True)
    assert "feature_means" in test_stats
    assert "feature_stds" in test_stats
    assert "feature_cols" in test_stats
    print(f"✅ File hợp lệ, có thể sử dụng với FlexibleLGBTarget")


if __name__ == "__main__":
    main()

