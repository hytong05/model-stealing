#!/usr/bin/env python3
"""
Script test module truy vấn target model.

Script này:
1. Load features và ground truth labels từ parquet file
2. Truy vấn target model để lấy predictions
3. So sánh predictions với ground truth
4. Tính các metrics (accuracy, precision, recall, F1, confusion matrix)
5. Lưu kết quả vào file
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.targets.oracle_client import LocalOracleClient, create_oracle_from_name
from sklearn.preprocessing import RobustScaler
from pathlib import Path


def get_feature_columns(parquet_path: str, label_col: str = "Label") -> list:
    """Lấy danh sách feature columns từ parquet file."""
    pq_file = pq.ParquetFile(parquet_path)
    return [name for name in pq_file.schema.names if name != label_col]


def load_data_with_labels(
    parquet_path: str,
    feature_cols: list,
    label_col: str = "Label",
    max_samples: Optional[int] = None,
    batch_size: int = 10000,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load features và ground truth labels từ parquet file.
    Loại bỏ các samples có label = -1 (unlabeled).
    """
    pq_file = pq.ParquetFile(parquet_path)
    all_X = []
    all_y = []
    rows_processed = 0

    print(f"📂 Đang load dữ liệu từ {parquet_path}...")
    for batch in pq_file.iter_batches(batch_size=batch_size, columns=feature_cols + [label_col]):
        if max_samples and rows_processed >= max_samples:
            break

        batch_df = batch.to_pandas()

        # Lấy labels
        if label_col in batch_df.columns:
            labels = batch_df[label_col].values
        else:
            alt_cols = [col for col in batch_df.columns if col.lower() == label_col.lower()]
            if alt_cols:
                labels = batch_df[alt_cols[0]].values
            else:
                raise KeyError(f"Label column '{label_col}' không tồn tại trong batch.")

        # Loại bỏ label -1 (unlabeled)
        valid_mask = labels != -1
        if not np.any(valid_mask):
            continue

        batch_X = batch_df[feature_cols].values.astype(np.float32)[valid_mask]
        batch_y = labels[valid_mask].astype(np.int32)

        # Xử lý NaN/Inf
        batch_X = np.nan_to_num(batch_X, nan=0.0, posinf=0.0, neginf=0.0)

        all_X.append(batch_X)
        all_y.append(batch_y)
        rows_processed += len(batch_y)

        if max_samples and rows_processed >= max_samples:
            # Cắt bớt batch cuối nếu cần
            excess = rows_processed - max_samples
            if excess > 0:
                all_X[-1] = all_X[-1][:-excess]
                all_y[-1] = all_y[-1][:-excess]
            break

    X = np.vstack(all_X) if all_X else np.array([]).reshape(0, len(feature_cols))
    y = np.concatenate(all_y) if all_y else np.array([], dtype=np.int32)

    print(f"✅ Đã load {len(y):,} samples (đã loại bỏ label -1)")
    print(f"   Features shape: {X.shape}")
    print(f"   Labels distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

    return X, y


def load_normalization_stats(normalization_stats_path: Optional[str], model_path: Path) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Load normalization stats từ file .npz.
    Tự động tìm file nếu không được chỉ định.
    """
    if normalization_stats_path:
        stats_path = Path(normalization_stats_path).expanduser().resolve()
    else:
        # Tự động tìm file normalization_stats.npz trong cùng thư mục với model
        model_dir = model_path.parent
        model_name = model_path.stem
        possible_paths = [
            model_dir / f"{model_name}.npz",
            model_dir / f"{model_name}_normalization_stats.npz",
            model_dir / "normalization_stats.npz",
        ]
        stats_path = None
        for path in possible_paths:
            if path.exists():
                stats_path = path
                break
    
    if stats_path is None or not stats_path.exists():
        return None, None
    
    print(f"   📂 Đang load normalization stats từ {stats_path.name}...")
    stats = np.load(stats_path, allow_pickle=True)
    feature_means = stats.get("feature_means")
    feature_stds = stats.get("feature_stds")
    
    if feature_means is None or feature_stds is None:
        print(f"   ⚠️  File không chứa feature_means hoặc feature_stds")
        return None, None
    
    print(f"   ✅ Đã load normalization stats: {feature_means.shape[0]} features")
    return feature_means, feature_stds


def normalize_features(X: np.ndarray, feature_means: np.ndarray, feature_stds: np.ndarray) -> np.ndarray:
    """
    Normalize features giống như trong notebook CEE.ipynb:
    X = (X - feature_means) / feature_stds
    """
    # Đảm bảo feature_means và feature_stds có cùng số features với X
    if feature_means.shape[0] > X.shape[1]:
        # Cắt bỏ features thừa
        feature_means = feature_means[:X.shape[1]]
        feature_stds = feature_stds[:X.shape[1]]
    elif feature_means.shape[0] < X.shape[1]:
        # Cắt bỏ features thừa từ X
        X = X[:, :feature_means.shape[0]]
    
    # Normalize: (X - mean) / std
    X_normalized = (X - feature_means) / feature_stds
    
    # Xử lý NaN/Inf giống như trong notebook
    X_normalized = np.nan_to_num(X_normalized, nan=0.0, posinf=0.0, neginf=0.0)
    
    return X_normalized


def query_oracle(
    oracle_client: LocalOracleClient,
    X: np.ndarray,
    model_type: str,
    model_path: Path,
    normalization_stats_path: Optional[str] = None,
    batch_size: int = 1024,
) -> np.ndarray:
    """
    Truy vấn oracle để lấy predictions.
    
    QUAN TRỌNG: Oracle client đã tự động xử lý normalization và feature alignment.
    Attacker chỉ cần gửi raw features, oracle sẽ tự động:
    - Normalize features (nếu có normalization stats)
    - Align feature dimensions
    - Trả về binary predictions (0 hoặc 1)
    """
    print(f"\n🚀 Đang truy vấn oracle module...")
    print(f"   ℹ️  Oracle client sẽ tự động xử lý normalization và feature alignment")

    # Query oracle theo batch - oracle client tự động xử lý mọi thứ
    num_samples = X.shape[0]
    predictions = np.zeros(num_samples, dtype=np.int32)

    for start in range(0, num_samples, batch_size):
        end = min(start + batch_size, num_samples)
        batch = X[start:end]
        predictions[start:end] = oracle_client.predict(batch)
        if (start // batch_size) % 50 == 0 or end == num_samples:
            print(f"   … processed {end:,}/{num_samples:,} ({end/num_samples*100:.1f}%)")

    return predictions


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Tính các metrics đánh giá."""
    accuracy = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    # Tính AUC nếu có thể (cần probabilities, nhưng oracle chỉ trả về binary)
    # Với binary predictions, không thể tính AUC chính xác
    try:
        # Thử lấy probabilities nếu oracle hỗ trợ
        auc = None
    except:
        auc = None

    # Chuyển đổi class distribution sang Python native types
    true_dist = np.unique(y_true, return_counts=True)
    pred_dist = np.unique(y_pred, return_counts=True)
    class_dist_true = {int(k): int(v) for k, v in zip(true_dist[0], true_dist[1])}
    class_dist_pred = {int(k): int(v) for k, v in zip(pred_dist[0], pred_dist[1])}

    metrics = {
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
        "total_samples": int(len(y_true)),
        "class_distribution_true": class_dist_true,
        "class_distribution_pred": class_dist_pred,
    }

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Test module truy vấn target model và so sánh với ground truth"
    )
    parser.add_argument(
        "--parquet-path",
        type=str,
        required=True,
        help="Đường dẫn tới parquet file có features và labels",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Đường dẫn tới target model (h5 hoặc lgb). Nếu không cung cấp, sẽ dùng --model-name.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Tên model (CEE, LEE, CSE, LSE). Nếu cung cấp, sẽ tự động tìm model và detect type.",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["h5", "lgb"],
        default=None,
        help="Loại model: 'h5' (Keras) hoặc 'lgb' (LightGBM). Chỉ cần nếu dùng --model-path.",
    )
    parser.add_argument(
        "--label-col",
        type=str,
        default="Label",
        help="Tên cột label trong parquet (mặc định: 'Label')",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Giới hạn số samples để test (mặc định: tất cả)",
    )
    parser.add_argument(
        "--normalization-stats-path",
        type=str,
        default=None,
        help="Đường dẫn tới file normalization_stats.npz (cho Keras/LightGBM). Nếu None, sẽ tự động tìm trong cùng thư mục với model.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold binary cho oracle (mặc định: 0.5)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Thư mục lưu kết quả (mặc định: output/test_oracle/)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="Batch size cho query oracle (mặc định: 1024)",
    )

    args = parser.parse_args()

    # Validate arguments
    if args.model_name is None and args.model_path is None:
        raise ValueError("❌ Phải cung cấp --model-name hoặc --model-path")
    
    if args.model_name is not None and args.model_path is not None:
        raise ValueError("❌ Chỉ cung cấp --model-name HOẶC --model-path, không phải cả hai")
    
    if args.model_path is not None and args.model_type is None:
        raise ValueError("❌ Khi dùng --model-path, phải cung cấp --model-type")

    # Resolve paths
    parquet_path = Path(args.parquet_path).expanduser().resolve()
    if not parquet_path.exists():
        raise FileNotFoundError(f"❌ Không tìm thấy parquet file: {parquet_path}")

    # Xử lý model path/name
    use_model_name = args.model_name is not None
    if use_model_name:
        model_name = args.model_name.upper().strip()
        model_path = None  # Sẽ được tạo từ tên
        model_type = None  # Sẽ được auto-detect
    else:
    model_path = Path(args.model_path).expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"❌ Không tìm thấy model file: {model_path}")
        model_type = args.model_type
        model_name = None

    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser()
    else:
        output_dir = PROJECT_ROOT / "output" / "test_oracle"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("TEST MODULE TRUY VẤN TARGET MODEL")
    print("=" * 80)
    print(f"\n📋 Cấu hình:")
    print(f"   Parquet file: {parquet_path}")
    if use_model_name:
        print(f"   Model name: {model_name} (tự động detect type và path)")
    else:
    print(f"   Model file: {model_path}")
        print(f"   Model type: {model_type.upper()}")
    print(f"   Label column: {args.label_col}")
    if args.max_samples:
        print(f"   Max samples: {args.max_samples:,}")
    print(f"   Output directory: {output_dir}")

    # Load feature columns
    print(f"\n📂 Đang đọc feature columns...")
    feature_cols = get_feature_columns(str(parquet_path), args.label_col)
    print(f"✅ Tìm thấy {len(feature_cols)} feature columns")

    # Load data với ground truth labels
    X, y_true = load_data_with_labels(
        str(parquet_path),
        feature_cols,
        args.label_col,
        max_samples=args.max_samples,
    )

    if len(y_true) == 0:
        raise ValueError("❌ Không có samples hợp lệ (tất cả đều có label -1)")

    # Khởi tạo oracle client
    print(f"\n🔄 Đang khởi tạo oracle client...")
    
    if use_model_name:
        # Sử dụng tên model - tự động detect mọi thứ
        oracle_client = create_oracle_from_name(
            model_name=model_name,
            threshold=args.threshold,
            feature_dim=X.shape[1],
        )
        # Lấy model_path và model_type từ oracle client để hiển thị
        model_path = Path(oracle_client.model_path)
        model_type = oracle_client.model_type
    else:
        # Sử dụng đường dẫn thủ công
        normalization_stats_path = args.normalization_stats_path
        if normalization_stats_path is None:
            # Tự động tìm file normalization_stats.npz trong cùng thư mục với model
            model_dir = model_path.parent
            model_name_from_path = model_path.stem
            possible_paths = [
                model_dir / f"{model_name_from_path}.npz",
                model_dir / f"{model_name_from_path}_normalization_stats.npz",
                model_dir / "normalization_stats.npz",
            ]
            for path in possible_paths:
                if path.exists():
                    normalization_stats_path = str(path)
                    print(f"📂 Tự động tìm thấy normalization stats: {path.name}")
                    break
            
            # Chỉ raise error nếu là LightGBM (bắt buộc)
            if normalization_stats_path is None and model_type == "lgb":
                raise FileNotFoundError(
                    f"❌ Không tìm thấy file normalization stats. "
                    f"LightGBM model cần normalization stats. "
                    f"Vui lòng cung cấp --normalization-stats-path hoặc đặt file trong: {model_dir}"
                )
        
    oracle_client = LocalOracleClient(
            model_type=model_type,
        model_path=str(model_path),
            normalization_stats_path=normalization_stats_path,
        threshold=args.threshold,
        feature_dim=X.shape[1],
    )
    
    print(f"✅ Oracle client đã sẵn sàng")
    print(f"   Model type: {model_type.upper()}")
    print(f"   Model path: {model_path}")

    # Query oracle để lấy predictions
    y_pred = query_oracle(
        oracle_client, 
        X, 
        model_type, 
        model_path,
        normalization_stats_path=None,  # Đã được xử lý trong oracle client
        batch_size=args.batch_size
    )

    # Tính metrics
    print(f"\n📊 Đang tính metrics...")
    metrics = calculate_metrics(y_true, y_pred)

    # In kết quả
    print("\n" + "=" * 80)
    print("KẾT QUẢ ĐÁNH GIÁ")
    print("=" * 80)
    print(f"\n📈 Metrics:")
    print(f"   Accuracy:           {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"   Balanced Accuracy:  {metrics['balanced_accuracy']:.4f} ({metrics['balanced_accuracy']*100:.2f}%)")
    print(f"   Precision:          {metrics['precision']:.4f}")
    print(f"   Recall:             {metrics['recall']:.4f}")
    print(f"   F1 Score:           {metrics['f1_score']:.4f}")

    print(f"\n📊 Confusion Matrix:")
    cm = metrics["confusion_matrix"]
    print(f"   True Negative (TN):  {cm['true_negative']:,}")
    print(f"   False Positive (FP): {cm['false_positive']:,}")
    print(f"   False Negative (FN): {cm['false_negative']:,}")
    print(f"   True Positive (TP):  {cm['true_positive']:,}")

    print(f"\n📊 Class Distribution:")
    print(f"   Ground Truth: {metrics['class_distribution_true']}")
    print(f"   Predictions:  {metrics['class_distribution_pred']}")

    # Classification report
    print(f"\n📋 Classification Report:")
    print(classification_report(y_true, y_pred, target_names=["Benign", "Malware"]))

    # Lưu kết quả
    output_json = output_dir / "test_results.json"
    output_txt = output_dir / "test_results.txt"

    # Lưu JSON
    results = {
        "config": {
            "parquet_path": str(parquet_path),
            "model_path": str(model_path),
            "model_name": model_name if use_model_name else None,
            "model_type": model_type,
            "label_col": args.label_col,
            "max_samples": args.max_samples,
            "threshold": args.threshold,
        },
        "metrics": metrics,
    }

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Lưu text report
    with open(output_txt, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("TEST MODULE TRUY VẤN TARGET MODEL - KẾT QUẢ\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"📋 Cấu hình:\n")
        f.write(f"   Parquet file: {parquet_path}\n")
        if use_model_name:
            f.write(f"   Model name: {model_name} (tự động detect type và path)\n")
        else:
        f.write(f"   Model file: {model_path}\n")
        f.write(f"   Model type: {model_type.upper()}\n")
        f.write(f"   Label column: {args.label_col}\n")
        if args.max_samples:
            f.write(f"   Max samples: {args.max_samples:,}\n")
        f.write(f"   Threshold: {args.threshold}\n")
        f.write(f"\n📈 Metrics:\n")
        f.write(f"   Accuracy:           {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)\n")
        f.write(f"   Balanced Accuracy:  {metrics['balanced_accuracy']:.4f} ({metrics['balanced_accuracy']*100:.2f}%)\n")
        f.write(f"   Precision:          {metrics['precision']:.4f}\n")
        f.write(f"   Recall:             {metrics['recall']:.4f}\n")
        f.write(f"   F1 Score:           {metrics['f1_score']:.4f}\n")
        f.write(f"\n📊 Confusion Matrix:\n")
        f.write(f"   True Negative (TN):  {cm['true_negative']:,}\n")
        f.write(f"   False Positive (FP): {cm['false_positive']:,}\n")
        f.write(f"   False Negative (FN): {cm['false_negative']:,}\n")
        f.write(f"   True Positive (TP):  {cm['true_positive']:,}\n")
        f.write(f"\n📊 Class Distribution:\n")
        f.write(f"   Ground Truth: {metrics['class_distribution_true']}\n")
        f.write(f"   Predictions:  {metrics['class_distribution_pred']}\n")
        f.write(f"\n📋 Classification Report:\n")
        f.write(classification_report(y_true, y_pred, target_names=["Benign", "Malware"]))

    print(f"\n✅ Đã lưu kết quả:")
    print(f"   JSON: {output_json}")
    print(f"   Text: {output_txt}")


if __name__ == "__main__":
    main()

