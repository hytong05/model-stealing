"""
Script để đánh giá độ tương đồng giữa target model và surrogate model
Sử dụng dữ liệu từ test_ember_2018_v2_features_label_minus1.parquet (unlabeled data)
Khi không có ground truth labels, chỉ tính Agreement (không tính Accuracy)
"""
# QUAN TRỌNG: Set environment variable TRƯỚC KHI import TensorFlow để dùng legacy Keras
import os
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # 3 = FATAL only (ẩn ERROR, WARNING, INFO)

import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import joblib
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
from sklearn.preprocessing import RobustScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.targets.oracle_client import create_oracle_from_name, LocalOracleClient
from src.models.dnn import create_dnn2
import tensorflow as tf


def get_feature_columns(parquet_path: str, label_col: str = "Label") -> list:
    """Lấy danh sách feature columns từ parquet file."""
    pq_file = pq.ParquetFile(parquet_path)
    return [name for name in pq_file.schema.names if name != label_col]


def load_test_data(parquet_path: str, feature_cols: list, max_samples: int = 10000, load_labels: bool = False):
    """
    Load dữ liệu test từ parquet file.
    
    Args:
        parquet_path: Đường dẫn đến file parquet
        feature_cols: Danh sách feature columns
        max_samples: Số lượng samples tối đa
        load_labels: Nếu True, sẽ load cả labels và chỉ lấy samples có label 0 hoặc 1 (ground truth)
                    Nếu False, load tất cả samples (bỏ qua labels)
    
    Returns:
        X: Feature array (n_samples, n_features)
        y_true: Ground truth labels (n_samples,) - chỉ trả về nếu load_labels=True
    """
    pq_file = pq.ParquetFile(parquet_path)
    all_X = []
    all_y = [] if load_labels else None
    rows_loaded = 0
    
    print(f"🔄 Đang load dữ liệu từ {parquet_path}...")
    
    for batch in pq_file.iter_batches(batch_size=5000, columns=feature_cols + ["Label"]):
        if rows_loaded >= max_samples:
            break
            
        batch_df = batch.to_pandas()
        
        if load_labels:
            # Load cả labels và chỉ lấy samples có label 0 hoặc 1 (ground truth)
            y = batch_df["Label"].values.astype(np.int32)
            valid_mask = (y >= 0) & (y <= 1)  # Chỉ lấy label 0 hoặc 1
            
            if valid_mask.sum() == 0:
                continue
            
            batch_df = batch_df[valid_mask]
            y = y[valid_mask]
            
            X = batch_df[feature_cols].values.astype(np.float32)
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            
            all_X.append(X)
            all_y.append(y)
            rows_loaded += len(X)
        else:
            # Lấy tất cả samples (không quan tâm nhãn vì sẽ query target model)
            X = batch_df[feature_cols].values.astype(np.float32)
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            
            all_X.append(X)
            rows_loaded += len(X)
        
        if rows_loaded % 5000 == 0:
            print(f"  Đã load {rows_loaded:,}/{max_samples:,} samples...")
        
        if rows_loaded >= max_samples:
            break
    
    if all_X:
        X_concat = np.concatenate(all_X, axis=0)
        if len(X_concat) > max_samples:
            X_concat = X_concat[:max_samples]
        
        if load_labels:
            y_concat = np.concatenate(all_y, axis=0) if all_y else None
            if y_concat is not None and len(y_concat) > max_samples:
                y_concat = y_concat[:max_samples]
            print(f"✅ Đã load {len(X_concat):,} samples với ground truth labels")
            return X_concat, y_concat
        else:
            print(f"✅ Đã load {len(X_concat):,} samples")
            return X_concat
    else:
        if load_labels:
            return np.empty((0, len(feature_cols)), dtype=np.float32), np.empty((0,), dtype=np.int32)
        else:
            return np.empty((0, len(feature_cols)), dtype=np.float32)


def load_dualDNN_surrogate_model(model_path: str, scaler_path: str, feature_dim: int = 2381, threshold: float = 0.5):
    """
    Load dualDNN surrogate model và scaler.
    dualDNN cần 2 inputs: (X_scaled, y_true) khi predict.
    """
    # Environment variables đã được set ở đầu file
    
    # Load scaler
    if scaler_path and os.path.exists(scaler_path):
        print(f"🔄 Đang load scaler từ {scaler_path}...")
        scaler = joblib.load(scaler_path)
        print(f"✅ Đã load scaler")
    else:
        print(f"⚠️  Không tìm thấy scaler tại {scaler_path}, tạo RobustScaler mới...")
        scaler = RobustScaler()
        # Note: scaler sẽ cần được fit trước khi sử dụng
    
    # Load model - rebuild architecture và load weights
    # Lưu ý: Model dualDNN được train với Keras 2, nên cần rebuild architecture
    # thay vì load trực tiếp để tránh compatibility issues
    print(f"🔄 Đang load dualDNN model từ {model_path}...")
    print(f"    (Đang rebuild architecture và load weights do Keras version compatibility)")
    model = None
    try:
        # Rebuild model architecture
        model = create_dnn2(seed=42, mc=False, input_shape=(feature_dim,))
        # Load weights từ file .h5
        model.load_weights(model_path)
        print(f"✅ Đã rebuild architecture và load weights thành công")
    except Exception as e:
        print(f"❌ Không thể load model: {e}")
        print(f"    Đang thử cách khác...")
        # Fallback: thử load trực tiếp (nếu môi trường hỗ trợ)
        try:
            model = tf.keras.models.load_model(model_path, compile=False)
            print(f"✅ Đã load model trực tiếp (fallback)")
        except Exception as e2:
            print(f"❌ Không thể load model với bất kỳ cách nào:")
            print(f"   Lỗi rebuild: {e}")
            print(f"   Lỗi load trực tiếp: {e2}")
            raise e2
    
    print(f"✅ Đã load dualDNN model")
    
    def predict(X, y_true):
        """
        Predict với dualDNN model.
        
        Args:
            X: Raw feature array (shape: [n_samples, n_features])
            y_true: Ground truth labels hoặc target predictions (shape: [n_samples,])
        
        Returns:
            (predictions, probabilities): predictions là binary classes, probabilities là raw outputs
        """
        # Scale data với RobustScaler và clip về [-5, 5]
        X_scaled = scaler.transform(X)
        X_scaled = np.clip(X_scaled, -5, 5)
        
        # dualDNN cần y_true làm input thứ 2
        y_true_reshaped = y_true.reshape(-1, 1)  # Reshape thành [n_samples, 1]
        
        # Predict với 2 inputs: (X_scaled, y_true)
        probs = np.squeeze(model.predict((X_scaled, y_true_reshaped), verbose=0), axis=-1)
        
        # Nếu model output là 2D (softmax), lấy class 1
        if probs.ndim > 1 and probs.shape[-1] == 2:
            probs = probs[:, 1]
        
        predictions = (probs >= threshold).astype(int)
        return predictions, probs
    
    return predict, scaler


def evaluate_model_similarity(
    target_model,
    surrogate_predict,
    X_test,
    y_target,
    y_true=None,
    model_name: str = "",
    is_dualDNN: bool = False
):
    """
    Đánh giá độ tương đồng giữa target và surrogate model.
    
    Args:
        target_model: Target model oracle
        surrogate_predict: Surrogate model predict function
        X_test: Test features
        y_target: Predictions từ target model
        y_true: Ground truth labels (nếu có) - để tính Accuracy thật sự
        model_name: Tên model
        is_dualDNN: Có phải dualDNN model không
    
    Returns:
        metrics dict với accuracy (nếu có y_true) và agreement
    """
    print(f"\n🔄 Đang đánh giá {model_name}...")
    
    # Predict với surrogate
    # dualDNN cần y_target làm input thứ 2
    if is_dualDNN:
        y_surrogate, probs_surrogate = surrogate_predict(X_test, y_target)
    else:
        y_surrogate, probs_surrogate = surrogate_predict(X_test)
    
    # Tính metrics
    # Agreement = tỉ lệ nhất quán giữa target và surrogate predictions
    agreement = accuracy_score(y_target, y_surrogate)
    
    # Accuracy = tỉ lệ chính xác của surrogate so với ground truth (nếu có)
    if y_true is not None:
        accuracy = accuracy_score(y_true, y_surrogate)
    else:
        accuracy = None
    
    # Tính precision, recall, f1 cho agreement (target vs surrogate)
    precision_agreement, recall_agreement, f1_agreement, _ = precision_recall_fscore_support(
        y_target, y_surrogate, average="binary", zero_division=0
    )
    
    # Tính AUC dựa trên target predictions
    try:
        auc = roc_auc_score(y_target, probs_surrogate)
    except ValueError:
        auc = float("nan")
    
    # Confusion matrix (so sánh target predictions vs surrogate predictions)
    cm_agreement = confusion_matrix(y_target, y_surrogate)
    tn_agreement, fp_agreement, fn_agreement, tp_agreement = cm_agreement.ravel() if cm_agreement.size == 4 else (0, 0, 0, 0)
    
    # Tính metrics với ground truth nếu có
    if y_true is not None:
        accuracy = accuracy_score(y_true, y_surrogate)
        precision_accuracy, recall_accuracy, f1_accuracy, _ = precision_recall_fscore_support(
            y_true, y_surrogate, average="binary", zero_division=0
        )
        try:
            auc_accuracy = roc_auc_score(y_true, probs_surrogate)
        except ValueError:
            auc_accuracy = float("nan")
        cm_accuracy = confusion_matrix(y_true, y_surrogate)
        tn_accuracy, fp_accuracy, fn_accuracy, tp_accuracy = cm_accuracy.ravel() if cm_accuracy.size == 4 else (0, 0, 0, 0)
        true_dist = dict(zip(*np.unique(y_true, return_counts=True)))
    else:
        accuracy = None
        precision_accuracy = None
        recall_accuracy = None
        f1_accuracy = None
        auc_accuracy = None
        tn_accuracy = fp_accuracy = fn_accuracy = tp_accuracy = None
        true_dist = None
    
    # Phân bố predictions
    target_dist = dict(zip(*np.unique(y_target, return_counts=True)))
    surrogate_dist = dict(zip(*np.unique(y_surrogate, return_counts=True)))
    
    metrics = {
        "model_name": model_name,
        "accuracy": float(accuracy) if accuracy is not None else None,
        "agreement": float(agreement),
        "auc": float(auc) if not np.isnan(auc) else None,
        "auc_accuracy": float(auc_accuracy) if (auc_accuracy is not None and not np.isnan(auc_accuracy)) else None,
        "precision_agreement": float(precision_agreement),
        "recall_agreement": float(recall_agreement),
        "f1_agreement": float(f1_agreement),
        "precision_accuracy": float(precision_accuracy) if precision_accuracy is not None else None,
        "recall_accuracy": float(recall_accuracy) if recall_accuracy is not None else None,
        "f1_accuracy": float(f1_accuracy) if f1_accuracy is not None else None,
        "confusion_matrix_agreement": {
            "tn": int(tn_agreement),
            "fp": int(fp_agreement),
            "fn": int(fn_agreement),
            "tp": int(tp_agreement)
        },
        "confusion_matrix_accuracy": {
            "tn": int(tn_accuracy),
            "fp": int(fp_accuracy),
            "fn": int(fn_accuracy),
            "tp": int(tp_accuracy)
        } if tn_accuracy is not None else None,
        "target_distribution": {int(k): int(v) for k, v in target_dist.items()},
        "surrogate_distribution": {int(k): int(v) for k, v in surrogate_dist.items()},
        "ground_truth_distribution": {int(k): int(v) for k, v in true_dist.items()} if true_dist is not None else None,
    }
    
    # In kết quả
    print(f"  Agreement: {agreement:.4f} (tỉ lệ nhất quán: surrogate predictions khớp với target predictions)")
    if accuracy is not None:
        print(f"  Accuracy: {accuracy:.4f} (tỉ lệ chính xác: surrogate predictions khớp với ground truth)")
    else:
        print(f"  ⚠️  Accuracy: Không có ground truth labels để tính")
    print(f"  AUC (vs target): {auc:.4f}" if not np.isnan(auc) else "  AUC (vs target): NaN")
    if auc_accuracy is not None and not np.isnan(auc_accuracy):
        print(f"  AUC (vs ground truth): {auc_accuracy:.4f}")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Đánh giá độ tương đồng giữa target và surrogate model (dualDNN)")
    parser.add_argument("--surrogate_model_path", type=str, required=True,
                       help="Đường dẫn đến file surrogate model (dualDNN .h5)")
    parser.add_argument("--threshold", type=float, default=0.5,
                       help="Threshold để chuyển probabilities thành binary labels (mặc định: 0.5)")
    parser.add_argument("--test_parquet", type=str, default=None,
                       help="Đường dẫn đến test data parquet file (mặc định: ember_2018_v2 train data)")
    parser.add_argument("--target_model_path", type=str, default=None,
                       help="Đường dẫn đến target model (mặc định: artifacts/targets/LEE.lgb)")
    parser.add_argument("--target_model_name", type=str, default="LEE",
                       help="Tên target model (mặc định: LEE)")
    parser.add_argument("--scaler_path", type=str, default=None,
                       help="Đường dẫn đến scaler .joblib file (tự động tìm nếu không chỉ định)")
    
    args = parser.parse_args()
    
    # Xử lý đường dẫn surrogate model
    surrogate_model_path = args.surrogate_model_path
    if not Path(surrogate_model_path).is_absolute():
        surrogate_model_path = str((PROJECT_ROOT / surrogate_model_path).resolve())
    
    if not Path(surrogate_model_path).exists():
        raise FileNotFoundError(f"❌ Không tìm thấy surrogate model tại: {surrogate_model_path}")
    
    # Xử lý threshold
    threshold = args.threshold
    if not (0.0 <= threshold <= 1.0):
        raise ValueError(f"❌ Threshold phải nằm trong khoảng [0.0, 1.0], nhận được: {threshold}")
    
    # Xử lý test data path
    if args.test_parquet:
        test_parquet = args.test_parquet
        if not Path(test_parquet).is_absolute():
            test_parquet = str((PROJECT_ROOT / test_parquet).resolve())
    else:
        test_parquet = str(PROJECT_ROOT / "data" / "ember_2018_v2" / "train" / "train_ember_2018_v2_features_label_minus1.parquet")
    
    # Xử lý target model path
    if args.target_model_path:
        target_model_path = args.target_model_path
        if not Path(target_model_path).is_absolute():
            target_model_path = str((PROJECT_ROOT / target_model_path).resolve())
    else:
        target_model_path = str(PROJECT_ROOT / "artifacts" / "targets" / f"{args.target_model_name}.lgb")
    
    target_model_name = args.target_model_name
    
    # Tìm scaler - ưu tiên trong output directory tương ứng với model name
    if args.scaler_path:
        scaler_path = args.scaler_path
        if not Path(scaler_path).is_absolute():
            scaler_path = str((PROJECT_ROOT / scaler_path).resolve())
        if not Path(scaler_path).exists():
            raise FileNotFoundError(f"❌ Không tìm thấy scaler tại: {scaler_path}")
        scaler_path = Path(scaler_path)
    else:
        surrogate_name = Path(surrogate_model_path).stem  # LEE-ember-dualDNN-2000
        possible_scaler_paths = [
            PROJECT_ROOT / "output" / surrogate_name / "robust_scaler.joblib",  # Ưu tiên nhất
            Path(surrogate_model_path).parent / "robust_scaler.joblib",  # Cùng thư mục với model
            PROJECT_ROOT / "storage" / "dualDNN" / "robust_scaler.joblib",  # Thư mục storage
        ]
        
        scaler_path = None
        for path in possible_scaler_paths:
            if path.exists():
                scaler_path = path
                break
        
        # Nếu không tìm thấy, sẽ tạo scaler mới
        if scaler_path is None:
            scaler_path = PROJECT_ROOT / "output" / surrogate_name / "robust_scaler.joblib"  # Đường dẫn mặc định để hiển thị
    
    output_dir = PROJECT_ROOT / "logs" / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("ĐÁNH GIÁ ĐỘ TƯƠNG ĐỒNG GIỮA TARGET VÀ SURROGATE MODELS")
    print("=" * 80)
    print(f"Target Model: {target_model_path}")
    print(f"Target Model Name: {target_model_name}")
    print(f"Surrogate Model: {surrogate_model_path}")
    print(f"Test Data: {test_parquet}")
    print(f"Threshold: {threshold}")
    print(f"Scaler: {scaler_path}")
    
    # Load feature columns
    feature_cols = get_feature_columns(test_parquet)
    print(f"\n✅ Feature columns: {len(feature_cols)}")
    
    # Load test data (unlabeled - label -1)
    # Lưu ý: Dataset này không có ground truth labels, chỉ tính Agreement (không tính Accuracy)
    print(f"\n🔄 Đang load test data từ {test_parquet}...")
    print(f"    (Dataset với label -1, không có ground truth labels)")
    X_test = load_test_data(test_parquet, feature_cols, max_samples=10000, load_labels=False)
    y_true = None  # Không có ground truth labels trong dataset này
    
    if len(X_test) == 0:
        print("❌ Không có dữ liệu để test!")
        return
    
    print(f"✅ Đã load {len(X_test):,} samples (unlabeled data)")
    
    # Load target model (LightGBM)
    print(f"\n🔄 Đang load target model (LightGBM)...")
    try:
        # Dùng create_oracle_from_name với models_dir để tự động detect
        # Hàm này sẽ tự tìm model và normalization stats trong thư mục
        target_oracle = create_oracle_from_name(
            model_name=target_model_name,
            models_dir=str(PROJECT_ROOT / "artifacts" / "targets"),
            feature_dim=len(feature_cols)
        )
        print(f"✅ Đã load target model")
    except Exception as e:
        print(f"❌ Lỗi khi load target model: {e}")
        print(f"    Đang thử load trực tiếp với LocalOracleClient...")
        import traceback
        traceback.print_exc()
        
        # Fallback: thử load trực tiếp
        try:
            from src.targets.oracle_client import LocalOracleClient
            # Tìm normalization stats
            target_norm_stats_path = PROJECT_ROOT / "artifacts" / "targets" / f"{target_model_name}_normalization_stats.npz"
            if not target_norm_stats_path.exists():
                raise FileNotFoundError(f"Không tìm thấy normalization stats tại {target_norm_stats_path}")
            
            target_oracle = LocalOracleClient(
                model_type="lgb",
                model_path=target_model_path,
                normalization_stats_path=str(target_norm_stats_path),
                threshold=threshold,
                feature_dim=len(feature_cols)
            )
            print(f"✅ Đã load target model bằng LocalOracleClient trực tiếp")
        except Exception as e2:
            print(f"❌ Lỗi khi load trực tiếp: {e2}")
            traceback.print_exc()
            return
    
    # Query target model để lấy nhãn thực tế
    print(f"\n🔄 Đang query target model để lấy nhãn...")
    try:
        # FlexibleLGBTarget.predict_proba() trả về probability của class 1 (malware) - 1D array
        # Data sẽ được normalize tự động bởi FlexibleLGBTarget trước khi predict
        # (giống như trong notebook: X = (X - feature_means) / feature_stds)
        y_target_proba = target_oracle.predict_proba(X_test)
        
        # LightGBM predict trả về probability của class 1 (malware) - shape: (n_samples,)
        # Chuyển thành binary labels với threshold
        if y_target_proba.ndim == 1:
            # 1D array: probabilities của class 1
            y_target = (y_target_proba >= threshold).astype(int)
        elif y_target_proba.ndim == 2 and y_target_proba.shape[1] == 2:
            # 2D array với 2 columns: [prob_class_0, prob_class_1]
            y_target = (y_target_proba[:, 1] >= threshold).astype(int)
        else:
            # Fallback: xử lý như 1D
            y_target_proba_flat = np.squeeze(y_target_proba)
            y_target = (y_target_proba_flat >= threshold).astype(int)
        
        print(f"✅ Đã lấy nhãn từ target model")
        unique, counts = np.unique(y_target, return_counts=True)
        print(f"  Phân bố nhãn: {dict(zip(unique, counts))}")
    except Exception as e:
        print(f"❌ Lỗi khi query target model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Load surrogate model (dualDNN)
    print(f"\n🔄 Đang load surrogate model (dualDNN)...")
    if not Path(surrogate_model_path).exists():
        print(f"❌ Không tìm thấy surrogate model tại {surrogate_model_path}")
        return
    
    # Kiểm tra scaler path
    scaler_exists = scaler_path is not None and scaler_path.exists()
    if not scaler_exists:
        print(f"⚠️  Không tìm thấy scaler, sẽ tạo và fit scaler mới với dữ liệu test")
        print(f"    (Đã tìm trong: {[str(p) for p in possible_scaler_paths[:2]]})")
    
    try:
        surrogate_predict, scaler = load_dualDNN_surrogate_model(
            model_path=surrogate_model_path,
            scaler_path=str(scaler_path) if scaler_exists else None,
            feature_dim=len(feature_cols),
            threshold=threshold
        )
        
        # Nếu scaler chưa được fit (không tìm thấy file), cần fit với dữ liệu test
        if not scaler_exists:
            print(f"🔄 Đang fit scaler với dữ liệu test...")
            scaler.fit(X_test)
            print(f"✅ Đã fit scaler với {len(X_test):,} samples")
        
        print(f"✅ Đã load surrogate model")
    except Exception as e:
        print(f"❌ Lỗi khi load surrogate model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Đánh giá surrogate model
    all_results = []
    
    try:
        metrics = evaluate_model_similarity(
            target_oracle,
            surrogate_predict,
            X_test,
            y_target,
            y_true=y_true,  # Truyền ground truth labels để tính Accuracy
            model_name=Path(surrogate_model_path).parent.name,
            is_dualDNN=True
        )
        all_results.append(metrics)
    except Exception as e:
        print(f"\n❌ Lỗi khi đánh giá surrogate model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Tạo report
    print(f"\n{'='*80}")
    print("📊 TẠO BÁO CÁO")
    print(f"{'='*80}\n")
    
    report_path = output_dir / "surrogate_similarity_report.txt"
    report_md_path = output_dir / "surrogate_similarity_report.md"
    json_path = output_dir / "surrogate_similarity_results.json"
    
    # Text report
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("BÁO CÁO ĐÁNH GIÁ ĐỘ TƯƠNG ĐỒNG GIỮA TARGET VÀ SURROGATE MODELS\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("THÔNG TIN MODELS:\n")
        f.write("-" * 80 + "\n")
        f.write(f"Target Model: {target_model_path}\n")
        f.write(f"Surrogate Model: {surrogate_model_path}\n")
        if scaler_exists and scaler_path is not None:
            f.write(f"Scaler: {scaler_path}\n")
        else:
            f.write(f"Scaler: Không tìm thấy (đã tạo và fit mới với dữ liệu test)\n")
        f.write("\n")
        
        f.write("THÔNG TIN DỮ LIỆU TEST:\n")
        f.write("-" * 80 + "\n")
        f.write(f"File: {test_parquet}\n")
        f.write(f"Số samples: {len(X_test):,}\n")
        if y_true is not None:
            unique_true, counts_true = np.unique(y_true, return_counts=True)
            f.write(f"Phân bố ground truth labels: {dict(zip(unique_true, counts_true))}\n")
        unique_target, counts_target = np.unique(y_target, return_counts=True)
        f.write(f"Phân bố nhãn từ target model: {dict(zip(unique_target, counts_target))}\n\n")
        
        f.write("KẾT QUẢ ĐÁNH GIÁ:\n")
        f.write("-" * 80 + "\n\n")
        
        for result in all_results:
            f.write(f"{result['model_name'].upper().replace('_', ' ')}:\n")
            
            # Accuracy: so sánh với ground truth (nếu có)
            if result['accuracy'] is not None:
                f.write(f"  Accuracy: {result['accuracy']:.4f} ({result['accuracy']*100:.2f}%)\n")
                f.write(f"    → Độ chính xác của surrogate model so với ground truth labels\n")
            else:
                f.write(f"  Accuracy: Không có ground truth labels để tính\n")
            
            # Agreement: so sánh với target model predictions
            f.write(f"  Agreement: {result['agreement']:.4f} ({result['agreement']*100:.2f}%)\n")
            f.write(f"    → Độ nhất quán giữa surrogate và target model predictions\n")
            
            # AUC
            if result['auc'] is not None:
                f.write(f"  AUC (vs target): {result['auc']:.4f}\n")
            if result.get('auc_accuracy') is not None:
                f.write(f"  AUC (vs ground truth): {result['auc_accuracy']:.4f}\n")
            
            # Precision, Recall, F1 cho Agreement
            f.write(f"  Precision (agreement): {result.get('precision_agreement', result.get('precision', 0)):.4f}\n")
            f.write(f"  Recall (agreement): {result.get('recall_agreement', result.get('recall', 0)):.4f}\n")
            f.write(f"  F1-score (agreement): {result.get('f1_agreement', result.get('f1_score', 0)):.4f}\n")
            
            # Precision, Recall, F1 cho Accuracy (nếu có)
            if result.get('precision_accuracy') is not None:
                f.write(f"  Precision (accuracy): {result['precision_accuracy']:.4f}\n")
                f.write(f"  Recall (accuracy): {result['recall_accuracy']:.4f}\n")
                f.write(f"  F1-score (accuracy): {result['f1_accuracy']:.4f}\n")
            
            # Confusion Matrix cho Agreement
            cm_agreement = result.get('confusion_matrix_agreement', result.get('confusion_matrix'))
            if cm_agreement:
                f.write(f"  Confusion Matrix (Agreement - Target vs Surrogate):\n")
                f.write(f"    TN: {cm_agreement['tn']}, FP: {cm_agreement['fp']}\n")
                f.write(f"    FN: {cm_agreement['fn']}, TP: {cm_agreement['tp']}\n")
            
            # Confusion Matrix cho Accuracy (nếu có)
            if result.get('confusion_matrix_accuracy'):
                cm_accuracy = result['confusion_matrix_accuracy']
                f.write(f"  Confusion Matrix (Accuracy - Ground Truth vs Surrogate):\n")
                f.write(f"    TN: {cm_accuracy['tn']}, FP: {cm_accuracy['fp']}\n")
                f.write(f"    FN: {cm_accuracy['fn']}, TP: {cm_accuracy['tp']}\n")
            
            # Phân bố
            if result.get('ground_truth_distribution'):
                f.write(f"  Ground truth distribution: {result['ground_truth_distribution']}\n")
            f.write(f"  Target distribution: {result['target_distribution']}\n")
            f.write(f"  Surrogate distribution: {result['surrogate_distribution']}\n")
            f.write("\n")
        
        # Tóm tắt kết quả
        f.write("\n" + "=" * 80 + "\n")
        f.write("TÓM TẮT KẾT QUẢ:\n")
        f.write("=" * 80 + "\n\n")
        
        if all_results:
            result = all_results[0]
            f.write(f"Surrogate model đạt:\n")
            if result['accuracy'] is not None:
                f.write(f"  - Accuracy: {result['accuracy']:.4f} ({result['accuracy']*100:.2f}%) - so với ground truth\n")
            else:
                f.write(f"  - Accuracy: Không có ground truth labels\n")
            f.write(f"  - Agreement: {result['agreement']:.4f} ({result['agreement']*100:.2f}%) - so với target model\n")
            if result['auc'] is not None:
                f.write(f"  - AUC (vs target): {result['auc']:.4f}\n")
            if result.get('auc_accuracy') is not None:
                f.write(f"  - AUC (vs ground truth): {result['auc_accuracy']:.4f}\n")
            f.write(f"  - F1-score (agreement): {result.get('f1_agreement', result.get('f1_score', 0)):.4f}\n")
            if result.get('f1_accuracy') is not None:
                f.write(f"  - F1-score (accuracy): {result['f1_accuracy']:.4f}\n")
    
    # Markdown report
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write("# Báo Cáo Đánh Giá Độ Tương Đồng Giữa Target và Surrogate Models\n\n")
        
        f.write("## Thông Tin Models\n\n")
        f.write(f"- **Target Model**: `{target_model_path}`\n")
        f.write(f"- **Surrogate Model**: `{surrogate_model_path}`\n")
        if scaler_exists and scaler_path is not None:
            f.write(f"- **Scaler**: `{scaler_path}`\n")
        else:
            f.write(f"- **Scaler**: Không tìm thấy (đã tạo và fit mới với dữ liệu test)\n")
        f.write("\n")
        
        f.write("## Thông Tin Dữ Liệu Test\n\n")
        f.write(f"- **File**: `{test_parquet}`\n")
        f.write(f"- **Số samples**: {len(X_test):,}\n")
        if y_true is not None:
            unique_true, counts_true = np.unique(y_true, return_counts=True)
            f.write(f"- **Phân bố ground truth labels**: {dict(zip(unique_true, counts_true))}\n")
        unique_target, counts_target = np.unique(y_target, return_counts=True)
        f.write(f"- **Phân bố nhãn từ target model**: {dict(zip(unique_target, counts_target))}\n\n")
        
        f.write("## Kết Quả Đánh Giá\n\n")
        f.write("### Metric Definitions\n\n")
        f.write("- **Accuracy**: Độ chính xác của surrogate model so với ground truth labels\n")
        f.write("- **Agreement**: Độ nhất quán giữa surrogate và target model predictions\n\n")
        
        f.write("| Model | Accuracy | Agreement | AUC (vs target) | AUC (vs GT) | F1 (agreement) | F1 (accuracy) |\n")
        f.write("|-------|----------|-----------|-----------------|-------------|----------------|---------------|\n")
        
        for result in all_results:
            accuracy_str = f"{result['accuracy']:.4f}" if result['accuracy'] is not None else "N/A"
            auc_target_str = f"{result['auc']:.4f}" if result['auc'] is not None else "N/A"
            auc_gt_str = f"{result.get('auc_accuracy', 'N/A'):.4f}" if result.get('auc_accuracy') is not None else "N/A"
            f1_agreement = result.get('f1_agreement', result.get('f1_score', 0))
            f1_accuracy_str = f"{result['f1_accuracy']:.4f}" if result.get('f1_accuracy') is not None else "N/A"
            f.write(f"| {result['model_name']} | {accuracy_str} | "
                   f"{result['agreement']:.4f} | {auc_target_str} | {auc_gt_str} | "
                   f"{f1_agreement:.4f} | {f1_accuracy_str} |\n")
        
        f.write("\n## Chi Tiết Từng Model\n\n")
        
        for result in all_results:
            f.write(f"### {result['model_name'].replace('_', ' ').title()}\n\n")
            
            # Accuracy
            if result['accuracy'] is not None:
                f.write(f"- **Accuracy**: {result['accuracy']:.4f} ({result['accuracy']*100:.2f}%) - so với ground truth\n")
            else:
                f.write(f"- **Accuracy**: Không có ground truth labels để tính\n")
            
            # Agreement
            f.write(f"- **Agreement**: {result['agreement']:.4f} ({result['agreement']*100:.2f}%) - so với target model\n")
            
            # AUC
            if result['auc'] is not None:
                f.write(f"- **AUC (vs target)**: {result['auc']:.4f}\n")
            if result.get('auc_accuracy') is not None:
                f.write(f"- **AUC (vs ground truth)**: {result['auc_accuracy']:.4f}\n")
            
            # Precision, Recall, F1 cho Agreement
            f.write(f"- **Precision (agreement)**: {result.get('precision_agreement', result.get('precision', 0)):.4f}\n")
            f.write(f"- **Recall (agreement)**: {result.get('recall_agreement', result.get('recall', 0)):.4f}\n")
            f.write(f"- **F1-score (agreement)**: {result.get('f1_agreement', result.get('f1_score', 0)):.4f}\n")
            
            # Precision, Recall, F1 cho Accuracy (nếu có)
            if result.get('precision_accuracy') is not None:
                f.write(f"- **Precision (accuracy)**: {result['precision_accuracy']:.4f}\n")
                f.write(f"- **Recall (accuracy)**: {result['recall_accuracy']:.4f}\n")
                f.write(f"- **F1-score (accuracy)**: {result['f1_accuracy']:.4f}\n")
            f.write("\n")
            
            # Confusion Matrix cho Agreement
            cm_agreement = result.get('confusion_matrix_agreement', result.get('confusion_matrix'))
            if cm_agreement:
                f.write("**Confusion Matrix (Agreement - Target vs Surrogate):**\n\n")
                f.write(f"| | Predicted 0 | Predicted 1 |\n")
                f.write(f"|------|------------|-------------|\n")
                f.write(f"| Target 0 | {cm_agreement['tn']} | {cm_agreement['fp']} |\n")
                f.write(f"| Target 1 | {cm_agreement['fn']} | {cm_agreement['tp']} |\n\n")
            
            # Confusion Matrix cho Accuracy (nếu có)
            if result.get('confusion_matrix_accuracy'):
                cm_accuracy = result['confusion_matrix_accuracy']
                f.write("**Confusion Matrix (Accuracy - Ground Truth vs Surrogate):**\n\n")
                f.write(f"| | Predicted 0 | Predicted 1 |\n")
                f.write(f"|------|------------|-------------|\n")
                f.write(f"| Actual 0 | {cm_accuracy['tn']} | {cm_accuracy['fp']} |\n")
                f.write(f"| Actual 1 | {cm_accuracy['fn']} | {cm_accuracy['tp']} |\n\n")
            
            # Phân bố
            if result.get('ground_truth_distribution'):
                f.write(f"- **Ground truth distribution**: {result['ground_truth_distribution']}\n")
            f.write(f"- **Target distribution**: {result['target_distribution']}\n")
            f.write(f"- **Surrogate distribution**: {result['surrogate_distribution']}\n\n")
    
    # JSON results
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "test_info": {
                "file": test_parquet,
                "num_samples": int(len(X_test)),
                "target_distribution": {int(k): int(v) for k, v in dict(zip(*np.unique(y_target, return_counts=True))).items()}
            },
            "results": all_results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Đã tạo report:")
    print(f"   - Text report: {report_path}")
    print(f"   - Markdown report: {report_md_path}")
    print(f"   - JSON results: {json_path}")
    
    # In tóm tắt
    print(f"\n{'='*80}")
    print("TÓM TẮT KẾT QUẢ:")
    print(f"{'='*80}\n")
    
    if all_results:
        df = pd.DataFrame(all_results)
        # Chọn các cột có sẵn
        cols_to_show = ["model_name", "accuracy", "agreement", "auc"]
        if "f1_agreement" in df.columns:
            cols_to_show.append("f1_agreement")
        elif "f1_score" in df.columns:
            cols_to_show.append("f1_score")
        if "f1_accuracy" in df.columns and df["f1_accuracy"].notna().any():
            cols_to_show.append("f1_accuracy")
        print(df[cols_to_show].to_string(index=False))
    
    return all_results


if __name__ == "__main__":
    main()

