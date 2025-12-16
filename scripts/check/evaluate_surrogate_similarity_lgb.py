"""
Script để đánh giá độ tương đồng giữa target model và surrogate model (LightGBM)
Sử dụng dữ liệu từ test dataset (unlabeled data)
Khi không có ground truth labels, chỉ tính Agreement (không tính Accuracy)
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import lightgbm as lgb
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.targets.oracle_client import create_oracle_from_name, LocalOracleClient


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


def load_normalization_stats(stats_path: str):
    """Load normalization statistics từ file .npz"""
    stats = np.load(stats_path, allow_pickle=True)
    
    feature_means = stats['feature_means']
    feature_stds = stats['feature_stds']
    
    if 'feature_cols' in stats:
        feature_cols = stats['feature_cols'].tolist() if hasattr(stats['feature_cols'], 'tolist') else stats['feature_cols']
    else:
        feature_cols = None
    
    return feature_means, feature_stds, feature_cols


def normalize_features(X, feature_means, feature_stds, feature_cols=None, required_feature_dim=None):
    """
    Normalize features giống như LightGBM target model.
    
    Args:
        X: Raw features (n_samples, n_features)
        feature_means: Mean values cho normalization
        feature_stds: Std values cho normalization
        feature_cols: Danh sách feature columns (nếu có)
        required_feature_dim: Số features mà model yêu cầu
    
    Returns:
        X_normalized: Normalized features
    """
    X = np.asarray(X, dtype=np.float32)
    
    # Align features nếu cần
    if required_feature_dim is not None:
        current_dim = X.shape[1]
        if current_dim < required_feature_dim:
            # Pad với zeros
            padding = np.zeros((X.shape[0], required_feature_dim - current_dim), dtype=np.float32)
            X = np.concatenate([X, padding], axis=1)
        elif current_dim > required_feature_dim:
            # Truncate
            X = X[:, :required_feature_dim]
    
    # Normalize: (X - mean) / std
    # Tránh chia cho 0
    feature_stds_safe = np.where(feature_stds == 0, 1.0, feature_stds)
    X_normalized = (X - feature_means) / feature_stds_safe
    
    return X_normalized


def load_LightGBM_surrogate_model(model_path: str, normalization_stats_path: str = None, feature_dim: int = None, threshold: float = 0.5):
    """
    Load LightGBM surrogate model và normalization stats.
    
    Args:
        model_path: Đường dẫn đến file LightGBM model (.txt, .lgb, .pkl)
        normalization_stats_path: Đường dẫn đến file normalization stats (.npz)
        feature_dim: Số features (nếu không có sẽ auto-detect từ model)
    
    Returns:
        predict function: (X) -> (predictions, probabilities)
    """
    print(f"🔄 Đang load LightGBM surrogate model từ {model_path}...")
    
    # Load model
    try:
        model = lgb.Booster(model_file=model_path)
        model_actual_features = model.num_feature()
        print(f"✅ Đã load LightGBM model")
        print(f"   Model features (từ model file): {model_actual_features}")
        
        # Auto-detect feature_dim từ model nếu chưa có
        if feature_dim is None:
            feature_dim = model_actual_features
            print(f"   Auto-detected feature_dim: {feature_dim}")
        else:
            # So sánh feature_dim được truyền vào với số features thực tế của model
            print(f"   Feature_dim được truyền vào: {feature_dim}")
            if feature_dim != model_actual_features:
                print(f"   ⚠️  WARNING: Feature mismatch!")
                print(f"      - Model thực tế có: {model_actual_features} features")
                print(f"      - Feature_dim được truyền vào: {feature_dim} features")
                print(f"      - Sẽ sử dụng số features thực tế của model: {model_actual_features}")
                print(f"      - Data sẽ được align/truncate để khớp với model")
                # Sử dụng số features thực tế của model
                feature_dim = model_actual_features
            else:
                print(f"   ✅ Feature dimensions khớp: {feature_dim}")
    except Exception as e:
        print(f"❌ Không thể load LightGBM model: {e}")
        raise
    
    # Load normalization stats nếu có
    feature_means = None
    feature_stds = None
    if normalization_stats_path and Path(normalization_stats_path).exists():
        print(f"🔄 Đang load normalization stats từ {normalization_stats_path}...")
        try:
            feature_means, feature_stds, _ = load_normalization_stats(normalization_stats_path)
            print(f"✅ Đã load normalization stats")
            print(f"   Feature means shape: {feature_means.shape}")
            print(f"   Feature stds shape: {feature_stds.shape}")
        except Exception as e:
            print(f"⚠️  Không thể load normalization stats: {e}")
            print(f"   Sẽ predict không normalize")
    else:
        print(f"⚠️  Không có normalization stats, sẽ predict không normalize")
    
    def predict(X):
        """
        Predict với LightGBM model.
        
        Args:
            X: Raw feature array (shape: [n_samples, n_features])
        
        Returns:
            (predictions, probabilities): predictions là binary classes, probabilities là raw outputs
        """
        current_dim = X.shape[1]
        
        # Debug: In thông tin về feature dimensions
        if current_dim != feature_dim:
            print(f"   🔍 Debug predict: Input có {current_dim} features, model cần {feature_dim} features")
        
        # Normalize nếu có stats
        # Tạo local copies để tránh UnboundLocalError
        local_feature_means = feature_means
        local_feature_stds = feature_stds
        
        if local_feature_means is not None and local_feature_stds is not None:
            # Kiểm tra xem normalization stats có khớp với model không
            stats_dim = local_feature_means.shape[0]
            if stats_dim != feature_dim:
                print(f"   ⚠️  WARNING: Normalization stats có {stats_dim} features nhưng model cần {feature_dim} features")
                print(f"      - Sẽ chỉ normalize {min(stats_dim, feature_dim)} features đầu tiên")
                # Truncate stats nếu cần (tạo copies để không modify original)
                if stats_dim > feature_dim:
                    local_feature_means = local_feature_means[:feature_dim].copy()
                    local_feature_stds = local_feature_stds[:feature_dim].copy()
                else:
                    # Pad stats với zeros nếu cần (không nên xảy ra)
                    padding_means = np.zeros(feature_dim - stats_dim, dtype=local_feature_means.dtype)
                    padding_stds = np.ones(feature_dim - stats_dim, dtype=local_feature_stds.dtype)
                    local_feature_means = np.concatenate([local_feature_means, padding_means])
                    local_feature_stds = np.concatenate([local_feature_stds, padding_stds])
            
            X_normalized = normalize_features(X, local_feature_means, local_feature_stds, required_feature_dim=feature_dim)
        else:
            # Align features nếu cần (không normalize)
            if feature_dim is not None:
                if current_dim < feature_dim:
                    padding = np.zeros((X.shape[0], feature_dim - current_dim), dtype=np.float32)
                    X_normalized = np.concatenate([X, padding], axis=1)
                    print(f"   🔍 Debug: Đã pad {feature_dim - current_dim} features với zeros")
                elif current_dim > feature_dim:
                    X_normalized = X[:, :feature_dim]
                    print(f"   🔍 Debug: Đã truncate từ {current_dim} xuống {feature_dim} features")
                else:
                    X_normalized = X
            else:
                X_normalized = X
        
        # Verify final dimensions
        if X_normalized.shape[1] != feature_dim:
            raise ValueError(f"❌ Lỗi: X_normalized có {X_normalized.shape[1]} features nhưng model cần {feature_dim}")
        
        # Predict
        num_iteration = None
        if hasattr(model, 'best_iteration') and model.best_iteration is not None:
            if model.best_iteration > 0:
                num_iteration = model.best_iteration
        
        try:
            probs = model.predict(X_normalized, num_iteration=num_iteration)
        except Exception as e:
            print(f"   ❌ Lỗi khi predict: {e}")
            print(f"   🔍 Debug: X_normalized shape: {X_normalized.shape}, model.num_feature(): {model.num_feature()}")
            raise
        
        # Đảm bảo output là 1D array
        if probs.ndim > 1:
            probs = np.squeeze(probs)
        
        # Debug: In thống kê về probabilities
        if len(probs) > 0:
            print(f"   🔍 Debug predict: Probs range: [{probs.min():.4f}, {probs.max():.4f}], mean: {probs.mean():.4f}, std: {probs.std():.4f}")
            # Kiểm tra xem có phải tất cả probabilities đều < threshold không
            below_threshold = (probs < threshold).sum()
            above_threshold = (probs >= threshold).sum()
            print(f"   🔍 Debug predict: Probabilities < threshold ({threshold}): {below_threshold}, >= threshold: {above_threshold}")
        
        # LightGBM predict trả về probability của class 1 (malware)
        # Chuyển thành binary labels với threshold
        predictions = (probs >= threshold).astype(int)
        
        # Debug: In thống kê về predictions
        unique_preds, counts_preds = np.unique(predictions, return_counts=True)
        print(f"   🔍 Debug predict: Predictions distribution: {dict(zip(unique_preds, counts_preds))}")
        
        # Cảnh báo nếu tất cả predictions đều giống nhau
        if len(unique_preds) == 1:
            print(f"   ⚠️  WARNING: Tất cả predictions đều là {unique_preds[0]}!")
            print(f"      - Điều này cho thấy model có thể quá yếu hoặc được train không đúng")
            print(f"      - Có thể do feature mismatch (train trên dataset khác với test)")
            print(f"      - Hoặc model chỉ học được pattern trivial (luôn predict một class)")
        
        return predictions, probs
    
    return predict


def evaluate_model_similarity(
    target_model,
    surrogate_predict,
    X_test,
    y_target,
    y_true=None,
    model_name: str = ""
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
    
    Returns:
        metrics dict với accuracy (nếu có y_true) và agreement
    """
    print(f"\n🔄 Đang đánh giá {model_name}...")
    print(f"   Test data shape: {X_test.shape}")
    print(f"   Target predictions distribution: {dict(zip(*np.unique(y_target, return_counts=True)))}")
    
    # Predict với surrogate
    print(f"   Đang predict với surrogate model...")
    y_surrogate, probs_surrogate = surrogate_predict(X_test)
    
    # Debug: In thống kê về surrogate predictions
    print(f"   Surrogate predictions shape: {y_surrogate.shape}")
    print(f"   Surrogate probabilities shape: {probs_surrogate.shape}")
    unique_surr, counts_surr = np.unique(y_surrogate, return_counts=True)
    print(f"   Surrogate predictions distribution: {dict(zip(unique_surr, counts_surr))}")
    if len(probs_surrogate) > 0:
        print(f"   Surrogate probabilities stats: min={probs_surrogate.min():.4f}, max={probs_surrogate.max():.4f}, mean={probs_surrogate.mean():.4f}, std={probs_surrogate.std():.4f}")
    
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
    parser = argparse.ArgumentParser(description="Đánh giá độ tương đồng giữa target và surrogate model (LightGBM)")
    parser.add_argument("--surrogate_model_path", type=str, required=True,
                       help="Đường dẫn đến file surrogate model (LightGBM .txt, .lgb, .pkl)")
    parser.add_argument("--threshold", type=float, default=0.5,
                       help="Threshold để chuyển probabilities thành binary labels (mặc định: 0.5)")
    parser.add_argument("--test_parquet", type=str, default=None,
                       help="Đường dẫn đến test data parquet file (mặc định: ember_2018_v2 train data)")
    parser.add_argument("--target_model_path", type=str, default=None,
                       help="Đường dẫn đến target model (mặc định: artifacts/targets/LEE.lgb)")
    parser.add_argument("--target_model_name", type=str, default="LEE",
                       help="Tên target model (mặc định: LEE)")
    parser.add_argument("--normalization_stats_path", type=str, default=None,
                       help="Đường dẫn đến normalization stats .npz file (tự động tìm nếu không chỉ định)")
    
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
    
    # Tìm normalization stats - ưu tiên trong output directory
    if args.normalization_stats_path:
        normalization_stats_path = args.normalization_stats_path
        if not Path(normalization_stats_path).is_absolute():
            normalization_stats_path = str((PROJECT_ROOT / normalization_stats_path).resolve())
        if not Path(normalization_stats_path).exists():
            raise FileNotFoundError(f"❌ Không tìm thấy normalization stats tại: {normalization_stats_path}")
    else:
        surrogate_output_dir = Path(surrogate_model_path).parent
        possible_stats_paths = [
            surrogate_output_dir / "normalization_stats.npz",  # Ưu tiên nhất
            PROJECT_ROOT / "artifacts" / "targets" / f"{target_model_name}_normalization_stats.npz",  # Target model stats
            PROJECT_ROOT / "artifacts" / "targets" / "normalization_stats.npz",  # Generic stats
        ]
        
        normalization_stats_path = None
        for path in possible_stats_paths:
            if path.exists():
                normalization_stats_path = str(path.resolve())
                break
        
        # Nếu không tìm thấy, sẽ dùng target model stats
        if normalization_stats_path is None:
            normalization_stats_path = str(PROJECT_ROOT / "artifacts" / "targets" / f"{target_model_name}_normalization_stats.npz")
            if not Path(normalization_stats_path).exists():
                normalization_stats_path = None
    
    output_dir = PROJECT_ROOT / "logs" / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("ĐÁNH GIÁ ĐỘ TƯƠNG ĐỒNG GIỮA TARGET VÀ SURROGATE MODELS (LightGBM)")
    print("=" * 80)
    print(f"Target Model: {target_model_path}")
    print(f"Target Model Name: {target_model_name}")
    print(f"Surrogate Model: {surrogate_model_path}")
    print(f"Test Data: {test_parquet}")
    print(f"Threshold: {threshold}")
    if normalization_stats_path:
        print(f"Normalization Stats: {normalization_stats_path}")
    else:
        print(f"Normalization Stats: Không có (sẽ predict không normalize)")
    
    # Load feature columns
    feature_cols = get_feature_columns(test_parquet)
    print(f"\n✅ Feature columns: {len(feature_cols)}")
    
    # Load test data (unlabeled - label -1)
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
        y_target_proba = target_oracle.predict_proba(X_test)
        
        # LightGBM predict trả về probability của class 1 (malware) - shape: (n_samples,)
        # Chuyển thành binary labels với threshold
        if y_target_proba.ndim == 1:
            y_target = (y_target_proba >= threshold).astype(int)
        elif y_target_proba.ndim == 2 and y_target_proba.shape[1] == 2:
            y_target = (y_target_proba[:, 1] >= threshold).astype(int)
        else:
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
    
    # Load surrogate model (LightGBM)
    print(f"\n🔄 Đang load surrogate model (LightGBM)...")
    if not Path(surrogate_model_path).exists():
        print(f"❌ Không tìm thấy surrogate model tại {surrogate_model_path}")
        return
    
    try:
        surrogate_predict = load_LightGBM_surrogate_model(
            model_path=surrogate_model_path,
            normalization_stats_path=normalization_stats_path,
            feature_dim=len(feature_cols),
            threshold=threshold
        )
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
            y_true=y_true,
            model_name=Path(surrogate_model_path).parent.name
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
    
    report_path = output_dir / "surrogate_similarity_lgb_report.txt"
    report_md_path = output_dir / "surrogate_similarity_lgb_report.md"
    json_path = output_dir / "surrogate_similarity_lgb_results.json"
    
    # Text report
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("BÁO CÁO ĐÁNH GIÁ ĐỘ TƯƠNG ĐỒNG GIỮA TARGET VÀ SURROGATE MODELS (LightGBM)\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("THÔNG TIN MODELS:\n")
        f.write("-" * 80 + "\n")
        f.write(f"Target Model: {target_model_path}\n")
        f.write(f"Surrogate Model: {surrogate_model_path}\n")
        if normalization_stats_path:
            f.write(f"Normalization Stats: {normalization_stats_path}\n")
        else:
            f.write(f"Normalization Stats: Không có\n")
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
            
            if result['accuracy'] is not None:
                f.write(f"  Accuracy: {result['accuracy']:.4f} ({result['accuracy']*100:.2f}%)\n")
                f.write(f"    → Độ chính xác của surrogate model so với ground truth labels\n")
            else:
                f.write(f"  Accuracy: Không có ground truth labels để tính\n")
            
            f.write(f"  Agreement: {result['agreement']:.4f} ({result['agreement']*100:.2f}%)\n")
            f.write(f"    → Độ nhất quán giữa surrogate và target model predictions\n")
            
            if result['auc'] is not None:
                f.write(f"  AUC (vs target): {result['auc']:.4f}\n")
            if result.get('auc_accuracy') is not None:
                f.write(f"  AUC (vs ground truth): {result['auc_accuracy']:.4f}\n")
            
            f.write(f"  Precision (agreement): {result.get('precision_agreement', 0):.4f}\n")
            f.write(f"  Recall (agreement): {result.get('recall_agreement', 0):.4f}\n")
            f.write(f"  F1-score (agreement): {result.get('f1_agreement', 0):.4f}\n")
            
            if result.get('precision_accuracy') is not None:
                f.write(f"  Precision (accuracy): {result['precision_accuracy']:.4f}\n")
                f.write(f"  Recall (accuracy): {result['recall_accuracy']:.4f}\n")
                f.write(f"  F1-score (accuracy): {result['f1_accuracy']:.4f}\n")
            
            cm_agreement = result.get('confusion_matrix_agreement')
            if cm_agreement:
                f.write(f"  Confusion Matrix (Agreement - Target vs Surrogate):\n")
                f.write(f"    TN: {cm_agreement['tn']}, FP: {cm_agreement['fp']}\n")
                f.write(f"    FN: {cm_agreement['fn']}, TP: {cm_agreement['tp']}\n")
            
            if result.get('confusion_matrix_accuracy'):
                cm_accuracy = result['confusion_matrix_accuracy']
                f.write(f"  Confusion Matrix (Accuracy - Ground Truth vs Surrogate):\n")
                f.write(f"    TN: {cm_accuracy['tn']}, FP: {cm_accuracy['fp']}\n")
                f.write(f"    FN: {cm_accuracy['fn']}, TP: {cm_accuracy['tp']}\n")
            
            if result.get('ground_truth_distribution'):
                f.write(f"  Ground truth distribution: {result['ground_truth_distribution']}\n")
            f.write(f"  Target distribution: {result['target_distribution']}\n")
            f.write(f"  Surrogate distribution: {result['surrogate_distribution']}\n")
            f.write("\n")
        
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
            f.write(f"  - F1-score (agreement): {result.get('f1_agreement', 0):.4f}\n")
            if result.get('f1_accuracy') is not None:
                f.write(f"  - F1-score (accuracy): {result['f1_accuracy']:.4f}\n")
    
    # Markdown report
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write("# Báo Cáo Đánh Giá Độ Tương Đồng Giữa Target và Surrogate Models (LightGBM)\n\n")
        
        f.write("## Thông Tin Models\n\n")
        f.write(f"- **Target Model**: `{target_model_path}`\n")
        f.write(f"- **Surrogate Model**: `{surrogate_model_path}`\n")
        if normalization_stats_path:
            f.write(f"- **Normalization Stats**: `{normalization_stats_path}`\n")
        else:
            f.write(f"- **Normalization Stats**: Không có\n")
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
            f1_agreement = result.get('f1_agreement', 0)
            f1_accuracy_str = f"{result['f1_accuracy']:.4f}" if result.get('f1_accuracy') is not None else "N/A"
            f.write(f"| {result['model_name']} | {accuracy_str} | "
                   f"{result['agreement']:.4f} | {auc_target_str} | {auc_gt_str} | "
                   f"{f1_agreement:.4f} | {f1_accuracy_str} |\n")
        
        f.write("\n## Chi Tiết Từng Model\n\n")
        
        for result in all_results:
            f.write(f"### {result['model_name'].replace('_', ' ').title()}\n\n")
            
            if result['accuracy'] is not None:
                f.write(f"- **Accuracy**: {result['accuracy']:.4f} ({result['accuracy']*100:.2f}%) - so với ground truth\n")
            else:
                f.write(f"- **Accuracy**: Không có ground truth labels để tính\n")
            
            f.write(f"- **Agreement**: {result['agreement']:.4f} ({result['agreement']*100:.2f}%) - so với target model\n")
            
            if result['auc'] is not None:
                f.write(f"- **AUC (vs target)**: {result['auc']:.4f}\n")
            if result.get('auc_accuracy') is not None:
                f.write(f"- **AUC (vs ground truth)**: {result['auc_accuracy']:.4f}\n")
            
            f.write(f"- **Precision (agreement)**: {result.get('precision_agreement', 0):.4f}\n")
            f.write(f"- **Recall (agreement)**: {result.get('recall_agreement', 0):.4f}\n")
            f.write(f"- **F1-score (agreement)**: {result.get('f1_agreement', 0):.4f}\n")
            
            if result.get('precision_accuracy') is not None:
                f.write(f"- **Precision (accuracy)**: {result['precision_accuracy']:.4f}\n")
                f.write(f"- **Recall (accuracy)**: {result['recall_accuracy']:.4f}\n")
                f.write(f"- **F1-score (accuracy)**: {result['f1_accuracy']:.4f}\n")
            f.write("\n")
            
            cm_agreement = result.get('confusion_matrix_agreement')
            if cm_agreement:
                f.write("**Confusion Matrix (Agreement - Target vs Surrogate):**\n\n")
                f.write(f"| | Predicted 0 | Predicted 1 |\n")
                f.write(f"|------|------------|-------------|\n")
                f.write(f"| Target 0 | {cm_agreement['tn']} | {cm_agreement['fp']} |\n")
                f.write(f"| Target 1 | {cm_agreement['fn']} | {cm_agreement['tp']} |\n\n")
            
            if result.get('confusion_matrix_accuracy'):
                cm_accuracy = result['confusion_matrix_accuracy']
                f.write("**Confusion Matrix (Accuracy - Ground Truth vs Surrogate):**\n\n")
                f.write(f"| | Predicted 0 | Predicted 1 |\n")
                f.write(f"|------|------------|-------------|\n")
                f.write(f"| Actual 0 | {cm_accuracy['tn']} | {cm_accuracy['fp']} |\n")
                f.write(f"| Actual 1 | {cm_accuracy['fn']} | {cm_accuracy['tp']} |\n\n")
            
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
        cols_to_show = ["model_name", "accuracy", "agreement", "auc"]
        if "f1_agreement" in df.columns:
            cols_to_show.append("f1_agreement")
        if "f1_accuracy" in df.columns and df["f1_accuracy"].notna().any():
            cols_to_show.append("f1_accuracy")
        print(df[cols_to_show].to_string(index=False))
    
    return all_results


if __name__ == "__main__":
    main()

