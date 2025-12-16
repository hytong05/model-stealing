import json
import os
import sys
from pathlib import Path
import gc

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, balanced_accuracy_score
from sklearn.preprocessing import RobustScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.attackers import KerasAttacker, LGBAttacker, KerasDualAttacker, CNNAttacker, KNNAttacker, XGBoostAttacker, TabNetAttacker
from src.targets.oracle_client import LocalOracleClient, create_oracle_from_name
from src.sampling import entropy_sampling
from sklearn_extra.cluster import KMedoids


def _clip_scale(scaler: RobustScaler, X: np.ndarray) -> np.ndarray:
    """Scale data với RobustScaler và clip về [-5, 5] giống pipeline gốc."""
    transformed = scaler.transform(X)
    return np.clip(transformed, -5, 5)


def _pad_or_truncate_features(X: np.ndarray, target_dim: int) -> np.ndarray:
    """
    Pad hoặc truncate features để khớp với target_dim.
    
    Args:
        X: Input features array (n_samples, n_features) hoặc (n_features,) cho single sample
        target_dim: Số features mong muốn
    
    Returns:
        X được pad/truncate để khớp với target_dim
    """
    # Xử lý cả 1D và 2D arrays
    is_1d = len(X.shape) == 1
    if is_1d:
        X = X.reshape(1, -1)
    
    current_dim = X.shape[1]
    
    if current_dim == target_dim:
        # Không cần thay đổi
        return X[0] if is_1d else X
    elif current_dim > target_dim:
        # Truncate: Cắt bỏ features thừa (giữ N features đầu tiên)
        X_truncated = X[:, :target_dim]
        return X_truncated[0] if is_1d else X_truncated
    else:
        # Padding: Thêm zeros vào cuối
        n_samples = X.shape[0]
        n_pad = target_dim - current_dim
        pad_values = np.zeros((n_samples, n_pad), dtype=X.dtype)
        X_padded = np.hstack([X, pad_values])
        return X_padded[0] if is_1d else X_padded


def _resolve_optional_path(path_str: str | None) -> Path | None:
    if path_str is None:
        return None
    path_obj = Path(path_str)
    if not path_obj.is_absolute():
        path_obj = PROJECT_ROOT / path_obj
    return path_obj.resolve()


def get_feature_columns(parquet_path: str, label_col: str = "Label") -> list:
    """Lấy danh sách feature columns từ parquet file."""
    pq_file = pq.ParquetFile(parquet_path)
    return [name for name in pq_file.schema.names if name != label_col]


def load_data_from_parquet_stratified(
    parquet_path_label_0: str,
    parquet_path_label_1: str,
    feature_cols: list,
    label_col: str,
    take_rows: int = None,
    shuffle: bool = False,
    batch_size: int = 10000,
    seed: int = None,
) -> tuple:
    """
    Load dữ liệu từ 2 file đã chia sẵn theo label (label_0 và label_1) với stratified sampling.
    Đảm bảo cân bằng class (50/50).
    """
    print(f"  🔄 Loading từ 2 file đã chia sẵn theo label (stratified)...")
    print(f"     Class 0: {parquet_path_label_0}")
    print(f"     Class 1: {parquet_path_label_1}")
    
    # Load từ mỗi file
    X_0, y_0 = load_data_from_parquet(
        parquet_path_label_0, feature_cols, label_col, skip_rows=0, take_rows=None, shuffle=False, batch_size=batch_size, seed=None
    )
    X_1, y_1 = load_data_from_parquet(
        parquet_path_label_1, feature_cols, label_col, skip_rows=0, take_rows=None, shuffle=False, batch_size=batch_size, seed=None
    )
    
    print(f"  ✅ Loaded: {len(X_0)} samples class 0, {len(X_1)} samples class 1")
    
    # Stratified sampling: Lấy 50% từ mỗi class
    if take_rows is not None:
        samples_per_class = take_rows // 2
        rng = np.random.default_rng(seed)
        
        # Shuffle mỗi class
        indices_0 = np.arange(len(X_0))
        indices_1 = np.arange(len(X_1))
        rng.shuffle(indices_0)
        rng.shuffle(indices_1)
        
        # Lấy samples_per_class từ mỗi class
        selected_0 = indices_0[:min(samples_per_class, len(X_0))]
        selected_1 = indices_1[:min(samples_per_class, len(X_1))]
        
        # Nếu không đủ từ một class, lấy thêm từ class kia
        if len(selected_0) < samples_per_class:
            needed = samples_per_class - len(selected_0)
            selected_1 = indices_1[:min(samples_per_class + needed, len(X_1))]
        elif len(selected_1) < samples_per_class:
            needed = samples_per_class - len(selected_1)
            selected_0 = indices_0[:min(samples_per_class + needed, len(X_0))]
        
        X_0 = X_0[selected_0]
        y_0 = y_0[selected_0]
        X_1 = X_1[selected_1]
        y_1 = y_1[selected_1]
        
        print(f"  ✅ Selected: {len(X_0)} samples class 0, {len(X_1)} samples class 1")
    
    # Kết hợp
    X_all = np.vstack([X_0, X_1])
    y_all = np.concatenate([y_0, y_1])
    
    # Shuffle nếu cần
    if shuffle:
        print(f"  🔄 Đang shuffle {len(X_all):,} samples...")
        if seed is not None:
            rng = np.random.default_rng(seed)
            indices = rng.permutation(len(X_all))
        else:
            indices = np.random.permutation(len(X_all))
        X_all = X_all[indices]
        y_all = y_all[indices]
    
    return X_all, y_all


def load_data_from_parquet(
    parquet_path: str,
    feature_cols: list,
    label_col: str,
    skip_rows: int = 0,
    take_rows: int = None,
    shuffle: bool = False,
    batch_size: int = 10000,
    seed: int = None,
) -> tuple:
    """
    Load dữ liệu từ parquet file, loại bỏ label -1 và trả về X, y.
    Giống logic trong final_model.ipynb nhưng không normalize (sẽ normalize sau).
    
    Args:
        seed: Random seed cho shuffle. Nếu None thì dùng np.random không có seed (không reproducible).
    """
    pq_file = pq.ParquetFile(parquet_path)
    all_X = []
    all_y = []
    rows_seen = 0
    emitted = 0
    removed_total = 0
    batch_count = 0

    try:
        total_batches = (pq_file.metadata.num_rows + batch_size - 1) // batch_size

        for batch in pq_file.iter_batches(batch_size=batch_size, columns=feature_cols + [label_col]):
            batch_count += 1
            batch_len = len(batch)
            batch_start = rows_seen
            rows_seen += batch_len

            if rows_seen <= skip_rows:
                if batch_count % 50 == 0:
                    print(f"  ⏳ Đã xử lý {batch_count}/{total_batches} batches (đang skip)...")
                continue

            batch_df = batch.to_pandas()

            if skip_rows > batch_start:
                start_idx = skip_rows - batch_start
                batch_df = batch_df.iloc[start_idx:]

            if label_col in batch_df.columns:
                label_series = batch_df[label_col]
            else:
                alt_cols = [col for col in batch_df.columns if col.lower() == label_col.lower()]
                if alt_cols:
                    label_series = batch_df[alt_cols[0]]
                else:
                    raise KeyError(
                        f"Label column '{label_col}' không tồn tại. Các cột: {list(batch_df.columns)[:5]}..."
                    )

            # Loại bỏ label -1 (unlabeled)
            valid_mask = label_series != -1
            if not np.any(valid_mask):
                removed_total += len(label_series)
                del batch_df, label_series
                gc.collect()
                continue

            removed_total += int(np.sum(~valid_mask))
            batch_df = batch_df[valid_mask]
            label_series = label_series[valid_mask]

            if take_rows is not None:
                remaining = take_rows - emitted
                if remaining <= 0:
                    break
                if len(batch_df) > remaining:
                    batch_df = batch_df.iloc[:remaining]
                    label_series = label_series.iloc[:remaining]

            if batch_df.empty:
                del batch_df, label_series
                gc.collect()
                continue

            X = batch_df[feature_cols].values.astype(np.float32)
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            y = label_series.values.astype(np.int32)

            all_X.append(X)
            all_y.append(y)
            emitted += len(X)

            if batch_count % 20 == 0:
                del batch_df, label_series
                gc.collect()
                if take_rows is None:
                    print(f"  ⏳ Đã xử lý {batch_count}/{total_batches} batches, loaded {emitted:,} samples...")
                else:
                    print(
                        f"  ⏳ Đã xử lý {batch_count}/{total_batches} batches, loaded {emitted:,}/{take_rows:,} samples..."
                    )
            else:
                del batch_df, label_series

            if take_rows is not None and emitted >= take_rows:
                break

        if all_X:
            X_concat = np.concatenate(all_X, axis=0)
            y_concat = np.concatenate(all_y, axis=0)
            del all_X, all_y
            gc.collect()
        else:
            X_concat = np.empty((0, len(feature_cols)), dtype=np.float32)
            y_concat = np.empty((0,), dtype=np.int32)

        if shuffle and len(X_concat) > 0:
            print(f"  🔄 Đang shuffle {len(X_concat):,} samples...")
            if seed is not None:
                rng = np.random.default_rng(seed)
                indices = rng.permutation(len(X_concat))
            else:
                indices = np.random.permutation(len(X_concat))
            X_concat = X_concat[indices]
            y_concat = y_concat[indices]

        if removed_total > 0:
            print(f"  ⚠️  Đã loại bỏ {removed_total:,} samples có label -1 (unlabeled)")

        return X_concat, y_concat
    finally:
        del pq_file
        gc.collect()


def run_extraction(
    output_dir: Path,
    train_parquet: str = None,
    test_parquet: str = None,
    dataset: str = "ember",  # "ember" hoặc "somlap" - dataset để tấn công
    seed: int = 42,
    feature_dim: int = 2381,
    seed_size: int = None,  # Tự động tính từ total_budget nếu None
    val_size: int = None,    # Tự động tính từ total_budget nếu None
    eval_size: int = 4000,
    query_batch: int = None,  # Tự động tính từ total_budget nếu None
    num_rounds: int = None,  # Tự động tính từ total_budget nếu None
    total_budget: int = None,  # Tổng query budget (seed + val + AL queries). Nếu được chỉ định, sẽ tự động tính seed_size, val_size, query_batch, num_rounds
    num_epochs: int = 5,
    model_type: str = "h5",  # "h5" hoặc "lgb" - chỉ cần nếu dùng weights_path
    normalization_stats_path: str = None,  # Chỉ cần nếu dùng weights_path với model_type="lgb"
    attacker_type: str = None,  # "keras", "lgb", hoặc "dual" (dualDNN), None để tự động chọn theo model_type
    weights_path: str | None = None,
    model_name: str = None,  # Tên model (CEE, LEE, CSE, LSE) - ưu tiên hơn weights_path
    threshold_optimization_metric: str = "f1",  # "f1", "accuracy", "balanced_accuracy" - metric để tối ưu threshold
    fixed_threshold: float | None = None,  # Nếu không None, sử dụng threshold cố định thay vì tối ưu
    surrogate_dir: str | None = None,  # Cho phép override thư mục lưu surrogate
    surrogate_name: str | None = None,  # Cho phép override tên file surrogate (không extension)
) -> dict:
    output_dir = Path(output_dir)
    rng = np.random.default_rng(seed)
    pool_exhausted_flag = False
    over_budget_flag = False

    # Chỉ set TF environment variables nếu dùng Keras model
    if model_type == "h5" or attacker_type in ["keras", "dual"]:
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

    # Xác định label column dựa trên dataset
    dataset = dataset.lower()
    if dataset == "ember":
        label_col = "Label"
    elif dataset == "somlap":
        label_col = "class"
    else:
        raise ValueError(f"Dataset không được hỗ trợ: {dataset}. Chọn 'ember' hoặc 'somlap'")
    
    # Auto-detect attacker_type nếu không được chỉ định
    if attacker_type is None:
        attacker_type = "keras" if model_type == "h5" else "lgb"

    # QUAN TRỌNG: Tính toán seed_size, val_size, query_batch, num_rounds từ total_budget
    # Nếu total_budget được chỉ định, tự động tính các giá trị này
    if total_budget is not None:
        # Tính seed_size = 10% của total_budget
        calculated_seed_size = int(total_budget * 0.1)
        # Tính val_size = 20% của total_budget
        calculated_val_size = int(total_budget * 0.2)
        # Tính AL_queries = 70% của total_budget (phần còn lại)
        AL_queries = total_budget - calculated_seed_size - calculated_val_size
        
        # Sử dụng giá trị đã tính nếu không được chỉ định
        if seed_size is None:
            seed_size = calculated_seed_size
        if val_size is None:
            val_size = calculated_val_size
        
        # Tính query_batch và num_rounds từ AL_queries nếu chưa được chỉ định
        if query_batch is None or num_rounds is None:
            # Mặc định: chia AL_queries thành 5 rounds
            if num_rounds is None:
                num_rounds = 5
            if query_batch is None:
                query_batch = AL_queries // num_rounds
                # Đảm bảo query_batch × num_rounds = AL_queries (có thể làm tròn)
                if query_batch * num_rounds < AL_queries:
                    query_batch += 1
        
        print(f"\n📊 Query Budget Configuration (từ total_budget={total_budget:,}):")
        print(f"   Seed size: {seed_size:,} (10% của budget)")
        print(f"   Val size: {val_size:,} (20% của budget)")
        print(f"   AL queries: {AL_queries:,} (70% của budget)")
        print(f"   Query batch: {query_batch:,} queries/round")
        print(f"   Number of rounds: {num_rounds}")
        print(f"   Total queries (seed + val + AL): {seed_size + val_size + AL_queries:,}")
    else:
        # Nếu total_budget không được chỉ định, dùng giá trị mặc định
        if seed_size is None:
            seed_size = 2000
        if val_size is None:
            val_size = 1000
        if query_batch is None:
            query_batch = 2000
        if num_rounds is None:
            num_rounds = 5
        
        print(f"\n📊 Query Budget Configuration (giá trị mặc định):")
        print(f"   Seed size: {seed_size:,}")
        print(f"   Val size: {val_size:,}")
        print(f"   Query batch: {query_batch:,} queries/round")
        print(f"   Number of rounds: {num_rounds}")
        print(f"   AL queries: {query_batch * num_rounds:,}")
        print(f"   Total queries (seed + val + AL): {seed_size + val_size + query_batch * num_rounds:,}")

    # Debug: Log giá trị train_parquet và test_parquet trước khi xử lý
    print(f"\n🔍 DEBUG: dataset={dataset}, train_parquet={train_parquet}, test_parquet={test_parquet}")

    # Load dữ liệu từ parquet files (EMBER hoặc SOMLAP)
    # QUAN TRỌNG: Nếu train_parquet hoặc test_parquet đã được set (không phải None),
    # cần đảm bảo chúng phù hợp với dataset được chỉ định
    if train_parquet is not None or test_parquet is not None:
        # Nếu đã được set, cần validate xem có khớp với dataset không
        if train_parquet is not None:
            train_path = Path(train_parquet)
            # Kiểm tra xem file có phải là EMBER file khi dataset là somlap không
            if dataset == "somlap" and ("ember" in str(train_path).lower() or "ember_2018" in str(train_path)):
                print(f"⚠️  WARNING: train_parquet được set là EMBER file nhưng dataset là SOMLAP!")
                print(f"   ⚠️  Đang bỏ qua train_parquet và sử dụng dataset parameter để chọn file đúng")
                train_parquet = None
            elif dataset == "ember" and "somlap" in str(train_path).lower():
                print(f"⚠️  WARNING: train_parquet được set là SOMLAP file nhưng dataset là EMBER!")
                print(f"   ⚠️  Đang bỏ qua train_parquet và sử dụng dataset parameter để chọn file đúng")
                train_parquet = None
        
        if test_parquet is not None:
            test_path = Path(test_parquet)
            # Kiểm tra xem file có phải là EMBER file khi dataset là somlap không
            if dataset == "somlap" and ("ember" in str(test_path).lower() or "ember_2018" in str(test_path)):
                print(f"⚠️  WARNING: test_parquet được set là EMBER file nhưng dataset là SOMLAP!")
                print(f"   ⚠️  Đang bỏ qua test_parquet và sử dụng dataset parameter để chọn file đúng")
                test_parquet = None
            elif dataset == "ember" and "somlap" in str(test_path).lower():
                print(f"⚠️  WARNING: test_parquet được set là SOMLAP file nhưng dataset là EMBER!")
                print(f"   ⚠️  Đang bỏ qua test_parquet và sử dụng dataset parameter để chọn file đúng")
                test_parquet = None
    
    if train_parquet is None:
        if dataset == "ember":
            # EMBER dataset: Thử dùng file đã chia sẵn theo label trước
            train_parquet_label_0 = str(PROJECT_ROOT / "data" / "ember_2018_v2" / "train" / "train_ember_2018_v2_features_label_other_label_0.parquet")
            train_parquet_label_1 = str(PROJECT_ROOT / "data" / "ember_2018_v2" / "train" / "train_ember_2018_v2_features_label_other_label_1.parquet")
            # Fallback về file cũ nếu không có file mới
            train_parquet_old = str(PROJECT_ROOT / "data" / "train_ember_2018_v2_features_label_other.parquet")
            if Path(train_parquet_label_0).exists() and Path(train_parquet_label_1).exists():
                train_parquet = None  # Sẽ dùng stratified load từ 2 file
            elif Path(train_parquet_old).exists():
                train_parquet = train_parquet_old
            else:
                raise FileNotFoundError(f"Không tìm thấy EMBER train data tại: {train_parquet_label_0} hoặc {train_parquet_old}")
        elif dataset == "somlap":
            # SOMLAP dataset
            train_parquet_path = PROJECT_ROOT / "data" / "SOMLAP" / "SOMLAP DATASET_train.parquet"
            if train_parquet_path.exists():
                train_parquet = str(train_parquet_path)
            else:
                raise FileNotFoundError(f"Không tìm thấy SOMLAP train data tại: {train_parquet_path}")
    
    if test_parquet is None:
        if dataset == "ember":
            test_parquet_new = str(PROJECT_ROOT / "data" / "ember_2018_v2" / "test" / "test_ember_2018_v2_features_label_other.parquet")
            test_parquet_old = str(PROJECT_ROOT / "data" / "test_ember_2018_v2_features_label_other.parquet")
            if Path(test_parquet_new).exists():
                test_parquet = test_parquet_new
            elif Path(test_parquet_old).exists():
                test_parquet = test_parquet_old
            else:
                raise FileNotFoundError(f"Không tìm thấy EMBER test data tại: {test_parquet_new} hoặc {test_parquet_old}")
        elif dataset == "somlap":
            test_parquet_path = PROJECT_ROOT / "data" / "SOMLAP" / "SOMLAP DATASET_test.parquet"
            if test_parquet_path.exists():
                test_parquet = str(test_parquet_path)
            else:
                raise FileNotFoundError(f"Không tìm thấy SOMLAP test data tại: {test_parquet_path}")

    print("=" * 60)
    print(f"📊 Đang load dữ liệu {dataset.upper()}...")
    print("=" * 60)
    print(f"Dataset: {dataset.upper()}")
    print(f"Label column: {label_col}")
    print(f"Train file: {train_parquet if train_parquet else '(sẽ tự động chọn dựa trên dataset)'}")
    print(f"Test file: {test_parquet if test_parquet else '(sẽ tự động chọn dựa trên dataset)'}")

    # Lấy feature columns và xác định feature_dim thực tế của dataset attack
    # Nếu train_parquet là None (dùng stratified load từ 2 file - chỉ EMBER), dùng một trong 2 file hoặc test_parquet
    if train_parquet is None:
        # Chỉ có thể None với EMBER dataset (stratified loading)
        train_parquet_label_0 = str(PROJECT_ROOT / "data" / "ember_2018_v2" / "train" / "train_ember_2018_v2_features_label_other_label_0.parquet")
        if Path(train_parquet_label_0).exists():
            feature_cols = get_feature_columns(train_parquet_label_0, label_col)
        elif test_parquet is not None:
            feature_cols = get_feature_columns(test_parquet, label_col)
        else:
            raise ValueError("Không thể lấy feature columns: train_parquet là None và không có file label_0 hoặc test_parquet")
    else:
        feature_cols = get_feature_columns(train_parquet, label_col)
    
    # Lưu số features thực tế của dataset attack (để pad/truncate sau)
    dataset_attack_feature_dim = len(feature_cols)
    print(f"📊 Dataset attack ({dataset.upper()}) có {dataset_attack_feature_dim} features")
    
    # QUAN TRỌNG: KHÔNG tự động thay đổi feature_dim thành số features của dataset attack!
    # feature_dim phải được set dựa trên target model, không phải dataset attack
    # Dataset attack chỉ là nguồn dữ liệu, cần pad/truncate để khớp với target model
    print(f"📊 Specified feature_dim (từ target model hoặc default): {feature_dim}")
    
    # QUAN TRỌNG: Validate và log thông tin target model
    oracle_source = None
    required_feature_dim = None
    oracle_client = None
    model_file_name = None
    
    # Ưu tiên sử dụng model_name nếu được cung cấp
    if model_name is not None:
        print(f"\n🔄 Khởi tạo target model từ tên: {model_name.upper()}")
        print(f"   ℹ️  Sẽ tự động detect model type và tìm normalization stats...")
        print(f"   🔒 BLACK BOX: Attacker chỉ biết tên model, không biết implementation details")
        
        # Sử dụng create_oracle_from_name - tự động detect mọi thứ
        # Trả về BlackBoxOracleClient để ẩn implementation details
        oracle_client = create_oracle_from_name(
            model_name=model_name,
            threshold=0.5,
            feature_dim=feature_dim,
        )
        
        # Lấy thông tin từ oracle client (chỉ để logging, không dùng trong attack)
        # Trong black box attack thực tế, attacker không nên biết những thông tin này
        # Nhưng để logging/debugging, vẫn lấy từ internal oracle
        if hasattr(oracle_client, '_oracle'):
            internal_oracle = oracle_client._oracle
            weights_path_abs = internal_oracle.model_path
            model_type = internal_oracle.model_type
            # Khi dùng model_name, normalization_stats_path đã được tự động tìm và truyền vào oracle
            # Không cần kiểm tra lại ở đây
            normalization_stats_path = "auto-detected"  # Đánh dấu đã được tự động detect
        else:
            # Fallback nếu không có _oracle (trường hợp dùng LocalOracleClient trực tiếp)
            weights_path_abs = oracle_client.model_path
            model_type = oracle_client.model_type
            normalization_stats_path = getattr(oracle_client, 'normalization_stats_path', None)
        
        model_file_name = Path(weights_path_abs).name
        model_file_size = Path(weights_path_abs).stat().st_size / (1024 * 1024)  # MB
        oracle_source = weights_path_abs
        
        print(f"   ✅ Target model file: {model_file_name}")
        print(f"   ✅ Model path (absolute): {weights_path_abs}")
        print(f"   ✅ Model type: {model_type.upper()}")
        print(f"   ✅ Model size: {model_file_size:.2f} MB")
        print(f"   ⚠️  LƯU Ý: Thông tin trên chỉ để logging, attacker không nên biết trong black box attack thực tế")
        
        required_feature_dim = oracle_client.get_required_feature_dim()
    else:
        # Sử dụng cách cũ với weights_path
        if weights_path is None:
            raise ValueError("Phải cung cấp weights_path hoặc model_name cho oracle module.")
        weights_path_abs = str(Path(weights_path).resolve())
        if not Path(weights_path_abs).exists():
            raise FileNotFoundError(f"❌ Target model không tồn tại: {weights_path_abs}")
        
        model_file_name = Path(weights_path_abs).name
        model_file_size = Path(weights_path_abs).stat().st_size / (1024 * 1024)  # MB
        oracle_source = weights_path_abs
        
        print(f"\n🔄 Khởi tạo target model ({model_type.upper()}) với feature_dim={feature_dim}...")
        print(f"   ✅ Target model file: {model_file_name}")
        print(f"   ✅ Model path (absolute): {weights_path_abs}")
        print(f"   ✅ Model size: {model_file_size:.2f} MB")
        
        if weights_path != weights_path_abs:
            print(f"   ⚠️  Path được resolve: {weights_path} -> {weights_path_abs}")
        
        if model_type == "lgb":
            if normalization_stats_path is None:
                raise ValueError(
                    "normalization_stats_path phải được cung cấp khi model_type='lgb'. "
                    "Vui lòng cung cấp đường dẫn tới file normalization_stats.npz"
                )
            if isinstance(normalization_stats_path, str):
                stats_path_abs = str(Path(normalization_stats_path).resolve())
                if not Path(stats_path_abs).exists():
                    raise FileNotFoundError(f"❌ Normalization stats không tồn tại: {stats_path_abs}")
                normalization_stats_path = stats_path_abs
            
            print(f"   ✅ Normalization stats file: {Path(normalization_stats_path).name}")
            print(f"   ✅ Stats path (absolute): {normalization_stats_path}")
        else:
            normalization_stats_path = None
        
        # Tạo oracle client với weights_path (cách cũ)
        oracle_client = LocalOracleClient(
            model_type=model_type,
            model_path=weights_path_abs,
            normalization_stats_path=normalization_stats_path,
            threshold=0.5,
            feature_dim=feature_dim,
        )
        required_feature_dim = oracle_client.get_required_feature_dim()
    required_feature_dim = oracle_client.get_required_feature_dim()
    
    # QUAN TRỌNG: Surrogate model phải có số features bằng với target model
    # Nếu required_feature_dim không None, dùng nó cho surrogate model
    # Nếu None (có preprocessing layer), dùng feature_dim hiện tại
    if required_feature_dim is None:
        # Target model có preprocessing layer - sẽ tự động map
        surrogate_feature_dim = feature_dim
        print(f"   ✅ Target model có preprocessing layer - sẽ tự động map từ {feature_dim} đặc trưng")
        print(f"   ✅ Surrogate model sẽ dùng {surrogate_feature_dim} features (từ feature_dim)")
    else:
        # Target model yêu cầu số features cụ thể
        surrogate_feature_dim = required_feature_dim
        print(f"   ✅ Target model yêu cầu {required_feature_dim} đặc trưng")
        print(f"   ✅ Surrogate model sẽ dùng {surrogate_feature_dim} features (từ target model)")
        
        # So sánh với dataset attack
        if dataset_attack_feature_dim > required_feature_dim:
            print(f"   ⚠️  Dataset attack có {dataset_attack_feature_dim} đặc trưng, sẽ tự động cắt bỏ {dataset_attack_feature_dim - required_feature_dim} đặc trưng thừa")
            print(f"      (Giữ {required_feature_dim} features đầu tiên)")
        elif dataset_attack_feature_dim < required_feature_dim:
            print(f"   ⚠️  Dataset attack có {dataset_attack_feature_dim} đặc trưng, nhưng target model yêu cầu {required_feature_dim} đặc trưng")
            print(f"   ✅ Sẽ tự động PADDING thêm {required_feature_dim - dataset_attack_feature_dim} đặc trưng (zeros) trước khi query oracle và train surrogate")
            print(f"      Lưu ý: Padding có thể ảnh hưởng đến độ chính xác của attack")
        else:
            print(f"   ✅ Dataset attack có {dataset_attack_feature_dim} features, khớp với target model ({required_feature_dim})")

    # QUAN TRỌNG: Load đủ thief dataset để có thể chọn mẫu sau này
    # Pool KHÔNG được chọn trước, mà tích lũy dần từ seed
    # Cần load: seed + val + AL_queries + buffer (để đảm bảo đủ mẫu)
    AL_queries_needed = query_batch * num_rounds
    buffer_size = int(AL_queries_needed * 0.2)  # Buffer 20% để đảm bảo đủ mẫu
    seed_val_size = seed_size + val_size
    total_needed = seed_val_size + AL_queries_needed + buffer_size
    
    print(f"\n🔄 Đang load thief dataset ({total_needed:,} samples: {seed_size:,} seed + {val_size:,} val + {AL_queries_needed:,} AL queries + {buffer_size:,} buffer)...")
    
    # Load train data - xử lý khác nhau cho EMBER và SOMLAP
    if dataset == "ember":
        # EMBER: CẢI TIẾN: Sử dụng file đã chia sẵn theo label nếu có
        train_parquet_label_0 = str(PROJECT_ROOT / "data" / "ember_2018_v2" / "train" / "train_ember_2018_v2_features_label_other_label_0.parquet")
        train_parquet_label_1 = str(PROJECT_ROOT / "data" / "ember_2018_v2" / "train" / "train_ember_2018_v2_features_label_other_label_1.parquet")
        
        if Path(train_parquet_label_0).exists() and Path(train_parquet_label_1).exists():
            # Sử dụng file đã chia sẵn theo label (stratified loading)
            # QUAN TRỌNG: Load cả ground truth labels từ train data (không query oracle!)
            X_train_all, y_train_all_gt = load_data_from_parquet_stratified(
                train_parquet_label_0, train_parquet_label_1, feature_cols, label_col,
                take_rows=total_needed, shuffle=True, seed=seed
            )
            print(f"✅ Train data loaded (stratified): {X_train_all.shape}")
        else:
            # Fallback: Dùng file cũ
            if train_parquet is None:
                raise FileNotFoundError(f"Không tìm thấy EMBER train data files. Đã tìm tại:\n  - {train_parquet_label_0}\n  - {train_parquet_label_1}")
            X_train_all, y_train_all_gt = load_data_from_parquet(
                train_parquet, feature_cols, label_col, skip_rows=0, take_rows=total_needed, shuffle=True, seed=seed
            )
            print(f"✅ Train data loaded: {X_train_all.shape}")
    elif dataset == "somlap":
        # SOMLAP: Chỉ có 1 file duy nhất, không có stratified files
        if train_parquet is None:
            raise FileNotFoundError(f"Không tìm thấy SOMLAP train data file")
        X_train_all, y_train_all_gt = load_data_from_parquet(
            train_parquet, feature_cols, label_col, skip_rows=0, take_rows=total_needed, shuffle=True, seed=seed
        )
        print(f"✅ Train data loaded: {X_train_all.shape}")
    else:
        raise ValueError(f"Dataset không được hỗ trợ: {dataset}")

    train_dist = dict(zip(*np.unique(y_train_all_gt, return_counts=True)))
    print(f"   📊 Train data distribution (ground truth): {train_dist}")
    
    # QUAN TRỌNG: Pad/truncate data từ dataset attack để khớp với surrogate_feature_dim
    if dataset_attack_feature_dim != surrogate_feature_dim:
        print(f"\n🔄 Đang pad/truncate train data từ {dataset_attack_feature_dim} lên {surrogate_feature_dim} features...")
        X_train_all = _pad_or_truncate_features(X_train_all, surrogate_feature_dim)
        print(f"   ✅ Train data shape sau pad/truncate: {X_train_all.shape}")

    # CẢI TIẾN: Stratified split cho Seed và Val để cân bằng class
    # Sử dụng ground truth labels từ train data (KHÔNG query oracle!)
    print(f"\n🔄 Chia Seed và Val với stratified sampling (cân bằng class, dùng ground truth labels)...")
    rng = np.random.default_rng(seed)
    
    # Tách indices theo class (dùng ground truth labels)
    class_0_indices = np.where(y_train_all_gt == 0)[0]
    class_1_indices = np.where(y_train_all_gt == 1)[0]
    
    # Shuffle mỗi class
    rng.shuffle(class_0_indices)
    rng.shuffle(class_1_indices)
    
    # Chia seed: 50% từ mỗi class
    seed_per_class = seed_size // 2
    seed_class_0_idx = class_0_indices[:seed_per_class]
    seed_class_1_idx = class_1_indices[:min(seed_per_class, len(class_1_indices))]
    
    # Nếu không đủ class 1, lấy thêm từ class 0
    if len(seed_class_1_idx) < seed_per_class:
        needed = seed_per_class - len(seed_class_1_idx)
        seed_class_0_idx = np.concatenate([seed_class_0_idx, class_0_indices[seed_per_class:seed_per_class+needed]])
    
    seed_indices = np.concatenate([seed_class_0_idx, seed_class_1_idx])
    rng.shuffle(seed_indices)  # Shuffle lại để trộn classes
    
    # Cập nhật class indices (loại bỏ đã dùng cho seed)
    class_0_indices = class_0_indices[len(seed_class_0_idx):]
    class_1_indices = class_1_indices[len(seed_class_1_idx):]
    
    # Chia val: 50% từ mỗi class (từ phần còn lại)
    val_per_class = val_size // 2
    val_class_0_idx = class_0_indices[:val_per_class]
    val_class_1_idx = class_1_indices[:min(val_per_class, len(class_1_indices))]
    
    # Nếu không đủ class 1, lấy thêm từ class 0
    if len(val_class_1_idx) < val_per_class:
        needed = val_per_class - len(val_class_1_idx)
        val_class_0_idx = np.concatenate([val_class_0_idx, class_0_indices[val_per_class:val_per_class+needed]])
    
    val_indices = np.concatenate([val_class_0_idx, val_class_1_idx])
    rng.shuffle(val_indices)  # Shuffle lại để trộn classes
    
    # Lấy seed và val
    X_seed = X_train_all[seed_indices]
    X_val = X_train_all[val_indices]
    
    # QUAN TRỌNG: Phần còn lại giữ làm unlabeled pool (KHÔNG query trước)
    # Pool sẽ tích lũy dần từ seed, sau đó thêm các mẫu được AL chọn
    used_indices = np.concatenate([seed_indices, val_indices])
    unlabeled_pool_indices = np.setdiff1d(np.arange(len(X_train_all)), used_indices)
    X_unlabeled_pool = X_train_all[unlabeled_pool_indices]
    y_unlabeled_pool_gt = y_train_all_gt[unlabeled_pool_indices]  # Ground truth labels để pre-filtering
    
    # Kiểm tra xem có đủ unlabeled pool không
    AL_queries_needed = query_batch * num_rounds
    available_unlabeled = len(unlabeled_pool_indices)
    if available_unlabeled < AL_queries_needed:
        print(f"   ⚠️  CẢNH BÁO: Unlabeled pool ({available_unlabeled:,}) < AL queries cần ({AL_queries_needed:,})")
        print(f"   💡 Có thể sẽ thiếu queries trong quá trình active learning")
        if available_unlabeled < AL_queries_needed * 0.9:
            raise ValueError(
                f"Không đủ unlabeled pool! Available: {available_unlabeled:,}, "
                f"Required: {AL_queries_needed:,} (query_batch={query_batch:,} × num_rounds={num_rounds})"
            )
    
    # Log distribution (ground truth)
    seed_dist_gt = dict(zip(*np.unique(y_train_all_gt[seed_indices], return_counts=True)))
    val_dist_gt = dict(zip(*np.unique(y_train_all_gt[val_indices], return_counts=True)))
    unlabeled_pool_dist_gt = dict(zip(*np.unique(y_unlabeled_pool_gt, return_counts=True)))
    print(f"   ✅ Seed distribution (stratified, ground truth): {seed_dist_gt}")
    print(f"   ✅ Val distribution (stratified, ground truth): {val_dist_gt}")
    print(f"   ✅ Unlabeled pool distribution (ground truth from thief dataset): {unlabeled_pool_dist_gt}")
    print(f"   ✅ Unlabeled pool size: {X_unlabeled_pool.shape[0]:,} samples (sẽ được chọn dần trong AL)")
    print(f"      - AL queries cần: {AL_queries_needed:,} (query_batch={query_batch:,} × num_rounds={num_rounds})")
    del X_train_all
    gc.collect()

    # Load eval set từ test file
    # QUAN TRỌNG: Load cả ground truth labels để tính accuracy chính xác
    print(f"\n🔄 Đang load eval set ({eval_size:,} samples)...")
    # Test data có thể dùng file cũ hoặc file mới
    X_eval, y_eval_gt = load_data_from_parquet(
        test_parquet, feature_cols, label_col, skip_rows=0, take_rows=eval_size, shuffle=True, seed=seed
    )
    print(f"✅ Eval set: {X_eval.shape}")
    print(f"✅ Ground truth labels: {y_eval_gt.shape}")
    
    # QUAN TRỌNG: Pad/truncate eval data để khớp với surrogate_feature_dim
    if dataset_attack_feature_dim != surrogate_feature_dim:
        print(f"🔄 Đang pad/truncate eval data từ {dataset_attack_feature_dim} lên {surrogate_feature_dim} features...")
        X_eval = _pad_or_truncate_features(X_eval, surrogate_feature_dim)
        print(f"   ✅ Eval data shape sau pad/truncate: {X_eval.shape}")

    print(f"\n📊 Data split:")
    print(f"  Seed: {X_seed.shape[0]:,}")
    print(f"  Val: {X_val.shape[0]:,}")
    print(f"  Unlabeled pool: {X_unlabeled_pool.shape[0]:,}")
    print(f"  Eval: {X_eval.shape[0]:,}")

    # QUAN TRỌNG: Xử lý dữ liệu trước khi query oracle
    # QUAN TRỌNG: Scale data dựa trên MODEL_TYPE CỦA ORACLE (target model), KHÔNG phải attacker_type!
    # - Với Keras/H5 Oracle: Cần scale data với RobustScaler (model được train với scaled data)
    # - Với LightGBM Oracle: FlexibleLGBTarget sẽ tự động normalize nếu có normalization_stats_path
    #   KHÔNG được scale với RobustScaler - chỉ cần raw data!
    # - attacker_type chỉ ảnh hưởng đến cách train surrogate model, không ảnh hưởng đến cách query oracle
    scaler = None
    X_eval_s = None
    X_seed_s = None
    X_val_s = None
    X_pool_s = None
    
    # Lấy model_type thực tế của oracle (không phải attacker_type)
    oracle_model_type = model_type  # Nếu dùng model_name, model_type đã được detect từ oracle_client
    
    # Kiểm tra xem oracle có normalization_stats_path hay không
    oracle_has_normalization_stats = False
    if oracle_model_type == "h5":
        # Kiểm tra từ oracle client xem có normalization stats không
        if hasattr(oracle_client, '_oracle'):
            # BlackBoxOracleClient -> LocalOracleClient -> FlexibleKerasTarget
            internal_oracle = oracle_client._oracle._oracle if hasattr(oracle_client._oracle, '_oracle') else oracle_client._oracle
            oracle_has_normalization_stats = getattr(internal_oracle, 'use_normalization', False)
        elif hasattr(oracle_client, '_oracle') and hasattr(oracle_client._oracle, 'use_normalization'):
            # LocalOracleClient -> FlexibleKerasTarget
            oracle_has_normalization_stats = oracle_client._oracle.use_normalization
        # Hoặc kiểm tra từ normalization_stats_path variable
        if not oracle_has_normalization_stats:
            oracle_has_normalization_stats = (normalization_stats_path is not None and 
                                             normalization_stats_path != "auto-detected" and
                                             normalization_stats_path != "")
    
    
    if oracle_model_type == "h5":
        if oracle_has_normalization_stats:
            # Keras/H5 Oracle có normalization_stats: Oracle sẽ tự động normalize và clip
            # KHÔNG scale với RobustScaler - chỉ cần raw data!
            print(f"\n🔄 Oracle có normalization stats - Oracle sẽ tự động normalize và clip")
            print(f"   ⚠️  KHÔNG scale với RobustScaler - dùng raw data để query oracle")
            
            # Dùng raw data để query oracle (oracle sẽ tự normalize và clip)
            X_eval_s = X_eval
            X_seed_s = X_seed
            X_val_s = X_val
            
            # Lấy nhãn từ oracle (VỚI RAW DATA - oracle sẽ tự normalize và clip)
            print(f"\n🔄 Đang lấy nhãn từ oracle (với raw data - oracle sẽ tự normalize và clip)...")
            y_eval = oracle_client.predict(X_eval_s)
            y_seed = oracle_client.predict(X_seed_s)
            y_val = oracle_client.predict(X_val_s)
            
            # Nếu attacker cần scaled data cho training, tạo scaler riêng
            if attacker_type in ["keras", "dual", "cnn"]:
                print(f"\n⚠️  LƯU Ý: Oracle tự normalize, nhưng surrogate là {attacker_type} (cần scaled data)")
            elif attacker_type in ["xgb", "tabnet"]:
                print(f"\n✅ Oracle tự normalize, surrogate là {attacker_type} (không cần scaled data)")
                print(f"   🔄 Sẽ scale data riêng cho surrogate model training sau...")
                scaler = RobustScaler()
                scaler.fit(np.vstack([X_seed, X_val, X_unlabeled_pool]))
                # Tạo scaled version cho surrogate training
                X_eval_s = _clip_scale(scaler, X_eval)
                X_seed_s = _clip_scale(scaler, X_seed)
                X_val_s = _clip_scale(scaler, X_val)
        else:
            # Keras/H5 Oracle KHÔNG có normalization_stats: Cần scale với RobustScaler
            print(f"\n🔄 Đang scale dữ liệu trước khi query oracle (Keras/H5 Oracle không có normalization stats, cần scaled data)...")
            scaler = RobustScaler()
            scaler.fit(np.vstack([X_seed, X_val, X_unlabeled_pool]))

            X_eval_s = _clip_scale(scaler, X_eval)
            X_seed_s = _clip_scale(scaler, X_seed)
            X_val_s = _clip_scale(scaler, X_val)
            # X_unlabeled_pool_s sẽ được tạo sau trong AL loop nếu cần
            
            print(f"✅ Đã scale dữ liệu")
            print(f"   - X_eval_s shape: {X_eval_s.shape}")
            print(f"   - X_seed_s shape: {X_seed_s.shape}")
            print(f"   - X_val_s shape: {X_val_s.shape}")
            print(f"   - X_unlabeled_pool_s: sẽ được tạo sau trong AL loop nếu cần")
            
            # Lấy nhãn từ oracle (VỚI DỮ LIỆU ĐÃ SCALE)
            print(f"\n🔄 Đang lấy nhãn từ oracle (với dữ liệu đã scale cho Keras Oracle)...")
            y_eval = oracle_client.predict(X_eval_s)
            y_seed = oracle_client.predict(X_seed_s)
            y_val = oracle_client.predict(X_val_s)
    else:
        # Non-Keras Oracle (LightGBM / XGBoost / TabNet / sklearn):
        # - Oracle luôn nhận raw features
        # - Nếu cần normalize (LightGBM/TabNet), oracle sẽ tự xử lý nội bộ
        print(f"\n🔄 Đang lấy nhãn từ oracle ({oracle_model_type.upper()} Oracle - dùng raw features, preprocessing nội bộ nếu cần)...")
        y_eval = oracle_client.predict(X_eval)
        y_seed = oracle_client.predict(X_seed)
        y_val = oracle_client.predict(X_val)
        
        # Với các oracle không phải Keras, KHÔNG scale data khi query oracle
        X_eval_s = X_eval
        X_seed_s = X_seed
        X_val_s = X_val
        # X_unlabeled_pool_s sẽ được tạo sau trong AL loop nếu cần
        
        # Nếu attacker_type là keras/dual/cnn (cần scaled data cho training),
        # cần scale riêng cho surrogate model training sau này
        if attacker_type in ["keras", "dual", "cnn"]:
            print(f"\n⚠️  LƯU Ý: Oracle là {oracle_model_type.upper()} (raw data), nhưng surrogate là {attacker_type} (cần scaled data)")
        elif attacker_type in ["xgb", "tabnet"]:
            # Oracle (lgb/xgb/tabnet/sklearn) làm việc trực tiếp trên raw features.
            # Surrogate xgb/tabnet thường huấn luyện tốt hơn với dữ liệu đã được scale,
            # nên ta chỉ scale bản sao dùng cho training/inference surrogate.
            print(f"\n✅ Oracle là {oracle_model_type.upper()} (raw data), surrogate là {attacker_type} (oracle KHÔNG scale, chỉ surrogate được scale)")
            print(f"   🔄 Sẽ scale dữ liệu RIÊNG cho surrogate model training (không ảnh hưởng đến oracle)")
            scaler = RobustScaler()
            scaler.fit(np.vstack([X_seed, X_val, X_unlabeled_pool]))
            # Tạo scaled version cho surrogate training
            X_eval_s = _clip_scale(scaler, X_eval)
            X_seed_s = _clip_scale(scaler, X_seed)
            X_val_s = _clip_scale(scaler, X_val)
            # X_unlabeled_pool_s sẽ được tạo sau trong AL loop nếu cần
    print(f"✅ Oracle labels retrieved")
    eval_dist = dict(zip(*np.unique(y_eval, return_counts=True)))
    seed_dist = dict(zip(*np.unique(y_seed, return_counts=True)))
    val_dist = dict(zip(*np.unique(y_val, return_counts=True)))
    print(f"  Eval distribution: {eval_dist}")
    print(f"  Seed distribution: {seed_dist}")
    print(f"  Val distribution: {val_dist}")
    
    # QUAN TRỌNG: Đánh giá độ chính xác của oracle với ground truth
    # Điều này giúp giải thích sự khác biệt giữa val_accuracy (vs oracle) và final accuracy (vs ground truth)
    oracle_acc_vs_gt = accuracy_score(y_eval_gt, y_eval)
    print(f"\n📊 Đánh giá Oracle (Target Model):")
    print(f"   Oracle accuracy vs Ground Truth: {oracle_acc_vs_gt:.4f} ({oracle_acc_vs_gt*100:.2f}%)")
    print(f"   ⚠️  LƯU Ý: Val accuracy trong training được tính với oracle labels (không phải ground truth)")
    print(f"   ⚠️  Final accuracy được tính với ground truth labels")
    print(f"   💡 Nếu oracle không chính xác 100%, sẽ có sự khác biệt giữa val_accuracy và final accuracy")
    
    # KIỂM TRA: Nếu oracle predict tất cả là một class, có thể có vấn đề
    all_distributions = [eval_dist, seed_dist, val_dist]
    all_single_class = all(len(d) == 1 for d in all_distributions)
    if all_single_class:
        print(f"\n⚠️  CẢNH BÁO: Oracle đang predict tất cả là một class duy nhất!")
        print(f"   Điều này có thể do:")
        print(f"   1. Oracle threshold quá cao/thấp")
        print(f"   2. Dữ liệu thực sự chỉ có một class")
        print(f"   3. Oracle model có vấn đề")
        if not oracle_client.supports_probabilities():
            print(f"   ℹ️  Oracle không hỗ trợ probabilities -> bỏ qua điều chỉnh threshold tự động.")
        else:
            print(f"   💡 Sẽ thử kiểm tra probabilities và có thể điều chỉnh threshold...")
            try:
                test_sample_size = min(100, X_eval_s.shape[0])
                test_indices = rng.choice(X_eval_s.shape[0], size=test_sample_size, replace=False)
                test_data = X_eval_s[test_indices]
                test_probs = oracle_client.predict_proba(test_data)
                print(f"   📊 Test probabilities trên {test_sample_size} samples:")
                print(f"      Range: [{test_probs.min():.4f}, {test_probs.max():.4f}]")
                print(f"      Mean: {test_probs.mean():.4f}, Median: {np.median(test_probs):.4f}")
                print(f"      Threshold hiện tại: {oracle_client.get_threshold():.4f}")
                
                current_thresh = oracle_client.get_threshold()
                if test_probs.min() < current_thresh < test_probs.max():
                    print(f"   💡 Probabilities có cả dưới và trên threshold - có thể có cả 2 classes")
                    print(f"      Thử với threshold thấp hơn có thể giúp phân biệt tốt hơn")
                elif test_probs.max() < current_thresh:
                    suggested_threshold = np.percentile(test_probs, 50)
                    print(f"   ⚠️  TẤT CẢ probabilities đều dưới threshold {current_thresh}")
                    print(f"   💡 Đề xuất giảm threshold xuống {suggested_threshold:.4f} (median) để phân biệt classes")
                    print(f"   🔄 Đang điều chỉnh threshold...")
                    oracle_client.set_threshold(suggested_threshold)
                    test_predictions_new = oracle_client.predict(X_eval_s[test_indices])
                    test_dist_new = dict(zip(*np.unique(test_predictions_new, return_counts=True)))
                    print(f"   ✅ Với threshold mới {suggested_threshold:.4f}: {test_dist_new}")
                    
                    print(f"   🔄 Re-querying seed, val, eval với threshold mới...")
                    y_eval = oracle_client.predict(X_eval_s)
                    y_seed = oracle_client.predict(X_seed_s)
                    y_val = oracle_client.predict(X_val_s)
                    eval_dist = dict(zip(*np.unique(y_eval, return_counts=True)))
                    seed_dist = dict(zip(*np.unique(y_seed, return_counts=True)))
                    val_dist = dict(zip(*np.unique(y_val, return_counts=True)))
                    print(f"   ✅ Distribution sau khi điều chỉnh threshold:")
                    print(f"      Eval: {eval_dist}")
                    print(f"      Seed: {seed_dist}")
                    print(f"      Val: {val_dist}")
            except Exception as e:
                print(f"   ⚠️  Không thể kiểm tra probabilities: {e}")

    metrics_history = []
    # Khởi tạo labeled_X dựa trên attacker_type
    # KNN, LGB, XGB, và TabNet cần raw data, Keras/Dual/CNN cần scaled data
    if attacker_type in ["lgb", "knn", "xgb", "tabnet"]:
        labeled_X = X_seed  # Raw data cho LGB và KNN
    else:
        labeled_X = X_seed_s  # Scaled data cho Keras/Dual/CNN
    labeled_y = y_seed

    def evaluate(model, round_id: int, total_labels: int):
        probs = np.squeeze(model(X_eval_s), axis=-1)
        
        # Tối ưu threshold dựa trên F1-score với ground truth labels
        # Điều này quan trọng với class imbalance nghiêm trọng
        thresholds = np.linspace(0.1, 0.9, 81)
        best_f1 = -1
        best_threshold = 0.5
        best_preds = (probs >= 0.5).astype(int)
        
        for thresh in thresholds:
            preds_thresh = (probs >= thresh).astype(int)
            _, _, f1_thresh, _ = precision_recall_fscore_support(
                y_eval_gt, preds_thresh, average="binary", zero_division=0
            )
            if f1_thresh > best_f1:
                best_f1 = f1_thresh
                best_threshold = thresh
                best_preds = preds_thresh
        
        # Sử dụng threshold tối ưu
        preds = best_preds
        
        # QUAN TRỌNG: Agreement = so sánh predictions của surrogate với predictions của target model
        # Accuracy = so sánh predictions của surrogate với ground truth labels
        agreement = (preds == y_eval).mean()  # y_eval là predictions từ target model (oracle)
        acc = accuracy_score(y_eval_gt, preds)  # y_eval_gt là ground truth labels
        acc_vs_oracle = accuracy_score(y_eval, preds)  # Accuracy so với oracle (giống agreement nhưng dùng accuracy_score)
        balanced_acc = balanced_accuracy_score(y_eval_gt, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_eval_gt, preds, average="binary", zero_division=0
        )
        try:
            auc = roc_auc_score(y_eval_gt, probs)
        except ValueError:
            auc = float("nan")

        # Tính số queries thực tế (không tính seed và val)
        # Ở round 0, total_labels chỉ là seed_size, nên actual_queries = 0
        # Từ round 1 trở đi, total_labels = seed_size + queries_accumulated
        actual_queries = max(0, total_labels - seed_size)
        
        # Log metrics để giải thích sự khác biệt
        print(f"\n📊 Round {round_id} Evaluation:")
        print(f"   Agreement (vs Oracle): {agreement:.4f} ({agreement*100:.2f}%)")
        print(f"   Accuracy (vs Ground Truth): {acc:.4f} ({acc*100:.2f}%)")
        print(f"   Oracle Accuracy (vs Ground Truth): {oracle_acc_vs_gt:.4f} ({oracle_acc_vs_gt*100:.2f}%)")
        print(f"   💡 Giải thích: Val accuracy trong training ({agreement:.4f}) cao vì so với oracle labels")
        print(f"   💡 Final accuracy ({acc:.4f}) thấp hơn vì so với ground truth (oracle không hoàn hảo)")
        
        metrics = {
            "round": round_id,
            "labels_used": int(total_labels),
            "queries_used": int(actual_queries),  # Số queries thực tế (chỉ tính active learning)
            "optimal_threshold": float(best_threshold),
            "threshold_optimization_method": "f1_on_eval_set",  # Method dùng để tìm threshold
            "threshold_optimization_dataset": "eval_set_with_ground_truth",  # Dataset dùng để tìm threshold
            "surrogate_acc": float(acc),  # Accuracy vs Ground Truth
            "surrogate_acc_vs_oracle": float(acc_vs_oracle),  # Accuracy vs Oracle (tương tự agreement)
            "surrogate_balanced_acc": float(balanced_acc),  # Quan trọng với class imbalance
            "surrogate_auc": float(auc),
            "surrogate_precision": float(precision),
            "surrogate_recall": float(recall),
            "surrogate_f1": float(f1),
            "agreement_with_target": float(agreement),
            "oracle_acc_vs_gt": float(oracle_acc_vs_gt),  # Độ chính xác của oracle với ground truth
        }
        metrics_history.append(metrics)
        return metrics

    # QUAN TRỌNG: Theo nghiên cứu, dùng early_stopping=30 và num_epochs cao (100)
    # để model có đủ thời gian học và tránh underfitting
    # early_stopping=30: patience đủ lớn để vượt qua local minima
    # num_epochs: đủ epochs để model học tốt với nhiều dữ liệu (mặc định 100 theo nghiên cứu)
    if attacker_type == "lgb":
        # LightGBM attacker không cần scale data - đảm bảo labeled_X là raw data
        if labeled_X is X_seed_s:
            labeled_X = X_seed  # Chuyển sang raw data
        # Sử dụng hyperparameters tối ưu để khớp với target model
        attacker = LGBAttacker(seed=seed)
        attacker.train_model(labeled_X, labeled_y, X_val, y_val, boosting_rounds=2000, early_stopping=100)
        # Với LightGBM, không cần scale data để evaluate
        def evaluate_lgb(model, round_id, total_labels):
            probs = model(X_eval)
            # LightGBM predict trả về 1D array hoặc 2D array
            if probs.ndim > 1:
                probs = probs.flatten()
            
            # Tối ưu threshold dựa trên F1-score với ground truth labels
            # Điều này quan trọng với class imbalance nghiêm trọng
            thresholds = np.linspace(0.1, 0.9, 81)
            best_f1 = -1
            best_threshold = 0.5
            best_preds = (probs >= 0.5).astype(int)
            
            for thresh in thresholds:
                preds_thresh = (probs >= thresh).astype(int)
                _, _, f1_thresh, _ = precision_recall_fscore_support(
                    y_eval_gt, preds_thresh, average="binary", zero_division=0
                )
                if f1_thresh > best_f1:
                    best_f1 = f1_thresh
                    best_threshold = thresh
                    best_preds = preds_thresh
            
            # Sử dụng threshold tối ưu
            preds = best_preds
            
            # QUAN TRỌNG: Agreement = so sánh predictions của surrogate với predictions của target model
            # Accuracy = so sánh predictions của surrogate với ground truth labels
            agreement = (preds == y_eval).mean()  # y_eval là predictions từ target model (oracle)
            acc = accuracy_score(y_eval_gt, preds)  # y_eval_gt là ground truth labels
            acc_vs_oracle = accuracy_score(y_eval, preds)  # Accuracy so với oracle (giống agreement nhưng dùng accuracy_score)
            balanced_acc = balanced_accuracy_score(y_eval_gt, preds)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_eval_gt, preds, average="binary", zero_division=0
            )
            try:
                auc = roc_auc_score(y_eval_gt, probs)
            except ValueError:
                auc = float("nan")

            # Tính số queries thực tế (không tính seed, val không được thêm vào pool)
            actual_queries = max(0, total_labels - seed_size)
            
            # Log metrics để giải thích sự khác biệt
            print(f"\n📊 Round {round_id} Evaluation:")
            print(f"   Agreement (vs Oracle): {agreement:.4f} ({agreement*100:.2f}%)")
            print(f"   Accuracy (vs Ground Truth): {acc:.4f} ({acc*100:.2f}%)")
            print(f"   Oracle Accuracy (vs Ground Truth): {oracle_acc_vs_gt:.4f} ({oracle_acc_vs_gt*100:.2f}%)")
            print(f"   💡 Giải thích: Val accuracy trong training ({agreement:.4f}) cao vì so với oracle labels")
            print(f"   💡 Final accuracy ({acc:.4f}) thấp hơn vì so với ground truth (oracle không hoàn hảo)")
            
            metrics = {
                "round": round_id,
                "labels_used": int(total_labels),
                "queries_used": int(actual_queries),  # Số queries thực tế (chỉ tính active learning)
                "optimal_threshold": float(best_threshold),
                "threshold_optimization_method": "f1_on_eval_set",  # Method dùng để tìm threshold
                "threshold_optimization_dataset": "eval_set_with_ground_truth",  # Dataset dùng để tìm threshold
                "surrogate_acc": float(acc),  # Accuracy vs Ground Truth
                "surrogate_acc_vs_oracle": float(acc_vs_oracle),  # Accuracy vs Oracle (tương tự agreement)
                "surrogate_balanced_acc": float(balanced_acc),  # Quan trọng với class imbalance
                "surrogate_auc": float(auc),
                "surrogate_precision": float(precision),
                "surrogate_recall": float(recall),
                "surrogate_f1": float(f1),
                "agreement_with_target": float(agreement),
                "oracle_acc_vs_gt": float(oracle_acc_vs_gt),  # Độ chính xác của oracle với ground truth
            }
            metrics_history.append(metrics)
            return metrics
        
        evaluate = evaluate_lgb
        evaluate(attacker, round_id=0, total_labels=labeled_X.shape[0])
    elif attacker_type == "dual":
        # DualDNN attacker cần scale data và cả ground truth labels (oracle predictions)
        # QUAN TRỌNG: Sử dụng surrogate_feature_dim (từ target model), không phải dataset attack
        attacker = KerasDualAttacker(early_stopping=30, seed=seed, input_shape=(surrogate_feature_dim,))
        # DualDNN train với (X, y_true) - y_true là oracle labels
        attacker.train_model(labeled_X, labeled_y, labeled_y, X_val_s, y_val, y_val, num_epochs=num_epochs)
        
        def evaluate_dual(model, round_id, total_labels):
            # DualDNN cần cả X và y_true (oracle labels) khi predict
            # __call__ nhận 2 tham số riêng biệt (X, y_true), không phải tuple
            probs = np.squeeze(model(X_eval_s, y_eval), axis=-1)
            
            # Tối ưu threshold hoặc sử dụng threshold cố định
            if fixed_threshold is not None:
                # Sử dụng threshold cố định
                best_threshold = fixed_threshold
                preds = (probs >= best_threshold).astype(int)
                print(f"   🔧 Sử dụng threshold cố định: {best_threshold:.3f}")
            else:
                # Tối ưu threshold dựa trên metric được chọn
                thresholds = np.linspace(0.1, 0.9, 81)
                best_metric_value = -1
                best_threshold = 0.5
                best_preds = (probs >= 0.5).astype(int)
                
                for thresh in thresholds:
                    preds_thresh = (probs >= thresh).astype(int)
                    
                    # Tính metric dựa trên metric được chọn
                    if threshold_optimization_metric == "f1":
                        _, _, metric_value, _ = precision_recall_fscore_support(
                            y_eval_gt, preds_thresh, average="binary", zero_division=0
                        )
                    elif threshold_optimization_metric == "accuracy":
                        metric_value = accuracy_score(y_eval_gt, preds_thresh)
                    elif threshold_optimization_metric == "balanced_accuracy":
                        metric_value = balanced_accuracy_score(y_eval_gt, preds_thresh)
                    else:
                        raise ValueError(
                            f"Unknown threshold_optimization_metric: {threshold_optimization_metric}. "
                            f"Chọn một trong: 'f1', 'accuracy', 'balanced_accuracy'"
                        )
                    
                    if metric_value > best_metric_value:
                        best_metric_value = metric_value
                        best_threshold = thresh
                        best_preds = preds_thresh
                
                # Sử dụng threshold tối ưu
                preds = best_preds
                print(f"   🔧 Threshold tối ưu ({threshold_optimization_metric}): {best_threshold:.3f} (metric = {best_metric_value:.4f})")
            
            # Agreement và accuracy metrics
            agreement = (preds == y_eval).mean()
            acc = accuracy_score(y_eval_gt, preds)
            acc_vs_oracle = accuracy_score(y_eval, preds)
            balanced_acc = balanced_accuracy_score(y_eval_gt, preds)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_eval_gt, preds, average="binary", zero_division=0
            )
            try:
                auc = roc_auc_score(y_eval_gt, probs)
            except ValueError:
                auc = float("nan")
            
            # Tính số queries thực tế (không tính seed, val không được thêm vào pool)
            actual_queries = max(0, total_labels - seed_size)
            
            print(f"\n📊 Round {round_id} Evaluation (DualDNN):")
            print(f"   Agreement (vs Oracle): {agreement:.4f} ({agreement*100:.2f}%)")
            print(f"   Accuracy (vs Ground Truth): {acc:.4f} ({acc*100:.2f}%)")
            print(f"   Oracle Accuracy (vs Ground Truth): {oracle_acc_vs_gt:.4f} ({oracle_acc_vs_gt*100:.2f}%)")
            
            metrics = {
                "round": round_id,
                "labels_used": int(total_labels),
                "queries_used": int(actual_queries),
                "optimal_threshold": float(best_threshold),
                "threshold_optimization_method": "f1_on_eval_set",  # Method dùng để tìm threshold
                "threshold_optimization_dataset": "eval_set_with_ground_truth",  # Dataset dùng để tìm threshold
                "surrogate_acc": float(acc),
                "surrogate_acc_vs_oracle": float(acc_vs_oracle),
                "surrogate_balanced_acc": float(balanced_acc),
                "surrogate_auc": float(auc),
                "surrogate_precision": float(precision),
                "surrogate_recall": float(recall),
                "surrogate_f1": float(f1),
                "agreement_with_target": float(agreement),
                "oracle_acc_vs_gt": float(oracle_acc_vs_gt),
            }
            metrics_history.append(metrics)
            return metrics
        
        evaluate = evaluate_dual
        evaluate(attacker, round_id=0, total_labels=labeled_X.shape[0])
    elif attacker_type == "cnn":
        # CNN attacker cần scale data và reshape thành (n_samples, n_features, 1)
        # QUAN TRỌNG: Sử dụng surrogate_feature_dim (từ target model), không phải dataset attack
        attacker = CNNAttacker(early_stopping=30, seed=seed, input_shape=(surrogate_feature_dim, 1))
        attacker.train_model(labeled_X, labeled_y, X_val_s, y_val, num_epochs=num_epochs)
        # CNN dùng cùng evaluate function như keras (dùng scaled data và np.squeeze)
        evaluate(attacker, round_id=0, total_labels=labeled_X.shape[0])
    elif attacker_type == "knn":
        # KNN attacker dùng raw data để phù hợp với sklearn
        # Đảm bảo labeled_X là raw data
        if labeled_X is X_seed_s:
            labeled_X = X_seed  # Chuyển sang raw data
        # Lưu ý: KNN không có validation set trong training, nhưng vẫn cần X_val, y_val cho evaluate
        attacker = KNNAttacker(seed=seed)
        # KNN train với raw data
        attacker.train_model(labeled_X, labeled_y, X_val, y_val)
        
        def evaluate_knn(model, round_id, total_labels):
            # KNN có thể dùng raw hoặc scaled data - dùng raw để nhất quán
            probs = model(X_eval)
            # KNN predict trả về 1D array
            if probs.ndim > 1:
                probs = probs.flatten()
            
            # Tối ưu threshold dựa trên F1-score với ground truth labels
            thresholds = np.linspace(0.1, 0.9, 81)
            best_f1 = -1
            best_threshold = 0.5
            best_preds = (probs >= 0.5).astype(int)
            
            for thresh in thresholds:
                preds_thresh = (probs >= thresh).astype(int)
                _, _, f1_thresh, _ = precision_recall_fscore_support(
                    y_eval_gt, preds_thresh, average="binary", zero_division=0
                )
                if f1_thresh > best_f1:
                    best_f1 = f1_thresh
                    best_threshold = thresh
                    best_preds = preds_thresh
            
            preds = best_preds
            
            agreement = (preds == y_eval).mean()
            acc = accuracy_score(y_eval_gt, preds)
            acc_vs_oracle = accuracy_score(y_eval, preds)
            balanced_acc = balanced_accuracy_score(y_eval_gt, preds)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_eval_gt, preds, average="binary", zero_division=0
            )
            try:
                auc = roc_auc_score(y_eval_gt, probs)
            except ValueError:
                auc = float("nan")
            
            actual_queries = max(0, total_labels - seed_size)
            
            print(f"\n📊 Round {round_id} Evaluation (KNN):")
            print(f"   Agreement (vs Oracle): {agreement:.4f} ({agreement*100:.2f}%)")
            print(f"   Accuracy (vs Ground Truth): {acc:.4f} ({acc*100:.2f}%)")
            print(f"   Oracle Accuracy (vs Ground Truth): {oracle_acc_vs_gt:.4f} ({oracle_acc_vs_gt*100:.2f}%)")
            
            metrics = {
                "round": round_id,
                "labels_used": int(total_labels),
                "queries_used": int(actual_queries),
                "optimal_threshold": float(best_threshold),
                "threshold_optimization_method": "f1_on_eval_set",
                "threshold_optimization_dataset": "eval_set_with_ground_truth",
                "surrogate_acc": float(acc),
                "surrogate_acc_vs_oracle": float(acc_vs_oracle),
                "surrogate_balanced_acc": float(balanced_acc),
                "surrogate_auc": float(auc),
                "surrogate_precision": float(precision),
                "surrogate_recall": float(recall),
                "surrogate_f1": float(f1),
                "agreement_with_target": float(agreement),
                "oracle_acc_vs_gt": float(oracle_acc_vs_gt),
            }
            metrics_history.append(metrics)
            return metrics
        
        evaluate = evaluate_knn
        evaluate(attacker, round_id=0, total_labels=labeled_X.shape[0])
    elif attacker_type == "xgb":
        # XGBoost attacker dùng raw data (giống LGB)
        # Đảm bảo labeled_X là raw data
        if labeled_X is X_seed_s:
            labeled_X = X_seed  # Chuyển sang raw data
        attacker = XGBoostAttacker(seed=seed)
        attacker.train_model(labeled_X, labeled_y, X_val, y_val, boosting_rounds=200, early_stopping=20)
        
        # XGBoost dùng raw data để evaluate (giống LGB)
        def evaluate_xgb(model, round_id, total_labels):
            probs = model(X_eval)
            # XGBoost predict trả về 1D array
            if probs.ndim > 1:
                probs = probs.flatten()
            
            # Tối ưu threshold dựa trên F1-score với ground truth labels
            thresholds = np.linspace(0.1, 0.9, 81)
            best_f1 = -1
            best_threshold = 0.5
            best_preds = (probs >= 0.5).astype(int)
            
            for thresh in thresholds:
                preds_thresh = (probs >= thresh).astype(int)
                _, _, f1_thresh, _ = precision_recall_fscore_support(
                    y_eval_gt, preds_thresh, average="binary", zero_division=0
                )
                if f1_thresh > best_f1:
                    best_f1 = f1_thresh
                    best_threshold = thresh
                    best_preds = preds_thresh
            
            preds = best_preds
            
            agreement = (preds == y_eval).mean()
            acc = accuracy_score(y_eval_gt, preds)
            acc_vs_oracle = accuracy_score(y_eval, preds)
            balanced_acc = balanced_accuracy_score(y_eval_gt, preds)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_eval_gt, preds, average="binary", zero_division=0
            )
            
            try:
                auc = roc_auc_score(y_eval_gt, probs)
            except ValueError:
                auc = float("nan")
            
            metrics = {
                "round": round_id,
                "labels_used": total_labels,
                "surrogate_acc": float(acc),
                "surrogate_balanced_acc": float(balanced_acc),
                "surrogate_auc": float(auc),
                "surrogate_precision": float(precision),
                "surrogate_recall": float(recall),
                "surrogate_f1": float(f1),
                "agreement_with_target": float(agreement),
                "optimal_threshold": float(best_threshold),
                "oracle_acc_vs_gt": float(oracle_acc_vs_gt),
            }
            metrics_history.append(metrics)
            return metrics
        
        evaluate = evaluate_xgb
        evaluate(attacker, round_id=0, total_labels=labeled_X.shape[0])
    elif attacker_type == "tabnet":
        # TabNet attacker dùng raw data (giống LGB và XGB)
        # Đảm bảo labeled_X là raw data
        if labeled_X is X_seed_s:
            labeled_X = X_seed  # Chuyển sang raw data
        attacker = TabNetAttacker(seed=seed)
        # Tăng số epoch và patience để TabNet học tốt hơn (nhất là khi có nhiều queries)
        attacker.train_model(
            labeled_X,
            labeled_y,
            X_val,
            y_val,
            max_epochs=100,
            patience=100000,  # effectively disable early stopping
            batch_size=2048,
        )
        
        # TabNet dùng raw data để evaluate (giống LGB và XGB)
        def evaluate_tabnet(model, round_id, total_labels):
            probs = model(X_eval)
            # TabNet predict_proba trả về 1D hoặc 2D array
            if probs.ndim > 1:
                probs = probs.flatten()
            
            # Tối ưu threshold dựa trên F1-score với ground truth labels
            thresholds = np.linspace(0.1, 0.9, 81)
            best_f1 = -1
            best_threshold = 0.5
            best_preds = (probs >= 0.5).astype(int)
            
            for thresh in thresholds:
                preds_thresh = (probs >= thresh).astype(int)
                _, _, f1_thresh, _ = precision_recall_fscore_support(
                    y_eval_gt, preds_thresh, average="binary", zero_division=0
                )
                if f1_thresh > best_f1:
                    best_f1 = f1_thresh
                    best_threshold = thresh
                    best_preds = preds_thresh
            
            preds = best_preds
            
            agreement = (preds == y_eval).mean()
            acc = accuracy_score(y_eval_gt, preds)
            acc_vs_oracle = accuracy_score(y_eval, preds)
            balanced_acc = balanced_accuracy_score(y_eval_gt, preds)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_eval_gt, preds, average="binary", zero_division=0
            )
            
            try:
                auc = roc_auc_score(y_eval_gt, probs)
            except ValueError:
                auc = float("nan")
            
            metrics = {
                "round": round_id,
                "labels_used": total_labels,
                "surrogate_acc": float(acc),
                "surrogate_balanced_acc": float(balanced_acc),
                "surrogate_auc": float(auc),
                "surrogate_precision": float(precision),
                "surrogate_recall": float(recall),
                "surrogate_f1": float(f1),
                "agreement_with_target": float(agreement),
                "optimal_threshold": float(best_threshold),
                "oracle_acc_vs_gt": float(oracle_acc_vs_gt),
            }
            metrics_history.append(metrics)
            return metrics
        
        evaluate = evaluate_tabnet
        evaluate(attacker, round_id=0, total_labels=labeled_X.shape[0])
    else:
        # Keras attacker cần scale data
        # QUAN TRỌNG: Sử dụng surrogate_feature_dim (từ target model), không phải dataset attack
        attacker = KerasAttacker(early_stopping=30, seed=seed, input_shape=(surrogate_feature_dim,))
        attacker.train_model(labeled_X, labeled_y, X_val_s, y_val, num_epochs=num_epochs)
        evaluate(attacker, round_id=0, total_labels=labeled_X.shape[0])

    # Track tổng queries để đảm bảo chính xác
    # QUAN TRỌNG: Query budget = seed + val + AL queries
    AL_queries_target = query_batch * num_rounds
    total_queries_target = seed_size + val_size + AL_queries_target  # Tổng query budget
    total_queries_accumulated = seed_size + val_size  # Seed và val đã được query
    # Cho phép lệch tối đa 10% (dư chứ không được thiếu)
    min_queries_acceptable = int(total_queries_target * 0.9)  # Ít nhất 90% của target
    max_queries_acceptable = int(total_queries_target * 1.1)  # Tối đa 110% của target
    
    print(f"\n📋 Query Budget Tracking:")
    print(f"   Seed queries: {seed_size:,} (đã query)")
    print(f"   Val queries: {val_size:,} (đã query)")
    print(f"   AL queries target: {AL_queries_target:,} ({query_batch:,} queries/round × {num_rounds} rounds)")
    print(f"   Total query budget: {total_queries_target:,} (seed + val + AL)")
    print(f"   Queries đã dùng: {total_queries_accumulated:,} (seed + val)")
    print(f"   Queries còn lại: {AL_queries_target:,} (AL queries)")
    print(f"   📊 Cho phép lệch: {min_queries_acceptable:,} - {max_queries_acceptable:,} queries (90% - 110%)")
    print(f"   ⚠️  Quan trọng: Không được thiếu queries! (tối thiểu {min_queries_acceptable:,})")
    
    # Kiểm tra unlabeled pool có đủ không
    if X_unlabeled_pool.shape[0] < AL_queries_target:
        print(f"\n⚠️  CẢNH BÁO: Unlabeled pool ({X_unlabeled_pool.shape[0]:,}) < AL queries dự kiến ({AL_queries_target:,})")
        print(f"   💡 Unlabeled pool sẽ cạn kiệt trước khi đạt đủ queries. Sẽ cố gắng lấy tối đa có thể.")
    
    # Scale unlabeled pool nếu cần (cho oracle query và attacker training)
    X_unlabeled_pool_s = None
    if oracle_model_type == "h5" or attacker_type in ["keras", "dual", "cnn"]:
        # Cần scale cho oracle (nếu h5) hoặc attacker (nếu keras/dual/cnn)
        if scaler is None:
            # Tạo scaler từ seed, val, và unlabeled pool
            scaler = RobustScaler()
            scaler.fit(np.vstack([X_seed, X_val, X_unlabeled_pool]))
        X_unlabeled_pool_s = _clip_scale(scaler, X_unlabeled_pool)
    
    for query_round in range(1, num_rounds + 1):
        # Kiểm tra xem còn cần bao nhiêu AL queries nữa
        AL_queries_remaining_needed = AL_queries_target - (total_queries_accumulated - seed_size - val_size)
        
        # Nếu đã đạt đủ AL queries, dừng lại
        if total_queries_accumulated >= total_queries_target:
            print(f"\n✅ Đã đạt đủ query budget ({total_queries_target:,}). Dừng active learning.")
            break
        
        # Nếu unlabeled pool còn lại ít hơn query_batch, vẫn cố gắng lấy tối đa có thể
        unlabeled_pool_remaining = X_unlabeled_pool.shape[0]
        queries_to_get_this_round = min(query_batch, unlabeled_pool_remaining, AL_queries_remaining_needed)
        
        if queries_to_get_this_round <= 0:
            print(f"\n⚠️  Round {query_round}: Không còn queries để lấy. Unlabeled pool: {unlabeled_pool_remaining}, Cần: {AL_queries_remaining_needed}")
            break
        
        if unlabeled_pool_remaining < query_batch:
            print(f"\n⚠️  Round {query_round}: Unlabeled pool còn lại ({unlabeled_pool_remaining}) < query_batch ({query_batch}).")
            print(f"   🔄 Sẽ lấy tối đa {queries_to_get_this_round} queries từ unlabeled pool còn lại.")
        
        # CẢI TIẾN: Stratified Entropy Sampling với Pre-filtering bằng Thief Dataset Labels
        # Giả định: Mẫu trong thief dataset đã biết nhãn, mẫu tương tự trong pool sẽ có nhãn tương tự
        # Sử dụng labels của thief dataset để pre-filter pool trước khi query oracle
        # Sau đó query oracle để xác nhận labels thực tế
        # Vẫn giữ logic cân bằng class
        print(f"\n🔄 Round {query_round}: Đang chọn queries bằng Stratified Entropy Sampling với Pre-filtering (thief dataset labels)...")
        
        # QUAN TRỌNG: Tách riêng unlabeled pool để query oracle và pool để train attacker
        # - Pool để query oracle: dựa trên oracle_model_type (raw data cho LightGBM, scaled cho Keras)
        # - Pool để train attacker: dựa trên attacker_type (scaled cho keras/dual, raw cho lgb)
        # Oracle query PHẢI dùng data phù hợp với oracle model, không phải attacker model!
        
        # Unlabeled pool để query oracle - dựa trên oracle_model_type
        # QUAN TRỌNG: Nếu oracle có normalization stats, dùng raw data (oracle tự normalize)
        if oracle_model_type == "h5":
            if oracle_has_normalization_stats:
                # Keras Oracle có normalization stats: dùng raw data (oracle tự normalize và clip)
                pool_for_oracle = X_unlabeled_pool
            else:
                # Keras Oracle không có normalization stats: cần scaled data
                pool_for_oracle = X_unlabeled_pool_s if X_unlabeled_pool_s is not None else X_unlabeled_pool
        else:
            # LightGBM Oracle: cần raw data (KHÔNG scale!)
            pool_for_oracle = X_unlabeled_pool
        
        # Unlabeled pool để train attacker - dựa trên attacker_type
        # xgb và tabnet không cần scaling (giống lgb và knn)
        if attacker_type in ["keras", "dual", "cnn"]:
            # Keras/Dual/CNN attacker: cần scaled data
            pool_for_entropy = X_unlabeled_pool_s if X_unlabeled_pool_s is not None else X_unlabeled_pool
        else:
            # LightGBM/KNN attacker: cần raw data
            pool_for_entropy = X_unlabeled_pool
        
        pool_size = pool_for_oracle.shape[0]  # Dùng pool_for_oracle để pre-filter
        
        # BƯỚC 1: Pre-filtering dựa trên labels của thief dataset
        # Sử dụng y_unlabeled_pool_gt (labels từ thief dataset) để chọn pool cân bằng TRƯỚC khi query oracle
        print(f"   🔄 Pre-filtering unlabeled pool dựa trên labels của thief dataset...")
        pool_dist_gt_current = dict(zip(*np.unique(y_unlabeled_pool_gt, return_counts=True)))
        print(f"   📊 Unlabeled pool distribution (thief dataset labels): {pool_dist_gt_current}")
        
        # Chọn subset từ pool dựa trên labels của thief dataset để đảm bảo cân bằng
        # Mục tiêu: Chọn đủ samples từ mỗi class để có thể chọn queries cân bằng sau này
        query_pool_size = min(pool_size, max(20000, queries_to_get_this_round * 10))
        
        # Stratified sampling từ pool dựa trên thief dataset labels
        # Lấy 50% từ mỗi class (nếu có đủ)
        queries_per_class_for_pool = query_pool_size // 2
        
        pool_query_idx_list = []
        for class_label in [0, 1]:
            class_indices_in_pool = np.where(y_unlabeled_pool_gt == class_label)[0]
            if len(class_indices_in_pool) == 0:
                continue
            
            # Lấy tối đa queries_per_class_for_pool từ class này
            n_select_from_class = min(queries_per_class_for_pool, len(class_indices_in_pool))
            selected_indices = rng.choice(class_indices_in_pool, size=n_select_from_class, replace=False)
            pool_query_idx_list.append(selected_indices)
        
        if len(pool_query_idx_list) > 0:
            # Kết hợp indices từ cả 2 classes
            pool_query_idx = np.concatenate(pool_query_idx_list)
            rng.shuffle(pool_query_idx)  # Shuffle để trộn classes
        else:
            # Fallback: Nếu không có class nào, dùng toàn bộ unlabeled pool
            pool_query_idx = np.arange(pool_size)
        
        # Đảm bảo không vượt quá query_pool_size
        if len(pool_query_idx) > query_pool_size:
            pool_query_idx = pool_query_idx[:query_pool_size]
        
        # Lấy data từ pool_for_oracle (raw/scaled tùy oracle model type) để query oracle
        # QUAN TRỌNG: Oracle query PHẢI dùng pool_for_oracle, không phải pool_for_entropy!
        X_pool_query = pool_for_oracle[pool_query_idx]
        y_pool_query_gt = y_unlabeled_pool_gt[pool_query_idx]  # Labels từ thief dataset (ground truth của unlabeled pool)
        
        # Log distribution sau pre-filtering
        pool_query_dist_gt = dict(zip(*np.unique(y_pool_query_gt, return_counts=True)))
        print(f"   ✅ Pre-filtered pool: {len(pool_query_idx)} samples (from {pool_size} total pool)")
        print(f"   📊 Pre-filtered distribution (thief dataset labels): {pool_query_dist_gt}")
        print(f"   🔍 Using {'scaled' if oracle_model_type == 'h5' else 'raw'} data for oracle query (oracle is {oracle_model_type.upper()})")
        
        # BƯỚC 2: Query oracle để lấy labels thực tế từ target model
        # Điều này xác nhận labels thực tế, có thể khác với thief dataset labels
        # QUAN TRỌNG: Oracle query dùng X_pool_query từ pool_for_oracle (đúng data type cho oracle)
        print(f"   🔄 Querying oracle để lấy labels thực tế từ target model...")
        y_pool_query = oracle_client.predict(X_pool_query)
        pool_query_dist = dict(zip(*np.unique(y_pool_query, return_counts=True)))
        print(f"   📊 Pool distribution (oracle labels): {pool_query_dist}")
        
        # So sánh labels từ thief dataset vs oracle
        agreement_thief_oracle = np.mean(y_pool_query_gt == y_pool_query)
        print(f"   📊 Agreement (thief labels vs oracle labels): {agreement_thief_oracle:.4f} ({agreement_thief_oracle*100:.2f}%)")
        if agreement_thief_oracle < 0.7:
            print(f"   ⚠️  WARNING: Thief labels và oracle labels khác nhau nhiều (>30%)")
            print(f"   💡 Pre-filtering dựa trên thief labels có thể không chính xác, nhưng vẫn dùng oracle labels cho chọn queries")
        else:
            print(f"   ✅ Thief labels và oracle labels khá khớp - pre-filtering hiệu quả")
        
        # BƯỚC 3: Tính entropy cho tất cả samples trong pool đã query
        # QUAN TRỌNG: Sử dụng oracle labels (y_pool_query) để chọn queries, không phải thief labels
        # vì chúng ta cần labels thực tế từ target model để đảm bảo accuracy
        # Với dualDNN, cần oracle labels cho entropy sampling
        
        # QUAN TRỌNG: Để tính entropy cho attacker, cần dùng pool_for_entropy (scaled cho keras/dual/cnn)
        # Nhưng X_pool_query là từ pool_for_oracle (raw cho LightGBM oracle)
        # Cần map về pool_for_entropy để tính entropy đúng
        # xgb và tabnet không cần scaling (giống lgb và knn)
        if attacker_type in ["keras", "dual", "cnn"] and oracle_model_type == "lgb":
            # Oracle là LightGBM (raw), nhưng attacker là keras/dual/cnn (cần scaled)
            # Cần lấy scaled version của X_pool_query để tính entropy
            X_pool_query_for_entropy = pool_for_entropy[pool_query_idx]
        else:
            # Oracle và attacker cùng data type, dùng X_pool_query trực tiếp
            X_pool_query_for_entropy = X_pool_query
        
        pool_labels_for_entropy = y_pool_query if attacker_type == "dual" else np.zeros(X_pool_query.shape[0])
        dual_flag = (attacker_type == "dual")
        
        # Tính entropy cho tất cả samples
        entropy_candidates = X_pool_query.shape[0]
        q_idx_all = entropy_sampling(
            attacker, 
            X_pool_query_for_entropy,  # Dùng scaled data nếu attacker là keras/dual
            pool_labels_for_entropy,
            n_instances=entropy_candidates,
            dual=dual_flag
        )
        
        # BƯỚC 4: Chọn queries cân bằng từ mỗi class dựa trên oracle labels
        # Mục tiêu: 50% class 0, 50% class 1 (hoặc tỷ lệ gần nhất có thể)
        queries_per_class = queries_to_get_this_round // 2
        query_idx_list = []
        
        for class_label in [0, 1]:
            # Lọc indices của class này trong q_idx_all
            # q_idx_all là indices trong X_pool_query, đã được sắp xếp theo entropy giảm dần
            class_mask = y_pool_query[q_idx_all] == class_label
            class_indices_in_q = np.where(class_mask)[0]  # Indices trong q_idx_all
            
            if len(class_indices_in_q) == 0:
                print(f"   ⚠️  Không tìm thấy class {class_label} trong pool")
                continue
            
            # Chọn queries_per_class samples có entropy cao nhất từ class này
            # (q_idx_all đã được sắp xếp theo entropy giảm dần)
            n_select = min(queries_per_class, len(class_indices_in_q))
            selected_indices_in_q = class_indices_in_q[:n_select]
            
            # Map từ indices trong q_idx_all sang indices trong X_pool_query
            # q_idx_all là indices trong X_pool_query (đã được sắp xếp theo entropy)
            selected_indices_in_pool_query = q_idx_all[selected_indices_in_q]
            
            # Map về indices trong pool gốc (pool_for_entropy)
            # pool_query_idx là indices trong pool gốc đã được pre-filter
            # selected_indices_in_pool_query là indices trong X_pool_query (subset)
            selected_pool_indices = pool_query_idx[selected_indices_in_pool_query]
            
            query_idx_list.append(selected_pool_indices)
            print(f"   ✅ Class {class_label}: Chọn {n_select}/{len(class_indices_in_q)} samples (entropy cao nhất)")
        
        # Kết hợp queries từ cả 2 classes
        if len(query_idx_list) > 0:
            # query_idx_list chứa indices trong unlabeled pool gốc (đã được map từ pool_query_idx)
            query_idx_in_unlabeled_pool = np.concatenate(query_idx_list)
        else:
            # Fallback: Nếu không có class nào, dùng entropy sampling thông thường từ pool_query
            print(f"   ⚠️  Fallback: Dùng entropy sampling thông thường từ pool_query (không có class nào)")
            entropy_candidates = min(10000, X_pool_query_for_entropy.shape[0])
            
            # Với dualDNN, dùng labels đã query từ oracle
            pool_labels_for_entropy = y_pool_query if attacker_type == "dual" else np.zeros(X_pool_query_for_entropy.shape[0])
            
            # Tính entropy trên pool_query (đã query oracle)
            q_idx = entropy_sampling(
                attacker, 
                X_pool_query_for_entropy, 
                pool_labels_for_entropy,
                n_instances=entropy_candidates,
                dual=dual_flag
            )
            X_med = X_pool_query_for_entropy[q_idx]
            num_clusters = min(queries_to_get_this_round, X_med.shape[0])
            if num_clusters > 0:
                kmed = KMedoids(n_clusters=num_clusters, init='k-medoids++', random_state=seed)
                kmed.fit(X_med)
                query_idx_in_med = kmed.medoid_indices_
                query_idx_in_pool_query = q_idx[query_idx_in_med]
            else:
                query_idx_in_pool_query = q_idx[:min(queries_to_get_this_round, len(q_idx))]
            
            # Map về indices trong unlabeled pool gốc
            query_idx_in_unlabeled_pool = pool_query_idx[query_idx_in_pool_query]
        
        # Đảm bảo không vượt quá queries_to_get_this_round
        if len(query_idx_in_unlabeled_pool) > queries_to_get_this_round:
            query_idx_in_unlabeled_pool = query_idx_in_unlabeled_pool[:queries_to_get_this_round]
        
        print(f"   ✅ Đã chọn {len(query_idx_in_unlabeled_pool)} queries (target: {queries_to_get_this_round})")

        # Lấy data và labels cho queries đã chọn
        # query_idx_in_unlabeled_pool là indices trong unlabeled pool gốc
        # Cần map về indices trong pool_query_idx để lấy labels từ y_pool_query
        
        # Tìm vị trí của query_idx_in_unlabeled_pool trong pool_query_idx
        # pool_query_idx là indices trong unlabeled pool gốc (đã được pre-filter)
        sorted_pool_query_idx = np.argsort(pool_query_idx)
        sorted_pool_query_values = pool_query_idx[sorted_pool_query_idx]
        positions = np.searchsorted(sorted_pool_query_values, query_idx_in_unlabeled_pool, side='left')
        valid_mask = (positions < len(sorted_pool_query_values)) & (sorted_pool_query_values[positions] == query_idx_in_unlabeled_pool)
        
        if not np.all(valid_mask):
            # Fallback: Query oracle trực tiếp cho các mẫu đã chọn
            print(f"   ⚠️  Một số queries không có trong pool_query, sẽ query oracle trực tiếp...")
            if oracle_model_type == "h5":
                X_query_for_oracle = X_unlabeled_pool_s[query_idx_in_unlabeled_pool] if X_unlabeled_pool_s is not None else X_unlabeled_pool[query_idx_in_unlabeled_pool]
            else:
                X_query_for_oracle = X_unlabeled_pool[query_idx_in_unlabeled_pool]
            y_query = oracle_client.predict(X_query_for_oracle)
        else:
            # Lấy labels từ y_pool_query (đã query oracle trong bước pre-filtering)
            query_positions_in_pool_query = sorted_pool_query_idx[positions]
            y_query = y_pool_query[query_positions_in_pool_query]
        
        # Lấy data phù hợp với attacker_type (scaled cho keras/dual/cnn, raw cho lgb/knn/xgb/tabnet)
        if attacker_type in ["keras", "dual", "cnn"]:
            # Attacker cần scaled data
            X_query_s = X_unlabeled_pool_s[query_idx_in_unlabeled_pool] if X_unlabeled_pool_s is not None else X_unlabeled_pool[query_idx_in_unlabeled_pool]
        else:
            # Attacker cần raw data (lgb, knn)
            X_query_s = X_unlabeled_pool[query_idx_in_unlabeled_pool]

        # Log class distribution để kiểm tra
        query_dist = dict(zip(*np.unique(y_query, return_counts=True)))
        print(f"   📊 Query distribution (sau stratified sampling): {query_dist}")
        
        # KIỂM TRA VÀ CÂN BẰNG CLASS DISTRIBUTION
        # Theo nghiên cứu: Class imbalance có thể làm model bias về class đa số
        # Giải pháp: Đảm bảo mỗi class có ít nhất 30% queries (hoặc tối thiểu 100 samples)
        total_queries = len(y_query)
        if total_queries > 0:
            max_class_ratio = max(query_dist.values()) / total_queries
            min_class_samples = min(query_dist.values()) if len(query_dist) > 1 else 0
            min_required_samples = max(100, int(query_batch * 0.3))  # Tối thiểu 30% hoặc 100 samples
            
            if max_class_ratio > 0.7 or min_class_samples < min_required_samples:
                print(f"   ⚠️  Class imbalance: Một class chiếm {max_class_ratio*100:.1f}%, class thiểu số có {min_class_samples} samples")
                print(f"   💡 Cần tối thiểu {min_required_samples} samples cho mỗi class")
                
                # Cân bằng bằng cách lấy thêm samples từ class thiểu số
                if len(query_dist) == 2:
                    minority_class = min(query_dist.items(), key=lambda x: x[1])[0]
                    majority_class = max(query_dist.items(), key=lambda x: x[1])[0]
                    minority_count = query_dist[minority_class]
                    
                    if minority_count < min_required_samples:
                        needed_samples = min_required_samples - minority_count
                        print(f"   🔄 Cần thêm {needed_samples} samples từ class {minority_class}...")
                        
                        # Query oracle trên toàn bộ unlabeled pool còn lại để tìm class thiểu số
                        remaining_pool_size = X_unlabeled_pool.shape[0]
                        if remaining_pool_size > needed_samples:
                            # Tăng sample_size để tìm đủ class thiểu số (có thể pool chủ yếu là class đa số)
                            # Ước tính: nếu class thiểu số chiếm ~10%, cần query ~10x để tìm đủ
                            sample_size = min(needed_samples * 10, remaining_pool_size)
                            candidate_idx = rng.choice(remaining_pool_size, size=sample_size, replace=False)
                            
                            # Lấy data phù hợp với oracle type
                            if oracle_model_type == "h5":
                                X_candidates = X_unlabeled_pool_s[candidate_idx] if X_unlabeled_pool_s is not None else X_unlabeled_pool[candidate_idx]
                            else:
                                X_candidates = X_unlabeled_pool[candidate_idx]
                            
                            y_candidates = oracle_client.predict(X_candidates)
                            
                            # Lọc chỉ lấy class thiểu số
                            minority_mask = y_candidates == minority_class
                            minority_found = np.sum(minority_mask)
                            
                            if minority_found >= needed_samples:
                                # Lấy đủ samples từ class thiểu số
                                minority_candidate_idx = candidate_idx[minority_mask][:needed_samples]
                                
                                # Lấy data phù hợp với attacker type
                                if attacker_type in ["keras", "dual", "cnn"]:
                                    X_additional = X_unlabeled_pool_s[minority_candidate_idx] if X_unlabeled_pool_s is not None else X_unlabeled_pool[minority_candidate_idx]
                                else:
                                    X_additional = X_unlabeled_pool[minority_candidate_idx]
                                
                                y_additional = y_candidates[minority_mask][:needed_samples]
                                
                                X_query_s = np.vstack([X_query_s, X_additional])
                                y_query = np.concatenate([y_query, y_additional])
                                # Thêm vào query_idx_in_unlabeled_pool
                                query_idx_in_unlabeled_pool = np.concatenate([query_idx_in_unlabeled_pool, minority_candidate_idx])
                                
                                balanced_dist = dict(zip(*np.unique(y_query, return_counts=True)))
                                print(f"   ✅ Đã cân bằng: {balanced_dist}")
                            else:
                                print(f"   ⚠️  Chỉ tìm thấy {minority_found}/{needed_samples} samples từ class {minority_class}")
                                if minority_found > 0:
                                    minority_candidate_idx = candidate_idx[minority_mask]
                                    
                                    # Lấy data phù hợp với attacker type
                                    if attacker_type in ["keras", "dual", "cnn"]:
                                        X_additional = X_unlabeled_pool_s[minority_candidate_idx] if X_unlabeled_pool_s is not None else X_unlabeled_pool[minority_candidate_idx]
                                    else:
                                        X_additional = X_unlabeled_pool[minority_candidate_idx]
                                    
                                    y_additional = y_candidates[minority_mask]
                                    X_query_s = np.vstack([X_query_s, X_additional])
                                    y_query = np.concatenate([y_query, y_additional])
                                    query_idx_in_unlabeled_pool = np.concatenate([query_idx_in_unlabeled_pool, minority_candidate_idx])
                                    
                                    final_dist = dict(zip(*np.unique(y_query, return_counts=True)))
                                    final_ratio = min(final_dist.values()) / sum(final_dist.values())
                                    print(f"   ✅ Đã thêm {minority_found} samples, distribution: {final_dist} (minority ratio: {final_ratio*100:.1f}%)")
                                else:
                                    print(f"   ⚠️  Không tìm thấy samples từ class {minority_class} trong unlabeled pool còn lại")
                                    print(f"   💡 Có thể unlabeled pool còn lại chủ yếu là class {majority_class}")
                elif len(query_dist) == 1:
                    print(f"   ⚠️  CẢNH BÁO: Chỉ có 1 class trong queries! Model sẽ không học được phân biệt 2 classes")
                    # Thử lấy thêm một số random samples để đảm bảo có cả 2 classes
                    remaining_pool_size = X_unlabeled_pool.shape[0]
                    if remaining_pool_size > 0:
                        additional_samples = min(200, remaining_pool_size, query_batch // 2)  # Lấy thêm 50% hoặc tối đa 200
                        additional_idx = rng.choice(remaining_pool_size, size=additional_samples, replace=False)
                        
                        # Lấy data phù hợp với oracle type
                        if oracle_model_type == "h5":
                            X_additional_for_query = X_unlabeled_pool_s[additional_idx] if X_unlabeled_pool_s is not None else X_unlabeled_pool[additional_idx]
                        else:
                            X_additional_for_query = X_unlabeled_pool[additional_idx]
                        
                        y_additional = oracle_client.predict(X_additional_for_query)
                        additional_dist = dict(zip(*np.unique(y_additional, return_counts=True)))
                        print(f"   🔄 Lấy thêm {additional_samples} random samples: {additional_dist}")
                        
                        # Thêm vào queries nếu có class mới
                        if len(additional_dist) > len(query_dist) or any(c not in query_dist for c in additional_dist):
                            # Lấy data phù hợp với attacker type
                            if attacker_type in ["keras", "dual", "cnn"]:
                                X_additional = X_unlabeled_pool_s[additional_idx] if X_unlabeled_pool_s is not None else X_unlabeled_pool[additional_idx]
                            else:
                                X_additional = X_unlabeled_pool[additional_idx]
                            
                            X_query_s = np.vstack([X_query_s, X_additional])
                            y_query = np.concatenate([y_query, y_additional])
                            query_idx_in_unlabeled_pool = np.concatenate([query_idx_in_unlabeled_pool, additional_idx])
                            print(f"   ✅ Đã thêm samples, distribution mới: {dict(zip(*np.unique(y_query, return_counts=True)))}")

        # QUAN TRỌNG: Đảm bảo số queries chính xác = queries_to_get_this_round
        # KHÔNG BAO GIỜ được thiếu queries trừ khi pool thực sự cạn kiệt!
        actual_queries = len(y_query)
        
        # Tính queries còn cần để đạt target
        queries_remaining_needed = total_queries_target - total_queries_accumulated
        
        # Mục tiêu queries cho round này: không vượt quá queries_remaining_needed và không vượt quá 110% của query_batch
        max_queries_this_round = min(int(query_batch * 1.1), queries_remaining_needed) if queries_remaining_needed > 0 else int(query_batch * 1.1)
        min_queries_this_round = queries_to_get_this_round  # Ít nhất phải đạt mục tiêu cho round này
        
        # QUAN TRỌNG: Nếu thiếu queries, BẮT BUỘC phải bổ sung từ unlabeled pool
        # Chỉ chấp nhận thiếu nếu unlabeled pool thực sự cạn kiệt
        if actual_queries < min_queries_this_round:
            # QUAN TRỌNG: Nếu có ít hơn mục tiêu, BẮT BUỘC phải bổ sung
            needed_samples = min_queries_this_round - actual_queries
            print(f"   ⚠️  CHỈ CÓ {actual_queries}/{min_queries_this_round} queries. CẦN BỔ SUNG {needed_samples} queries!")
            
            remaining_pool_size = X_unlabeled_pool.shape[0]
            if remaining_pool_size >= needed_samples:
                # Lấy thêm random samples từ unlabeled pool còn lại
                additional_idx = rng.choice(remaining_pool_size, size=needed_samples, replace=False)
                
                # Lấy data phù hợp với oracle type để query
                if oracle_model_type == "h5":
                    X_additional_for_query = X_unlabeled_pool_s[additional_idx] if X_unlabeled_pool_s is not None else X_unlabeled_pool[additional_idx]
                else:
                    X_additional_for_query = X_unlabeled_pool[additional_idx]
                
                y_additional = oracle_client.predict(X_additional_for_query)
                
                # Lấy data phù hợp với attacker type
                if attacker_type in ["keras", "dual", "cnn"]:
                    X_additional = X_unlabeled_pool_s[additional_idx] if X_unlabeled_pool_s is not None else X_unlabeled_pool[additional_idx]
                else:
                    X_additional = X_unlabeled_pool[additional_idx]
                
                X_query_s = np.vstack([X_query_s, X_additional])
                y_query = np.concatenate([y_query, y_additional])
                query_idx_in_unlabeled_pool = np.concatenate([query_idx_in_unlabeled_pool, additional_idx])
                
                print(f"   ✅ Đã bổ sung {needed_samples} queries từ unlabeled pool. Total: {len(y_query)}")
                actual_queries = len(y_query)
            else:
                # Unlabeled pool không đủ, lấy tất cả còn lại
                pool_exhausted_flag = True
                if remaining_pool_size > 0:
                    # Lấy data phù hợp với oracle type để query
                    if oracle_model_type == "h5":
                        X_additional_for_query = X_unlabeled_pool_s if X_unlabeled_pool_s is not None else X_unlabeled_pool
                    else:
                        X_additional_for_query = X_unlabeled_pool
                    
                    y_additional = oracle_client.predict(X_additional_for_query)
                    
                    # Lấy data phù hợp với attacker type
                    if attacker_type in ["keras", "dual", "cnn"]:
                        X_additional = X_unlabeled_pool_s if X_unlabeled_pool_s is not None else X_unlabeled_pool
                    else:
                        X_additional = X_unlabeled_pool
                    
                    X_query_s = np.vstack([X_query_s, X_additional])
                    y_query = np.concatenate([y_query, y_additional])
                    all_indices = np.arange(X_unlabeled_pool.shape[0])
                    query_idx_in_unlabeled_pool = np.concatenate([query_idx_in_unlabeled_pool, all_indices])
                    
                    actual_queries = len(y_query)
                    print(f"   ⚠️  Unlabeled pool còn lại chỉ có {remaining_pool_size} samples. Đã lấy tất cả.")
                    print(f"   📊 Total queries trong round này: {actual_queries} (mục tiêu: {min_queries_this_round})")
                    if actual_queries < min_queries_this_round:
                        missing = min_queries_this_round - actual_queries
                        print(f"   ❌ VẪN THIẾU {missing} queries do unlabeled pool cạn kiệt!")
                else:
                    pool_exhausted_flag = True
                    print(f"   ❌ LỖI NGHIÊM TRỌNG: Unlabeled pool đã cạn kiệt! Chỉ có {actual_queries} queries thay vì {min_queries_this_round}")
                    print(f"   ❌ Thiếu {min_queries_this_round - actual_queries} queries! Điều này sẽ ảnh hưởng nghiêm trọng đến hiệu suất!")
        
        # Giới hạn tối đa: không vượt quá max_queries_this_round (110% của query_batch hoặc queries còn cần)
        if actual_queries > max_queries_this_round:
            print(f"   ⚠️  Class balancing đã thêm {actual_queries - max_queries_this_round} queries (vượt quá 110%).")
            print(f"   🔄 Giới hạn lại về {max_queries_this_round} queries.")
            X_query_s = X_query_s[:max_queries_this_round]
            y_query = y_query[:max_queries_this_round]
            query_idx_in_unlabeled_pool = query_idx_in_unlabeled_pool[:max_queries_this_round]
            actual_queries = max_queries_this_round
            final_dist = dict(zip(*np.unique(y_query, return_counts=True)))
            print(f"   📊 Query distribution sau khi giới hạn: {final_dist}")
        
        final_query_count = actual_queries
        
        # QUAN TRỌNG: Verify số queries trước khi thêm vào labeled set
        queries_this_round = len(y_query)
        total_queries_accumulated += queries_this_round
        if total_queries_accumulated > total_queries_target:
            over_budget_flag = True
        
        # Kiểm tra xem có đạt mục tiêu không
        if queries_this_round >= min_queries_this_round:
            status = "✅"
        else:
            status = "⚠️"
        
        print(f"   {status} Round {query_round}: Đã chọn {queries_this_round} queries (mục tiêu: {min_queries_this_round}, tối đa: {max_queries_this_round})")
        print(f"   📊 Tổng queries tích lũy: {total_queries_accumulated:,}/{total_queries_target:,} ({total_queries_accumulated/total_queries_target*100:.1f}%)")
        
        # QUAN TRỌNG: Verify queries_this_round đạt mục tiêu trước khi xóa từ unlabeled pool
        # Nếu thiếu queries và unlabeled pool vẫn còn, phải cảnh báo nghiêm trọng
        if queries_this_round < min_queries_this_round:
            missing = min_queries_this_round - queries_this_round
            pool_remaining_before_delete = X_unlabeled_pool.shape[0]
            print(f"\n   ❌ LỖI NGHIÊM TRỌNG: Round {query_round} chỉ có {queries_this_round} queries thay vì {min_queries_this_round}!")
            print(f"   ❌ Thiếu {missing} queries! Điều này sẽ ảnh hưởng nghiêm trọng đến hiệu suất!")
            print(f"   💡 Unlabeled pool còn lại trước khi xóa: {pool_remaining_before_delete:,} samples")
            print(f"   💡 Kiểm tra logic bổ sung queries hoặc unlabeled pool size ban đầu!")
            # KHÔNG raise error vì có thể unlabeled pool thực sự cạn kiệt, nhưng cảnh báo rõ ràng
        
        # QUAN TRỌNG: Thêm vào labeled pool (pool tích lũy dần)
        labeled_X = np.vstack([labeled_X, X_query_s])
        labeled_y = np.concatenate([labeled_y, y_query])

        # Xóa từ unlabeled pool (đảm bảo query_idx_in_unlabeled_pool unique)
        query_idx_unique = np.unique(query_idx_in_unlabeled_pool)
        X_unlabeled_pool = np.delete(X_unlabeled_pool, query_idx_unique, axis=0)
        # QUAN TRỌNG: Cũng xóa labels tương ứng từ y_unlabeled_pool_gt (thief dataset labels)
        y_unlabeled_pool_gt = np.delete(y_unlabeled_pool_gt, query_idx_unique, axis=0)
        
        if X_unlabeled_pool_s is not None:
            # X_unlabeled_pool_s có sẵn cho Keras, dualDNN, và CNN
            X_unlabeled_pool_s = np.delete(X_unlabeled_pool_s, query_idx_unique, axis=0)
        
        print(f"   📊 Unlabeled pool còn lại: {X_unlabeled_pool.shape[0]:,} samples")

        # QUAN TRỌNG: Re-train từ đầu trên toàn bộ dữ liệu tích lũy
        # Theo nghiên cứu: Huấn luyện lại từ đầu giúp model học lại phân phối tổng thể,
        # giảm thiểu việc bị lệch theo phân phối của lô dữ liệu mới nhất
        print(f"   🔄 Re-training model với {labeled_X.shape[0]:,} labeled samples...")
        
        if attacker_type == "lgb":
            attacker = LGBAttacker(seed=seed)
            # Sử dụng hyperparameters tối ưu để khớp với target model
            attacker.train_model(labeled_X, labeled_y, X_val, y_val, boosting_rounds=2000, early_stopping=100)
        elif attacker_type == "dual":
            # QUAN TRỌNG: Sử dụng surrogate_feature_dim (từ target model), không phải dataset attack
            attacker = KerasDualAttacker(early_stopping=30, seed=seed, input_shape=(surrogate_feature_dim,))
            # DualDNN train với (X, y, y_true) - y_true là oracle labels
            attacker.train_model(labeled_X, labeled_y, labeled_y, X_val_s, y_val, y_val, num_epochs=num_epochs)
        elif attacker_type == "cnn":
            # QUAN TRỌNG: Sử dụng surrogate_feature_dim (từ target model), không phải dataset attack
            attacker = CNNAttacker(early_stopping=30, seed=seed, input_shape=(surrogate_feature_dim, 1))
            attacker.train_model(labeled_X, labeled_y, X_val_s, y_val, num_epochs=num_epochs)
        elif attacker_type == "knn":
            attacker = KNNAttacker(seed=seed)
            # KNN dùng raw data (labeled_X thay vì scaled)
            attacker.train_model(labeled_X, labeled_y, X_val, y_val)
        elif attacker_type == "xgb":
            attacker = XGBoostAttacker(seed=seed)
            # XGBoost dùng raw data (labeled_X thay vì scaled)
            attacker.train_model(labeled_X, labeled_y, X_val, y_val, boosting_rounds=200, early_stopping=20)
        elif attacker_type == "tabnet":
            attacker = TabNetAttacker(seed=seed)
            # TabNet dùng raw data (labeled_X thay vì scaled)
            attacker.train_model(
                labeled_X,
                labeled_y,
                X_val,
                y_val,
                max_epochs=100,
                patience=100000,  # effectively disable early stopping
                batch_size=2048,
            )
        else:
            # QUAN TRỌNG: Sử dụng surrogate_feature_dim (từ target model), không phải dataset attack
            attacker = KerasAttacker(early_stopping=30, seed=seed, input_shape=(surrogate_feature_dim,))
            attacker.train_model(labeled_X, labeled_y, X_val_s, y_val, num_epochs=num_epochs)

        evaluate(attacker, round_id=query_round, total_labels=labeled_X.shape[0])
    
    # Kiểm tra tổng queries cuối cùng
    final_total_queries = total_queries_accumulated
    diff = final_total_queries - total_queries_target
    diff_percent = (diff / total_queries_target * 100) if total_queries_target > 0 else 0
    query_gap_reason = "on_target"
    if final_total_queries < total_queries_target:
        query_gap_reason = "pool_exhausted" if pool_exhausted_flag else "stopped_before_target"
    elif final_total_queries > total_queries_target:
        query_gap_reason = "over_budget" if over_budget_flag else "extra_queries"
    
    print(f"\n{'='*80}")
    print(f"📊 TỔNG KẾT QUERIES:")
    print(f"{'='*80}")
    AL_queries_actual = final_total_queries - seed_size - val_size
    print(f"   Seed queries: {seed_size:,} (đã query)")
    print(f"   Val queries: {val_size:,} (đã query)")
    print(f"   AL queries dự kiến: {AL_queries_target:,} ({query_batch:,} queries/round × {num_rounds} rounds)")
    print(f"   AL queries thực tế: {AL_queries_actual:,}")
    print(f"   Total query budget dự kiến: {total_queries_target:,} (seed + val + AL)")
    print(f"   Total queries thực tế: {final_total_queries:,}")
    print(f"   Chênh lệch: {diff:+,} queries ({diff_percent:+.2f}%)")
    print(f"   Ngưỡng chấp nhận: {min_queries_acceptable:,} - {max_queries_acceptable:,} (90% - 110%)")
    
    if final_total_queries == total_queries_target:
        print(f"   ✅ Số queries chính xác 100%!")
    elif final_total_queries >= min_queries_acceptable and final_total_queries <= max_queries_acceptable:
        if diff > 0:
            print(f"   ✅ Số queries trong ngưỡng chấp nhận (dư {abs(diff):,} queries)")
        else:
            print(f"   ⚠️  Số queries trong ngưỡng chấp nhận nhưng thiếu {abs(diff):,} queries ({abs(diff_percent):.2f}%)")
    elif final_total_queries < min_queries_acceptable:
        print(f"   ❌ LỖI NGHIÊM TRỌNG: SỐ QUERIES THIẾU QUÁ NHIỀU! ({abs(diff_percent):.2f}% thiếu)")
        print(f"   ❌ Thiếu {abs(diff):,} queries! Điều này sẽ ảnh hưởng NGHIÊM TRỌNG đến hiệu suất tấn công!")
        print(f"   💡 Lý do có thể: Pool đã cạn kiệt trước khi đạt đủ queries")
        print(f"   ⚠️  Cần kiểm tra lại:")
        print(f"      - Pool size ban đầu có đủ không? (cần ít nhất {total_queries_target:,} với buffer 20%)")
        print(f"      - Logic bổ sung queries có hoạt động đúng không?")
        print(f"      - Có thể cần tăng pool size hoặc điều chỉnh query_batch/num_rounds")
        # KHÔNG raise error vì vẫn muốn có kết quả, nhưng cảnh báo rõ ràng
    else:
        print(f"   ⚠️  Số queries vượt quá 10% (dư {diff:,} queries, {diff_percent:.2f}%)")
    print(f"{'='*80}\n")

    output_dir.mkdir(parents=True, exist_ok=True)
    target_surrogate_dir = _resolve_optional_path(surrogate_dir) if surrogate_dir else output_dir.resolve()
    target_surrogate_dir.mkdir(parents=True, exist_ok=True)
    surrogate_basename = surrogate_name if surrogate_name else "surrogate_model"
    surrogate_path = target_surrogate_dir / surrogate_basename
    attacker.save_model(str(surrogate_path))
    
    # Lấy extension phù hợp với model type
    if attacker_type == "lgb":
        surrogate_model_path = f"{surrogate_path}.txt"
    elif attacker_type == "knn":
        surrogate_model_path = f"{surrogate_path}.pkl"
    elif attacker_type == "xgb":
        surrogate_model_path = f"{surrogate_path}.json"
    elif attacker_type == "tabnet":
        surrogate_model_path = f"{surrogate_path}.zip"
    else:
        # Keras, dualDNN, và CNN đều dùng .h5
        surrogate_model_path = f"{surrogate_path}.h5"

    joblib_path = output_dir / "robust_scaler.joblib"
    try:
        if scaler is not None:
            import joblib
            joblib.dump(scaler, joblib_path)
        else:
            joblib_path = None
    except Exception:
        joblib_path = None

    df_metrics = pd.DataFrame(metrics_history)
    metrics_csv = output_dir / "extraction_metrics.csv"
    df_metrics.to_csv(metrics_csv, index=False)

    #region agent log
    try:
        import json as _json, time as _time
        from pathlib import Path as _Path
        _log_payload = {
            "sessionId": "debug-session",
            "runId": "pre-fix",
            "hypothesisId": "H2",
            "location": "extract_final_model.py:metrics_csv_write",
            "message": "metrics_history final snapshot before summary",
            "data": {
                "output_dir": str(output_dir),
                "metrics_len": len(metrics_history),
                "final_metrics": metrics_history[-1] if metrics_history else None,
                "metrics_csv": str(metrics_csv),
                "metrics_csv_exists_before": _Path(metrics_csv).exists(),
            },
            "timestamp": int(_time.time() * 1000),
        }
        with open("/home/hytong/Documents/model_extraction_malware/.cursor/debug.log", "a", encoding="utf-8") as _f:
            _f.write(_json.dumps(_log_payload, ensure_ascii=False) + "\n")
    except Exception:
        pass
    #endregion

    summary = {
        "oracle_source": oracle_source,
        "model_file_name": model_file_name,
        "model_type": model_type,
        "normalization_stats_path": normalization_stats_path,
        "attacker_type": attacker_type,
        "surrogate_model_path": surrogate_model_path,
        "scaler_path": str(joblib_path) if joblib_path else None,
        "metrics_csv": str(metrics_csv),
        "metrics": metrics_history,
        "seed_size": int(seed_size),
        "val_size": int(val_size),
        "query_batch": int(query_batch),
        "num_rounds": int(num_rounds),
        "total_queries_target": int(total_queries_target),
        "total_queries_actual": int(final_total_queries),
        "query_gap_reason": query_gap_reason,
    }

    summary_path = output_dir / "extraction_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    SUMMARY = run_extraction(
        weights_path=str(PROJECT_ROOT / "artifacts" / "targets" / "final_model.h5"),
        output_dir=PROJECT_ROOT / "output",
        seed=42,
    )
    print(json.dumps(SUMMARY["metrics"][-1], indent=2))

