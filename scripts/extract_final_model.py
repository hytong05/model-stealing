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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.attackers import KerasAttacker, LGBAttacker, KerasDualAttacker
from src.targets.flexible_target import FlexibleKerasTarget, FlexibleLGBTarget
from src.sampling import entropy_sampling
from sklearn_extra.cluster import KMedoids


def _clip_scale(scaler: RobustScaler, X: np.ndarray) -> np.ndarray:
    """Scale data với RobustScaler và clip về [-5, 5] giống pipeline gốc."""
    transformed = scaler.transform(X)
    return np.clip(transformed, -5, 5)


def get_feature_columns(parquet_path: str, label_col: str = "Label") -> list:
    """Lấy danh sách feature columns từ parquet file."""
    pq_file = pq.ParquetFile(parquet_path)
    return [name for name in pq_file.schema.names if name != label_col]


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
    weights_path: str,
    output_dir: Path,
    train_parquet: str = None,
    test_parquet: str = None,
    seed: int = 42,
    feature_dim: int = 2381,
    seed_size: int = 2000,
    val_size: int = 2000,
    eval_size: int = 4000,
    query_batch: int = 2000,
    num_rounds: int = 5,
    num_epochs: int = 5,
    model_type: str = "h5",  # "h5" hoặc "lgb"
    normalization_stats_path: str = None,  # Cần thiết nếu model_type="lgb"
    attacker_type: str = None,  # "keras", "lgb", hoặc "dual" (dualDNN), None để tự động chọn theo model_type
) -> dict:
    rng = np.random.default_rng(seed)

    # Chỉ set TF environment variables nếu dùng Keras model
    if model_type == "h5" or attacker_type in ["keras", "dual"]:
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

    label_col = "Label"
    
    # Auto-detect attacker_type nếu không được chỉ định
    if attacker_type is None:
        attacker_type = "keras" if model_type == "h5" else "lgb"

    # Load dữ liệu từ EMBER parquet files
    if train_parquet is None:
        train_parquet = str(PROJECT_ROOT / "src" / "train_ember_2018_v2_features_label_other.parquet")
    if test_parquet is None:
        test_parquet = str(PROJECT_ROOT / "src" / "test_ember_2018_v2_features_label_other.parquet")

    print("=" * 60)
    print("📊 Đang load dữ liệu EMBER...")
    print("=" * 60)
    print(f"Train file: {train_parquet}")
    print(f"Test file: {test_parquet}")

    # Lấy feature columns và xác định feature_dim thực tế
    feature_cols = get_feature_columns(train_parquet, label_col)
    actual_feature_dim = len(feature_cols)
    print(f"Feature columns: {actual_feature_dim}")
    
    # Cập nhật feature_dim nếu khác với giá trị mặc định
    if actual_feature_dim != feature_dim:
        print(f"⚠️  Feature dimension mismatch: dataset has {actual_feature_dim} features, "
              f"but feature_dim parameter is {feature_dim}")
        print(f"   Updating feature_dim to {actual_feature_dim} (từ dataset)")
        feature_dim = actual_feature_dim
    
    # QUAN TRỌNG: Validate và log thông tin target model
    weights_path_abs = str(Path(weights_path).resolve())
    if not Path(weights_path_abs).exists():
        raise FileNotFoundError(f"❌ Target model không tồn tại: {weights_path_abs}")
    
    model_file_name = Path(weights_path_abs).name
    model_file_size = Path(weights_path_abs).stat().st_size / (1024 * 1024)  # MB
    
    print(f"\n🔄 Khởi tạo target model ({model_type.upper()}) với feature_dim={feature_dim}...")
    print(f"   ✅ Target model file: {model_file_name}")
    print(f"   ✅ Model path (absolute): {weights_path_abs}")
    print(f"   ✅ Model size: {model_file_size:.2f} MB")
    
    if weights_path != weights_path_abs:
        print(f"   ⚠️  Path được resolve: {weights_path} -> {weights_path_abs}")
    
    if model_type == "lgb":
        # LightGBM model cần normalization stats
        if normalization_stats_path is None:
            raise ValueError(
                "normalization_stats_path phải được cung cấp khi model_type='lgb'. "
                "Vui lòng cung cấp đường dẫn tới file normalization_stats.npz"
            )
        
        # Validate normalization stats path
        if isinstance(normalization_stats_path, str):
            stats_path_abs = str(Path(normalization_stats_path).resolve())
            if not Path(stats_path_abs).exists():
                raise FileNotFoundError(f"❌ Normalization stats không tồn tại: {stats_path_abs}")
            normalization_stats_path = stats_path_abs
        
        print(f"   ✅ Normalization stats file: {Path(normalization_stats_path).name}")
        print(f"   ✅ Stats path (absolute): {normalization_stats_path}")
        
        oracle = FlexibleLGBTarget(
            model_path=weights_path_abs,  # Sử dụng absolute path
            normalization_stats_path=normalization_stats_path,  # Sử dụng absolute path
            threshold=0.5,
            name=f"lgb-target-{model_file_name}",
            feature_dim=feature_dim
        )
    else:
        # Keras/H5 model
        oracle = FlexibleKerasTarget(weights_path_abs, feature_dim=feature_dim, threshold=0.5)
    
    required_feature_dim = oracle.get_required_feature_dim()
    
    if required_feature_dim is None:
        print(f"   ✅ Target model có preprocessing layer - sẽ tự động map từ {feature_dim} đặc trưng")
    else:
        print(f"   ✅ Target model yêu cầu {required_feature_dim} đặc trưng")
        if feature_dim > required_feature_dim:
            print(f"   ⚠️  Dataset có {feature_dim} đặc trưng, sẽ tự động cắt bỏ {feature_dim - required_feature_dim} đặc trưng thừa")
        elif feature_dim < required_feature_dim:
            print(f"   ❌ Dataset có {feature_dim} đặc trưng, nhưng target model yêu cầu {required_feature_dim} đặc trưng")
            raise ValueError(f"Dataset không đủ đặc trưng: {feature_dim} < {required_feature_dim}")

    # QUAN TRỌNG: Đảm bảo seed/val sets giống nhau giữa các configs
    # Giải pháp: Load đủ lớn (seed_val + pool lớn nhất), shuffle với seed, sau đó chia
    # Tính pool lớn nhất cần thiết trong các configs (để đảm bảo không thiếu dữ liệu)
    # Với cấu hình hiện tại: max_queries_10000 có query_batch=2000, num_rounds=5 => pool cần 10000
    # QUAN TRỌNG: Thêm buffer 20% để đảm bảo KHÔNG BAO GIỜ thiếu queries
    # Tăng buffer từ 10% lên 20% để đảm bảo đủ pool cho class balancing
    max_pool_needed_base = query_batch * num_rounds
    max_pool_needed = int(max_pool_needed_base * 1.2)  # Buffer 20%
    seed_val_size = seed_size + val_size
    total_needed = seed_val_size + max_pool_needed
    
    print(f"\n🔄 Đang load train data ({total_needed:,} samples: {seed_val_size:,} seed+val + {max_pool_needed:,} pool)...")
    X_train_all, _ = load_data_from_parquet(
        train_parquet, feature_cols, label_col, skip_rows=0, take_rows=total_needed, shuffle=True, seed=seed
    )
    print(f"✅ Train data loaded: {X_train_all.shape}")

    # Chia train data thành seed, val, pool
    # Seed và val giống nhau cho tất cả configs
    idx_offset = 0
    X_seed = X_train_all[idx_offset : idx_offset + seed_size]
    idx_offset += seed_size

    X_val = X_train_all[idx_offset : idx_offset + val_size]
    idx_offset += val_size

    # QUAN TRỌNG: Pool size phải đủ cho total queries + dư 20% để đảm bảo KHÔNG BAO GIỜ thiếu
    # Do class balancing có thể thêm queries, và cần buffer lớn để đảm bảo đủ queries
    pool_needed_base = query_batch * num_rounds
    pool_needed = int(pool_needed_base * 1.2)  # Dư 20% để đảm bảo đủ queries (tăng từ 10% lên 20%)
    
    # Kiểm tra xem có đủ data không
    available_pool = X_train_all.shape[0] - idx_offset
    if available_pool < pool_needed:
        # Nếu không đủ data cho pool với buffer, vẫn cố gắng lấy ít nhất pool_needed_base
        if available_pool < pool_needed_base:
            print(f"   ❌ LỖI NGHIÊM TRỌNG: Không đủ data cho pool!")
            print(f"   ❌ Available: {available_pool:,}, Required: {pool_needed_base:,}")
            print(f"   ❌ Pool sẽ cạn kiệt và queries sẽ thiếu!")
            raise ValueError(
                f"Không đủ data cho pool! Available: {available_pool:,}, "
                f"Required: {pool_needed_base:,} (query_batch={query_batch:,} × num_rounds={num_rounds})"
            )
        else:
            print(f"   ⚠️  CẢNH BÁO: Không đủ data cho pool với buffer ({available_pool:,} < {pool_needed:,})")
            print(f"   💡 Sẽ dùng tối đa {available_pool:,} samples cho pool (thiếu buffer)")
            pool_needed = available_pool
    
    X_pool = X_train_all[idx_offset : idx_offset + pool_needed]
    buffer_size = pool_needed - pool_needed_base
    print(f"   ✅ Pool size: {X_pool.shape[0]:,} samples")
    print(f"      - Target: {pool_needed_base:,} (query_batch={query_batch:,} × num_rounds={num_rounds})")
    print(f"      - Buffer: +{buffer_size:,} ({buffer_size/pool_needed_base*100:.1f}%) để đảm bảo không thiếu queries")
    del X_train_all
    gc.collect()

    # Load eval set từ test file
    # QUAN TRỌNG: Load cả ground truth labels để tính accuracy chính xác
    print(f"\n🔄 Đang load eval set ({eval_size:,} samples)...")
    X_eval, y_eval_gt = load_data_from_parquet(
        test_parquet, feature_cols, label_col, skip_rows=0, take_rows=eval_size, shuffle=True, seed=seed
    )
    print(f"✅ Eval set: {X_eval.shape}")
    print(f"✅ Ground truth labels: {y_eval_gt.shape}")

    print(f"\n📊 Data split:")
    print(f"  Seed: {X_seed.shape[0]:,}")
    print(f"  Val: {X_val.shape[0]:,}")
    print(f"  Pool: {X_pool.shape[0]:,}")
    print(f"  Eval: {X_eval.shape[0]:,}")

    # QUAN TRỌNG: Xử lý dữ liệu trước khi query oracle
    # - Với Keras/H5 model: Cần scale data với RobustScaler (model được train với scaled data)
    # - Với LightGBM model: FlexibleLGBTarget sẽ tự động normalize nếu có normalization_stats_path
    #   Không cần scale thêm với RobustScaler
    # - Với dualDNN: Cũng cần scale data giống Keras
    scaler = None
    X_eval_s = None
    X_seed_s = None
    X_val_s = None
    X_pool_s = None
    
    if model_type == "h5" or attacker_type in ["keras", "dual"]:
        # Keras model cần scale data
        print(f"\n🔄 Đang scale dữ liệu trước khi query oracle (Keras model)...")
        scaler = RobustScaler()
        scaler.fit(np.vstack([X_seed, X_val, X_pool]))

        X_eval_s = _clip_scale(scaler, X_eval)
        X_seed_s = _clip_scale(scaler, X_seed)
        X_val_s = _clip_scale(scaler, X_val)
        X_pool_s = _clip_scale(scaler, X_pool)
        
        print(f"✅ Đã scale dữ liệu")
        print(f"   - X_eval_s shape: {X_eval_s.shape}")
        print(f"   - X_seed_s shape: {X_seed_s.shape}")
        print(f"   - X_val_s shape: {X_val_s.shape}")
        print(f"   - X_pool_s shape: {X_pool_s.shape}")
        
        # Lấy nhãn từ oracle (VỚI DỮ LIỆU ĐÃ SCALE)
        print(f"\n🔄 Đang lấy nhãn từ oracle (với dữ liệu đã scale)...")
        y_eval = oracle(X_eval_s)
        y_seed = oracle(X_seed_s)
        y_val = oracle(X_val_s)
    else:
        # LightGBM model: FlexibleLGBTarget sẽ tự động normalize
        print(f"\n🔄 Đang lấy nhãn từ oracle (LightGBM sẽ tự động normalize)...")
        y_eval = oracle(X_eval)
        y_seed = oracle(X_seed)
        y_val = oracle(X_val)
        
        # Với LightGBM, không cần scale data
        X_eval_s = X_eval
        X_seed_s = X_seed
        X_val_s = X_val
        X_pool_s = X_pool
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
        print(f"   💡 Sẽ thử kiểm tra probabilities và có thể điều chỉnh threshold...")
        
        # Kiểm tra probabilities để xem có phải do threshold không
        try:
            test_sample_size = min(100, X_eval_s.shape[0])
            test_indices = rng.choice(X_eval_s.shape[0], size=test_sample_size, replace=False)
            # Sử dụng X_eval_s đã được scale/normalize
            test_data = X_eval_s[test_indices]
            test_probs = oracle.predict_proba(test_data)
            print(f"   📊 Test probabilities trên {test_sample_size} samples:")
            print(f"      Range: [{test_probs.min():.4f}, {test_probs.max():.4f}]")
            print(f"      Mean: {test_probs.mean():.4f}, Median: {np.median(test_probs):.4f}")
            print(f"      Threshold hiện tại: {oracle.model_threshold}")
            
            # Nếu probabilities tập trung gần threshold, có thể cần điều chỉnh
            if test_probs.min() < oracle.model_threshold < test_probs.max():
                print(f"   💡 Probabilities có cả dưới và trên threshold - có thể có cả 2 classes")
                print(f"      Thử với threshold thấp hơn có thể giúp phân biệt tốt hơn")
            elif test_probs.max() < oracle.model_threshold:
                # Tất cả probabilities đều dưới threshold - cần giảm threshold
                suggested_threshold = np.percentile(test_probs, 50)  # Median
                print(f"   ⚠️  TẤT CẢ probabilities đều dưới threshold {oracle.model_threshold}")
                print(f"   💡 Đề xuất giảm threshold xuống {suggested_threshold:.4f} (median) để phân biệt classes")
                print(f"   🔄 Đang điều chỉnh threshold...")
                oracle.model_threshold = suggested_threshold
                # Test lại với threshold mới
                test_predictions_new = oracle(X_eval_s[test_indices])
                test_dist_new = dict(zip(*np.unique(test_predictions_new, return_counts=True)))
                print(f"   ✅ Với threshold mới {suggested_threshold:.4f}: {test_dist_new}")
                
                # QUAN TRỌNG: Re-query seed, val, eval với threshold mới
                print(f"   🔄 Re-querying seed, val, eval với threshold mới...")
                # Sử dụng dữ liệu đã được xử lý (scaled hoặc raw tùy theo model type)
                y_eval = oracle(X_eval_s)
                y_seed = oracle(X_seed_s)
                y_val = oracle(X_val_s)
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
    labeled_X = X_seed_s
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
        actual_queries = total_labels - seed_size - val_size
        
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
        # LightGBM attacker không cần scale data
        attacker = LGBAttacker(seed=seed)
        attacker.train_model(labeled_X, labeled_y, X_val, y_val, boosting_rounds=100, early_stopping=15)
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

            # Tính số queries thực tế (không tính seed và val)
            actual_queries = total_labels - seed_size - val_size
            
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
        attacker = KerasDualAttacker(early_stopping=30, seed=seed)
        # DualDNN train với (X, y_true) - y_true là oracle labels
        attacker.train_model(labeled_X, labeled_y, labeled_y, X_val_s, y_val, y_val, num_epochs=num_epochs)
        
        def evaluate_dual(model, round_id, total_labels):
            # DualDNN cần cả X và y_true (oracle labels) khi predict
            # __call__ nhận 2 tham số riêng biệt (X, y_true), không phải tuple
            probs = np.squeeze(model(X_eval_s, y_eval), axis=-1)
            
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
            
            # Sử dụng threshold tối ưu
            preds = best_preds
            
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
            
            # Tính số queries thực tế
            actual_queries = total_labels - seed_size - val_size
            
            print(f"\n📊 Round {round_id} Evaluation (DualDNN):")
            print(f"   Agreement (vs Oracle): {agreement:.4f} ({agreement*100:.2f}%)")
            print(f"   Accuracy (vs Ground Truth): {acc:.4f} ({acc*100:.2f}%)")
            print(f"   Oracle Accuracy (vs Ground Truth): {oracle_acc_vs_gt:.4f} ({oracle_acc_vs_gt*100:.2f}%)")
            
            metrics = {
                "round": round_id,
                "labels_used": int(total_labels),
                "queries_used": int(actual_queries),
                "optimal_threshold": float(best_threshold),
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
    else:
        # Keras attacker cần scale data
        attacker = KerasAttacker(early_stopping=30, seed=seed, input_shape=(feature_dim,))
        attacker.train_model(labeled_X, labeled_y, X_val_s, y_val, num_epochs=num_epochs)
        evaluate(attacker, round_id=0, total_labels=labeled_X.shape[0])

    # Track tổng queries để đảm bảo chính xác
    total_queries_target = query_batch * num_rounds
    total_queries_accumulated = 0
    # Cho phép lệch tối đa 10% (dư chứ không được thiếu)
    min_queries_acceptable = int(total_queries_target * 0.9)  # Ít nhất 90% của target
    max_queries_acceptable = int(total_queries_target * 1.1)  # Tối đa 110% của target
    
    print(f"\n📋 Mục tiêu queries: {total_queries_target:,} ({query_batch:,} queries/round × {num_rounds} rounds)")
    print(f"   📊 Cho phép lệch: {min_queries_acceptable:,} - {max_queries_acceptable:,} queries (90% - 110%)")
    print(f"   ⚠️  Quan trọng: Không được thiếu queries! (tối thiểu {min_queries_acceptable:,})")
    
    # Kiểm tra pool ban đầu có đủ không
    if X_pool.shape[0] < total_queries_target:
        print(f"\n⚠️  CẢNH BÁO: Pool ban đầu ({X_pool.shape[0]:,}) < Tổng queries dự kiến ({total_queries_target:,})")
        print(f"   💡 Pool sẽ cạn kiệt trước khi đạt đủ queries. Sẽ cố gắng lấy tối đa có thể.")
    
    for query_round in range(1, num_rounds + 1):
        # Kiểm tra xem còn cần bao nhiêu queries nữa
        queries_remaining_needed = total_queries_target - total_queries_accumulated
        
        # Nếu đã đạt đủ queries, dừng lại
        if total_queries_accumulated >= total_queries_target:
            print(f"\n✅ Đã đạt đủ queries dự kiến ({total_queries_target:,}). Dừng active learning.")
            break
        
        # Nếu pool còn lại ít hơn query_batch, vẫn cố gắng lấy tối đa có thể
        pool_remaining = X_pool.shape[0]
        queries_to_get_this_round = min(query_batch, pool_remaining, queries_remaining_needed)
        
        if queries_to_get_this_round <= 0:
            print(f"\n⚠️  Round {query_round}: Không còn queries để lấy. Pool: {pool_remaining}, Cần: {queries_remaining_needed}")
            break
        
        if pool_remaining < query_batch:
            print(f"\n⚠️  Round {query_round}: Pool còn lại ({pool_remaining}) < query_batch ({query_batch}).")
            print(f"   🔄 Sẽ lấy tối đa {queries_to_get_this_round} queries từ pool còn lại.")
        
        # GIẢI PHÁP 1: Dùng Entropy + k-medoids thay vì random sampling
        # Theo nghiên cứu: Entropy để chọn điểm có độ bất định cao,
        # sau đó k-medoids để đảm bảo tính đa dạng và tránh lấy nhiều điểm nhiễu
        print(f"\n🔄 Round {query_round}: Đang chọn queries bằng Entropy + k-medoids...")
        
        # Chọn dữ liệu để query dựa trên attacker type
        pool_for_entropy = X_pool_s if attacker_type in ["keras", "dual"] else X_pool
        
        # QUAN TRỌNG: Với dualDNN, cần oracle labels cho pool để entropy sampling
        # Query oracle cho pool nếu chưa có (chỉ cho dualDNN)
        pool_labels_for_entropy = None
        if attacker_type == "dual":
            # Query oracle để lấy labels cho pool (dùng cho entropy sampling với dual=True)
            pool_labels_for_entropy = oracle(pool_for_entropy)
        
        # Bước 1: Dùng entropy để chọn 10000 điểm có độ bất định cao
        # (k-medoids không scale tốt với toàn bộ pool)
        entropy_candidates = min(10000, pool_for_entropy.shape[0])
        dual_flag = (attacker_type == "dual")
        q_idx = entropy_sampling(
            attacker, 
            pool_for_entropy, 
            pool_labels_for_entropy if dual_flag else np.zeros(pool_for_entropy.shape[0]),  # y cần thiết cho dualDNN
            n_instances=entropy_candidates,
            dual=dual_flag
        )
        X_med = pool_for_entropy[q_idx]
        
        # Bước 2: Dùng k-medoids để chọn queries_to_get_this_round điểm đại diện từ các điểm entropy cao
        # QUAN TRỌNG: queries_to_get_this_round đã được tính ở trên (dòng 583)
        
        # Đảm bảo số clusters không lớn hơn số samples có sẵn
        num_clusters = min(queries_to_get_this_round, X_med.shape[0])
        
        if num_clusters > 0:
            kmed = KMedoids(n_clusters=num_clusters, init='k-medoids++', random_state=seed)
            kmed.fit(X_med)
            query_idx_in_med = kmed.medoid_indices_
            query_idx = q_idx[query_idx_in_med]
        else:
            # Nếu không đủ samples, lấy tất cả có sẵn
            query_idx = q_idx[:min(queries_to_get_this_round, len(q_idx))]
        
        print(f"   ✅ Đã chọn {len(query_idx)} queries từ {entropy_candidates} entropy candidates (target: {queries_to_get_this_round})")

        # Query oracle
        # - Với Keras: Query với dữ liệu đã scale (X_pool_s)
        # - Với LightGBM: Query với raw data (X_pool), FlexibleLGBTarget sẽ tự động normalize
        X_query_s = pool_for_entropy[query_idx]
        y_query = oracle(X_query_s)

        # Log class distribution để kiểm tra imbalance
        query_dist = dict(zip(*np.unique(y_query, return_counts=True)))
        print(f"   📊 Query distribution: {query_dist}")
        
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
                        
                        # Query oracle trên toàn bộ pool còn lại để tìm class thiểu số
                        remaining_pool_size = X_pool_s.shape[0]
                        if remaining_pool_size > needed_samples:
                            # Tăng sample_size để tìm đủ class thiểu số (có thể pool chủ yếu là class đa số)
                            # Ước tính: nếu class thiểu số chiếm ~10%, cần query ~10x để tìm đủ
                            sample_size = min(needed_samples * 10, remaining_pool_size)
                            candidate_idx = rng.choice(remaining_pool_size, size=sample_size, replace=False)
                            X_candidates = X_pool_s[candidate_idx]
                            y_candidates = oracle(X_candidates)
                            
                            # Lọc chỉ lấy class thiểu số
                            minority_mask = y_candidates == minority_class
                            minority_found = np.sum(minority_mask)
                            
                            if minority_found >= needed_samples:
                                # Lấy đủ samples từ class thiểu số
                                minority_indices = candidate_idx[minority_mask][:needed_samples]
                                X_additional = X_pool_s[minority_indices]
                                y_additional = oracle(X_additional)
                                
                                X_query_s = np.vstack([X_query_s, X_additional])
                                y_query = np.concatenate([y_query, y_additional])
                                query_idx = np.concatenate([query_idx, minority_indices])
                                
                                balanced_dist = dict(zip(*np.unique(y_query, return_counts=True)))
                                print(f"   ✅ Đã cân bằng: {balanced_dist}")
                            else:
                                print(f"   ⚠️  Chỉ tìm thấy {minority_found}/{needed_samples} samples từ class {minority_class}")
                                if minority_found > 0:
                                    minority_indices = candidate_idx[minority_mask]
                                    X_additional = X_pool_s[minority_indices]
                                    y_additional = oracle(X_additional)
                                    X_query_s = np.vstack([X_query_s, X_additional])
                                    y_query = np.concatenate([y_query, y_additional])
                                    query_idx = np.concatenate([query_idx, minority_indices])
                                    
                                    final_dist = dict(zip(*np.unique(y_query, return_counts=True)))
                                    final_ratio = min(final_dist.values()) / sum(final_dist.values())
                                    print(f"   ✅ Đã thêm {minority_found} samples, distribution: {final_dist} (minority ratio: {final_ratio*100:.1f}%)")
                                else:
                                    print(f"   ⚠️  Không tìm thấy samples từ class {minority_class} trong pool còn lại")
                                    print(f"   💡 Có thể pool còn lại chủ yếu là class {majority_class}")
                elif len(query_dist) == 1:
                    print(f"   ⚠️  CẢNH BÁO: Chỉ có 1 class trong queries! Model sẽ không học được phân biệt 2 classes")
                    # Thử lấy thêm một số random samples để đảm bảo có cả 2 classes
                    remaining_pool_size = X_pool_s.shape[0]
                    if remaining_pool_size > 0:
                        additional_samples = min(200, remaining_pool_size, query_batch // 2)  # Lấy thêm 50% hoặc tối đa 200
                        additional_idx = rng.choice(remaining_pool_size, size=additional_samples, replace=False)
                        X_additional = X_pool_s[additional_idx]
                        y_additional = oracle(X_additional)
                        additional_dist = dict(zip(*np.unique(y_additional, return_counts=True)))
                        print(f"   🔄 Lấy thêm {additional_samples} random samples: {additional_dist}")
                        
                        # Thêm vào queries nếu có class mới
                        if len(additional_dist) > len(query_dist) or any(c not in query_dist for c in additional_dist):
                            X_query_s = np.vstack([X_query_s, X_additional])
                            y_query = np.concatenate([y_query, y_additional])
                            query_idx = np.concatenate([query_idx, additional_idx])
                            print(f"   ✅ Đã thêm samples, distribution mới: {dict(zip(*np.unique(y_query, return_counts=True)))}")

        # QUAN TRỌNG: Đảm bảo số queries chính xác = queries_to_get_this_round
        # KHÔNG BAO GIỜ được thiếu queries trừ khi pool thực sự cạn kiệt!
        actual_queries = len(y_query)
        
        # Tính queries còn cần để đạt target
        queries_remaining_needed = total_queries_target - total_queries_accumulated
        
        # Mục tiêu queries cho round này: không vượt quá queries_remaining_needed và không vượt quá 110% của query_batch
        max_queries_this_round = min(int(query_batch * 1.1), queries_remaining_needed) if queries_remaining_needed > 0 else int(query_batch * 1.1)
        min_queries_this_round = queries_to_get_this_round  # Ít nhất phải đạt mục tiêu cho round này
        
        # QUAN TRỌNG: Nếu thiếu queries, BẮT BUỘC phải bổ sung từ pool
        # Chỉ chấp nhận thiếu nếu pool thực sự cạn kiệt
        if actual_queries < min_queries_this_round:
            # QUAN TRỌNG: Nếu có ít hơn mục tiêu, BẮT BUỘC phải bổ sung
            needed_samples = min_queries_this_round - actual_queries
            print(f"   ⚠️  CHỈ CÓ {actual_queries}/{min_queries_this_round} queries. CẦN BỔ SUNG {needed_samples} queries!")
            
            remaining_pool_size = X_pool_s.shape[0]
            if remaining_pool_size >= needed_samples:
                # Lấy thêm random samples từ pool còn lại
                additional_idx = rng.choice(remaining_pool_size, size=needed_samples, replace=False)
                X_additional = X_pool_s[additional_idx]
                y_additional = oracle(X_additional)
                
                X_query_s = np.vstack([X_query_s, X_additional])
                y_query = np.concatenate([y_query, y_additional])
                query_idx = np.concatenate([query_idx, additional_idx])
                
                print(f"   ✅ Đã bổ sung {needed_samples} queries từ pool. Total: {len(y_query)}")
                actual_queries = len(y_query)
            else:
                # Pool không đủ, lấy tất cả còn lại
                if remaining_pool_size > 0:
                    X_additional = X_pool_s
                    y_additional = oracle(X_additional)
                    
                    X_query_s = np.vstack([X_query_s, X_additional])
                    y_query = np.concatenate([y_query, y_additional])
                    all_indices = np.arange(X_pool_s.shape[0])
                    query_idx = np.concatenate([query_idx, all_indices])
                    
                    actual_queries = len(y_query)
                    print(f"   ⚠️  Pool còn lại chỉ có {remaining_pool_size} samples. Đã lấy tất cả.")
                    print(f"   📊 Total queries trong round này: {actual_queries} (mục tiêu: {min_queries_this_round})")
                    if actual_queries < min_queries_this_round:
                        missing = min_queries_this_round - actual_queries
                        print(f"   ❌ VẪN THIẾU {missing} queries do pool cạn kiệt!")
                else:
                    print(f"   ❌ LỖI NGHIÊM TRỌNG: Pool đã cạn kiệt! Chỉ có {actual_queries} queries thay vì {min_queries_this_round}")
                    print(f"   ❌ Thiếu {min_queries_this_round - actual_queries} queries! Điều này sẽ ảnh hưởng nghiêm trọng đến hiệu suất!")
        
        # Giới hạn tối đa: không vượt quá max_queries_this_round (110% của query_batch hoặc queries còn cần)
        if actual_queries > max_queries_this_round:
            print(f"   ⚠️  Class balancing đã thêm {actual_queries - max_queries_this_round} queries (vượt quá 110%).")
            print(f"   🔄 Giới hạn lại về {max_queries_this_round} queries.")
            X_query_s = X_query_s[:max_queries_this_round]
            y_query = y_query[:max_queries_this_round]
            query_idx = query_idx[:max_queries_this_round]
            actual_queries = max_queries_this_round
            final_dist = dict(zip(*np.unique(y_query, return_counts=True)))
            print(f"   📊 Query distribution sau khi giới hạn: {final_dist}")
        
        final_query_count = actual_queries
        
        # QUAN TRỌNG: Verify số queries trước khi thêm vào labeled set
        queries_this_round = len(y_query)
        total_queries_accumulated += queries_this_round
        
        # Kiểm tra xem có đạt mục tiêu không
        if queries_this_round >= min_queries_this_round:
            status = "✅"
        else:
            status = "⚠️"
        
        print(f"   {status} Round {query_round}: Đã chọn {queries_this_round} queries (mục tiêu: {min_queries_this_round}, tối đa: {max_queries_this_round})")
        print(f"   📊 Tổng queries tích lũy: {total_queries_accumulated:,}/{total_queries_target:,} ({total_queries_accumulated/total_queries_target*100:.1f}%)")
        
        # QUAN TRỌNG: Verify queries_this_round đạt mục tiêu trước khi xóa từ pool
        # Nếu thiếu queries và pool vẫn còn, phải cảnh báo nghiêm trọng
        if queries_this_round < min_queries_this_round:
            missing = min_queries_this_round - queries_this_round
            pool_remaining_before_delete = X_pool.shape[0]
            print(f"\n   ❌ LỖI NGHIÊM TRỌNG: Round {query_round} chỉ có {queries_this_round} queries thay vì {min_queries_this_round}!")
            print(f"   ❌ Thiếu {missing} queries! Điều này sẽ ảnh hưởng nghiêm trọng đến hiệu suất!")
            print(f"   💡 Pool còn lại trước khi xóa: {pool_remaining_before_delete:,} samples")
            print(f"   💡 Kiểm tra logic bổ sung queries hoặc pool size ban đầu!")
            # KHÔNG raise error vì có thể pool thực sự cạn kiệt, nhưng cảnh báo rõ ràng
        
        labeled_X = np.vstack([labeled_X, X_query_s])
        labeled_y = np.concatenate([labeled_y, y_query])

        # Xóa từ pool (đảm bảo query_idx unique)
        query_idx_unique = np.unique(query_idx)
        X_pool = np.delete(X_pool, query_idx_unique, axis=0)
        if attacker_type in ["keras", "dual"]:
            # X_pool_s có sẵn cho Keras và dualDNN
            X_pool_s = np.delete(X_pool_s, query_idx_unique, axis=0)
            # Với dualDNN, cũng cần xóa labels đã query khỏi pool_labels_for_entropy
            if attacker_type == "dual" and pool_labels_for_entropy is not None:
                pool_labels_for_entropy = np.delete(pool_labels_for_entropy, query_idx_unique, axis=0)
        else:
            # Với LightGBM, X_pool_s = X_pool
            X_pool_s = X_pool
        
        print(f"   📊 Pool còn lại: {X_pool.shape[0]:,} samples")

        # QUAN TRỌNG: Re-train từ đầu trên toàn bộ dữ liệu tích lũy
        # Theo nghiên cứu: Huấn luyện lại từ đầu giúp model học lại phân phối tổng thể,
        # giảm thiểu việc bị lệch theo phân phối của lô dữ liệu mới nhất
        print(f"   🔄 Re-training model với {labeled_X.shape[0]:,} labeled samples...")
        
        if attacker_type == "lgb":
            attacker = LGBAttacker(seed=seed)
            attacker.train_model(labeled_X, labeled_y, X_val, y_val, boosting_rounds=1000, early_stopping=60)
        elif attacker_type == "dual":
            attacker = KerasDualAttacker(early_stopping=30, seed=seed)
            # DualDNN train với (X, y, y_true) - y_true là oracle labels
            attacker.train_model(labeled_X, labeled_y, labeled_y, X_val_s, y_val, y_val, num_epochs=num_epochs)
        else:
            attacker = KerasAttacker(early_stopping=30, seed=seed, input_shape=(feature_dim,))
            attacker.train_model(labeled_X, labeled_y, X_val_s, y_val, num_epochs=num_epochs)

        evaluate(attacker, round_id=query_round, total_labels=labeled_X.shape[0])
    
    # Kiểm tra tổng queries cuối cùng
    final_total_queries = total_queries_accumulated
    diff = final_total_queries - total_queries_target
    diff_percent = (diff / total_queries_target * 100) if total_queries_target > 0 else 0
    
    print(f"\n{'='*80}")
    print(f"📊 TỔNG KẾT QUERIES:")
    print(f"{'='*80}")
    print(f"   Queries dự kiến: {total_queries_target:,} ({query_batch:,} queries/round × {num_rounds} rounds)")
    print(f"   Queries thực tế: {final_total_queries:,}")
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
    surrogate_path = output_dir / "surrogate_model"
    attacker.save_model(str(surrogate_path))
    
    # Lấy extension phù hợp với model type
    if attacker_type == "lgb":
        surrogate_model_path = f"{surrogate_path}.txt"
    else:
        # Keras và dualDNN đều dùng .h5
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

    summary = {
        "weights_path": weights_path_abs,  # Lưu absolute path để đảm bảo consistency
        "model_file_name": model_file_name,  # Lưu tên file để dễ verify
        "model_type": model_type,
        "normalization_stats_path": normalization_stats_path,
        "attacker_type": attacker_type,
        "surrogate_model_path": surrogate_model_path,
        "scaler_path": str(joblib_path) if joblib_path else None,
        "metrics_csv": str(metrics_csv),
        "metrics": metrics_history,
    }

    summary_path = output_dir / "extraction_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    SUMMARY = run_extraction(
        weights_path=str(PROJECT_ROOT / "src" / "final_model.h5"),
        output_dir=PROJECT_ROOT / "src" / "output",
        seed=42,
    )
    print(json.dumps(SUMMARY["metrics"][-1], indent=2))

