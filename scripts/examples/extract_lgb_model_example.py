"""
Ví dụ script để thực hiện model extraction attack với target model LightGBM (.lgb)

Script này minh họa cách sử dụng FlexibleLGBTarget để tấn công model extraction
với target model là file .lgb (có normalization stats).

Usage:
    python scripts/examples/extract_lgb_model_example.py \
        --model_path path/to/model.lgb \
        --normalization_stats_path path/to/normalization_stats.npz \
        --output_dir path/to/output \
        --train_parquet path/to/train.parquet \
        --test_parquet path/to/test.parquet
"""
import json
import os
import sys
from pathlib import Path
import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.targets import FlexibleLGBTarget
from src.attackers import LGBAttacker
from src.sampling import entropy_sampling
from sklearn_extra.cluster import KMedoids


def load_data_simple(parquet_path, feature_cols, label_col, n_samples=None):
    """Load dữ liệu đơn giản từ parquet (ví dụ)"""
    import pyarrow.parquet as pq
    
    pq_file = pq.ParquetFile(parquet_path)
    batches = []
    count = 0
    
    for batch in pq_file.iter_batches(batch_size=10000, columns=feature_cols + [label_col]):
        batch_df = batch.to_pandas()
        
        # Loại bỏ label -1
        valid_mask = batch_df[label_col] != -1
        batch_df = batch_df[valid_mask]
        
        if len(batch_df) == 0:
            continue
        
        X = batch_df[feature_cols].values.astype(np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        y = batch_df[label_col].values.astype(np.int32)
        
        batches.append((X, y))
        count += len(X)
        
        if n_samples is not None and count >= n_samples:
            break
    
    if not batches:
        raise ValueError("No valid data found")
    
    X_all = np.vstack([b[0] for b in batches])
    y_all = np.concatenate([b[1] for b in batches])
    
    if n_samples is not None and len(X_all) > n_samples:
        X_all = X_all[:n_samples]
        y_all = y_all[:n_samples]
    
    return X_all, y_all


def main():
    parser = argparse.ArgumentParser(description="Model extraction attack với LightGBM target model")
    parser.add_argument("--model_path", type=str, required=True, 
                       help="Đường dẫn tới file model .lgb")
    parser.add_argument("--normalization_stats_path", type=str, required=True,
                       help="Đường dẫn tới file normalization_stats.npz")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Thư mục để lưu kết quả")
    parser.add_argument("--train_parquet", type=str, required=True,
                       help="Đường dẫn tới file train.parquet")
    parser.add_argument("--test_parquet", type=str, required=True,
                       help="Đường dẫn tới file test.parquet")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--seed_size", type=int, default=2000,
                       help="Số samples trong seed set")
    parser.add_argument("--val_size", type=int, default=2000,
                       help="Số samples trong validation set")
    parser.add_argument("--query_batch", type=int, default=2000,
                       help="Số queries mỗi round")
    parser.add_argument("--num_rounds", type=int, default=5,
                       help="Số rounds của active learning")
    parser.add_argument("--threshold", type=float, default=0.5,
                       help="Threshold cho binary classification")
    
    args = parser.parse_args()
    
    # Tạo output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("🔍 MODEL EXTRACTION ATTACK VỚI LIGHTGBM TARGET MODEL")
    print("=" * 60)
    
    # Load normalization stats để lấy feature_cols
    print(f"\n🔄 Đang load normalization stats từ {args.normalization_stats_path}...")
    stats = np.load(args.normalization_stats_path, allow_pickle=True)
    feature_cols = stats['feature_cols'].tolist() if hasattr(stats['feature_cols'], 'tolist') else stats['feature_cols']
    print(f"✅ Đã load {len(feature_cols)} feature columns")
    
    # Load target model với normalization stats
    print(f"\n🔄 Đang load target model từ {args.model_path}...")
    oracle = FlexibleLGBTarget(
        model_path=args.model_path,
        normalization_stats_path=args.normalization_stats_path,
        threshold=args.threshold,
        name="lgb-target"
    )
    print(f"✅ Target model yêu cầu {oracle.get_required_feature_dim()} features")
    
    # Load dữ liệu
    print(f"\n🔄 Đang load dữ liệu từ {args.train_parquet}...")
    X_train, y_train = load_data_simple(
        args.train_parquet, 
        feature_cols, 
        "Label",
        n_samples=args.seed_size + args.val_size + args.query_batch * args.num_rounds
    )
    print(f"✅ Train data: {X_train.shape}")
    
    print(f"\n🔄 Đang load test data từ {args.test_parquet}...")
    X_test, y_test = load_data_simple(
        args.test_parquet,
        feature_cols,
        "Label",
        n_samples=4000
    )
    print(f"✅ Test data: {X_test.shape}")
    
    # Chia dữ liệu
    rng = np.random.default_rng(args.seed)
    indices = rng.permutation(len(X_train))
    
    seed_end = args.seed_size
    val_end = seed_end + args.val_size
    pool_start = val_end
    
    X_seed = X_train[indices[:seed_end]]
    y_seed_true = y_train[indices[:seed_end]]
    
    X_val = X_train[indices[seed_end:val_end]]
    y_val_true = y_train[indices[seed_end:val_end]]
    
    X_pool = X_train[indices[pool_start:]]
    
    # Query oracle để lấy labels (VỚI DỮ LIỆU ĐÃ NORMALIZE)
    # FlexibleLGBTarget sẽ tự động normalize nếu có normalization_stats_path
    print(f"\n🔄 Đang query oracle để lấy labels...")
    y_seed = oracle(X_seed)
    y_val = oracle(X_val)
    y_test_target = oracle(X_test)
    
    print(f"✅ Seed distribution: {dict(zip(*np.unique(y_seed, return_counts=True)))}")
    print(f"✅ Val distribution: {dict(zip(*np.unique(y_val, return_counts=True)))}")
    print(f"✅ Test distribution: {dict(zip(*np.unique(y_test_target, return_counts=True)))}")
    
    # Train surrogate model
    metrics_history = []
    labeled_X = X_seed.copy()
    labeled_y = y_seed.copy()
    
    def evaluate(attacker, round_id, total_labels):
        """Evaluate surrogate model"""
        probs = attacker(X_test)
        preds = (probs >= 0.5).astype(int)
        
        agreement = (preds == y_test_target).mean()
        acc = accuracy_score(y_test_target, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test_target, preds, average="binary", zero_division=0
        )
        try:
            auc = roc_auc_score(y_test_target, probs)
        except ValueError:
            auc = float("nan")
        
        metrics = {
            "round": round_id,
            "labels_used": int(total_labels),
            "surrogate_acc": float(acc),
            "surrogate_auc": float(auc),
            "surrogate_precision": float(precision),
            "surrogate_recall": float(recall),
            "surrogate_f1": float(f1),
            "agreement_with_target": float(agreement),
        }
        metrics_history.append(metrics)
        
        print(f"\n📊 Round {round_id} Metrics:")
        print(f"   Labels used: {total_labels}")
        print(f"   Accuracy: {acc:.4f}")
        print(f"   Agreement: {agreement:.4f}")
        print(f"   AUC: {auc:.4f}")
        print(f"   F1: {f1:.4f}")
        
        return metrics
    
    # Initial training
    print(f"\n🔄 Đang train surrogate model ban đầu...")
    attacker = LGBAttacker(seed=args.seed)
    attacker.train_model(labeled_X, labeled_y, X_val, y_val, boosting_rounds=100, early_stopping=15)
    evaluate(attacker, round_id=0, total_labels=len(labeled_X))
    
    # Active learning rounds
    for round_id in range(1, args.num_rounds + 1):
        if len(X_pool) < args.query_batch:
            print(f"⚠️  Pool còn {len(X_pool)} samples, không đủ cho round {round_id}")
            break
        
        print(f"\n{'='*60}")
        print(f"🔄 Round {round_id}/{args.num_rounds}")
        print(f"{'='*60}")
        
        # Entropy sampling + k-medoids
        print(f"🔄 Đang chọn queries bằng Entropy + k-medoids...")
        entropy_candidates = min(10000, len(X_pool))
        q_idx = entropy_sampling(
            attacker,
            X_pool,
            np.zeros(len(X_pool)),  # y không cần thiết cho entropy
            n_instances=entropy_candidates
        )
        X_med = X_pool[q_idx]
        
        kmed = KMedoids(n_clusters=args.query_batch, init='k-medoids++', random_state=args.seed)
        kmed.fit(X_med)
        query_idx_in_med = kmed.medoid_indices_
        query_idx = q_idx[query_idx_in_med]
        
        X_query = X_pool[query_idx]
        y_query = oracle(X_query)  # Query với dữ liệu đã normalize tự động
        
        print(f"✅ Đã chọn {len(query_idx)} queries")
        print(f"   Query distribution: {dict(zip(*np.unique(y_query, return_counts=True)))}")
        
        # Thêm vào labeled set
        labeled_X = np.vstack([labeled_X, X_query])
        labeled_y = np.concatenate([labeled_y, y_query])
        
        # Xóa từ pool
        X_pool = np.delete(X_pool, query_idx, axis=0)
        
        # Re-train
        print(f"🔄 Đang re-train surrogate model với {len(labeled_X):,} labeled samples...")
        attacker = LGBAttacker(seed=args.seed)
        attacker.train_model(labeled_X, labeled_y, X_val, y_val, boosting_rounds=1000, early_stopping=60)
        evaluate(attacker, round_id=round_id, total_labels=len(labeled_X))
    
    # Save results
    print(f"\n{'='*60}")
    print(f"💾 Đang lưu kết quả...")
    print(f"{'='*60}")
    
    surrogate_path = output_dir / "surrogate_model"
    attacker.save_model(str(surrogate_path))
    print(f"✅ Surrogate model: {surrogate_path}.txt")
    
    df_metrics = pd.DataFrame(metrics_history)
    metrics_csv = output_dir / "extraction_metrics.csv"
    df_metrics.to_csv(metrics_csv, index=False)
    print(f"✅ Metrics CSV: {metrics_csv}")
    
    summary = {
        "model_path": args.model_path,
        "normalization_stats_path": args.normalization_stats_path,
        "surrogate_model_path": f"{surrogate_path}.txt",
        "metrics_csv": str(metrics_csv),
        "metrics": metrics_history,
    }
    
    summary_path = output_dir / "extraction_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"✅ Summary JSON: {summary_path}")
    
    print(f"\n{'='*60}")
    print(f"✅ HOÀN TẤT MODEL EXTRACTION ATTACK!")
    print(f"{'='*60}")
    print(f"\n📊 Metrics cuối cùng:")
    if metrics_history:
        final_metrics = metrics_history[-1]
        print(f"   Round: {final_metrics['round']}")
        print(f"   Labels used: {final_metrics['labels_used']}")
        print(f"   Accuracy: {final_metrics['surrogate_acc']:.4f}")
        print(f"   Agreement: {final_metrics['agreement_with_target']:.4f}")
        print(f"   AUC: {final_metrics['surrogate_auc']:.4f}")
        print(f"   F1: {final_metrics['surrogate_f1']:.4f}")


if __name__ == "__main__":
    main()

