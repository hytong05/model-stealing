"""
Script để đánh giá độ tương đồng giữa target model và các surrogate models
sử dụng dữ liệu từ train_ember_2018_v2_features_label_minus1.parquet
"""
import json
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.targets import KerasCNNTarget
from src.attackers import KerasAttacker


def get_feature_columns(parquet_path: str, label_col: str = "Label") -> list:
    """Lấy danh sách feature columns từ parquet file."""
    pq_file = pq.ParquetFile(parquet_path)
    return [name for name in pq_file.schema.names if name != label_col]


def load_test_data(parquet_path: str, feature_cols: list, max_samples: int = 10000):
    """Load dữ liệu test từ parquet file (bỏ qua nhãn -1)."""
    pq_file = pq.ParquetFile(parquet_path)
    all_X = []
    rows_loaded = 0
    
    print(f"🔄 Đang load dữ liệu từ {parquet_path}...")
    
    for batch in pq_file.iter_batches(batch_size=5000, columns=feature_cols + ["Label"]):
        if rows_loaded >= max_samples:
            break
            
        batch_df = batch.to_pandas()
        
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
        print(f"✅ Đã load {len(X_concat):,} samples")
        return X_concat
    else:
        return np.empty((0, len(feature_cols)), dtype=np.float32)


def load_surrogate_model(model_path: str, scaler_path: str, feature_dim: int = 2381):
    """Load surrogate model và scaler."""
    import joblib
    import tensorflow as tf
    
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
    
    # Load scaler
    scaler = joblib.load(scaler_path)
    
    # Load model với compile=False để tránh lỗi compatibility giữa các version Keras
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
    except Exception as e:
        # Nếu vẫn lỗi, thử load với safe_mode=False
        try:
            model = tf.keras.models.load_model(model_path, compile=False, safe_mode=False)
        except Exception as e2:
            # Nếu vẫn lỗi, thử load weights thủ công
            print(f"⚠️  Không thể load model với compile=False, thử cách khác: {e2}")
            raise e2
    
    def predict(X):
        # Scale data
        X_scaled = scaler.transform(X)
        X_scaled = np.clip(X_scaled, -5, 5)
        # Predict
        probs = np.squeeze(model.predict(X_scaled, verbose=0), axis=-1)
        # Nếu model output là 2D (softmax), lấy class 1
        if probs.ndim > 1 and probs.shape[-1] == 2:
            probs = probs[:, 1]
        return (probs >= 0.5).astype(int), probs
    
    return predict


def evaluate_model_similarity(
    target_model,
    surrogate_predict,
    X_test,
    y_target,
    model_name: str
):
    """Đánh giá độ tương đồng giữa target và surrogate model."""
    print(f"\n🔄 Đang đánh giá {model_name}...")
    
    # Predict với surrogate
    y_surrogate, probs_surrogate = surrogate_predict(X_test)
    
    # Tính metrics
    accuracy = accuracy_score(y_target, y_surrogate)
    agreement = (y_target == y_surrogate).mean()
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_target, y_surrogate, average="binary", zero_division=0
    )
    
    try:
        auc = roc_auc_score(y_target, probs_surrogate)
    except ValueError:
        auc = float("nan")
    
    # Confusion matrix
    cm = confusion_matrix(y_target, y_surrogate)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    
    # Phân bố predictions
    target_dist = dict(zip(*np.unique(y_target, return_counts=True)))
    surrogate_dist = dict(zip(*np.unique(y_surrogate, return_counts=True)))
    
    metrics = {
        "model_name": model_name,
        "accuracy": float(accuracy),
        "agreement": float(agreement),
        "auc": float(auc) if not np.isnan(auc) else None,
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp)
        },
        "target_distribution": {int(k): int(v) for k, v in target_dist.items()},
        "surrogate_distribution": {int(k): int(v) for k, v in surrogate_dist.items()},
    }
    
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Agreement: {agreement:.4f}")
    print(f"  AUC: {auc:.4f}" if not np.isnan(auc) else "  AUC: NaN")
    
    return metrics


def main():
    # Đường dẫn
    test_parquet = str(PROJECT_ROOT / "data" / "train_ember_2018_v2_features_label_minus1.parquet")
    target_model_path = str(PROJECT_ROOT / "src" / "final_model.h5")
    output_dir = PROJECT_ROOT / "logs" / "evaluation"
    
    # Các surrogate models
    surrogate_configs = [
        {
            "name": "surrogate_model.h5 (attack_run)",
            "model_path": PROJECT_ROOT / "output" / "attack_run" / "surrogate_model.h5",
            "scaler_path": PROJECT_ROOT / "output" / "attack_run" / "robust_scaler.joblib",
        },
        {
            "name": "surrogate_model.h5 (attack_run_5000)",
            "model_path": PROJECT_ROOT / "output" / "attack_run_5000" / "surrogate_model.h5",
            "scaler_path": PROJECT_ROOT / "output" / "attack_run_5000" / "robust_scaler.joblib",
        },
    ]
    
    print("=" * 80)
    print("ĐÁNH GIÁ ĐỘ TƯƠNG ĐỒNG GIỮA TARGET VÀ SURROGATE MODELS")
    print("=" * 80)
    
    # Load feature columns
    feature_cols = get_feature_columns(test_parquet)
    print(f"\nFeature columns: {len(feature_cols)}")
    
    # Load test data
    X_test = load_test_data(test_parquet, feature_cols, max_samples=10000)
    
    if len(X_test) == 0:
        print("❌ Không có dữ liệu để test!")
        return
    
    # Load target model
    print(f"\n🔄 Đang load target model...")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
    
    target_model = KerasCNNTarget(target_model_path, feature_dim=len(feature_cols))
    
    # Query target model để lấy nhãn thực tế
    print(f"\n🔄 Đang query target model để lấy nhãn...")
    y_target = target_model(X_test)
    print(f"✅ Đã lấy nhãn từ target model")
    print(f"  Phân bố nhãn: {dict(zip(*np.unique(y_target, return_counts=True)))}")
    
    # Đánh giá từng surrogate model
    all_results = []
    
    for config in surrogate_configs:
        if not config["model_path"].exists() or not config["scaler_path"].exists():
            print(f"\n⚠️  Không tìm thấy model hoặc scaler cho {config['name']}")
            continue
        
        try:
            surrogate_predict = load_surrogate_model(
                str(config["model_path"]),
                str(config["scaler_path"]),
                feature_dim=len(feature_cols)
            )
            
            metrics = evaluate_model_similarity(
                target_model,
                surrogate_predict,
                X_test,
                y_target,
                config["name"]
            )
            
            all_results.append(metrics)
            
        except Exception as e:
            print(f"\n❌ Lỗi khi đánh giá {config['name']}: {e}")
            import traceback
            traceback.print_exc()
    
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
        
        f.write("THÔNG TIN DỮ LIỆU TEST:\n")
        f.write("-" * 80 + "\n")
        f.write(f"File: {test_parquet}\n")
        f.write(f"Số samples: {len(X_test):,}\n")
        f.write(f"Phân bố nhãn từ target model: {dict(zip(*np.unique(y_target, return_counts=True)))}\n\n")
        
        f.write("KẾT QUẢ ĐÁNH GIÁ:\n")
        f.write("-" * 80 + "\n\n")
        
        for result in all_results:
            f.write(f"{result['model_name'].upper().replace('_', ' ')}:\n")
            f.write(f"  Accuracy: {result['accuracy']:.4f} ({result['accuracy']*100:.2f}%)\n")
            f.write(f"  Agreement: {result['agreement']:.4f} ({result['agreement']*100:.2f}%)\n")
            if result['auc'] is not None:
                f.write(f"  AUC: {result['auc']:.4f}\n")
            else:
                f.write(f"  AUC: NaN\n")
            f.write(f"  Precision: {result['precision']:.4f}\n")
            f.write(f"  Recall: {result['recall']:.4f}\n")
            f.write(f"  F1-score: {result['f1_score']:.4f}\n")
            f.write(f"  Confusion Matrix:\n")
            f.write(f"    TN: {result['confusion_matrix']['tn']}, FP: {result['confusion_matrix']['fp']}\n")
            f.write(f"    FN: {result['confusion_matrix']['fn']}, TP: {result['confusion_matrix']['tp']}\n")
            f.write(f"  Target distribution: {result['target_distribution']}\n")
            f.write(f"  Surrogate distribution: {result['surrogate_distribution']}\n")
            f.write("\n")
        
        # So sánh
        f.write("\n" + "=" * 80 + "\n")
        f.write("SO SÁNH CÁC SURROGATE MODELS:\n")
        f.write("=" * 80 + "\n\n")
        
        if all_results:
            best_acc = max(all_results, key=lambda x: x['accuracy'])
            best_agreement = max(all_results, key=lambda x: x['agreement'])
            best_auc = max([r for r in all_results if r['auc'] is not None], key=lambda x: x['auc'], default=None)
            
            f.write(f"Best Accuracy: {best_acc['model_name']} ({best_acc['accuracy']:.4f})\n")
            f.write(f"Best Agreement: {best_agreement['model_name']} ({best_agreement['agreement']:.4f})\n")
            if best_auc:
                f.write(f"Best AUC: {best_auc['model_name']} ({best_auc['auc']:.4f})\n")
    
    # Markdown report
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write("# Báo Cáo Đánh Giá Độ Tương Đồng Giữa Target và Surrogate Models\n\n")
        
        f.write("## Thông Tin Dữ Liệu Test\n\n")
        f.write(f"- **File**: `{test_parquet}`\n")
        f.write(f"- **Số samples**: {len(X_test):,}\n")
        f.write(f"- **Phân bố nhãn từ target model**: {dict(zip(*np.unique(y_target, return_counts=True)))}\n\n")
        
        f.write("## Kết Quả Đánh Giá\n\n")
        f.write("| Model | Accuracy | Agreement | AUC | Precision | Recall | F1 |\n")
        f.write("|-------|----------|-----------|-----|-----------|--------|----|\n")
        
        for result in all_results:
            auc_str = f"{result['auc']:.4f}" if result['auc'] is not None else "N/A"
            f.write(f"| {result['model_name']} | {result['accuracy']:.4f} | "
                   f"{result['agreement']:.4f} | {auc_str} | {result['precision']:.4f} | "
                   f"{result['recall']:.4f} | {result['f1_score']:.4f} |\n")
        
        f.write("\n## Chi Tiết Từng Model\n\n")
        
        for result in all_results:
            f.write(f"### {result['model_name'].replace('_', ' ').title()}\n\n")
            f.write(f"- **Accuracy**: {result['accuracy']:.4f} ({result['accuracy']*100:.2f}%)\n")
            f.write(f"- **Agreement**: {result['agreement']:.4f} ({result['agreement']*100:.2f}%)\n")
            if result['auc'] is not None:
                f.write(f"- **AUC**: {result['auc']:.4f}\n")
            else:
                f.write(f"- **AUC**: NaN\n")
            f.write(f"- **Precision**: {result['precision']:.4f}\n")
            f.write(f"- **Recall**: {result['recall']:.4f}\n")
            f.write(f"- **F1-score**: {result['f1_score']:.4f}\n\n")
            
            f.write("**Confusion Matrix:**\n\n")
            f.write(f"| | Predicted 0 | Predicted 1 |\n")
            f.write(f"|------|------------|-------------|\n")
            f.write(f"| Actual 0 | {result['confusion_matrix']['tn']} | {result['confusion_matrix']['fp']} |\n")
            f.write(f"| Actual 1 | {result['confusion_matrix']['fn']} | {result['confusion_matrix']['tp']} |\n\n")
            
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
        print(df[["model_name", "accuracy", "agreement", "auc", "f1_score"]].to_string(index=False))
    
    return all_results


if __name__ == "__main__":
    main()

