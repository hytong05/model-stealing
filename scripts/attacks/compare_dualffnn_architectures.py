"""
Script để so sánh 3 kiến trúc dualFFNN: dualFFNN, dualFFNN-1 (deeper), dualFFNN-2 (narrower)
Cố định số queries tấn công là 1000 và so sánh kết quả giữa 3 kiến trúc.
"""
import json
import os
import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.dnn import create_dnn2, create_dnn2_deeper, create_dnn2_narrower
from src.attackers import AbstractAttacker


class KerasDualAttackerOriginal(AbstractAttacker):
    """KerasDualAttacker với kiến trúc dualFFNN gốc (create_dnn2)"""
    def __init__(self, early_stopping=30, seed=42, mc=False, input_shape=(2381,)):
        self.model = create_dnn2(seed=seed, mc=mc, input_shape=input_shape)
        self.checkpoint_filepath = '/tmp/checkpoint2_original.weights.h5'
        self.model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=self.checkpoint_filepath,
            save_weights_only=True,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True)
        self.early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=early_stopping)

    def train_model(self, X, y, y_true, X_val, y_val, y_val_true, num_epochs):
        self.model.fit((X, y_true), y,
            batch_size=128, 
            epochs=num_epochs, 
            validation_data=((X_val, y_val_true), y_val),
            callbacks=[self.model_checkpoint_callback, self.early_stopping])  
        self.model.load_weights(self.checkpoint_filepath)          

    def __call__(self, X, y_true):
        return self.model.predict((X, y_true), verbose=0)

    def save_model(self, path):
        self.model.save(path+".h5")


class KerasDualAttackerDeeper(AbstractAttacker):
    """KerasDualAttacker với kiến trúc dualFFNN-1 (deeper - create_dnn2_deeper)"""
    def __init__(self, early_stopping=30, seed=42, mc=False, input_shape=(2381,)):
        self.model = create_dnn2_deeper(seed=seed, mc=mc, input_shape=input_shape)
        self.checkpoint_filepath = '/tmp/checkpoint2_deeper.weights.h5'
        self.model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=self.checkpoint_filepath,
            save_weights_only=True,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True)
        self.early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=early_stopping)

    def train_model(self, X, y, y_true, X_val, y_val, y_val_true, num_epochs):
        self.model.fit((X, y_true), y,
            batch_size=128, 
            epochs=num_epochs, 
            validation_data=((X_val, y_val_true), y_val),
            callbacks=[self.model_checkpoint_callback, self.early_stopping])  
        self.model.load_weights(self.checkpoint_filepath)          

    def __call__(self, X, y_true):
        return self.model.predict((X, y_true), verbose=0)

    def save_model(self, path):
        self.model.save(path+".h5")


class KerasDualAttackerNarrower(AbstractAttacker):
    """KerasDualAttacker với kiến trúc dualFFNN-2 (narrower - create_dnn2_narrower)"""
    def __init__(self, early_stopping=30, seed=42, mc=False, input_shape=(2381,)):
        self.model = create_dnn2_narrower(seed=seed, mc=mc, input_shape=input_shape)
        self.checkpoint_filepath = '/tmp/checkpoint2_narrower.weights.h5'
        self.model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=self.checkpoint_filepath,
            save_weights_only=True,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True)
        self.early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=early_stopping)

    def train_model(self, X, y, y_true, X_val, y_val, y_val_true, num_epochs):
        self.model.fit((X, y_true), y,
            batch_size=128, 
            epochs=num_epochs, 
            validation_data=((X_val, y_val_true), y_val),
            callbacks=[self.model_checkpoint_callback, self.early_stopping])  
        self.model.load_weights(self.checkpoint_filepath)          

    def __call__(self, X, y_true):
        return self.model.predict((X, y_true), verbose=0)

    def save_model(self, path):
        self.model.save(path+".h5")


def _resolve_path(path_str: str) -> Path:
    """Resolve path (relative to PROJECT_ROOT nếu cần)."""
    path_obj = Path(path_str)
    if not path_obj.is_absolute():
        path_obj = PROJECT_ROOT / path_obj
    return path_obj.resolve()


def run_extraction_with_architecture(
    architecture_name: str,
    attacker_class,
    output_dir: Path,
    train_parquet=None,
    test_parquet=None,
    dataset: str = "ember",
    seed: int = 42,
    eval_size: int = 4000,
    total_budget: int = 1000,
    num_epochs: int = 100,
    model_type: str = None,
    normalization_stats_path: str = None,
    weights_path: str | None = None,
    model_name: str = None,
    threshold_optimization_metric: str = "f1",
    fixed_threshold: float | None = None,
    surrogate_name: str | None = None,
):
    """
    Chạy extraction với attacker class tùy chỉnh.
    
    Args:
        architecture_name: Tên kiến trúc (dualFFNN, dualFFNN-1, dualFFNN-2)
        attacker_class: Class của attacker (KerasDualAttackerOriginal, KerasDualAttackerDeeper, KerasDualAttackerNarrower)
        ... (các tham số khác giống run_extraction)
    """
    # Monkey-patch KerasDualAttacker trong cả src.attackers và extract_final_model
    import src.attackers as attackers_module
    original_attacker_class = attackers_module.KerasDualAttacker
    attackers_module.KerasDualAttacker = attacker_class
    
    # Import và patch trong extract_final_model module
    import scripts.attacks.extract_final_model as extract_module
    original_extract_attacker = extract_module.KerasDualAttacker
    extract_module.KerasDualAttacker = attacker_class
    
    from scripts.attacks.extract_final_model import run_extraction
    
    try:
        # Gọi run_extraction với attacker_type="dual"
        summary = run_extraction(
            output_dir=output_dir,
            train_parquet=train_parquet,
            test_parquet=test_parquet,
            dataset=dataset,
            seed=seed,
            eval_size=eval_size,
            total_budget=total_budget,
            num_epochs=num_epochs,
            model_type=model_type,
            normalization_stats_path=normalization_stats_path,
            attacker_type="dual",
            weights_path=weights_path,
            model_name=model_name,
            threshold_optimization_metric=threshold_optimization_metric,
            fixed_threshold=fixed_threshold,
            surrogate_name=surrogate_name,
        )
        
        summary['architecture_name'] = architecture_name
        return summary
    finally:
        # Khôi phục class gốc
        attackers_module.KerasDualAttacker = original_attacker_class
        extract_module.KerasDualAttacker = original_extract_attacker


def main():
    parser = argparse.ArgumentParser(
        description="So sánh 3 kiến trúc dualFFNN với số queries cố định 1000"
    )
    parser.add_argument("--model_name", type=str, default=None,
                       help="Tên model (CEE, LEE, CSE, LSE). Ưu tiên hơn --model_path.")
    parser.add_argument("--model_path", type=str, default=None,
                       help="Đường dẫn tới file model (.h5 hoặc .lgb). Chỉ dùng nếu không có --model_name")
    parser.add_argument("--model_type", type=str, choices=["h5", "lgb"], default=None,
                       help="Loại model: 'h5' (Keras) hoặc 'lgb' (LightGBM). Chỉ cần nếu dùng --model_path")
    parser.add_argument("--normalization_stats_path", type=str, default=None,
                       help="Đường dẫn tới file normalization_stats.npz. Chỉ cần nếu dùng --model_path với model_type='lgb'")
    parser.add_argument("--dataset", type=str, choices=["ember", "somlap"], default="ember",
                       help="Dataset để tấn công: 'ember' (mặc định) hoặc 'somlap'")
    parser.add_argument("--threshold_optimization_metric", type=str, choices=["f1", "accuracy", "balanced_accuracy"], default="f1",
                       help="Metric để tối ưu threshold: 'f1' (mặc định), 'accuracy', hoặc 'balanced_accuracy'")
    parser.add_argument("--fixed_threshold", type=float, default=None,
                       help="Sử dụng threshold cố định thay vì tối ưu (ví dụ: 0.5)")
    parser.add_argument("--total_queries", type=int, default=1000,
                       help="Tổng số queries tấn công (mặc định: 1000)")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.model_name is None and args.model_path is None:
        # Tự động tìm model file nếu không được chỉ định
        pass
    elif args.model_name is not None and args.model_path is not None:
        raise ValueError("❌ Chỉ cung cấp --model_name HOẶC --model_path, không phải cả hai")
    
    # Xử lý model_name hoặc model_path
    model_name = args.model_name.upper().strip() if args.model_name else None
    weights_path = None
    
    if model_name is not None:
        print(f"✅ Sử dụng model name: {model_name}")
        print(f"   Sẽ tự động detect model type và tìm normalization stats")
    elif args.model_path is None:
        # Tự động tìm model
        possible_models = [
            PROJECT_ROOT / "artifacts" / "targets" / "CEE.h5",
            PROJECT_ROOT / "artifacts" / "targets" / "CSE.h5",
            PROJECT_ROOT / "artifacts" / "targets" / "LEE.lgb",
            PROJECT_ROOT / "artifacts" / "targets" / "LSE.lgb",
        ]
        
        for model_path in possible_models:
            if model_path.exists():
                weights_path = str(model_path.resolve())
                print(f"✅ Tự động tìm thấy model: {weights_path}")
                break
        
        if weights_path is None:
            raise FileNotFoundError(
                f"Không tìm thấy file model nào. Vui lòng chỉ định bằng --model_path. "
                f"Đã tìm tại: {[str(p) for p in possible_models]}"
            )
    else:
        weights_path_obj = Path(args.model_path)
        if not weights_path_obj.is_absolute():
            weights_path = str((PROJECT_ROOT / args.model_path).resolve())
        else:
            weights_path = str(weights_path_obj.resolve())
        
        if not Path(weights_path).exists():
            raise FileNotFoundError(
                f"❌ Model file không tồn tại: {weights_path}\n"
                f"   Đã thử resolve từ: {args.model_path}"
            )
    
    if model_name is None:
        # Xử lý model_type nếu cần
        if args.model_type is None:
            model_path_obj = Path(weights_path)
            if model_path_obj.suffix.lower() in ['.lgb', '.txt', '.d5']:
                args.model_type = "lgb"
                print(f"✅ Tự động phát hiện model type: LGB (từ extension {model_path_obj.suffix})")
            elif model_path_obj.suffix.lower() in ['.h5', '.hdf5']:
                args.model_type = "h5"
                print(f"✅ Tự động phát hiện model type: H5 (từ extension {model_path_obj.suffix})")
            else:
                args.model_type = "h5"
                print(f"⚠️  Không thể phát hiện model type từ extension, mặc định: H5")
    
    # Xử lý normalization_stats_path nếu cần
    normalization_stats_path = None
    if model_name is None and args.model_type in ["lgb", "sklearn"] and args.normalization_stats_path is None:
        model_path_obj = Path(weights_path)
        model_name_without_ext = model_path_obj.stem
        possible_stats_paths = [
            model_path_obj.parent / f"{model_name_without_ext}.npz",
            model_path_obj.parent / f"{model_name_without_ext}_normalization_stats.npz",
            model_path_obj.parent / "normalization_stats.npz",
            PROJECT_ROOT / "artifacts" / "targets" / "normalization_stats.npz",
        ]
        
        for stats_path in possible_stats_paths:
            if stats_path.exists():
                normalization_stats_path = str(stats_path.resolve())
                print(f"✅ Tự động tìm thấy normalization stats: {normalization_stats_path}")
                break
    elif args.normalization_stats_path is not None:
        stats_path_obj = Path(args.normalization_stats_path)
        if not stats_path_obj.is_absolute():
            normalization_stats_path = str((PROJECT_ROOT / args.normalization_stats_path).resolve())
        else:
            normalization_stats_path = str(stats_path_obj.resolve())
        
        if not Path(normalization_stats_path).exists():
            raise FileNotFoundError(
                f"❌ Normalization stats file không tồn tại: {normalization_stats_path}"
            )
    
    # Xác định tên model target
    if model_name:
        target_model_name = model_name.upper()
    else:
        if weights_path:
            target_model_name = Path(weights_path).stem.upper()
        else:
            target_model_name = "UNKNOWN"
    
    dataset_name = args.dataset.lower()
    total_queries = args.total_queries
    
    # Cấu hình 3 kiến trúc
    architectures = [
        {
            "name": "dualFFNN",
            "description": "dualFFNN - Kiến trúc gốc",
            "attacker_class": KerasDualAttackerOriginal,
        },
        {
            "name": "dualFFNN-1",
            "description": "dualFFNN-1 - Deeper Network (2382→2382→1024→512→128→64→32→1)",
            "attacker_class": KerasDualAttackerDeeper,
        },
        {
            "name": "dualFFNN-2",
            "description": "dualFFNN-2 - Narrower Network (2382→1024→512→256→64→1)",
            "attacker_class": KerasDualAttackerNarrower,
        },
    ]
    
    base_output_dir = PROJECT_ROOT / "output"
    
    # Tạo thư mục output chính cho comparison
    comparison_output_dir = base_output_dir / f"dualFFNN_comparison_{target_model_name}_{dataset_name}_{total_queries}"
    comparison_output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("BẮT ĐẦU SO SÁNH 3 KIẾN TRÚC DUALFFNN")
    print("=" * 80)
    print(f"\n📋 Cấu hình:")
    print(f"   ✅ Target model: {target_model_name}")
    print(f"   ✅ Dataset: {args.dataset.upper()}")
    print(f"   ✅ Tổng queries: {total_queries:,}")
    print(f"   ✅ Số kiến trúc: {len(architectures)}")
    print(f"   ✅ Output directory: {comparison_output_dir}")
    print("=" * 80)
    
    results = []
    train_parquet = None
    test_parquet = None
    
    for arch_config in architectures:
        arch_name = arch_config["name"]
        arch_description = arch_config["description"]
        attacker_class = arch_config["attacker_class"]
        
        print(f"\n{'='*80}")
        print(f"🔬 KIẾN TRÚC: {arch_name}")
        print(f"   {arch_description}")
        print(f"{'='*80}\n")
        
        # Tạo thư mục output cho từng kiến trúc
        arch_output_dir = comparison_output_dir / arch_name
        arch_output_dir.mkdir(parents=True, exist_ok=True)
        
        surrogate_name = f"surrogate_{target_model_name}_{arch_name}"
        
        try:
            summary = run_extraction_with_architecture(
                architecture_name=arch_name,
                attacker_class=attacker_class,
                output_dir=arch_output_dir,
                train_parquet=train_parquet,
                test_parquet=test_parquet,
                dataset=args.dataset,
                seed=42,
                eval_size=4000,
                total_budget=total_queries,
                num_epochs=100,
                model_type=args.model_type,
                normalization_stats_path=normalization_stats_path,
                weights_path=weights_path if model_name is None else None,
                model_name=model_name,
                threshold_optimization_metric=args.threshold_optimization_metric,
                fixed_threshold=args.fixed_threshold,
                surrogate_name=surrogate_name,
            )
            
            # Lấy metrics cuối cùng
            final_metrics = summary["metrics"][-1] if summary["metrics"] else {}
            
            result = {
                "architecture": arch_name,
                "description": arch_description,
                "total_queries": total_queries,
                "actual_queries_used": summary.get("total_queries_actual", total_queries),
                "query_batch": summary.get("query_batch", 0),
                "num_rounds": summary.get("num_rounds", 0),
                "seed_size": summary.get("seed_size", 0),
                "val_size": summary.get("val_size", 0),
                "total_labels_used": final_metrics.get("labels_used", 0),
                "optimal_threshold": final_metrics.get("optimal_threshold", 0.5),
                "final_accuracy": final_metrics.get("surrogate_acc", 0.0),
                "final_balanced_accuracy": final_metrics.get("surrogate_balanced_acc", 0.0),
                "final_auc": final_metrics.get("surrogate_auc", float("nan")),
                "final_precision": final_metrics.get("surrogate_precision", 0.0),
                "final_recall": final_metrics.get("surrogate_recall", 0.0),
                "final_f1": final_metrics.get("surrogate_f1", 0.0),
                "final_agreement": final_metrics.get("agreement_with_target", 0.0),
                "output_dir": str(arch_output_dir),
                "metrics_csv": summary.get("metrics_csv", ""),
                "surrogate_model_path": summary.get("surrogate_model_path", ""),
            }
            
            results.append(result)
            
            print(f"\n{'='*80}")
            print(f"✅ Hoàn thành {arch_name}")
            print(f"{'='*80}")
            print(f"   Accuracy: {result['final_accuracy']:.4f}")
            print(f"   Balanced Accuracy: {result['final_balanced_accuracy']:.4f}")
            print(f"   F1-score: {result['final_f1']:.4f}")
            print(f"   Agreement: {result['final_agreement']:.4f}")
            print(f"   Optimal Threshold: {result['optimal_threshold']:.4f}")
            
        except Exception as e:
            print(f"\n❌ Lỗi khi chạy {arch_name}: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "architecture": arch_name,
                "description": arch_description,
                "error": str(e)
            })
    
    # Tạo report so sánh
    print(f"\n{'='*80}")
    print("📊 TẠO REPORT SO SÁNH")
    print(f"{'='*80}\n")
    
    report_path = comparison_output_dir / "comparison_report.txt"
    report_md_path = comparison_output_dir / "comparison_report.md"
    
    # Tạo report text
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("BÁO CÁO SO SÁNH 3 KIẾN TRÚC DUALFFNN\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("TÓM TẮT:\n")
        f.write("-" * 80 + "\n")
        f.write(f"Target model: {target_model_name}\n")
        f.write(f"Dataset: {args.dataset.upper()}\n")
        f.write(f"Tổng queries: {total_queries:,}\n")
        f.write(f"Số kiến trúc: {len(architectures)}\n\n")
        
        f.write("KẾT QUẢ SO SÁNH:\n")
        f.write("-" * 80 + "\n\n")
        
        for result in results:
            if "error" in result:
                f.write(f"❌ {result['architecture']}: LỖI - {result['error']}\n\n")
            else:
                f.write(f"✅ {result['architecture']} ({result['description']}):\n")
                f.write(f"   - Queries thực tế: {result.get('actual_queries_used', total_queries):,}\n")
                f.write(f"   - Labels sử dụng: {result['total_labels_used']:,}\n")
                f.write(f"   - Accuracy: {result['final_accuracy']:.4f} ({result['final_accuracy']*100:.2f}%)\n")
                f.write(f"   - Balanced Accuracy: {result['final_balanced_accuracy']:.4f} ({result['final_balanced_accuracy']*100:.2f}%)\n")
                f.write(f"   - F1-score: {result['final_f1']:.4f}\n")
                f.write(f"   - Agreement: {result['final_agreement']:.4f} ({result['final_agreement']*100:.2f}%)\n")
                f.write(f"   - Optimal Threshold: {result['optimal_threshold']:.4f}\n")
                if not pd.isna(result['final_auc']):
                    f.write(f"   - AUC: {result['final_auc']:.4f}\n")
                f.write(f"   - Precision: {result['final_precision']:.4f}\n")
                f.write(f"   - Recall: {result['final_recall']:.4f}\n")
                f.write(f"   - Output: {result['output_dir']}\n\n")
        
        # Tìm kiến trúc tốt nhất cho từng metric
        if all("error" not in r for r in results):
            f.write("\n" + "=" * 80 + "\n")
            f.write("KIẾN TRÚC TỐT NHẤT:\n")
            f.write("=" * 80 + "\n\n")
            
            metrics_to_compare = [
                ("final_accuracy", "Accuracy"),
                ("final_balanced_accuracy", "Balanced Accuracy"),
                ("final_f1", "F1-score"),
                ("final_agreement", "Agreement"),
            ]
            
            for metric_key, metric_name in metrics_to_compare:
                best_result = max(results, key=lambda x: x.get(metric_key, 0))
                f.write(f"{metric_name}: {best_result['architecture']} ({best_result[metric_key]:.4f})\n")
    
    # Tạo report Markdown
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write("# Báo Cáo So Sánh 3 Kiến Trúc dualFFNN\n\n")
        f.write("## Tóm Tắt\n\n")
        f.write(f"- **Target model:** {target_model_name}\n")
        f.write(f"- **Dataset:** {args.dataset.upper()}\n")
        f.write(f"- **Tổng queries:** {total_queries:,}\n")
        f.write(f"- **Số kiến trúc:** {len(architectures)}\n\n")
        
        f.write("## Bảng So Sánh\n\n")
        f.write("| Kiến trúc | Queries | Labels | Accuracy | Balanced Acc | F1 | Agreement | Threshold | AUC |\n")
        f.write("|-----------|---------|--------|----------|--------------|----|-----------|-----------|-----|\n")
        
        for result in results:
            if "error" not in result:
                auc_str = f"{result['final_auc']:.4f}" if not pd.isna(result['final_auc']) else "N/A"
                actual_queries = result.get('actual_queries_used', total_queries)
                balanced_acc = result.get('final_balanced_accuracy', 0.0)
                threshold = result.get('optimal_threshold', 0.5)
                f.write(f"| {result['architecture']} | {actual_queries:,} | {result['total_labels_used']:,} | "
                       f"{result['final_accuracy']:.4f} | {balanced_acc:.4f} | {result['final_f1']:.4f} | "
                       f"{result['final_agreement']:.4f} | {threshold:.3f} | {auc_str} |\n")
            else:
                f.write(f"| {result['architecture']} | ERROR | - | - | - | - | - | - | - |\n")
        
        f.write("\n## Chi Tiết Từng Kiến Trúc\n\n")
        
        for result in results:
            if "error" not in result:
                f.write(f"### {result['architecture']}\n\n")
                f.write(f"**Mô tả:** {result['description']}\n\n")
                f.write(f"- Query batch: {result['query_batch']:,}\n")
                f.write(f"- Số rounds: {result['num_rounds']}\n")
                f.write(f"- Queries thực tế: {result.get('actual_queries_used', total_queries):,}\n")
                f.write(f"- Tổng labels sử dụng: {result['total_labels_used']:,}\n\n")
                
                f.write("**Metrics:**\n\n")
                f.write(f"- Accuracy: {result['final_accuracy']:.4f} ({result['final_accuracy']*100:.2f}%)\n")
                f.write(f"- Balanced Accuracy: {result['final_balanced_accuracy']:.4f} ({result['final_balanced_accuracy']*100:.2f}%)\n")
                f.write(f"- F1-score: {result['final_f1']:.4f}\n")
                f.write(f"- Optimal Threshold: {result['optimal_threshold']:.4f}\n")
                f.write(f"- Agreement: {result['final_agreement']:.4f} ({result['final_agreement']*100:.2f}%)\n")
                if not pd.isna(result['final_auc']):
                    f.write(f"- AUC: {result['final_auc']:.4f}\n")
                f.write(f"- Precision: {result['final_precision']:.4f}\n")
                f.write(f"- Recall: {result['final_recall']:.4f}\n\n")
                
                f.write("**Files:**\n\n")
                f.write(f"- Metrics CSV: `{result['metrics_csv']}`\n")
                f.write(f"- Surrogate model: `{result['surrogate_model_path']}`\n")
                f.write(f"- Output directory: `{result['output_dir']}`\n\n")
    
    # Lưu JSON summary
    json_path = comparison_output_dir / "comparison_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Đã tạo report:")
    print(f"   - Text report: {report_path}")
    print(f"   - Markdown report: {report_md_path}")
    print(f"   - JSON summary: {json_path}")
    
    # In tóm tắt ra console
    print(f"\n{'='*80}")
    print("TÓM TẮT KẾT QUẢ:")
    print(f"{'='*80}\n")
    
    df_results = pd.DataFrame([r for r in results if "error" not in r])
    if not df_results.empty:
        df_display = df_results[["architecture", "final_accuracy", "final_balanced_accuracy", 
                                 "final_f1", "final_agreement", "optimal_threshold"]].copy()
        print(df_display.to_string(index=False))
    
    return results


if __name__ == "__main__":
    main()
