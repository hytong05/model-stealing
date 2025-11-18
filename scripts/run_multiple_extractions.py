"""
Script để chạy extraction với các số lượng queries khác nhau và tạo report

Hỗ trợ cả target model .h5 (Keras) và .lgb (LightGBM)
"""
import json
import sys
import argparse
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.extract_final_model import run_extraction


def main():
    parser = argparse.ArgumentParser(description="Chạy model extraction với nhiều cấu hình")
    parser.add_argument("--model_path", type=str, default=None,
                       help="Đường dẫn tới file model (.h5 hoặc .lgb). Mặc định: tự động tìm")
    parser.add_argument("--model_type", type=str, choices=["h5", "lgb"], default=None,
                       help="Loại model: 'h5' (Keras) hoặc 'lgb' (LightGBM). Mặc định: tự động phát hiện từ extension")
    parser.add_argument("--normalization_stats_path", type=str, default=None,
                       help="Đường dẫn tới file normalization_stats.npz. Mặc định: tự động tìm")
    parser.add_argument("--attacker_type", type=str, choices=["keras", "lgb", "dual"], default=None,
                       help="Loại surrogate model: 'keras' (DNN), 'lgb' (LightGBM), hoặc 'dual' (dualDNN). Mặc định: tự động theo model_type")
    parser.add_argument("--auto_create_stats", action="store_true", default=False,
                       help="Tự động tạo file normalization stats nếu không tìm thấy (chỉ cho model .lgb)")
    args = parser.parse_args()
    
    # Tự động tìm model file nếu không được chỉ định
    if args.model_path is None:
        # Thử tìm các file model phổ biến
        possible_models = [
            PROJECT_ROOT / "src" / "final_model.h5",
            PROJECT_ROOT / "src" / "final_model_LEE.lgb",
            PROJECT_ROOT / "src" / "final_model_LSE.lgb",
            PROJECT_ROOT / "src" / "best_model.lgb",
            PROJECT_ROOT / "src" / "final_model.lgb",
        ]
        
        weights_path = None
        for model_path in possible_models:
            if model_path.exists():
                weights_path = str(model_path.resolve())  # Convert to absolute path
                print(f"✅ Tự động tìm thấy model: {weights_path}")
                break
        
        if weights_path is None:
            raise FileNotFoundError(
                f"Không tìm thấy file model nào. Vui lòng chỉ định bằng --model_path. "
                f"Đã tìm tại: {[str(p) for p in possible_models]}"
            )
    else:
        # Convert user-provided path to absolute path
        weights_path_obj = Path(args.model_path)
        if not weights_path_obj.is_absolute():
            weights_path = str((PROJECT_ROOT / args.model_path).resolve())
        else:
            weights_path = str(weights_path_obj.resolve())
        
        # Validate model file exists
        if not Path(weights_path).exists():
            raise FileNotFoundError(
                f"❌ Model file không tồn tại: {weights_path}\n"
                f"   Đã thử resolve từ: {args.model_path}"
            )
    
    # Tự động phát hiện model_type từ extension nếu không được chỉ định
    if args.model_type is None:
        model_path_obj = Path(weights_path)
        if model_path_obj.suffix.lower() in ['.lgb', '.txt', '.pkl', '.d5']:
            args.model_type = "lgb"
            print(f"✅ Tự động phát hiện model type: LGB (từ extension {model_path_obj.suffix})")
        elif model_path_obj.suffix.lower() in ['.h5', '.hdf5']:
            args.model_type = "h5"
            print(f"✅ Tự động phát hiện model type: H5 (từ extension {model_path_obj.suffix})")
        else:
            # Mặc định là h5
            args.model_type = "h5"
            print(f"⚠️  Không thể phát hiện model type từ extension, mặc định: H5")
    
    # QUAN TRỌNG: Đảm bảo weights_path là absolute path và validate
    weights_path_abs = str(Path(weights_path).resolve())
    weights_path = weights_path_abs  # Update để dùng cho phần còn lại
    
    # Validate model file exists
    if not Path(weights_path).exists():
        raise FileNotFoundError(f"❌ Model file không tồn tại: {weights_path}")
    
    # Get model info for verification
    model_path_obj = Path(weights_path)
    model_name = model_path_obj.name
    model_size = model_path_obj.stat().st_size / (1024 * 1024)  # MB
    
    print(f"\n✅ Đã xác nhận target model:")
    print(f"   Path (absolute): {weights_path}")
    print(f"   File name: {model_name}")
    print(f"   File size: {model_size:.2f} MB")
    
    # Kiểm tra normalization stats cho LightGBM
    if args.model_type == "lgb" and args.normalization_stats_path is None:
        # Tự động tìm file normalization stats dựa trên tên model
        model_name_without_ext = model_path_obj.stem  # Lấy tên file không có extension
        
        # Thử các pattern phổ biến:
        # 1. final_model_LEE.npz (cùng tên với model)
        # 2. final_model_LEE_normalization_stats.npz
        # 3. normalization_stats.npz (mặc định)
        possible_stats_paths = [
            model_path_obj.parent / f"{model_name_without_ext}.npz",
            model_path_obj.parent / f"{model_name_without_ext}_normalization_stats.npz",
            model_path_obj.parent / "normalization_stats.npz",
            PROJECT_ROOT / "src" / "normalization_stats.npz",
        ]
        
        normalization_stats_path = None
        for stats_path in possible_stats_paths:
            if stats_path.exists():
                normalization_stats_path = str(stats_path.resolve())  # Absolute path
                print(f"✅ Tự động tìm thấy normalization stats: {normalization_stats_path}")
                print(f"   Stats file: {Path(normalization_stats_path).name}")
                break
        
        # Nếu không tìm thấy và cho phép auto-create
        if normalization_stats_path is None and args.auto_create_stats:
            print(f"\n⚠️  KHÔNG TÌM THẤY file normalization stats!")
            print(f"   🔄 Đang tự động tạo file normalization stats...")
            try:
                # Import function để tạo stats
                from scripts.create_normalization_stats import (
                    get_feature_columns,
                    compute_normalization_stats,
                )

                # Tìm training parquet file
                train_parquet = PROJECT_ROOT / "data" / "train_ember_2018_v2_features_label_other.parquet"
                if not train_parquet.exists():
                    raise FileNotFoundError(f"Không tìm thấy training data: {train_parquet}")

                # Tạo file stats với tên tương ứng với model
                output_stats_path = model_path_obj.parent / f"{model_name_without_ext}_normalization_stats.npz"
                label_col = "Label"

                print(f"   📊 Đang đọc features từ {train_parquet}...")
                feature_cols = get_feature_columns(str(train_parquet), label_col)

                print(f"   📊 Đang tính normalization stats...")
                feature_means, feature_stds = compute_normalization_stats(
                    str(train_parquet), feature_cols, label_col, sample_size=50000, batch_size=2048
                )

                print(f"   💾 Đang lưu vào {output_stats_path}...")
                import numpy as np

                np.savez(
                    str(output_stats_path),
                    feature_means=feature_means,
                    feature_stds=feature_stds,
                    feature_cols=np.array(feature_cols, dtype=object),
                )

                normalization_stats_path = str(output_stats_path.resolve())  # Absolute path
                print(f"   ✅ Đã tạo file normalization stats: {normalization_stats_path}")
                print(f"   Stats file: {Path(normalization_stats_path).name} (cho model {model_name})")

            except Exception as e:
                print(f"   ❌ Lỗi khi tạo normalization stats: {e}")
                import traceback

                traceback.print_exc()
                print(f"\n   💡 Vui lòng tạo thủ công bằng:")
                print(f"      python scripts/create_normalization_stats.py \\")
                print(f"          --output_path {model_path_obj.parent / f'{model_name_without_ext}_normalization_stats.npz'}")
                print(f"   hoặc chỉ định đường dẫn đã có sẵn qua --normalization_stats_path")
                raise
    else:
        # User provided normalization_stats_path - convert to absolute
        if args.normalization_stats_path is not None:
            stats_path_obj = Path(args.normalization_stats_path)
            if not stats_path_obj.is_absolute():
                normalization_stats_path = str((PROJECT_ROOT / args.normalization_stats_path).resolve())
            else:
                normalization_stats_path = str(stats_path_obj.resolve())
            
            # Validate stats file exists
            if not Path(normalization_stats_path).exists():
                raise FileNotFoundError(
                    f"❌ Normalization stats file không tồn tại: {normalization_stats_path}\n"
                    f"   Đã thử resolve từ: {args.normalization_stats_path}"
                )
        else:
            normalization_stats_path = None
    
    base_output_dir = PROJECT_ROOT / "output"
    
    # Đường dẫn data files
    train_parquet = str(PROJECT_ROOT / "data" / "train_ember_2018_v2_features_label_other.parquet")
    test_parquet = str(PROJECT_ROOT / "data" / "test_ember_2018_v2_features_label_other.parquet")
    
    # Tạo tên output directory dựa trên model type
    model_suffix = args.model_type.upper()
    
    # Các cấu hình khác nhau
    # Lưu ý: total_queries = query_batch × num_rounds (chỉ tính số queries trong active learning rounds)
    # Labels sử dụng = seed_size + val_size + total_queries
    configurations = [
        {
            "name": f"max_queries_10000_{model_suffix}",
            "query_batch": 2000,
            "num_rounds": 5,
            "total_queries": 10000,  # 2000 × 5 = 10000
            "description": "Tổng 10,000 queries (2000 queries/round × 5 rounds)"
        },
        {
            "name": f"max_queries_5000_{model_suffix}",
            "query_batch": 1250,
            "num_rounds": 4,
            "total_queries": 5000,  # 1250 × 4 = 5000
            "description": "Tổng 5,000 queries (1250 queries/round × 4 rounds)"
        },
        {
            "name": f"max_queries_2000_{model_suffix}",
            "query_batch": 2000,
            "num_rounds": 1,
            "total_queries": 2000,  # 2000 × 1 = 2000
            "description": "Tổng 2,000 queries (2000 queries/round × 1 round)"
        }
    ]
    
    results = []
    
    print("=" * 80)
    print("BẮT ĐẦU CHẠY EXTRACTION VỚI CÁC CẤU HÌNH KHÁC NHAU")
    print("=" * 80)
    print(f"\n📋 Cấu hình chung cho TẤT CẢ configs:")
    print(f"   ✅ Target model: {Path(weights_path).name}")
    print(f"      Path (absolute): {weights_path}")
    print(f"      Model type: {args.model_type.upper()}")
    if normalization_stats_path:
        print(f"   ✅ Normalization stats: {Path(normalization_stats_path).name}")
        print(f"      Path (absolute): {normalization_stats_path}")
    else:
        print(f"   ℹ️  Normalization stats: Không sử dụng (Keras model)")
    if args.attacker_type:
        print(f"   Attacker type: {args.attacker_type.upper()}")
    else:
        print(f"   Attacker type: Tự động ({args.model_type.upper()})")
    print("=" * 80)
    print(f"\n⚠️  LƯU Ý: Tất cả các configs sẽ tấn công CÙNG MỘT target model: {Path(weights_path).name}")
    print("=" * 80)
    
    for config in configurations:
        print(f"\n{'='*80}")
        print(f"🔬 CẤU HÌNH: {config['name']}")
        print(f"   {config['description']}")
        print(f"{'='*80}\n")
        
        output_dir = base_output_dir / config["name"]
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # QUAN TRỌNG: Verify lại model path cho mỗi config để đảm bảo không bị nhầm lẫn
        if not Path(weights_path).exists():
            raise FileNotFoundError(
                f"❌ LỖI NGHIÊM TRỌNG: Target model không tồn tại khi chạy config {config['name']}!\n"
                f"   Model path: {weights_path}\n"
                f"   Có thể model đã bị xóa hoặc di chuyển trong quá trình chạy."
            )
        
        print(f"\n🔍 Xác nhận target model cho config {config['name']}:")
        print(f"   ✅ Model file: {Path(weights_path).name}")
        print(f"   ✅ Path: {weights_path}")
        if normalization_stats_path:
            if not Path(normalization_stats_path).exists():
                raise FileNotFoundError(
                    f"❌ LỖI NGHIÊM TRỌNG: Normalization stats không tồn tại!\n"
                    f"   Stats path: {normalization_stats_path}"
                )
            print(f"   ✅ Normalization stats: {Path(normalization_stats_path).name}")
        
        try:
            summary = run_extraction(
                weights_path=weights_path,  # Đảm bảo là absolute path
                output_dir=output_dir,
                train_parquet=train_parquet,
                test_parquet=test_parquet,
                seed=42,
                seed_size=2000,
                val_size=1000,
                eval_size=4000,
                query_batch=config["query_batch"],
                num_rounds=config["num_rounds"],
                num_epochs=100,  # Theo nghiên cứu: 100 epochs với early_stopping=30 (chỉ dùng cho Keras)
                model_type=args.model_type,
                normalization_stats_path=normalization_stats_path,  # Đảm bảo là absolute path
                attacker_type=args.attacker_type,
            )
            
            # QUAN TRỌNG: Verify model trong summary khớp với model đã chỉ định
            # Để đảm bảo không bị nhầm lẫn target model
            if "weights_path" in summary:
                summary_model_path = summary["weights_path"]
                summary_model_name = summary.get("model_file_name", Path(summary_model_path).name)
                expected_model_name = Path(weights_path).name
                
                # Verify bằng absolute path
                if Path(summary_model_path).resolve() != Path(weights_path).resolve():
                    print(f"\n⚠️  CẢNH BÁO: Summary model path ({summary_model_path}) != Model path được chỉ định ({weights_path})")
                    print(f"   Tuy nhiên sẽ tiếp tục vì có thể do resolve path.")
                
                # Verify bằng tên file để chắc chắn không bị nhầm model
                if summary_model_name != expected_model_name:
                    print(f"\n❌ LỖI NGHIÊM TRỌNG: Model file name không khớp!")
                    print(f"   Summary model: {summary_model_name}")
                    print(f"   Expected model: {expected_model_name}")
                    print(f"   Có thể đã bị nhầm lẫn model!")
                    raise ValueError(
                        f"Model file name không khớp: summary có {summary_model_name} "
                        f"nhưng expected là {expected_model_name}. "
                        f"Có thể đã tấn công sai target model!"
                    )
                
                print(f"   ✅ Verified: Model trong summary khớp ({summary_model_name})")
            
            # Lấy metrics cuối cùng
            final_metrics = summary["metrics"][-1] if summary["metrics"] else {}
            
            # Lấy số queries thực tế từ metrics (không tính seed và val)
            actual_queries_used = final_metrics.get("queries_used", config["total_queries"])
            
            result = {
                "config_name": config["name"],
                "description": config["description"],
                "total_queries": config["total_queries"],  # Số queries dự kiến
                "actual_queries_used": actual_queries_used,  # Số queries thực tế
                "query_batch": config["query_batch"],
                "num_rounds": config["num_rounds"],
                "total_labels_used": final_metrics.get("labels_used", 0),
                "optimal_threshold": final_metrics.get("optimal_threshold", 0.5),
                "final_accuracy": final_metrics.get("surrogate_acc", 0.0),
                "final_balanced_accuracy": final_metrics.get("surrogate_balanced_acc", 0.0),  # Quan trọng với class imbalance
                "final_auc": final_metrics.get("surrogate_auc", float("nan")),
                "final_precision": final_metrics.get("surrogate_precision", 0.0),
                "final_recall": final_metrics.get("surrogate_recall", 0.0),
                "final_f1": final_metrics.get("surrogate_f1", 0.0),
                "final_agreement": final_metrics.get("agreement_with_target", 0.0),
                "output_dir": str(output_dir),
                "metrics_csv": summary.get("metrics_csv", ""),
                "surrogate_model_path": summary.get("surrogate_model_path", ""),
            }
            
            results.append(result)
            
            print(f"\n✅ Hoàn thành {config['name']}")
            print(f"   Accuracy: {result['final_accuracy']:.4f}")
            print(f"   Balanced Accuracy: {result['final_balanced_accuracy']:.4f} (quan trọng với class imbalance)")
            print(f"   F1-score: {result['final_f1']:.4f}")
            print(f"   Agreement: {result['final_agreement']:.4f}")
            print(f"   Optimal Threshold: {result['optimal_threshold']:.4f}")
            
        except Exception as e:
            print(f"\n❌ Lỗi khi chạy {config['name']}: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "config_name": config["name"],
                "description": config["description"],
                "error": str(e)
            })
    
    # Tạo report
    print(f"\n{'='*80}")
    print("📊 TẠO REPORT")
    print(f"{'='*80}\n")
    
    report_path = base_output_dir / "extraction_comparison_report.txt"
    report_md_path = base_output_dir / "extraction_comparison_report.md"
    
    # Tạo report text
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("BÁO CÁO SO SÁNH CÁC SURROGATE MODELS\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("TÓM TẮT:\n")
        f.write("-" * 80 + "\n")
        f.write(f"Đã chạy extraction với {len(configurations)} cấu hình khác nhau:\n\n")
        
        for result in results:
            if "error" in result:
                f.write(f"❌ {result['config_name']}: LỖI - {result['error']}\n")
            else:
                f.write(f"✅ {result['config_name']}:\n")
                f.write(f"   - Queries dự kiến: {result['total_queries']:,}\n")
                f.write(f"   - Queries thực tế: {result.get('actual_queries_used', result['total_queries']):,}\n")
                f.write(f"   - Labels sử dụng (bao gồm seed+val): {result['total_labels_used']:,}\n")
                f.write(f"   - Accuracy: {result['final_accuracy']:.4f} ({result['final_accuracy']*100:.2f}%)\n")
                f.write(f"   - Balanced Accuracy: {result.get('final_balanced_accuracy', 0.0):.4f} ({result.get('final_balanced_accuracy', 0.0)*100:.2f}%) [quan trọng với class imbalance]\n")
                f.write(f"   - F1-score: {result['final_f1']:.4f}\n")
                f.write(f"   - Agreement: {result['final_agreement']:.4f} ({result['final_agreement']*100:.2f}%)\n")
                f.write(f"   - Optimal Threshold: {result.get('optimal_threshold', 0.5):.4f}\n")
                if not pd.isna(result['final_auc']):
                    f.write(f"   - AUC: {result['final_auc']:.4f}\n")
                f.write(f"   - Precision: {result['final_precision']:.4f}\n")
                f.write(f"   - Recall: {result['final_recall']:.4f}\n")
                f.write(f"   - Output: {result['output_dir']}\n")
                f.write("\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("CHI TIẾT TỪNG CẤU HÌNH:\n")
        f.write("=" * 80 + "\n\n")
        
        for result in results:
            if "error" not in result:
                f.write(f"\n{result['config_name'].upper().replace('_', ' ')}:\n")
                f.write("-" * 80 + "\n")
                f.write(f"Mô tả: {result['description']}\n")
                f.write(f"Query batch: {result['query_batch']:,}\n")
                f.write(f"Số rounds: {result['num_rounds']}\n")
                f.write(f"Queries dự kiến: {result['total_queries']:,}\n")
                f.write(f"Queries thực tế: {result.get('actual_queries_used', result['total_queries']):,}\n")
                f.write(f"Tổng labels sử dụng (bao gồm seed+val): {result['total_labels_used']:,}\n\n")
                
                f.write("Metrics cuối cùng:\n")
                f.write(f"  - Accuracy: {result['final_accuracy']:.4f} ({result['final_accuracy']*100:.2f}%)\n")
                f.write(f"  - Balanced Accuracy: {result.get('final_balanced_accuracy', 0.0):.4f} ({result.get('final_balanced_accuracy', 0.0)*100:.2f}%) [quan trọng với class imbalance]\n")
                f.write(f"  - F1-score: {result['final_f1']:.4f}\n")
                f.write(f"  - Optimal Threshold: {result.get('optimal_threshold', 0.5):.4f}\n")
                f.write(f"  - Agreement: {result['final_agreement']:.4f} ({result['final_agreement']*100:.2f}%)\n")
                if not pd.isna(result['final_auc']):
                    f.write(f"  - AUC: {result['final_auc']:.4f}\n")
                f.write(f"  - Precision: {result['final_precision']:.4f}\n")
                f.write(f"  - Recall: {result['final_recall']:.4f}\n")
                f.write(f"\nFiles:\n")
                f.write(f"  - Metrics CSV: {result['metrics_csv']}\n")
                f.write(f"  - Surrogate model: {result['surrogate_model_path']}\n")
                f.write(f"  - Output directory: {result['output_dir']}\n")
    
    # Tạo report Markdown
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write("# Báo Cáo So Sánh Các Surrogate Models\n\n")
        f.write("## Tóm Tắt\n\n")
        f.write(f"Đã chạy extraction với {len(configurations)} cấu hình khác nhau về số lượng queries.\n\n")
        
        f.write("## Bảng So Sánh\n\n")
        f.write("| Cấu hình | Queries | Labels | Accuracy | Balanced Acc | F1 | Agreement | Threshold | AUC |\n")
        f.write("|----------|---------|--------|----------|--------------|----|-----------|-----------|-----|\n")
        
        for result in results:
            if "error" not in result:
                auc_str = f"{result['final_auc']:.4f}" if not pd.isna(result['final_auc']) else "N/A"
                actual_queries = result.get('actual_queries_used', result['total_queries'])
                balanced_acc = result.get('final_balanced_accuracy', 0.0)
                threshold = result.get('optimal_threshold', 0.5)
                f.write(f"| {result['config_name']} | {actual_queries:,} | {result['total_labels_used']:,} | "
                       f"{result['final_accuracy']:.4f} | {balanced_acc:.4f} | {result['final_f1']:.4f} | "
                       f"{result['final_agreement']:.4f} | {threshold:.3f} | {auc_str} |\n")
            else:
                f.write(f"| {result['config_name']} | ERROR | - | - | - | - | - | - | - |\n")
        
        f.write("\n## Chi Tiết Từng Cấu Hình\n\n")
        
        for result in results:
            if "error" not in result:
                f.write(f"### {result['config_name'].replace('_', ' ').title()}\n\n")
                f.write(f"**Mô tả:** {result['description']}\n\n")
                f.write(f"- Query batch: {result['query_batch']:,}\n")
                f.write(f"- Số rounds: {result['num_rounds']}\n")
                f.write(f"- Queries dự kiến: {result['total_queries']:,}\n")
                f.write(f"- Queries thực tế: {result.get('actual_queries_used', result['total_queries']):,}\n")
                f.write(f"- Tổng labels sử dụng (bao gồm seed+val): {result['total_labels_used']:,}\n\n")
                
                f.write("**Metrics cuối cùng:**\n\n")
                f.write(f"- Accuracy: {result['final_accuracy']:.4f} ({result['final_accuracy']*100:.2f}%)\n")
                f.write(f"- Balanced Accuracy: {result.get('final_balanced_accuracy', 0.0):.4f} ({result.get('final_balanced_accuracy', 0.0)*100:.2f}%) [quan trọng với class imbalance]\n")
                f.write(f"- F1-score: {result['final_f1']:.4f}\n")
                f.write(f"- Optimal Threshold: {result.get('optimal_threshold', 0.5):.4f}\n")
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
    json_path = base_output_dir / "extraction_comparison_summary.json"
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
        df_display = df_results.copy()
        if 'actual_queries_used' not in df_display.columns:
            df_display['actual_queries_used'] = df_display['total_queries']
        print(df_display[["config_name", "actual_queries_used", "total_labels_used", 
                          "final_accuracy", "final_balanced_accuracy", "final_f1", 
                          "final_agreement", "optimal_threshold"]].to_string(index=False))
    
    return results


if __name__ == "__main__":
    main()

