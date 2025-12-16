"""
Script để chạy extraction với các số lượng queries khác nhau và tạo report

Hỗ trợ cả target model .h5 (Keras) và .lgb (LightGBM)
"""
import json
import os
import sys
import argparse
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _resolve_path(path_str: str) -> Path:
    """Resolve path (relative to PROJECT_ROOT nếu cần)."""
    path_obj = Path(path_str)
    if not path_obj.is_absolute():
        path_obj = PROJECT_ROOT / path_obj
    return path_obj.resolve()


def _format_template(template: str, context: dict, template_name: str) -> str:
    """Helper để format template và báo lỗi rõ ràng nếu placeholder sai."""
    try:
        return template.format(**context)
    except KeyError as exc:
        missing = exc.args[0]
        available = ", ".join(sorted(context.keys()))
        raise ValueError(
            f"Placeholder {{{missing}}} trong {template_name} không khả dụng. "
            f"Các placeholder hợp lệ: {available}"
        ) from exc

from scripts.attacks.extract_final_model import run_extraction

# Known compatibility matrix: model_name -> list of compatible datasets
MODEL_DATASET_COMPATIBILITY = {
    "CEE": ["ember"],
    "CSE": ["ember"],
    "LEE": ["ember"],
    "LSE": ["ember"],
    "CNN": ["ember"],
    "KNN": ["ember"],
    "XGBOOST": ["ember"],
    "XGBOOST-EMBER": ["ember"],
    "DUALFFNN": ["ember"],
    "DUALFFNN-EMBER": ["ember"],
    "TABNET": ["ember"],
    "TABNET-EMBER": ["ember"],
    # Add new models here: "LEE_SOMLAP": ["somlap"],
}

def validate_model_dataset_compatibility(model_name: str, dataset: str):
    """
    Validate compatibility between model and dataset before running attack.
    Chỉ warning, không block vì padding/truncate sẽ tự động xử lý feature mismatch.
    
    Args:
        model_name: Name of target model (e.g., "LEE", "CEE")
        dataset: Name of attack dataset (e.g., "ember", "somlap")
    """
    model_name_upper = model_name.upper().strip()
    dataset_lower = dataset.lower().strip()
    
    if model_name_upper in MODEL_DATASET_COMPATIBILITY:
        compatible_datasets = MODEL_DATASET_COMPATIBILITY[model_name_upper]
        if dataset_lower not in [d.lower() for d in compatible_datasets]:
            compatible_str = ", ".join(compatible_datasets)
            print(f"\n⚠️  WARNING: Model '{model_name}' thường được train trên: {compatible_str}")
            print(f"   Đang sử dụng dataset '{dataset}' - có thể có feature mismatch")
            print(f"   📌 Padding/Truncate sẽ tự động xử lý nếu có sự khác biệt về số features")
            print(f"   💡 Nếu muốn tối ưu, nên sử dụng: --dataset {compatible_str}\n")
    # Unknown model - will be validated later by extract_final_model based on feature dimensions


def _create_individual_report(output_dir: Path, result: dict, config: dict):
    """
    Tạo report riêng cho từng config trong folder output của config đó
    
    Args:
        output_dir: Thư mục output của config
        result: Kết quả của config (chứa metrics)
        config: Cấu hình config (chứa description, query_batch, etc.)
    """
    # Report paths trong folder output của config
    report_txt_path = output_dir / "extraction_report.txt"
    report_md_path = output_dir / "extraction_report.md"
    report_json_path = output_dir / "extraction_report.json"
    
    # Tạo report text
    with open(report_txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write(f"BÁO CÁO EXTRACTION ATTACK\n")
        f.write(f"Cấu hình: {result['config_name']}\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("THÔNG TIN CẤU HÌNH:\n")
        f.write("-" * 80 + "\n")
        f.write(f"Mô tả: {config.get('description', result.get('description', 'N/A'))}\n")
        f.write(f"Query batch: {result['query_batch']:,}\n")
        f.write(f"Số rounds: {result['num_rounds']}\n")
        f.write(f"Queries dự kiến: {result['total_queries']:,}\n")
        f.write(f"Queries thực tế: {result.get('actual_queries_used', result['total_queries']):,}\n")
        if result.get("query_gap_reason"):
            f.write(f"Ghi chú queries: {result['query_gap_reason']}\n")
        f.write(f"Tổng labels sử dụng (bao gồm seed+val): {result['total_labels_used']:,}\n\n")
        
        f.write("KẾT QUẢ METRICS:\n")
        f.write("-" * 80 + "\n")
        f.write(f"Accuracy: {result['final_accuracy']:.4f} ({result['final_accuracy']*100:.2f}%)\n")
        f.write(f"Balanced Accuracy: {result.get('final_balanced_accuracy', 0.0):.4f} ({result.get('final_balanced_accuracy', 0.0)*100:.2f}%) [quan trọng với class imbalance]\n")
        f.write(f"F1-score: {result['final_f1']:.4f}\n")
        f.write(f"Optimal Threshold: {result.get('optimal_threshold', 0.5):.4f}\n")
        f.write(f"Agreement: {result['final_agreement']:.4f} ({result['final_agreement']*100:.2f}%)\n")
        if not pd.isna(result.get('final_auc', float('nan'))):
            f.write(f"AUC: {result['final_auc']:.4f}\n")
        f.write(f"Precision: {result['final_precision']:.4f}\n")
        f.write(f"Recall: {result['final_recall']:.4f}\n\n")
        
        f.write("FILES:\n")
        f.write("-" * 80 + "\n")
        f.write(f"Metrics CSV: {result.get('metrics_csv', 'N/A')}\n")
        f.write(f"Surrogate model: {result.get('surrogate_model_path', 'N/A')}\n")
        f.write(f"Output directory: {result['output_dir']}\n")
    
    # Tạo report Markdown
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write(f"# Báo Cáo Extraction Attack: {result['config_name']}\n\n")
        
        f.write("## Thông Tin Cấu Hình\n\n")
        f.write(f"- **Mô tả:** {config.get('description', result.get('description', 'N/A'))}\n")
        f.write(f"- **Query batch:** {result['query_batch']:,}\n")
        f.write(f"- **Số rounds:** {result['num_rounds']}\n")
        f.write(f"- **Queries dự kiến:** {result['total_queries']:,}\n")
        f.write(f"- **Queries thực tế:** {result.get('actual_queries_used', result['total_queries']):,}\n")
        if result.get("query_gap_reason"):
            f.write(f"- **Ghi chú queries:** {result['query_gap_reason']}\n")
        f.write(f"- **Tổng labels sử dụng (bao gồm seed+val):** {result['total_labels_used']:,}\n\n")
        
        f.write("## Kết Quả Metrics\n\n")
        f.write(f"- **Accuracy:** {result['final_accuracy']:.4f} ({result['final_accuracy']*100:.2f}%)\n")
        f.write(f"- **Balanced Accuracy:** {result.get('final_balanced_accuracy', 0.0):.4f} ({result.get('final_balanced_accuracy', 0.0)*100:.2f}%) [quan trọng với class imbalance]\n")
        f.write(f"- **F1-score:** {result['final_f1']:.4f}\n")
        f.write(f"- **Optimal Threshold:** {result.get('optimal_threshold', 0.5):.4f}\n")
        f.write(f"- **Agreement:** {result['final_agreement']:.4f} ({result['final_agreement']*100:.2f}%)\n")
        if not pd.isna(result.get('final_auc', float('nan'))):
            f.write(f"- **AUC:** {result['final_auc']:.4f}\n")
        f.write(f"- **Precision:** {result['final_precision']:.4f}\n")
        f.write(f"- **Recall:** {result['final_recall']:.4f}\n\n")
        
        f.write("## Files\n\n")
        f.write(f"- **Metrics CSV:** `{result.get('metrics_csv', 'N/A')}`\n")
        f.write(f"- **Surrogate model:** `{result.get('surrogate_model_path', 'N/A')}`\n")
        f.write(f"- **Output directory:** `{result['output_dir']}`\n")
    
    # Tạo report JSON
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump({
            "config": result,
            "description": config.get('description', result.get('description', 'N/A'))
        }, f, indent=2, ensure_ascii=False)
    
    print(f"   📄 Đã lưu report trong folder output:")
    print(f"      - Text: {report_txt_path.name}")
    print(f"      - Markdown: {report_md_path.name}")
    print(f"      - JSON: {report_json_path.name}")


def main():
    parser = argparse.ArgumentParser(description="Chạy model extraction với nhiều cấu hình")
    parser.add_argument("--model_name", type=str, default=None,
                       help="Tên model (CEE, LEE, CSE, LSE, CNN, KNN, XGBOOST, DUALFFNN, TABNET). Ưu tiên hơn --model_path. Sẽ tự động detect type và tìm normalization stats.")
    parser.add_argument("--model_path", type=str, default=None,
                       help="Đường dẫn tới file model (.h5, .lgb, .json, .pt, hoặc .zip). Chỉ dùng nếu không có --model_name")
    parser.add_argument("--model_type", type=str, choices=["h5", "lgb", "xgboost", "pytorch", "tabnet"], default=None,
                       help="Loại model: 'h5' (Keras), 'lgb' (LightGBM), 'xgboost' (XGBoost), 'pytorch' (PyTorch/dualFFNN), hoặc 'tabnet' (TabNet). Chỉ cần nếu dùng --model_path")
    parser.add_argument("--normalization_stats_path", type=str, default=None,
                       help="Đường dẫn tới file normalization_stats.npz. Cần cho model_type='lgb', 'xgboost', 'pytorch', hoặc 'tabnet'")
    parser.add_argument("--attacker_type", type=str, choices=["keras", "lgb", "dual", "cnn", "knn", "xgb", "tabnet"], required=True,
                       help="Loại surrogate model: 'keras' (DNN), 'lgb' (LightGBM), 'dual' (dualDNN), 'cnn' (CNN), 'knn' (KNN), 'xgb' (XGBoost), hoặc 'tabnet' (TabNet). BẮT BUỘC phải chỉ định.")
    parser.add_argument("--dataset", type=str, choices=["ember", "somlap"], default="ember",
                       help="Dataset để tấn công: 'ember' (mặc định) hoặc 'somlap'")
    parser.add_argument("--threshold_optimization_metric", type=str, choices=["f1", "accuracy", "balanced_accuracy"], default="f1",
                       help="Metric để tối ưu threshold cho dualDNN: 'f1' (mặc định), 'accuracy', hoặc 'balanced_accuracy'")
    parser.add_argument("--fixed_threshold", type=float, default=None,
                       help="Sử dụng threshold cố định thay vì tối ưu (ví dụ: 0.5). Chỉ áp dụng cho dualDNN.")
    parser.add_argument("--auto_create_stats", action="store_true", default=False,
                       help="Tự động tạo file normalization stats nếu không tìm thấy (chỉ cho model .lgb)")
    parser.add_argument("--surrogate_dir_template", type=str, default=None,
                       help="Template thư mục lưu surrogate. Có thể dùng {config}, {attacker}, {model}, {model_type}. Mặc định: output/<config>")
    parser.add_argument("--surrogate_name_template", type=str, default=None,
                       help="Template tên file surrogate (không extension). Hỗ trợ {config}, {attacker}, {model}, {model_type}. Mặc định: surrogate_model")
    args = parser.parse_args()
    
    # Validate arguments
    if args.model_name is None and args.model_path is None:
    # Tự động tìm model file nếu không được chỉ định
        pass
    elif args.model_name is not None and args.model_path is not None:
        raise ValueError("❌ Chỉ cung cấp --model-name HOẶC --model-path, không phải cả hai")
    
    # attacker_type đã được argparse validate (required=True)
    
    # Xử lý model_name hoặc model_path
    model_name = args.model_name.upper().strip() if args.model_name else None
    weights_path = None
    
    # Validate model-dataset compatibility early (before loading data)
    if model_name is not None:
        validate_model_dataset_compatibility(model_name, args.dataset)
        print(f"✅ Model-dataset compatibility check passed: {model_name} <-> {args.dataset}")
    
    if model_name is not None:
        # Sử dụng model_name - sẽ tự động detect mọi thứ
        print(f"✅ Sử dụng model name: {model_name}")
        print(f"   Sẽ tự động detect model type và tìm normalization stats")
        # weights_path và model_type sẽ được xử lý trong run_extraction
    elif args.model_path is None:
        possible_models = [
            PROJECT_ROOT / "artifacts" / "targets" / "CEE.h5",
            PROJECT_ROOT / "artifacts" / "targets" / "CSE.h5",
            PROJECT_ROOT / "artifacts" / "targets" / "LEE.lgb",
            PROJECT_ROOT / "artifacts" / "targets" / "LSE.lgb",
            PROJECT_ROOT / "artifacts" / "targets" / "CNN.h5",
            PROJECT_ROOT / "artifacts" / "targets" / "KNN.pkl",
            PROJECT_ROOT / "artifacts" / "targets" / "xgboost_ember.json",
            PROJECT_ROOT / "artifacts" / "targets" / "dualffnn_ember_full.pt",
            PROJECT_ROOT / "artifacts" / "targets" / "tabnet_ember.zip",
            PROJECT_ROOT / "artifacts" / "targets" / "final_model.h5",
            PROJECT_ROOT / "artifacts" / "targets" / "final_model.lgb",
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
        # Xử lý model_path (cách cũ)
        if args.model_type is None:
            model_path_obj = Path(weights_path)
            suffix_lower = model_path_obj.suffix.lower()
            
            if suffix_lower in ['.lgb', '.txt', '.d5']:
                args.model_type = "lgb"
                print(f"✅ Tự động phát hiện model type: LGB (từ extension {model_path_obj.suffix})")
            elif suffix_lower == '.json':
                # XGBoost model
                args.model_type = "xgboost"
                print(f"✅ Tự động phát hiện model type: XGBoost (từ extension {model_path_obj.suffix})")
            elif suffix_lower == '.pt':
                # PyTorch model (dualFFNN)
                args.model_type = "pytorch"
                print(f"✅ Tự động phát hiện model type: PyTorch (từ extension {model_path_obj.suffix})")
            elif suffix_lower == '.zip':
                # TabNet model
                args.model_type = "tabnet"
                print(f"✅ Tự động phát hiện model type: TabNet (từ extension {model_path_obj.suffix})")
            elif suffix_lower == '.pkl':
                # .pkl có thể là LightGBM hoặc sklearn - sẽ được auto-detect trong create_oracle_from_name
                # Tạm thời để None để auto-detect
                args.model_type = None
                print(f"✅ File .pkl - sẽ tự động detect model type (LightGBM hoặc sklearn) trong create_oracle_from_name")
            elif suffix_lower in ['.h5', '.hdf5']:
                args.model_type = "h5"
                print(f"✅ Tự động phát hiện model type: H5 (từ extension {model_path_obj.suffix})")
            else:
                args.model_type = "h5"
                print(f"⚠️  Không thể phát hiện model type từ extension, mặc định: H5")
    
    # Chỉ resolve weights_path nếu không dùng model_name
    if model_name is None:
        weights_path_abs = str(Path(weights_path).resolve())
        weights_path = weights_path_abs
        
        if not Path(weights_path).exists():
            raise FileNotFoundError(f"❌ Model file không tồn tại: {weights_path}")
        
        # Get model info for verification
        model_path_obj = Path(weights_path)
        model_name_from_path = model_path_obj.name
        model_size = model_path_obj.stat().st_size / (1024 * 1024)  # MB
        
        print(f"\n✅ Đã xác nhận target model:")
        print(f"   Path (absolute): {weights_path}")
        print(f"   File name: {model_name_from_path}")
        print(f"   File size: {model_size:.2f} MB")
    else:
        # Sử dụng model_name - không cần xử lý paths ở đây
        model_path_obj = None
        model_name_from_path = model_name
        model_size = None
        args.model_type = None  # Sẽ được auto-detect

    model_identifier = model_name_from_path if model_name_from_path else "UNKNOWN_TARGET"
    model_type_label = (args.model_type.upper() if args.model_type else "AUTO")
    template_context_base = {
        "attacker": args.attacker_type,
        "model": model_identifier,
        "model_type": model_type_label,
    }
    
    normalization_stats_path = None
    # Các model types cần normalization stats: lgb, sklearn, xgboost, pytorch (dualFFNN), tabnet
    needs_normalization = args.model_type in ["lgb", "sklearn", "xgboost", "pytorch", "tabnet"]
    if model_name is None and needs_normalization and args.normalization_stats_path is None:
        model_name_without_ext = model_path_obj.stem
        # Xử lý special cases: dualffnn_ember_full.pt -> dualffnn_ember
        if model_name_without_ext.endswith("_full"):
            model_name_without_ext = model_name_without_ext[:-5]  # Remove "_full"
        
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
                print(f"   Stats file: {Path(normalization_stats_path).name}")
                break
        
        if normalization_stats_path is None and args.auto_create_stats:
            print(f"\n⚠️  KHÔNG TÌM THẤY file normalization stats!")
            print(f"   🔄 Đang tự động tạo file normalization stats...")
            try:
                from scripts.data.create_normalization_stats import (
                    get_feature_columns,
                    compute_normalization_stats,
                )

                # Thử dùng file mới trong ember_2018_v2 trước
                train_parquet_new = PROJECT_ROOT / "data" / "ember_2018_v2" / "train" / "train_ember_2018_v2_features_label_other.parquet"
                train_parquet_old = PROJECT_ROOT / "data" / "train_ember_2018_v2_features_label_other.parquet"
                if train_parquet_new.exists():
                    train_parquet = train_parquet_new
                elif train_parquet_old.exists():
                    train_parquet = train_parquet_old
                else:
                    raise FileNotFoundError(f"Không tìm thấy training data tại: {train_parquet_new} hoặc {train_parquet_old}")

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

                normalization_stats_path = str(output_stats_path.resolve())
                print(f"   ✅ Đã tạo file normalization stats: {normalization_stats_path}")
                print(f"   Stats file: {Path(normalization_stats_path).name} (cho model {model_name})")

            except Exception as e:
                print(f"   ❌ Lỗi khi tạo normalization stats: {e}")
                import traceback

                traceback.print_exc()
                print(f"\n   💡 Vui lòng tạo thủ công bằng:")
                print(f"      python scripts/data/create_normalization_stats.py \\")
                print(f"          --output_path {model_path_obj.parent / f'{model_name_without_ext}_normalization_stats.npz'}")
                print(f"   hoặc chỉ định đường dẫn đã có sẵn qua --normalization_stats_path")
                raise
    else:
        if args.normalization_stats_path is not None:
            stats_path_obj = Path(args.normalization_stats_path)
            if not stats_path_obj.is_absolute():
                normalization_stats_path = str((PROJECT_ROOT / args.normalization_stats_path).resolve())
            else:
                normalization_stats_path = str(stats_path_obj.resolve())
            
            if not Path(normalization_stats_path).exists():
                raise FileNotFoundError(
                    f"❌ Normalization stats file không tồn tại: {normalization_stats_path}\n"
                    f"   Đã thử resolve từ: {args.normalization_stats_path}"
                )
        else:
            normalization_stats_path = None
    
    base_output_dir = PROJECT_ROOT / "output"
    
    # Đường dẫn data files - để None để run_extraction() tự xử lý dựa trên dataset parameter
    # run_extraction() sẽ tự động load đúng file dựa trên dataset (ember hoặc somlap)
    train_parquet = None  # Sẽ được set bởi run_extraction() dựa trên dataset parameter
    test_parquet = None   # Sẽ được set bởi run_extraction() dựa trên dataset parameter
    
    # Xác định tên model target để dùng trong output folder name
    if model_name:
        target_model_name = model_name.upper()
    else:
        # Nếu không có model_name, lấy từ file name (bỏ extension)
        if weights_path:
            target_model_name = Path(weights_path).stem.upper()
        else:
            target_model_name = "UNKNOWN"
    
    # Format attacker type cho output folder name
    attacker_name_map = {
        "keras": "DNN",
        "lgb": "LGB",
        "dual": "dualDNN",
        "cnn": "CNN",
        "knn": "KNN",
        "xgb": "XGB",
        "tabnet": "TabNet"
    }
    attacker_name_display = attacker_name_map.get(args.attacker_type.lower(), args.attacker_type.upper())
    
    # Dataset name (lowercase cho folder name theo yêu cầu)
    dataset_name = args.dataset.lower()
    
    # Helper function để tạo output folder name theo format: [targetmodel]-[dataset]-[surrogate]-[queries]
    def create_output_folder_name(target_model: str, dataset: str, attacker: str, total_queries: int) -> str:
        """Tạo tên folder output theo format: TARGETMODEL-dataset-ATTACKER-queries"""
        return f"{target_model}-{dataset}-{attacker}-{total_queries}"
    
    # Các cấu hình khác nhau
    # Lưu ý: total_budget = seed_size (10%) + val_size (20%) + AL_queries (70%)
    # AL_queries = query_batch × num_rounds (chỉ tính số queries trong active learning rounds)
    # Labels sử dụng = seed_size + val_size + AL_queries = total_budget
    # TEST MODE: Sử dụng config nhỏ để test nhanh
    test_mode = os.environ.get("EXTRACTION_TEST_MODE", "false").lower() == "true"
    
    if test_mode:
        total_queries = 100
        config_name = create_output_folder_name(target_model_name, dataset_name, attacker_name_display, total_queries)
        configurations = [
            {
                "name": config_name,
                "total_budget": total_queries,  # 100 queries total (seed + val + AL)
                "description": "TEST: Tổng 100 queries (seed + val + AL queries)"
            }
        ]
    else:
        # Config 1: 200 queries
        total_queries_1 = 200
        config_name_1 = create_output_folder_name(target_model_name, dataset_name, attacker_name_display, total_queries_1)
        # Config 2: 1000 queries
        total_queries_2 = 1000
        config_name_2 = create_output_folder_name(target_model_name, dataset_name, attacker_name_display, total_queries_2)
        # Config 3: 5000 queries
        total_queries_3 = 5000
        config_name_3 = create_output_folder_name(target_model_name, dataset_name, attacker_name_display, total_queries_3)
        
        configurations = [
            {
                "name": config_name_1,
                "total_budget": total_queries_1,  # 200 queries total (seed + val + AL)
                "description": "Tổng 200 queries (seed + val + AL queries)"
            },
            {
                "name": config_name_2,
                "total_budget": total_queries_2,  # 1000 queries total (seed + val + AL)
                "description": "Tổng 1,000 queries (seed + val + AL queries)"
            },
            {
                "name": config_name_3,
                "total_budget": total_queries_3,  # 5000 queries total (seed + val + AL)
                "description": "Tổng 5,000 queries (seed + val + AL queries)"
            }
        ]
    
    results = []
    
    print("=" * 80)
    print("BẮT ĐẦU CHẠY EXTRACTION VỚI CÁC CẤU HÌNH KHÁC NHAU")
    print("=" * 80)
    print(f"\n📋 Cấu hình chung cho TẤT CẢ configs:")
    print(f"   ✅ Target model: {target_model_name}")
    if model_name:
        print(f"      (tự động detect type và path)")
    else:
        if weights_path:
            print(f"      File: {Path(weights_path).name}")
            print(f"      Path (absolute): {weights_path}")
            if args.model_type:
                print(f"      Model type: {args.model_type.upper()}")
    print(f"   ✅ Dataset: {args.dataset.upper()}")
    if normalization_stats_path:
        print(f"   ✅ Normalization stats: {Path(normalization_stats_path).name}")
        print(f"      Path (absolute): {normalization_stats_path}")
    elif model_name:
        print(f"   ℹ️  Normalization stats: Tự động tìm (nếu có)")
    else:
        print(f"   ℹ️  Normalization stats: Không sử dụng (Keras model)")
    print(f"   ✅ Attacker type (surrogate model): {attacker_name_display}")
    print(f"   📁 Output folder format: {target_model_name}-{dataset_name}-{attacker_name_display}-[queries]")
    print("=" * 80)
    target_model_display = model_name if model_name else (Path(weights_path).name if weights_path else "Unknown")
    print(f"\n⚠️  LƯU Ý: Tất cả các configs sẽ tấn công CÙNG MỘT target model: {target_model_display}")
    print("=" * 80)
    
    for config in configurations:
        print(f"\n{'='*80}")
        print(f"🔬 CẤU HÌNH: {config['name']}")
        print(f"   {config['description']}")
        print(f"{'='*80}\n")
        
        output_dir = base_output_dir / config["name"]
        output_dir.mkdir(parents=True, exist_ok=True)
        
        template_context = dict(template_context_base)
        template_context["config"] = config["name"]
        surrogate_dir_override = None
        surrogate_name_override = None
        if args.surrogate_dir_template:
            formatted_dir = _format_template(
                args.surrogate_dir_template,
                template_context,
                "surrogate_dir_template"
            )
            surrogate_dir_override = str(_resolve_path(formatted_dir))
        if args.surrogate_name_template:
            surrogate_name_override = _format_template(
                args.surrogate_name_template,
                template_context,
                "surrogate_name_template"
            )

        if model_name is None:
            if not Path(weights_path).exists():
                raise FileNotFoundError(
                    f"❌ LỖI NGHIÊM TRỌNG: Target model không tồn tại khi chạy config {config['name']}!\n"
                    f"   Model path: {weights_path}\n"
                    f"   Có thể model đã bị xóa hoặc di chuyển trong quá trình chạy."
                )
        
        print(f"\n🔍 Xác nhận target model cho config {config['name']}:")
        if model_name:
            print(f"   ✅ Model name: {model_name} (sẽ tự động detect)")
        else:
            print(f"   ✅ Model file: {Path(weights_path).name}")
            print(f"   ✅ Path: {weights_path}")
        if normalization_stats_path:
            if not Path(normalization_stats_path).exists():
                raise FileNotFoundError(
                    f"❌ LỖI NGHIÊM TRỌNG: Normalization stats không tồn tại!\n"
                    f"   Stats path: {normalization_stats_path}"
                )
            print(f"   ✅ Normalization stats: {Path(normalization_stats_path).name}")
        if surrogate_dir_override:
            print(f"   📁 Surrogate dir override: {surrogate_dir_override}")
        if surrogate_name_override:
            print(f"   📄 Surrogate name override: {surrogate_name_override}")
        
        try:
            summary = run_extraction(
                output_dir=output_dir,
                train_parquet=train_parquet,
                test_parquet=test_parquet,
                dataset=args.dataset,  # Dataset để tấn công: "ember" hoặc "somlap"
                seed=42,
                eval_size=4000,
                total_budget=config["total_budget"],  # Tổng query budget (seed + val + AL queries)
                num_epochs=100,  # Theo nghiên cứu: 100 epochs với early_stopping=30 (chỉ dùng cho Keras)
                model_type=args.model_type,
                normalization_stats_path=normalization_stats_path,  # Đảm bảo là absolute path
                attacker_type=args.attacker_type,
                weights_path=weights_path if model_name is None else None,
                model_name=model_name,
                threshold_optimization_metric=args.threshold_optimization_metric,
                fixed_threshold=args.fixed_threshold,
                surrogate_dir=surrogate_dir_override,
                surrogate_name=surrogate_name_override,
            )
            
            oracle_source = summary.get("oracle_source")
            if oracle_source is None:
                raise ValueError("Summary không chứa oracle_source để verify.")
            summary_model_path = Path(oracle_source)
            summary_model_name = summary.get("model_file_name", summary_model_path.name)
            
            if model_name is None:
                # Với weights_path, verify cả path và name
                expected_model_name = Path(weights_path).name
                if weights_path and summary_model_path.resolve() != Path(weights_path).resolve():
                    print(f"\n⚠️  CẢNH BÁO: Summary model path ({summary_model_path}) != Model path được chỉ định ({weights_path})")
                    print(f"   Tuy nhiên sẽ tiếp tục vì có thể do resolve path.")
                
                if summary_model_name != expected_model_name:
                    raise ValueError(
                        f"Model file name không khớp: summary có {summary_model_name} nhưng expected là {expected_model_name}."
                    )
            else:
                # Với model_name, chỉ cần verify tên model khớp
                # Xác định extension dựa trên model_name hoặc summary_model_name
                model_name_upper = model_name.upper()
                summary_name_lower = summary_model_name.lower()
                
                # Map model names to expected extensions
                expected_extensions = {
                    "XGBOOST": ".json",
                    "DUALFFNN": ".pt",
                    "TABNET": ".zip",
                }
                
                # Tìm extension từ summary hoặc dùng default
                if any(ext in summary_name_lower for ext in [".lgb", ".txt", ".d5"]):
                    expected_ext = ".lgb"
                elif ".json" in summary_name_lower:
                    expected_ext = ".json"
                elif ".pt" in summary_name_lower:
                    expected_ext = ".pt"
                elif ".zip" in summary_name_lower:
                    expected_ext = ".zip"
                elif ".h5" in summary_name_lower or ".hdf5" in summary_name_lower:
                    expected_ext = ".h5"
                elif model_name_upper in expected_extensions:
                    expected_ext = expected_extensions[model_name_upper]
                else:
                    expected_ext = ".h5"  # Default
                
                expected_model_name = f"{model_name}{expected_ext}".lower()
                if summary_model_name.lower() != expected_model_name:
                    print(f"\n⚠️  CẢNH BÁO: Summary model name ({summary_model_name}) != Expected model name ({expected_model_name})")
                    print(f"   Tuy nhiên sẽ tiếp tục vì có thể do extension khác nhau hoặc naming convention.")
                if not summary_model_name.upper().startswith(model_name.upper()):
                    print(f"\n⚠️  CẢNH BÁO: Summary model name ({summary_model_name}) không bắt đầu với model name ({model_name})")
                    print(f"   Tuy nhiên sẽ tiếp tục vì có thể do naming convention.")
            
            print(f"   ✅ Verified: Model trong summary khớp ({summary_model_name})")
            
            # Lấy metrics cuối cùng
            final_metrics = summary["metrics"][-1] if summary["metrics"] else {}
            
            # Lấy số queries thực tế từ metrics (không tính seed và val)
            # queries_used trong metrics chỉ tính AL queries, không tính seed và val
            # Nếu không có trong metrics, tính từ summary hoặc config
            if "queries_used" in final_metrics:
                actual_queries_used = final_metrics["queries_used"]
            else:
                # Fallback: Tính AL queries từ summary hoặc config
                query_batch_from_summary = summary.get("query_batch", 0)
                num_rounds_from_summary = summary.get("num_rounds", 0)
                if query_batch_from_summary > 0 and num_rounds_from_summary > 0:
                    actual_queries_used = query_batch_from_summary * num_rounds_from_summary
                else:
                    # Tính từ total_budget: AL_queries = total_budget - seed - val
                    seed_size_from_summary = summary.get("seed_size", 0)
                    val_size_from_summary = summary.get("val_size", 0)
                    if seed_size_from_summary > 0 and val_size_from_summary > 0:
                        actual_queries_used = config["total_budget"] - seed_size_from_summary - val_size_from_summary
                    else:
                        # Fallback cuối cùng: dùng total_budget (sai nhưng tốt hơn là crash)
                        actual_queries_used = config["total_budget"]
            
            result = {
                "config_name": config["name"],
                "description": config["description"],
                "total_queries": config["total_budget"],  # Total query budget (seed + val + AL)
                "actual_queries_used": summary.get("total_queries_actual", actual_queries_used),
                "query_batch": summary.get("query_batch", 0),  # Lấy từ summary
                "num_rounds": summary.get("num_rounds", 0),  # Lấy từ summary
                "seed_size": summary.get("seed_size", 0),
                "val_size": summary.get("val_size", 0),
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
                "query_gap_reason": summary.get("query_gap_reason"),
            }

            #region agent log
            try:
                import json as _json, time as _time
                _log_payload = {
                    "sessionId": "debug-session",
                    "runId": "pre-fix",
                    "hypothesisId": "H1",
                    "location": "run_multiple_extractions.py:result_build",
                    "message": "summary to result mapping",
                    "data": {
                        "config": config["name"],
                        "summary_last_metrics": summary.get("metrics", [])[-1] if summary.get("metrics") else None,
                        "final_accuracy": result["final_accuracy"],
                        "final_agreement": result["final_agreement"],
                        "optimal_threshold": result["optimal_threshold"],
                        "metrics_csv": result["metrics_csv"],
                        "surrogate_model_path": result["surrogate_model_path"],
                        "total_queries_actual": summary.get("total_queries_actual"),
                        "query_batch": summary.get("query_batch"),
                        "num_rounds": summary.get("num_rounds"),
                    },
                    "timestamp": int(_time.time() * 1000),
                }
                with open("/home/hytong/Documents/model_extraction_malware/.cursor/debug.log", "a", encoding="utf-8") as _f:
                    _f.write(_json.dumps(_log_payload, ensure_ascii=False) + "\n")
            except Exception:
                pass
            #endregion
            
            results.append(result)
            
            # Tạo report riêng cho từng config trong folder output của config đó
            _create_individual_report(output_dir, result, config)
            
            print(f"\n{'='*80}")
            print(f"✅ Hoàn thành {config['name']}")
            print(f"{'='*80}")
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
    
    #region agent log
    try:
        import json as _json, time as _time
        _log_payload = {
            "sessionId": "debug-session",
            "runId": "pre-fix",
            "hypothesisId": "H3",
            "location": "run_multiple_extractions.py:before_report",
            "message": "results collected before report",
            "data": {
                "results_count": len(results),
                "configs": [r.get("config_name") for r in results],
                "final_metrics_list": [
                    {
                        "config": r.get("config_name"),
                        "final_accuracy": r.get("final_accuracy"),
                        "final_agreement": r.get("final_agreement"),
                        "optimal_threshold": r.get("optimal_threshold"),
                        "metrics_csv": r.get("metrics_csv"),
                    }
                    for r in results
                    if "error" not in r
                ],
            },
            "timestamp": int(_time.time() * 1000),
        }
        with open("/home/hytong/Documents/model_extraction_malware/.cursor/debug.log", "a", encoding="utf-8") as _f:
            _f.write(_json.dumps(_log_payload, ensure_ascii=False) + "\n")
    except Exception:
        pass
    #endregion

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
                if result.get("query_gap_reason"):
                    f.write(f"   - Ghi chú queries: {result['query_gap_reason']}\n")
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
                if result.get("query_gap_reason"):
                    f.write(f"Lý do chênh lệch queries: {result['query_gap_reason']}\n")
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
                if result.get("query_gap_reason"):
                    f.write(f"- Ghi chú queries: {result['query_gap_reason']}\n")
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

