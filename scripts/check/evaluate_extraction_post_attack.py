"""
Script để tự động đánh giá surrogate models sau extraction attack
Đánh giá trên dataset trung lập (label -1) để tính agreement chính xác
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.check.evaluate_surrogate_similarity_lgb import (
    load_LightGBM_surrogate_model,
    load_test_data,
    get_feature_columns,
    evaluate_model_similarity,
    find_optimal_threshold_for_agreement
)
from src.targets.oracle_client import create_oracle_from_name, LocalOracleClient


def find_surrogate_models(output_dir: Path):
    """
    Tìm tất cả surrogate models trong output directory.
    
    Args:
        output_dir: Đường dẫn đến output directory
    
    Returns:
        List of (model_path, output_dir) tuples
    """
    surrogate_models = []
    
    # Tìm trong output_dir và các subdirectories
    for model_file in output_dir.rglob("surrogate_model.txt"):
        surrogate_models.append((model_file, model_file.parent))
    
    # Cũng tìm .lgb files
    for model_file in output_dir.rglob("surrogate_model.lgb"):
        surrogate_models.append((model_file, model_file.parent))
    
    return surrogate_models


def load_extraction_metadata(output_dir: Path):
    """
    Load metadata từ extraction report.
    
    Args:
        output_dir: Đường dẫn đến output directory của extraction
    
    Returns:
        dict với metadata (threshold, model_name, etc.)
    """
    metadata = {
        "optimal_threshold": None,
        "target_model_name": None,
        "config_name": None
    }
    
    # Thử đọc từ extraction_report.json
    report_json = output_dir / "extraction_report.json"
    if report_json.exists():
        try:
            with open(report_json, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if 'config' in data:
                    config = data['config']
                    metadata["optimal_threshold"] = config.get("optimal_threshold")
                    metadata["config_name"] = config.get("config_name")
        except Exception as e:
            print(f"⚠️  Không thể đọc extraction_report.json: {e}")
    
    # Thử đọc từ extraction_metrics.csv
    metrics_csv = output_dir / "extraction_metrics.csv"
    if metrics_csv.exists() and metadata["optimal_threshold"] is None:
        try:
            df = pd.read_csv(metrics_csv)
            if 'optimal_threshold' in df.columns and len(df) > 0:
                metadata["optimal_threshold"] = float(df['optimal_threshold'].iloc[-1])
        except Exception as e:
            print(f"⚠️  Không thể đọc extraction_metrics.csv: {e}")
    
    # Thử đọc từ extraction_report.txt
    report_txt = output_dir / "extraction_report.txt"
    if report_txt.exists() and metadata["optimal_threshold"] is None:
        try:
            with open(report_txt, 'r', encoding='utf-8') as f:
                content = f.read()
                import re
                match = re.search(r'Optimal Threshold:\s*([\d.]+)', content)
                if match:
                    metadata["optimal_threshold"] = float(match.group(1))
        except Exception as e:
            print(f"⚠️  Không thể đọc extraction_report.txt: {e}")
    
    # Tìm target model name từ config name
    if metadata["config_name"]:
        # Format: LEE-ember-LGB-10000 -> LEE
        parts = metadata["config_name"].split("-")
        if len(parts) > 0:
            metadata["target_model_name"] = parts[0]
    
    return metadata


def evaluate_single_model(
    surrogate_model_path: Path,
    output_dir: Path,
    neutral_dataset_path: str,
    target_model_name: str = "LEE",
    max_samples: int = 10000
):
    """
    Đánh giá một surrogate model trên dataset trung lập.
    
    Args:
        surrogate_model_path: Đường dẫn đến surrogate model
        output_dir: Output directory của extraction
        neutral_dataset_path: Đường dẫn đến dataset trung lập (label -1)
        target_model_name: Tên target model
        max_samples: Số lượng samples tối đa để đánh giá
    
    Returns:
        dict với kết quả đánh giá
    """
    print(f"\n{'='*80}")
    print(f"ĐÁNH GIÁ SURROGATE MODEL: {output_dir.name}")
    print(f"{'='*80}")
    
    # Load metadata từ extraction
    metadata = load_extraction_metadata(output_dir)
    extraction_threshold = metadata.get("optimal_threshold")
    if extraction_threshold is None:
        extraction_threshold = 0.5
        print(f"⚠️  Không tìm thấy threshold từ extraction, dùng mặc định: {extraction_threshold}")
    else:
        print(f"✅ Đã đọc threshold từ extraction: {extraction_threshold}")
    
    # Load feature columns
    feature_cols = get_feature_columns(neutral_dataset_path)
    print(f"✅ Feature columns: {len(feature_cols)}")
    
    # Load dataset trung lập
    print(f"\n🔄 Đang load dataset trung lập...")
    X_test = load_test_data(neutral_dataset_path, feature_cols, max_samples=max_samples, load_labels=False)
    
    if len(X_test) == 0:
        print("❌ Không có dữ liệu để test!")
        return None
    
    print(f"✅ Đã load {len(X_test):,} samples")
    
    # Load target model
    print(f"\n🔄 Đang load target model...")
    try:
        target_oracle = create_oracle_from_name(
            model_name=target_model_name,
            models_dir=str(PROJECT_ROOT / "artifacts" / "targets"),
            feature_dim=len(feature_cols)
        )
        print(f"✅ Đã load target model: {target_model_name}")
    except Exception as e:
        print(f"❌ Lỗi khi load target model: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Query target model để lấy predictions
    print(f"\n🔄 Đang query target model...")
    try:
        y_target_proba = target_oracle.predict_proba(X_test)
        
        # Chuyển thành binary labels với threshold từ extraction
        if y_target_proba.ndim == 1:
            y_target = (y_target_proba >= extraction_threshold).astype(int)
        elif y_target_proba.ndim == 2 and y_target_proba.shape[1] == 2:
            y_target = (y_target_proba[:, 1] >= extraction_threshold).astype(int)
        else:
            y_target_proba_flat = np.squeeze(y_target_proba)
            y_target = (y_target_proba_flat >= extraction_threshold).astype(int)
        
        unique, counts = np.unique(y_target, return_counts=True)
        print(f"✅ Target predictions distribution: {dict(zip(unique, counts))}")
    except Exception as e:
        print(f"❌ Lỗi khi query target model: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Tìm normalization stats
    normalization_stats_path = None
    possible_stats_paths = [
        output_dir / "normalization_stats.npz",
        PROJECT_ROOT / "artifacts" / "targets" / f"{target_model_name}_normalization_stats.npz",
    ]
    
    for path in possible_stats_paths:
        if path.exists():
            normalization_stats_path = str(path.resolve())
            break
    
    # Load surrogate model
    print(f"\n🔄 Đang load surrogate model...")
    try:
        surrogate_predict = load_LightGBM_surrogate_model(
            model_path=str(surrogate_model_path),
            normalization_stats_path=normalization_stats_path,
            feature_dim=len(feature_cols),
            threshold=extraction_threshold
        )
        print(f"✅ Đã load surrogate model")
    except Exception as e:
        print(f"❌ Lỗi khi load surrogate model: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Đánh giá
    print(f"\n🔄 Đang đánh giá...")
    try:
        metrics = evaluate_model_similarity(
            target_oracle,
            surrogate_predict,
            X_test,
            y_target,
            y_true=None,  # Không có ground truth trên dataset trung lập
            model_name=output_dir.name,
            extraction_threshold=extraction_threshold,
            find_optimal_threshold=True
        )
        
        # Thêm metadata
        metrics["extraction_metadata"] = metadata
        metrics["neutral_dataset"] = neutral_dataset_path
        metrics["evaluation_samples"] = len(X_test)
        
        return metrics
    except Exception as e:
        print(f"❌ Lỗi khi đánh giá: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Tự động đánh giá surrogate models sau extraction trên dataset trung lập"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Đường dẫn đến output directory (mặc định: output/)"
    )
    parser.add_argument(
        "--neutral_dataset",
        type=str,
        default=None,
        help="Đường dẫn đến dataset trung lập (mặc định: train_ember_2018_v2_features_label_minus1.parquet)"
    )
    parser.add_argument(
        "--target_model_name",
        type=str,
        default="LEE",
        help="Tên target model (mặc định: LEE)"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=10000,
        help="Số lượng samples tối đa để đánh giá (mặc định: 10000)"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Đánh giá một model cụ thể (nếu không chỉ định, sẽ scan tất cả)"
    )
    
    args = parser.parse_args()
    
    # Xử lý output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
        if not output_dir.is_absolute():
            output_dir = PROJECT_ROOT / output_dir
    else:
        output_dir = PROJECT_ROOT / "output"
    
    if not output_dir.exists():
        print(f"❌ Không tìm thấy output directory: {output_dir}")
        return
    
    # Xử lý neutral dataset path
    if args.neutral_dataset:
        neutral_dataset_path = args.neutral_dataset
        if not Path(neutral_dataset_path).is_absolute():
            neutral_dataset_path = str((PROJECT_ROOT / neutral_dataset_path).resolve())
    else:
        neutral_dataset_path = str(
            PROJECT_ROOT / "data" / "ember_2018_v2" / "train" / "train_ember_2018_v2_features_label_minus1.parquet"
        )
    
    if not Path(neutral_dataset_path).exists():
        print(f"❌ Không tìm thấy dataset trung lập: {neutral_dataset_path}")
        return
    
    print("=" * 80)
    print("ĐÁNH GIÁ TỰ ĐỘNG SURROGATE MODELS TRÊN DATASET TRUNG LẬP")
    print("=" * 80)
    print(f"Output Directory: {output_dir}")
    print(f"Neutral Dataset: {neutral_dataset_path}")
    print(f"Target Model: {args.target_model_name}")
    print(f"Max Samples: {args.max_samples}")
    print("=" * 80)
    
    # Tìm surrogate models
    if args.model_path:
        # Đánh giá một model cụ thể
        model_path = Path(args.model_path)
        if not model_path.is_absolute():
            model_path = PROJECT_ROOT / model_path
        
        if not model_path.exists():
            print(f"❌ Không tìm thấy model: {model_path}")
            return
        
        model_output_dir = model_path.parent
        surrogate_models = [(model_path, model_output_dir)]
    else:
        # Scan tất cả models
        print(f"\n🔍 Đang tìm surrogate models trong {output_dir}...")
        surrogate_models = find_surrogate_models(output_dir)
        print(f"✅ Tìm thấy {len(surrogate_models)} surrogate model(s)")
    
    if len(surrogate_models) == 0:
        print("❌ Không tìm thấy surrogate model nào!")
        return
    
    # Đánh giá từng model
    all_results = []
    for model_path, model_output_dir in surrogate_models:
        result = evaluate_single_model(
            surrogate_model_path=model_path,
            output_dir=model_output_dir,
            neutral_dataset_path=neutral_dataset_path,
            target_model_name=args.target_model_name,
            max_samples=args.max_samples
        )
        
        if result is not None:
            all_results.append(result)
            
            # Lưu kết quả vào output directory của model
            result_path = model_output_dir / "neutral_evaluation_results.json"
            with open(result_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"\n✅ Đã lưu kết quả vào: {result_path}")
            
            # Tạo report markdown
            report_path = model_output_dir / "neutral_evaluation_report.md"
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(f"# Báo Cáo Đánh Giá Trên Dataset Trung Lập\n\n")
                f.write(f"**Model**: {result['model_name']}\n\n")
                f.write(f"## Kết Quả\n\n")
                f.write(f"- **Agreement**: {result['agreement']:.4f} ({result['agreement']*100:.2f}%)\n")
                if result.get('optimal_threshold') is not None:
                    f.write(f"- **Optimal Threshold**: {result['optimal_threshold']:.4f}\n")
                    f.write(f"- **Agreement với Optimal Threshold**: {result.get('optimal_agreement', result['agreement']):.4f}\n")
                if result.get('extraction_threshold') is not None:
                    f.write(f"- **Threshold từ Extraction**: {result['extraction_threshold']:.4f}\n")
                    if result.get('agreement_with_extraction_threshold') is not None:
                        f.write(f"- **Agreement với Threshold từ Extraction**: {result['agreement_with_extraction_threshold']:.4f}\n")
                f.write(f"- **AUC**: {result.get('auc', 'N/A')}\n")
                f.write(f"- **F1 (agreement)**: {result.get('f1_agreement', 0):.4f}\n")
                f.write(f"\n## So Sánh\n\n")
                if result.get('extraction_metadata'):
                    metadata = result['extraction_metadata']
                    if metadata.get('optimal_threshold'):
                        f.write(f"- **Threshold trong extraction**: {metadata['optimal_threshold']:.4f}\n")
                        if result.get('optimal_threshold'):
                            diff = abs(result['optimal_threshold'] - metadata['optimal_threshold'])
                            f.write(f"- **Chênh lệch threshold**: {diff:.4f}\n")
            print(f"✅ Đã tạo report: {report_path}")
    
    # Tạo tổng hợp
    if len(all_results) > 1:
        print(f"\n{'='*80}")
        print("TỔNG HỢP KẾT QUẢ")
        print(f"{'='*80}\n")
        
        summary_data = []
        for result in all_results:
            summary_data.append({
                "model_name": result['model_name'],
                "agreement": result['agreement'],
                "optimal_threshold": result.get('optimal_threshold'),
                "extraction_threshold": result.get('extraction_threshold'),
                "threshold_difference": abs(result.get('optimal_threshold', 0) - result.get('extraction_threshold', 0)) if result.get('optimal_threshold') and result.get('extraction_threshold') else None,
                "auc": result.get('auc'),
                "f1_agreement": result.get('f1_agreement', 0)
            })
        
        df = pd.DataFrame(summary_data)
        print(df.to_string(index=False))
        
        # Lưu tổng hợp
        summary_path = output_dir / "neutral_evaluation_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
        print(f"\n✅ Đã lưu tổng hợp vào: {summary_path}")
    
    print(f"\n{'='*80}")
    print("✅ HOÀN THÀNH ĐÁNH GIÁ")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

