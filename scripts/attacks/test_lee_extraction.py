#!/usr/bin/env python3
"""
Test script để kiểm tra extraction với LEE model sử dụng model_name
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.attacks.extract_final_model import run_extraction

# Test với config nhỏ
output_dir = PROJECT_ROOT / "output" / "test_lee_extraction"
output_dir.mkdir(parents=True, exist_ok=True)

# Ưu tiên file mới trong ember_2018_v2
train_parquet_new = PROJECT_ROOT / "data" / "ember_2018_v2" / "train" / "train_ember_2018_v2_features_label_other.parquet"
train_parquet_old = PROJECT_ROOT / "data" / "train_ember_2018_v2_features_label_other.parquet"
train_parquet = str(train_parquet_new if train_parquet_new.exists() else train_parquet_old)

test_parquet_new = PROJECT_ROOT / "data" / "ember_2018_v2" / "test" / "test_ember_2018_v2_features_label_other.parquet"
test_parquet_old = PROJECT_ROOT / "data" / "test_ember_2018_v2_features_label_other.parquet"
test_parquet = str(test_parquet_new if test_parquet_new.exists() else test_parquet_old)

print("=" * 80)
print("TEST EXTRACTION VỚI LEE MODEL (SỬ DỤNG MODEL_NAME)")
print("=" * 80)

try:
    summary = run_extraction(
        output_dir=output_dir,
        train_parquet=train_parquet,
        test_parquet=test_parquet,
        seed=42,
        seed_size=100,  # Nhỏ để test nhanh
        val_size=50,
        eval_size=200,
        query_batch=50,  # Nhỏ để test nhanh
        num_rounds=1,  # Chỉ 1 round để test nhanh
        num_epochs=5,  # Nhỏ để test nhanh
        model_name="LEE",  # Chỉ cần tên model!
        attacker_type="lgb",
    )
    
    print("\n✅ Extraction hoàn tất!")
    print(f"   Oracle source: {summary.get('oracle_source')}")
    print(f"   Model file: {summary.get('model_file_name')}")
    if summary.get("metrics"):
        final_metrics = summary["metrics"][-1]
        print(f"   Final accuracy: {final_metrics.get('surrogate_acc', 0):.4f}")
        print(f"   Final agreement: {final_metrics.get('agreement_with_target', 0):.4f}")
    
except Exception as e:
    print(f"\n❌ Lỗi: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

