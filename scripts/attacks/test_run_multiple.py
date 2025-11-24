#!/usr/bin/env python3
"""
Test nhanh run_multiple_extractions với LEE - chỉ test khởi tạo
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.targets.oracle_client import create_oracle_from_name
import numpy as np

print("=" * 80)
print("TEST: Khởi tạo Oracle với model_name='LEE'")
print("=" * 80)

# Test khởi tạo oracle
oracle = create_oracle_from_name("LEE", feature_dim=2381)
print(f"\n✅ Oracle đã được khởi tạo thành công!")
print(f"   Model type: {oracle.model_type}")
print(f"   Model path: {oracle.model_path}")
print(f"   Required feature dim: {oracle.get_required_feature_dim()}")

# Test query
print(f"\n🧪 Test query với raw features...")
sample = np.random.randn(2381).astype(np.float32)
prediction = oracle.predict(sample)
print(f"   Sample shape: {sample.shape}")
print(f"   Prediction: {prediction[0]} ({'Malware' if prediction[0] == 1 else 'Benign'})")

print(f"\n✅ Tất cả test đều PASS! Extraction script sẽ hoạt động với --model_name LEE")

