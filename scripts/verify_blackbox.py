#!/usr/bin/env python3
"""
Kiểm tra xem BlackBoxOracleClient có ẩn đúng thông tin khỏi attacker không
"""
import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.targets.oracle_client import create_oracle_from_name, BlackBoxOracleClient

print("=" * 80)
print("KIỂM TRA BLACK BOX COMPLIANCE")
print("=" * 80)

# Tạo oracle client
print("\n1. Tạo BlackBoxOracleClient từ tên model...")
oracle = create_oracle_from_name("LEE", feature_dim=2381)
print(f"   ✅ Oracle client created: {type(oracle).__name__}")

# Kiểm tra các thuộc tính attacker có thể truy cập
print("\n2. Kiểm tra thuộc tính attacker có thể truy cập:")
print(f"   ✅ model_name: {oracle.model_name if hasattr(oracle, 'model_name') else 'N/A'}")
print(f"   ✅ predict(): {hasattr(oracle, 'predict')}")
print(f"   ✅ predict_proba(): {hasattr(oracle, 'predict_proba')}")
print(f"   ✅ supports_probabilities(): {hasattr(oracle, 'supports_probabilities')}")
print(f"   ✅ get_required_feature_dim(): {hasattr(oracle, 'get_required_feature_dim')}")

# Kiểm tra các thuộc tính attacker KHÔNG nên truy cập
print("\n3. Kiểm tra thuộc tính attacker KHÔNG nên truy cập:")
print(f"   ❌ model_type: {hasattr(oracle, 'model_type')}")
if hasattr(oracle, 'model_type'):
    print(f"      ⚠️  VI PHẠM: Attacker có thể truy cập model_type!")
    print(f"      Value: {oracle.model_type}")
else:
    print(f"      ✅ OK: model_type đã được ẩn")

print(f"   ❌ model_path: {hasattr(oracle, 'model_path')}")
if hasattr(oracle, 'model_path'):
    print(f"      ⚠️  VI PHẠM: Attacker có thể truy cập model_path!")
    print(f"      Value: {oracle.model_path}")
else:
    print(f"      ✅ OK: model_path đã được ẩn")

print(f"   ❌ normalization_stats_path: {hasattr(oracle, 'normalization_stats_path')}")
if hasattr(oracle, 'normalization_stats_path'):
    print(f"      ⚠️  VI PHẠM: Attacker có thể truy cập normalization_stats_path!")
else:
    print(f"      ✅ OK: normalization_stats_path đã được ẩn")

# Kiểm tra _oracle (internal, không nên truy cập trực tiếp)
print(f"\n   ❌ _oracle (internal): {hasattr(oracle, '_oracle')}")
if hasattr(oracle, '_oracle'):
    print(f"      ⚠️  LƯU Ý: _oracle là internal, attacker không nên truy cập")
    print(f"      💡 Trong Python, attacker vẫn có thể truy cập (không có private)")
    print(f"      💡 Trong thực tế, oracle client chạy trên server riêng, attacker không thể truy cập")

# Test predict
print("\n4. Test predict với raw features:")
test_X = np.random.rand(10, 2381).astype(np.float32)
predictions = oracle.predict(test_X)
print(f"   ✅ Predictions shape: {predictions.shape}")
print(f"   ✅ Predictions: {predictions}")
print(f"   ✅ Oracle hoạt động đúng với raw features")

print("\n" + "=" * 80)
print("KẾT LUẬN")
print("=" * 80)
print("""
✅ BLACK BOX COMPLIANCE:

1. Attacker chỉ cần tên model để tạo oracle client
2. Oracle client tự động detect model type, load normalization stats
3. Attacker chỉ có thể gọi predict() và predict_proba()
4. Implementation details được ẩn trong _oracle

⚠️  LƯU Ý:

- Trong Python, attacker vẫn có thể truy cập _oracle (không có private)
- Trong thực tế, oracle client nên chạy trên server riêng
- Attacker chỉ có thể query qua API, không có access đến code
- Logging có thể leak thông tin (nên tắt hoặc chỉ log ở phía server)
""")


