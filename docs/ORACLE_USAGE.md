# Hướng Dẫn Sử Dụng Oracle - Cực Kỳ Đơn Giản

## Tổng Quan

Oracle client đã được đơn giản hóa tối đa. **Attacker chỉ cần tên model** và gửi raw features, oracle sẽ tự động xử lý mọi thứ.

## Cách Sử Dụng Đơn Giản Nhất

### 1. Chỉ cần tên model

```python
from src.targets.oracle_client import create_oracle_from_name

# Khởi tạo oracle - chỉ cần tên model!
oracle = create_oracle_from_name("LEE")  # hoặc "CEE", "CSE", "LSE"

# Query với raw features
sample = np.random.randn(2381).astype(np.float32)
prediction = oracle.predict(sample)  # Trả về 0 hoặc 1

print(f"Prediction: {prediction[0]}")  # 0 = Benign, 1 = Malware
```

### 2. Oracle tự động xử lý

Oracle sẽ tự động:
- ✅ Tìm model file (`.h5` hoặc `.lgb`)
- ✅ Detect model type (Keras hay LightGBM)
- ✅ Tìm normalization stats
- ✅ Normalize features
- ✅ Align feature dimensions
- ✅ Trả về binary prediction

**Attacker không cần biết:**
- ❌ Model type (h5 hay lgb)
- ❌ Đường dẫn model file
- ❌ Normalization stats
- ❌ Preprocessing steps

## Các Model Hỗ Trợ

| Tên Model | Type | File | Normalization Stats |
|-----------|------|------|---------------------|
| CEE | Keras | `CEE.h5` | `CEE_normalization_stats.npz` |
| LEE | LightGBM | `LEE.lgb` | `LEE_normalization_stats.npz` |
| CSE | Keras | `CSE.h5` | (tùy chọn) |
| LSE | LightGBM | `LSE.lgb` | `LSE_normalization_stats.npz` |

## Ví Dụ Đầy Đủ

```python
import numpy as np
from src.targets.oracle_client import create_oracle_from_name

# 1. Khởi tạo oracle
oracle = create_oracle_from_name("LEE")

# 2. Query một sample
sample = np.random.randn(2381).astype(np.float32)
prediction = oracle.predict(sample)
print(f"Prediction: {prediction[0]}")

# 3. Query batch
batch = np.random.randn(10, 2381).astype(np.float32)
predictions = oracle.predict(batch)
print(f"Predictions: {predictions}")

# 4. Query với probabilities (nếu cần)
if oracle.supports_probabilities():
    probs = oracle.predict_proba(batch)
    print(f"Probabilities: {probs}")
```

## Sử Dụng Trong Script

### Test Oracle Query

```bash
# Chỉ cần tên model - không cần model-type hay model-path
python config/test_oracle_query.py \
    --parquet-path data/test_ember_2018_v2_features_label_other.parquet \
    --model-name LEE \
    --max-samples 5000
```

### Trong Attack Script

```python
from src.targets.oracle_client import create_oracle_from_name

# Trong attack script
oracle = create_oracle_from_name("LEE")

# Query samples
for sample in samples:
    label = oracle.predict(sample)
    # Sử dụng label để train surrogate model
```

## Lợi Ích

1. **Đơn giản**: Chỉ cần tên model
2. **Tự động**: Oracle tự detect và xử lý mọi thứ
3. **Linh hoạt**: Hoạt động với cả Keras và LightGBM
4. **Gọn gàng**: Không cần biết chi tiết về preprocessing

## Lưu Ý

- Models phải được đặt trong `artifacts/targets/`
- Normalization stats (nếu có) phải cùng tên với model + `_normalization_stats.npz`
- Với LightGBM, normalization stats là **bắt buộc**
- Với Keras, normalization stats là **tùy chọn** (nếu có sẽ được dùng)

## Xem Thêm

- `examples/ultra_simple_oracle.py` - Ví dụ cực kỳ đơn giản
- `examples/simple_oracle_query.py` - Ví dụ đầy đủ hơn
- `config/test_oracle_query.py` - Script test oracle

