# Black Box Attack Compliance

## Tóm Tắt

Đã cải thiện code để đảm bảo tính chất **black box (hộp đen)** trong model extraction attack.

## Black Box Attack Requirements

### Attacker CHỈ được biết:
1. **Tên model** (hoặc API endpoint)
2. **Raw features** (có thể query)
3. **Predictions** (0 hoặc 1, hoặc probabilities nếu API cho phép)

### Attacker KHÔNG được biết:
1. ❌ Model type (Keras vs LightGBM)
2. ❌ Normalization statistics
3. ❌ Model architecture
4. ❌ Model parameters/weights
5. ❌ Training data của target model
6. ❌ Feature importance
7. ❌ Internal workings của model

### Oracle Client (của nhà cung cấp):
- ✅ Tự động detect model type
- ✅ Tự động load normalization stats
- ✅ Tự động xử lý preprocessing
- ✅ Chỉ expose `predict()` và `predict_proba()`
- ✅ Ẩn tất cả implementation details

## Cải Tiến Đã Thực Hiện

### 1. Tạo BlackBoxOracleClient

**File:** `src/targets/oracle_client.py`

```python
class BlackBoxOracleClient(BaseOracleClient):
    """
    Black Box Oracle Client - Ẩn hoàn toàn implementation details khỏi attacker.
    
    Attacker chỉ cần:
    - Tên model
    - Raw features
    - Nhận predictions
    """
    
    def __init__(self, model_name: str, ...):
        # Tự động detect mọi thứ, ẩn khỏi attacker
        self._oracle = create_oracle_from_name(...)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        # Chỉ expose predict(), ẩn implementation
        return self._oracle.predict(X)
```

### 2. Cập Nhật create_oracle_from_name

**File:** `src/targets/oracle_client.py`

- Trả về `BlackBoxOracleClient` thay vì `LocalOracleClient`
- Ẩn `model_type`, `normalization_stats_path` khỏi attacker
- Tự động detect mọi thứ

### 3. Cập Nhật Attack Script

**File:** `scripts/attacks/extract_final_model.py`

- Khi dùng `model_name`, tự động dùng `BlackBoxOracleClient`
- Attacker không cần biết `model_type` hay `normalization_stats_path`
- Chỉ cần tên model và raw features

## So Sánh

### Trước (Vi Phạm Black Box):

```python
# Attacker phải biết model_type và normalization_stats_path
oracle_client = LocalOracleClient(
    model_type="lgb",  # ❌ Attacker biết model type
    model_path=...,
    normalization_stats_path=...,  # ❌ Attacker biết normalization stats
    ...
)
```

### Sau (Black Box Compliant):

```python
# Attacker chỉ cần tên model
oracle_client = create_oracle_from_name(
    model_name="LEE",  # ✅ Chỉ cần tên model
    feature_dim=2381,
)
# Tự động detect model_type, load normalization_stats, etc.
```

## Lưu Ý

### 1. Ground Truth Labels từ Train Data
- ✅ **Hợp lệ**: Attacker sử dụng ground truth labels từ thief dataset
- 💡 Attacker kiểm soát thief dataset, có thể có labels của chính data của mình
- 💡 Đây không vi phạm black box assumption

### 2. get_required_feature_dim()
- ⚠️ **Có thể hợp lệ**: Trong black box attack thực tế, attacker có thể biết input size
- 💡 Thông qua API documentation hoặc trial-and-error
- 💡 Nhưng không nên biết model architecture hay normalization

### 3. Logging/Debugging
- ⚠️ **Chỉ để logging**: Một số thông tin (model_type, model_path) vẫn được log
- 💡 Trong black box attack thực tế, attacker không nên thấy những log này
- 💡 Có thể tắt logging hoặc chỉ log ở phía nhà cung cấp

## Kết Luận

✅ **Đã cải thiện** để đảm bảo tính chất black box:
- Attacker chỉ cần tên model
- Oracle client tự động xử lý mọi thứ
- Ẩn implementation details

⚠️ **Vẫn còn một số điểm**:
- Logging có thể leak thông tin (có thể tắt)
- `get_required_feature_dim()` có thể leak thông tin (có thể ẩn)

💡 **Trong thực tế**:
- Oracle client nên chạy trên server riêng (của nhà cung cấp)
- Attacker chỉ có thể query qua API
- Không có access đến code hay logs


