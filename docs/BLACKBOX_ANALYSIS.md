# Phân Tích Black Box Compliance

## Tóm Tắt

Đã kiểm tra và cải thiện code để đảm bảo tính chất **black box (hộp đen)** trong model extraction attack.

## Black Box Attack Requirements

### ✅ Attacker CHỈ được biết:
1. **Tên model** (hoặc API endpoint) - ✅ OK
2. **Raw features** (có thể query) - ✅ OK
3. **Predictions** (0 hoặc 1, hoặc probabilities) - ✅ OK

### ❌ Attacker KHÔNG được biết:
1. **Model type** (Keras vs LightGBM) - ✅ Đã ẩn
2. **Normalization statistics** - ✅ Đã ẩn
3. **Model architecture** - ✅ Đã ẩn
4. **Model parameters/weights** - ✅ Đã ẩn
5. **Training data của target model** - ✅ OK (attacker không có access)
6. **Feature importance** - ✅ OK
7. **Internal workings của model** - ✅ Đã ẩn

## Kiểm Tra Chi Tiết

### 1. Oracle Client Interface

**Trước (Vi Phạm):**
```python
# Attacker phải biết model_type và normalization_stats_path
oracle_client = LocalOracleClient(
    model_type="lgb",  # ❌ Attacker biết model type
    model_path=...,
    normalization_stats_path=...,  # ❌ Attacker biết normalization stats
)
```

**Sau (Black Box Compliant):**
```python
# Attacker chỉ cần tên model
oracle_client = create_oracle_from_name(
    model_name="LEE",  # ✅ Chỉ cần tên model
    feature_dim=2381,
)
# Tự động detect model_type, load normalization_stats, etc.
```

### 2. BlackBoxOracleClient

**Thuộc tính Attacker Có Thể Truy Cập:**
- ✅ `model_name`: Tên model (OK)
- ✅ `predict(X)`: Predict binary labels (OK)
- ✅ `predict_proba(X)`: Predict probabilities (OK)
- ✅ `supports_probabilities()`: Kiểm tra hỗ trợ probabilities (OK)
- ✅ `get_required_feature_dim()`: Lấy số features yêu cầu (OK - có thể biết qua API docs)

**Thuộc tính Attacker KHÔNG Thể Truy Cập:**
- ✅ `model_type`: Đã ẩn
- ✅ `model_path`: Đã ẩn
- ✅ `normalization_stats_path`: Đã ẩn
- ⚠️ `_oracle`: Internal (trong Python vẫn có thể truy cập, nhưng trong thực tế oracle chạy trên server riêng)

### 3. Ground Truth Labels từ Train Data

**✅ Hợp Lệ:**
- Attacker sử dụng ground truth labels từ thief dataset
- Attacker kiểm soát thief dataset, có thể có labels của chính data của mình
- Đây không vi phạm black box assumption

### 4. Logging

**⚠️ Lưu Ý:**
- Một số thông tin (model_type, model_path) vẫn được log trong `extract_final_model.py`
- Trong black box attack thực tế, attacker không nên thấy những log này
- **Giải pháp**: Logging chỉ nên ở phía nhà cung cấp (server), không expose cho attacker

## Kết Quả Kiểm Tra

```
✅ model_type: Đã ẩn
✅ model_path: Đã ẩn
✅ normalization_stats_path: Đã ẩn
✅ Oracle client chỉ expose predict() và predict_proba()
✅ Attacker chỉ cần tên model để tạo oracle client
```

## Cải Tiến Đã Thực Hiện

### 1. Tạo BlackBoxOracleClient
- Wrap `LocalOracleClient` để ẩn implementation details
- Chỉ expose `predict()` và `predict_proba()`
- Tự động detect model type, load normalization stats

### 2. Cập Nhật create_oracle_from_name
- Trả về `BlackBoxOracleClient` (mặc định `blackbox=True`)
- Tự động detect mọi thứ, ẩn khỏi attacker

### 3. Cập Nhật Attack Script
- Khi dùng `model_name`, tự động dùng `BlackBoxOracleClient`
- Attacker không cần biết `model_type` hay `normalization_stats_path`

## Lưu Ý Quan Trọng

### 1. Python Limitation
- Trong Python, không có true private attributes
- Attacker vẫn có thể truy cập `_oracle` (nhưng không nên)
- **Trong thực tế**: Oracle client chạy trên server riêng, attacker không có access đến code

### 2. Logging
- Logging có thể leak thông tin
- **Giải pháp**: Tắt logging hoặc chỉ log ở phía server
- Attacker không nên thấy logs về model_type, model_path, etc.

### 3. API Design
- Trong black box attack thực tế, oracle nên là một API endpoint
- Attacker chỉ có thể query qua HTTP/REST API
- Không có access đến code, logs, hay file system

## Kết Luận

✅ **Đã cải thiện** để đảm bảo tính chất black box:
- Attacker chỉ cần tên model
- Oracle client tự động xử lý mọi thứ
- Implementation details được ẩn

⚠️ **Vẫn còn một số điểm**:
- Logging có thể leak thông tin (có thể tắt)
- Python không có true private (nhưng trong thực tế oracle chạy trên server riêng)

💡 **Trong thực tế**:
- Oracle client nên chạy trên server riêng (của nhà cung cấp)
- Attacker chỉ có thể query qua API
- Không có access đến code hay logs


