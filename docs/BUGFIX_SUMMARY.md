# Tóm Tắt Các Vấn Đề Đã Sửa

## Vấn Đề Nghiêm Trọng: Oracle Query với Raw Data

### Mô Tả Vấn Đề
- **Vấn đề**: Oracle (target model) được query với **raw data** (chưa scale), nhưng target model được train với **scaled data**.
- **Hậu quả**: Oracle cho predictions hoàn toàn sai (0% agreement với ground truth khi query với raw data, nhưng 90% agreement khi query với scaled data).
- **Nguyên nhân**: Target model (`final_model.h5`) được train với dữ liệu đã normalize (RobustScaler + clip), nhưng code query oracle với raw data.

### Giải Pháp
1. **Scale data TRƯỚC KHI query oracle**:
   - Tạo scaler và fit trên seed+val+pool
   - Scale tất cả dữ liệu (eval, seed, val, pool) trước khi query oracle
   - Query oracle với dữ liệu đã scale

2. **Cập nhật code trong `extract_final_model.py`**:
   - Di chuyển việc scale data lên trước khi query oracle
   - Query oracle với `X_eval_s`, `X_seed_s`, `X_val_s`, `X_pool_s` (đã scale)
   - Train surrogate với dữ liệu đã scale (đã đúng từ trước)

### Thay Đổi Code
```python
# TRƯỚC (SAI):
y_eval = oracle(X_eval)  # Query với raw data
y_seed = oracle(X_seed)  # Query với raw data
y_val = oracle(X_val)    # Query với raw data
scaler = RobustScaler()
scaler.fit(...)
X_eval_s = _clip_scale(scaler, X_eval)  # Scale sau khi query

# SAU (ĐÚNG):
scaler = RobustScaler()
scaler.fit(...)
X_eval_s = _clip_scale(scaler, X_eval)  # Scale TRƯỚC
y_eval = oracle(X_eval_s)  # Query với scaled data
y_seed = oracle(X_seed_s)  # Query với scaled data
y_val = oracle(X_val_s)    # Query với scaled data
```

## Vấn Đề Phụ: KerasAttacker Hardcode Input Shape

### Mô Tả Vấn Đề
- **Vấn đề**: `KerasAttacker` hardcode `input_shape=(2381,)`, nhưng dataset có thể có số đặc trưng khác.
- **Hậu quả**: Nếu dataset có số đặc trưng khác 2381, model sẽ không khớp với dữ liệu.

### Giải Pháp
1. **Thêm parameter `input_shape` vào `KerasAttacker.__init__()`**:
   - Default là `(2381,)` để giữ backward compatibility
   - Cho phép truyền `input_shape` từ bên ngoài

2. **Cập nhật `extract_final_model.py`**:
   - Truyền `input_shape=(feature_dim,)` vào `KerasAttacker`
   - Đảm bảo model khớp với số đặc trưng thực tế của dataset

### Thay Đổi Code
```python
# TRƯỚC:
class KerasAttacker(AbstractAttacker):
    def __init__(self, early_stopping=30, seed=42, mc=False):
        self.model = create_dnn(seed=seed, input_shape=(2381,), mc=mc)

# SAU:
class KerasAttacker(AbstractAttacker):
    def __init__(self, early_stopping=30, seed=42, mc=False, input_shape=(2381,)):
        self.model = create_dnn(seed=seed, input_shape=input_shape, mc=mc)

# Sử dụng:
attacker = KerasAttacker(early_stopping=10, seed=seed, input_shape=(feature_dim,))
```

## Kết Quả Mong Đợi

Sau khi sửa các vấn đề trên:
1. **Oracle predictions đúng**: Oracle sẽ cho predictions đúng vì được query với dữ liệu đã scale (giống như khi train).
2. **Agreement tăng**: Agreement giữa surrogate và target sẽ tăng đáng kể (từ ~0% lên ~90%+).
3. **Accuracy tăng**: Surrogate model sẽ học được patterns đúng từ target model.
4. **Model khớp với dữ liệu**: KerasAttacker sẽ tự động khớp với số đặc trưng thực tế của dataset.

## Các File Đã Sửa

1. **`scripts/extract_final_model.py`**:
   - Scale data trước khi query oracle
   - Query oracle với scaled data
   - Truyền `input_shape` vào `KerasAttacker`

2. **`src/attackers/__init__.py`**:
   - Thêm parameter `input_shape` vào `KerasAttacker.__init__()`

## Kiểm Tra

Để kiểm tra xem vấn đề đã được sửa:
1. Chạy `scripts/extract_final_model.py` với dataset nhỏ
2. Kiểm tra oracle predictions distribution (nên có cả 0 và 1, không phải tất cả là 1)
3. Kiểm tra agreement giữa surrogate và target (nên > 80%)
4. Kiểm tra accuracy của surrogate model (nên tăng theo số lượng queries)

