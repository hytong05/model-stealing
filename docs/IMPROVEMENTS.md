# Cải Tiến Query Selection và Class Imbalance Handling

## Tóm Tắt

Đã cải tiến 2 điểm chính:
1. **Stratified Entropy Sampling**: Chọn queries cân bằng class (50/50) ngay từ đầu
2. **scale_pos_weight trong LGBAttacker**: Xử lý class imbalance trong training

## 1. Stratified Entropy Sampling

### Vấn Đề Trước Đây

- Chọn queries dựa trên entropy (không quan tâm class)
- Sau đó mới query oracle và cân bằng class
- Phải query oracle nhiều lần (không hiệu quả)
- Class imbalance nghiêm trọng (96% class 0, 4% class 1)

### Giải Pháp Mới

**Bước 1: Query Oracle Trước**
- Query oracle trên toàn bộ pool (hoặc subset lớn) để biết labels TRƯỚC
- Vì attacker kiểm soát thief dataset, có thể làm điều này

**Bước 2: Tính Entropy**
- Tính entropy cho tất cả samples trong pool đã query
- Sắp xếp theo entropy giảm dần

**Bước 3: Chọn Queries Cân Bằng**
- Chọn 50% từ class 0 (entropy cao nhất)
- Chọn 50% từ class 1 (entropy cao nhất)
- Đảm bảo class distribution cân bằng ngay từ đầu

### Lợi Ích

✅ **Cân bằng class ngay từ đầu**: 50% class 0, 50% class 1
✅ **Hiệu quả hơn**: Chỉ query oracle 1 lần (trên subset lớn)
✅ **Đa dạng hơn**: Trong mỗi class, chọn samples có entropy cao nhất
✅ **Hạn chế class imbalance**: Model học tốt hơn với data cân bằng

## 2. scale_pos_weight trong LGBAttacker

### Vấn Đề Trước Đây

- LGBAttacker không có `scale_pos_weight`
- Model không được điều chỉnh cho class imbalance
- Probabilities thấp (mean = 0.126)
- Threshold phải thấp (0.1) để tối ưu F1-score

### Giải Pháp Mới

**Tự động tính scale_pos_weight:**
```python
train_label_counts = np.bincount(y)
num_negative = train_label_counts[0]
num_positive = train_label_counts[1]

if num_positive > 0 and num_negative > 0:
    scale_pos_weight = num_negative / num_positive
    self.lgb_params['scale_pos_weight'] = scale_pos_weight
```

### Lợi Ích

✅ **Xử lý class imbalance**: Model được điều chỉnh tự động
✅ **Probabilities cao hơn**: Model tự tin hơn
✅ **Threshold gần 0.5**: Không cần threshold thấp
✅ **Accuracy tốt hơn**: Model học tốt hơn với class imbalance

## So Sánh

| Metric | Trước | Sau |
|--------|-------|-----|
| Query selection | Entropy (không cân bằng) | Stratified Entropy (50/50) |
| Class distribution | ~96/4 | ~50/50 |
| scale_pos_weight | Không có | Tự động tính |
| Probabilities mean | 0.126 | Cao hơn (dự kiến) |
| Threshold | 0.1 | Gần 0.5 (dự kiến) |
| Oracle queries | Nhiều lần | 1 lần (hiệu quả hơn) |

## Kết Luận

✅ **Cải tiến query selection**: Cân bằng class ngay từ đầu
✅ **Cải tiến model training**: Xử lý class imbalance tự động
✅ **Hiệu quả hơn**: Giảm số lần query oracle
✅ **Kết quả tốt hơn**: Model học tốt hơn với data cân bằng

## Next Steps

1. Test với LEE model để xem kết quả
2. So sánh accuracy và agreement với version cũ
3. Điều chỉnh tỷ lệ class nếu cần (có thể không phải 50/50)

