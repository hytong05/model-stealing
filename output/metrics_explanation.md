# Giải Thích Chi Tiết Các Thông Số Trong Bảng So Sánh

## Tổng Quan

Bảng so sánh đánh giá hiệu quả của các cấu hình model extraction attack khác nhau. Mỗi metric đo lường một khía cạnh khác nhau của quá trình tấn công và chất lượng surrogate model.

---

## 1. **Cấu hình (Configuration)**

**Định nghĩa:** Tên của cấu hình extraction attack, thể hiện số lượng queries tối đa được sử dụng.

**Ví dụ:**
- `max_queries_10000_H5`: Cấu hình với tối đa 10,000 queries, sử dụng model H5 (Keras)
- `max_queries_5000_H5`: Cấu hình với tối đa 5,000 queries
- `max_queries_2000_H5`: Cấu hình với tối đa 2,000 queries

**Ý nghĩa:** Giúp phân biệt các thí nghiệm khác nhau và so sánh hiệu quả theo số lượng queries.

---

## 2. **Queries**

**Định nghĩa:** Số lượng queries thực tế được gửi đến target model (oracle) trong quá trình active learning.

**Công thức:** `Queries = Tổng số samples được query từ oracle (không tính seed và validation set)`

**Ví dụ:**
- `max_queries_10000_H5`: 9,000 queries (dự kiến 10,000 nhưng thực tế chỉ có 9,000)
- `max_queries_5000_H5`: 4,000 queries
- `max_queries_2000_H5`: 1,000 queries

**Ý nghĩa:**
- **Quan trọng trong model extraction:** Số queries càng nhiều, attacker càng có nhiều thông tin về target model
- **Chi phí:** Mỗi query là một lần tương tác với target model (có thể tốn kém hoặc bị phát hiện)
- **Hiệu quả:** Cần cân bằng giữa số queries và chất lượng extraction

**Lưu ý:** Queries thực tế có thể thấp hơn dự kiến do pool data cạn kiệt hoặc class balancing.

---

## 3. **Labels**

**Định nghĩa:** Tổng số labels (nhãn) được sử dụng để train surrogate model, bao gồm:
- **Seed set:** Dữ liệu ban đầu (thường 2,000 samples)
- **Validation set:** Dữ liệu validation (thường 1,000 samples)
- **Queries:** Labels từ oracle qua các rounds của active learning

**Công thức:** `Labels = Seed size + Val size + Queries thực tế`

**Ví dụ:**
- `max_queries_10000_H5`: 12,000 labels = 2,000 (seed) + 1,000 (val) + 9,000 (queries)
- `max_queries_5000_H5`: 7,000 labels = 2,000 + 1,000 + 4,000
- `max_queries_2000_H5`: 4,000 labels = 2,000 + 1,000 + 1,000

**Ý nghĩa:**
- **Tổng dữ liệu training:** Số labels càng nhiều, model có thể học tốt hơn
- **So sánh với Queries:** Labels = Queries + Seed + Val, nên luôn lớn hơn Queries

---

## 4. **Accuracy**

**Định nghĩa:** Độ chính xác của surrogate model khi so sánh với **ground truth labels** (nhãn thực tế của dữ liệu).

**Công thức:** 
```
Accuracy = (Số dự đoán đúng) / (Tổng số samples)
         = (TP + TN) / (TP + TN + FP + FN)
```

Trong đó:
- **TP (True Positive):** Dự đoán đúng là positive (malware)
- **TN (True Negative):** Dự đoán đúng là negative (benign)
- **FP (False Positive):** Dự đoán sai là positive (false alarm)
- **FN (False Negative):** Dự đoán sai là negative (bỏ sót malware)

**Ví dụ:**
- `max_queries_10000_H5`: 0.4338 = 43.38% accuracy
- `max_queries_5000_H5`: 0.4065 = 40.65% accuracy
- `max_queries_2000_H5`: 0.3950 = 39.50% accuracy

**Ý nghĩa:**
- **Đánh giá hiệu suất thực tế:** Accuracy đo lường khả năng phân loại đúng của surrogate model với ground truth
- **Thấp trong trường hợp này:** Các giá trị ~40% cho thấy:
  - Surrogate model không học tốt từ oracle
  - **HOẶC** Oracle (target model) không chính xác với ground truth
  - **HOẶC** Có class imbalance nghiêm trọng (cần xem Balanced Accuracy)

**Lưu ý quan trọng:**
- Accuracy được tính với **ground truth**, không phải với oracle labels
- Nếu oracle không chính xác, accuracy sẽ thấp dù agreement cao
- Accuracy thấp (~40%) nhưng Agreement cao (~90%) cho thấy: **Oracle không chính xác với ground truth**

---

## 5. **Balanced Accuracy**

**Định nghĩa:** Độ chính xác cân bằng, tính trung bình accuracy của từng class riêng biệt. Quan trọng khi có **class imbalance** (một class chiếm đa số).

**Công thức:**
```
Balanced Accuracy = (Sensitivity + Specificity) / 2
                  = (TP/(TP+FN) + TN/(TN+FP)) / 2
```

Trong đó:
- **Sensitivity (Recall):** Tỷ lệ dự đoán đúng positive trong số tất cả positive thực tế
- **Specificity:** Tỷ lệ dự đoán đúng negative trong số tất cả negative thực tế

**Ví dụ:**
- `max_queries_10000_H5`: 0.4320 = 43.20% balanced accuracy
- `max_queries_5000_H5`: 0.4049 = 40.49% balanced accuracy
- `max_queries_2000_H5`: 0.3935 = 39.35% balanced accuracy

**Ý nghĩa:**
- **Quan trọng với class imbalance:** Nếu một class chiếm 90% dữ liệu, accuracy thường sẽ cao nhưng không phản ánh đúng khả năng phân loại
- **So sánh với Accuracy:** 
  - Nếu Balanced Accuracy ≈ Accuracy: Không có class imbalance nghiêm trọng
  - Nếu Balanced Accuracy << Accuracy: Có class imbalance, model bias về class đa số
- **Trong trường hợp này:** Balanced Accuracy ≈ Accuracy (~40%), cho thấy không có class imbalance nghiêm trọng, nhưng model vẫn không học tốt

**Khi nào quan trọng:**
- Dữ liệu không cân bằng (ví dụ: 90% benign, 10% malware)
- Cần đánh giá công bằng cho cả 2 classes
- Accuracy thông thường có thể gây hiểu lầm

---

## 6. **F1 Score**

**Định nghĩa:** Harmonic mean của Precision và Recall, cân bằng giữa độ chính xác và độ bao phủ.

**Công thức:**
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

Trong đó:
- **Precision:** Tỷ lệ dự đoán positive đúng trong số tất cả dự đoán positive
- **Recall:** Tỷ lệ dự đoán đúng positive trong số tất cả positive thực tế

**Ví dụ:**
- `max_queries_10000_H5`: 0.1391 = 13.91% F1 score
- `max_queries_5000_H5`: 0.1194 = 11.94% F1 score
- `max_queries_2000_H5`: 0.1276 = 12.76% F1 score

**Ý nghĩa:**
- **Đánh giá tổng hợp:** F1 cân bằng giữa Precision và Recall
- **Thấp trong trường hợp này:** F1 ~12-14% rất thấp, cho thấy:
  - Precision thấp: Nhiều false positives
  - Recall thấp: Bỏ sót nhiều malware
  - Model không học tốt để phân loại chính xác

**Khi nào quan trọng:**
- Cần cân bằng giữa false positives và false negatives
- Trong malware detection: Cả 2 lỗi đều quan trọng (false alarm và bỏ sót malware)
- Khi có class imbalance, F1 thường tốt hơn accuracy

**Mối quan hệ:**
- F1 thấp → Precision hoặc Recall thấp (hoặc cả 2)
- Với Precision = 0.2855 và Recall = 0.0920 (max_queries_10000_H5):
  - F1 = 2 × (0.2855 × 0.0920) / (0.2855 + 0.0920) ≈ 0.1391 ✓

---

## 7. **Agreement**

**Định nghĩa:** Tỷ lệ đồng ý giữa predictions của surrogate model và predictions của target model (oracle). **Đây là metric quan trọng nhất trong model extraction attack.**

**Công thức:**
```
Agreement = (Số predictions giống nhau) / (Tổng số samples)
          = (Surrogate predictions == Oracle predictions).mean()
```

**Ví dụ:**
- `max_queries_10000_H5`: 0.9133 = 91.33% agreement
- `max_queries_5000_H5`: 0.9275 = 92.75% agreement
- `max_queries_2000_H5`: 0.9028 = 90.28% agreement

**Ý nghĩa:**
- **Mục tiêu của model extraction:** Agreement cao = Surrogate model đã học tốt để bắt chước target model
- **Không phải accuracy:** Agreement so sánh với oracle, không phải ground truth
- **Trong trường hợp này:** Agreement ~90% cao, nhưng Accuracy ~40% thấp → **Oracle không chính xác với ground truth**

**So sánh với Accuracy:**
- **Agreement cao + Accuracy thấp:** 
  - Surrogate học tốt từ oracle ✓
  - Nhưng oracle không chính xác với ground truth ✗
  - → Model extraction thành công, nhưng target model không tốt

- **Agreement cao + Accuracy cao:**
  - Surrogate học tốt từ oracle ✓
  - Oracle cũng chính xác với ground truth ✓
  - → Model extraction thành công và target model tốt

**Khi nào quan trọng:**
- **Đánh giá thành công của attack:** Agreement là metric chính để đo lường extraction
- **Không cần ground truth:** Agreement chỉ cần oracle predictions
- **Thực tế:** Trong model extraction, attacker không có ground truth, chỉ có oracle responses

---

## 8. **Threshold (Optimal Threshold)**

**Định nghĩa:** Ngưỡng tối ưu để chuyển đổi probabilities thành binary predictions (0 hoặc 1).

**Công thức:** Threshold được tìm bằng cách tối ưu F1-score trên validation set:
```
For each threshold in [0.1, 0.2, ..., 0.9]:
    predictions = (probabilities >= threshold).astype(int)
    f1 = calculate_f1_score(ground_truth, predictions)
    
optimal_threshold = threshold với F1 cao nhất
```

**Ví dụ:**
- Tất cả cấu hình: 0.100 = 10% threshold

**Ý nghĩa:**
- **Thấp (0.1):** Model cần probability chỉ 10% để dự đoán là positive (malware)
- **Cho thấy:**
  - Model có xu hướng dự đoán probabilities thấp
  - Có thể do class imbalance (nhiều negative, ít positive)
  - Hoặc model không tự tin với predictions

**So sánh:**
- **Threshold = 0.5 (mặc định):** Cân bằng, dự đoán positive khi probability ≥ 50%
- **Threshold = 0.1 (trong trường hợp này):** Rất thấp, dự đoán positive khi probability ≥ 10%
  - → Model cần rất ít confidence để dự đoán malware
  - → Có thể do model không học tốt hoặc có bias

**Khi nào quan trọng:**
- **Tối ưu performance:** Threshold ảnh hưởng đến Precision, Recall, F1
- **Class imbalance:** Threshold thấp thường tốt hơn khi có nhiều negative
- **Cost-sensitive:** Nếu false positive đắt hơn false negative, tăng threshold

---

## 9. **AUC (Area Under ROC Curve)**

**Định nghĩa:** Diện tích dưới đường cong ROC (Receiver Operating Characteristic), đo lường khả năng phân biệt giữa 2 classes của model.

**Công thức:**
```
AUC = ∫ ROC_curve d(False Positive Rate)
```

ROC curve vẽ:
- **X-axis:** False Positive Rate (FPR) = FP / (FP + TN)
- **Y-axis:** True Positive Rate (TPR/Recall) = TP / (TP + FN)

**Giá trị:**
- **AUC = 1.0:** Perfect classifier (phân biệt hoàn hảo)
- **AUC = 0.5:** Random classifier (không tốt hơn đoán ngẫu nhiên)
- **AUC < 0.5:** Tệ hơn random (có thể đảo ngược predictions)

**Ví dụ:**
- `max_queries_10000_H5`: 0.4258 = 42.58% AUC
- `max_queries_5000_H5`: 0.2613 = 26.13% AUC
- `max_queries_2000_H5`: 0.2634 = 26.34% AUC

**Ý nghĩa:**
- **Rất thấp trong trường hợp này:** AUC ~26-43% < 50% (random)
- **Cho thấy:**
  - Model không phân biệt được giữa malware và benign
  - Có thể do:
    - Model không học tốt từ oracle
    - Oracle không chính xác với ground truth
    - Dữ liệu không đủ hoặc không đại diện

**So sánh với các metrics khác:**
- **AUC thấp + Accuracy thấp:** Model không học tốt
- **AUC thấp + Agreement cao:** 
  - Surrogate bắt chước oracle tốt
  - Nhưng oracle không phân biệt tốt giữa 2 classes
  - → Oracle có vấn đề, không phải surrogate

**Khi nào quan trọng:**
- **Đánh giá khả năng phân biệt:** AUC không phụ thuộc vào threshold
- **Class imbalance:** AUC tốt hơn accuracy khi có imbalance
- **So sánh models:** AUC cho phép so sánh models độc lập với threshold

---

## Tổng Kết và Phân Tích

### Pattern Quan Sát:

1. **Agreement cao (~90%):** Surrogate model học tốt để bắt chước oracle
2. **Accuracy thấp (~40%):** Oracle không chính xác với ground truth
3. **AUC rất thấp (~26-43%):** Oracle không phân biệt tốt giữa malware và benign
4. **F1 thấp (~12-14%):** Model không phân loại tốt với ground truth
5. **Threshold thấp (0.1):** Model cần ít confidence để dự đoán positive

### Kết Luận:

**Model extraction attack thành công** (Agreement ~90%), nhưng:
- **Target model (oracle) không chính xác** với ground truth
- **Surrogate đã học tốt** để bắt chước oracle, nhưng vì oracle không tốt nên surrogate cũng không tốt
- **Cần kiểm tra:** Oracle accuracy vs ground truth để xác nhận

### Khuyến Nghị:

1. **Kiểm tra Oracle:** Tính oracle accuracy vs ground truth để xác nhận oracle có vấn đề
2. **Cải thiện Oracle:** Nếu oracle không chính xác, cần retrain target model
3. **Đánh giá Extraction:** Agreement cao cho thấy extraction thành công, nhưng cần oracle tốt để có surrogate tốt
4. **Tăng Queries:** Có thể thử tăng queries để xem có cải thiện không (nhưng dựa vào kết quả, có vẻ không phải vấn đề về số lượng queries)

---

## Tài Liệu Tham Khảo

- **Accuracy:** Tỷ lệ dự đoán đúng tổng thể
- **Balanced Accuracy:** Accuracy cân bằng cho từng class
- **Precision:** Độ chính xác của dự đoán positive
- **Recall:** Độ bao phủ của dự đoán positive
- **F1 Score:** Cân bằng giữa Precision và Recall
- **AUC:** Khả năng phân biệt giữa 2 classes
- **Agreement:** Độ đồng ý giữa surrogate và oracle (metric chính trong model extraction)


