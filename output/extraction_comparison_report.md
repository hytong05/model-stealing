# Báo Cáo So Sánh Các Surrogate Models

## Tóm Tắt

Đã chạy extraction với 3 cấu hình khác nhau về số lượng queries.

## Bảng So Sánh

| Cấu hình | Queries | Labels | Accuracy | Balanced Acc | F1 | Agreement | Threshold | AUC |
|----------|---------|--------|----------|--------------|----|-----------|-----------|-----|
| LEE-somlap-dualDNN-10000 | 10,000 | 12,000 | 0.6947 | 0.7246 | 0.6757 | 0.9990 | 0.380 | 0.8467 |
| LEE-somlap-dualDNN-5000 | 5,000 | 7,000 | 0.7262 | 0.7386 | 0.6843 | 0.9972 | 0.410 | 0.8129 |
| LEE-somlap-dualDNN-2000 | 2,000 | 4,000 | 0.7212 | 0.7421 | 0.6905 | 0.9988 | 0.130 | 0.8054 |

## Chi Tiết Từng Cấu Hình

### Lee-Somlap-Dualdnn-10000

**Mô tả:** Tổng 10,000 queries (2000 queries/round × 5 rounds)

- Query batch: 2,000
- Số rounds: 5
- Queries dự kiến: 10,000
- Queries thực tế: 10,000
- Ghi chú queries: on_target
- Tổng labels sử dụng (bao gồm seed+val): 12,000

**Metrics cuối cùng:**

- Accuracy: 0.6947 (69.47%)
- Balanced Accuracy: 0.7246 (72.46%) [quan trọng với class imbalance]
- F1-score: 0.6757
- Optimal Threshold: 0.3800
- Agreement: 0.9990 (99.90%)
- AUC: 0.8467
- Precision: 0.5628
- Recall: 0.8452

**Files:**

- Metrics CSV: `/home/hytong/Documents/model_extraction_malware/output/LEE-somlap-dualDNN-10000/extraction_metrics.csv`
- Surrogate model: `/home/hytong/Documents/model_extraction_malware/output/LEE-somlap-dualDNN-10000/surrogate_model.h5`
- Output directory: `/home/hytong/Documents/model_extraction_malware/output/LEE-somlap-dualDNN-10000`

### Lee-Somlap-Dualdnn-5000

**Mô tả:** Tổng 5,000 queries (1250 queries/round × 4 rounds)

- Query batch: 1,250
- Số rounds: 4
- Queries dự kiến: 5,000
- Queries thực tế: 5,000
- Ghi chú queries: on_target
- Tổng labels sử dụng (bao gồm seed+val): 7,000

**Metrics cuối cùng:**

- Accuracy: 0.7262 (72.62%)
- Balanced Accuracy: 0.7386 (73.86%) [quan trọng với class imbalance]
- F1-score: 0.6843
- Optimal Threshold: 0.4100
- Agreement: 0.9972 (99.72%)
- AUC: 0.8129
- Precision: 0.6044
- Recall: 0.7887

**Files:**

- Metrics CSV: `/home/hytong/Documents/model_extraction_malware/output/LEE-somlap-dualDNN-5000/extraction_metrics.csv`
- Surrogate model: `/home/hytong/Documents/model_extraction_malware/output/LEE-somlap-dualDNN-5000/surrogate_model.h5`
- Output directory: `/home/hytong/Documents/model_extraction_malware/output/LEE-somlap-dualDNN-5000`

### Lee-Somlap-Dualdnn-2000

**Mô tả:** Tổng 2,000 queries (2000 queries/round × 1 round)

- Query batch: 2,000
- Số rounds: 1
- Queries dự kiến: 2,000
- Queries thực tế: 2,000
- Ghi chú queries: on_target
- Tổng labels sử dụng (bao gồm seed+val): 4,000

**Metrics cuối cùng:**

- Accuracy: 0.7212 (72.12%)
- Balanced Accuracy: 0.7421 (74.21%) [quan trọng với class imbalance]
- F1-score: 0.6905
- Optimal Threshold: 0.1300
- Agreement: 0.9988 (99.88%)
- AUC: 0.8054
- Precision: 0.5929
- Recall: 0.8266

**Files:**

- Metrics CSV: `/home/hytong/Documents/model_extraction_malware/output/LEE-somlap-dualDNN-2000/extraction_metrics.csv`
- Surrogate model: `/home/hytong/Documents/model_extraction_malware/output/LEE-somlap-dualDNN-2000/surrogate_model.h5`
- Output directory: `/home/hytong/Documents/model_extraction_malware/output/LEE-somlap-dualDNN-2000`

