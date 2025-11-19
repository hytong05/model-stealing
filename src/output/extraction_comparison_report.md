# Báo Cáo So Sánh Các Surrogate Models

## Tóm Tắt

Đã chạy extraction với 3 cấu hình khác nhau về số lượng queries.

## Bảng So Sánh

| Cấu hình | Tổng Queries | Labels Sử Dụng | Accuracy | Agreement | AUC | Precision | Recall | F1 |
|----------|--------------|----------------|----------|-----------|-----|-----------|--------|----|
| max_queries_10000 | 10,000 | 12,000 | 0.9855 | 0.9855 | 0.9601 | 0.9923 | 0.9929 | 0.9926 |
| max_queries_5000 | 5,000 | 7,000 | 0.9898 | 0.9898 | 0.9904 | 0.9911 | 0.9985 | 0.9948 |
| max_queries_2000 | 2,000 | 4,000 | 0.9858 | 0.9858 | 0.8798 | 0.9859 | 0.9997 | 0.9928 |

## Chi Tiết Từng Cấu Hình

### Max Queries 10000

**Mô tả:** Tối đa 10,000 queries (2000 queries/round × 5 rounds)

- Query batch: 2,000
- Số rounds: 5
- Tổng queries: 10,000
- Tổng labels sử dụng: 12,000

**Metrics cuối cùng:**

- Accuracy: 0.9855 (98.55%)
- Agreement: 0.9855 (98.55%)
- AUC: 0.9601
- Precision: 0.9923
- Recall: 0.9929
- F1-score: 0.9926

**Files:**

- Metrics CSV: `/home/hytong/Documents/model_extraction_malware/src/output/max_queries_10000/extraction_metrics.csv`
- Surrogate model: `/home/hytong/Documents/model_extraction_malware/src/output/max_queries_10000/surrogate_model.h5`
- Output directory: `/home/hytong/Documents/model_extraction_malware/src/output/max_queries_10000`

### Max Queries 5000

**Mô tả:** Tối đa 5,000 queries (1000 queries/round × 5 rounds)

- Query batch: 1,000
- Số rounds: 5
- Tổng queries: 5,000
- Tổng labels sử dụng: 7,000

**Metrics cuối cùng:**

- Accuracy: 0.9898 (98.98%)
- Agreement: 0.9898 (98.98%)
- AUC: 0.9904
- Precision: 0.9911
- Recall: 0.9985
- F1-score: 0.9948

**Files:**

- Metrics CSV: `/home/hytong/Documents/model_extraction_malware/src/output/max_queries_5000/extraction_metrics.csv`
- Surrogate model: `/home/hytong/Documents/model_extraction_malware/src/output/max_queries_5000/surrogate_model.h5`
- Output directory: `/home/hytong/Documents/model_extraction_malware/src/output/max_queries_5000`

### Max Queries 2000

**Mô tả:** Tối đa 2,000 queries (2000 queries/round × 1 round)

- Query batch: 2,000
- Số rounds: 1
- Tổng queries: 2,000
- Tổng labels sử dụng: 4,000

**Metrics cuối cùng:**

- Accuracy: 0.9858 (98.58%)
- Agreement: 0.9858 (98.58%)
- AUC: 0.8798
- Precision: 0.9859
- Recall: 0.9997
- F1-score: 0.9928

**Files:**

- Metrics CSV: `/home/hytong/Documents/model_extraction_malware/src/output/max_queries_2000/extraction_metrics.csv`
- Surrogate model: `/home/hytong/Documents/model_extraction_malware/src/output/max_queries_2000/surrogate_model.h5`
- Output directory: `/home/hytong/Documents/model_extraction_malware/src/output/max_queries_2000`

