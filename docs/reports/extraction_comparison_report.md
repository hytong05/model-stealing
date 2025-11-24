# Báo Cáo So Sánh Các Surrogate Models

## Tóm Tắt

Đã chạy extraction với 3 cấu hình khác nhau về số lượng queries.

## Bảng So Sánh

| Cấu hình | Queries | Labels | Accuracy | Balanced Acc | F1 | Agreement | Threshold | AUC |
|----------|---------|--------|----------|--------------|----|-----------|-----------|-----|
| max_queries_10000_H5 | 9,000 | 12,000 | 0.5022 | 0.4998 | 0.0000 | 0.9998 | 0.100 | 0.6630 |
| max_queries_5000_H5 | 4,000 | 7,000 | 0.5025 | 0.5000 | 0.0010 | 0.9995 | 0.750 | 0.5746 |
| max_queries_2000_H5 | 1,000 | 4,000 | 0.5025 | 0.5000 | 0.0010 | 0.9995 | 0.150 | 0.6365 |

## Chi Tiết Từng Cấu Hình

### Max Queries 10000 H5

**Mô tả:** Tổng 10,000 queries (2000 queries/round × 5 rounds)

- Query batch: 2,000
- Số rounds: 5
- Queries dự kiến: 10,000
- Queries thực tế: 9,000
- Tổng labels sử dụng (bao gồm seed+val): 12,000

**Metrics cuối cùng:**

- Accuracy: 0.5022 (50.22%)
- Balanced Accuracy: 0.4998 (49.98%) [quan trọng với class imbalance]
- F1-score: 0.0000
- Optimal Threshold: 0.1000
- Agreement: 0.9998 (99.98%)
- AUC: 0.6630
- Precision: 0.0000
- Recall: 0.0000

**Files:**

- Metrics CSV: `/home/hytong/Documents/model_extraction_malware/output/max_queries_10000_H5/extraction_metrics.csv`
- Surrogate model: `/home/hytong/Documents/model_extraction_malware/output/max_queries_10000_H5/surrogate_model.h5`
- Output directory: `/home/hytong/Documents/model_extraction_malware/output/max_queries_10000_H5`

### Max Queries 5000 H5

**Mô tả:** Tổng 5,000 queries (1250 queries/round × 4 rounds)

- Query batch: 1,250
- Số rounds: 4
- Queries dự kiến: 5,000
- Queries thực tế: 4,000
- Tổng labels sử dụng (bao gồm seed+val): 7,000

**Metrics cuối cùng:**

- Accuracy: 0.5025 (50.25%)
- Balanced Accuracy: 0.5000 (50.00%) [quan trọng với class imbalance]
- F1-score: 0.0010
- Optimal Threshold: 0.7500
- Agreement: 0.9995 (99.95%)
- AUC: 0.5746
- Precision: 0.5000
- Recall: 0.0005

**Files:**

- Metrics CSV: `/home/hytong/Documents/model_extraction_malware/output/max_queries_5000_H5/extraction_metrics.csv`
- Surrogate model: `/home/hytong/Documents/model_extraction_malware/output/max_queries_5000_H5/surrogate_model.h5`
- Output directory: `/home/hytong/Documents/model_extraction_malware/output/max_queries_5000_H5`

### Max Queries 2000 H5

**Mô tả:** Tổng 2,000 queries (2000 queries/round × 1 round)

- Query batch: 2,000
- Số rounds: 1
- Queries dự kiến: 2,000
- Queries thực tế: 1,000
- Tổng labels sử dụng (bao gồm seed+val): 4,000

**Metrics cuối cùng:**

- Accuracy: 0.5025 (50.25%)
- Balanced Accuracy: 0.5000 (50.00%) [quan trọng với class imbalance]
- F1-score: 0.0010
- Optimal Threshold: 0.1500
- Agreement: 0.9995 (99.95%)
- AUC: 0.6365
- Precision: 0.5000
- Recall: 0.0005

**Files:**

- Metrics CSV: `/home/hytong/Documents/model_extraction_malware/output/max_queries_2000_H5/extraction_metrics.csv`
- Surrogate model: `/home/hytong/Documents/model_extraction_malware/output/max_queries_2000_H5/surrogate_model.h5`
- Output directory: `/home/hytong/Documents/model_extraction_malware/output/max_queries_2000_H5`

