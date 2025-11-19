# Báo Cáo So Sánh Các Surrogate Models

## Tóm Tắt

Đã chạy extraction với 3 cấu hình khác nhau về số lượng queries.

## Bảng So Sánh

| Cấu hình | Queries | Labels | Accuracy | Balanced Acc | F1 | Agreement | Threshold | AUC |
|----------|---------|--------|----------|--------------|----|-----------|-----------|-----|
| max_queries_10000_H5 | 9,000 | 12,000 | 0.5360 | 0.5379 | 0.6625 | 0.9910 | 0.110 | 0.5453 |
| max_queries_5000_H5 | 4,000 | 7,000 | 0.5585 | 0.5590 | 0.6012 | 0.9647 | 0.100 | 0.5946 |
| max_queries_2000_H5 | 1,000 | 4,000 | 0.6120 | 0.6114 | 0.5535 | 0.9557 | 0.110 | 0.6332 |

## Chi Tiết Từng Cấu Hình

### Max Queries 10000 H5

**Mô tả:** Tổng 10,000 queries (2000 queries/round × 5 rounds)

- Query batch: 2,000
- Số rounds: 5
- Queries dự kiến: 10,000
- Queries thực tế: 9,000
- Tổng labels sử dụng (bao gồm seed+val): 12,000

**Metrics cuối cùng:**

- Accuracy: 0.5360 (53.60%)
- Balanced Accuracy: 0.5379 (53.79%) [quan trọng với class imbalance]
- F1-score: 0.6625
- Optimal Threshold: 0.1100
- Agreement: 0.9910 (99.10%)
- AUC: 0.5453
- Precision: 0.5191
- Recall: 0.9156

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

- Accuracy: 0.5585 (55.85%)
- Balanced Accuracy: 0.5590 (55.90%) [quan trọng với class imbalance]
- F1-score: 0.6012
- Optimal Threshold: 0.1000
- Agreement: 0.9647 (96.47%)
- AUC: 0.5946
- Precision: 0.5459
- Recall: 0.6688

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

- Accuracy: 0.6120 (61.20%)
- Balanced Accuracy: 0.6114 (61.14%) [quan trọng với class imbalance]
- F1-score: 0.5535
- Optimal Threshold: 0.1100
- Agreement: 0.9557 (95.58%)
- AUC: 0.6332
- Precision: 0.6474
- Recall: 0.4834

**Files:**

- Metrics CSV: `/home/hytong/Documents/model_extraction_malware/output/max_queries_2000_H5/extraction_metrics.csv`
- Surrogate model: `/home/hytong/Documents/model_extraction_malware/output/max_queries_2000_H5/surrogate_model.h5`
- Output directory: `/home/hytong/Documents/model_extraction_malware/output/max_queries_2000_H5`

